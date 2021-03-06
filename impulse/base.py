import numpy as np
from impulse.random_nums import rng
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm

from loguru import logger

import ray

from impulse.mhsampler import MHSampler, RayMHSampler # , GlobalFunctionActor
from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData

from numba import njit
# from numba.typed import List


@njit
def update_chains(res, chain, lnlike_arr, lnprob_arr, accept_arr, low_idx, high_idx):
    for ii in np.arange(len(res)):
        (chain[low_idx:high_idx, :, ii],
         lnlike_arr[low_idx:high_idx, ii],
         lnprob_arr[low_idx:high_idx, ii],
         accept_arr[low_idx:high_idx, ii]) = res[ii]
    return chain, lnlike_arr, lnprob_arr, accept_arr


class PTSampler():
    def __init__(self, lnlike, lnprior, x0, num_samples=1_000_000, buf_size=50_000, ntemps=8, ncores=1,
                 tmin=1, tmax=None, tstep=None, swap_count=100, ladder=None, tinf=False, adapt=True, thin=10,
                 amweight=30, scamweight=15, deweight=50, adapt_t0=100, adapt_nu=10, loop_iterations=1000,
                 groups=None, cov=None, outdir='./chains', temp_dir=None, ret_chain=False, resume=False):
        """
        :param lnlike: log-likelihood function
        :param lnprior: log-prior function
        :param x0: initial position
        :param buf_size: size of buffer [multiple of 10_000 >> ACL]
        :param ntemps: number of temperatures
        :param ncores: number of cores
        :param tmin: minimum temperature
        :param tmax: maximum temperature
        :param tstep: temperature step
        :param swap_count: number of iterations between swap proposals chains
        :param ladder: temperature ladder
        :param tinf: if True, ladder is set to infinite temperature
        :param adapt: if True, temperature ladder is adapted
        :param adapt_t0: initial temperature
        :param adapt_nu: temperature adaptation rate
        :param loop_iterations: number of iterations between loops
        :param groups: list of groups of parameters to be considered as independent
        :param cov: sample covariance matrix of the parameters
        :param outdir: output directory
        :param temp_dir: temperature directory
        :param ret_chain: if True, return chain
        :param resume: if True, resume from previous run
        """
        self.lnlike = lnlike
        self.lnprior = lnprior
        self.x0 = x0
        self.num_samples = int(num_samples)
        if ncores > 1:
            self.ndim = len(x0[0])
        else:
            self.ndim = len(x0) if type(x0) != list else len(x0[0])
        self.buf_size = buf_size
        self.ntemps = ntemps
        self.ncores = ncores
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.swap_count = swap_count
        self.ladder = ladder
        self.tinf = tinf
        self.adapt = adapt
        self.thin = thin
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.loop_iterations = loop_iterations
        self.groups = groups
        self.cov = cov
        self.outdir = outdir
        self.temp_dir = temp_dir
        self.ret_chain = ret_chain
        self.sample_count = 0
        self.amweight = amweight
        self.scamweight = scamweight
        self.deweight = deweight
        self.resume = resume

        if ray.is_initialized():
            ray.shutdown()

        if self.ncores > 1 and not ray.is_initialized():
            ray.init(num_cpus=self.ncores)
            # ray.init(local_mode=True)

        if ntemps > 1:  # PTSampler
            self._init_ptswap()
            self._init_pt_saves()
            self._init_pt_proposals()
            self._init_pt_sampler()

        else:  # MHSampler
            self._init_mh_save()
            self._init_mh_jumps()
            self._init_mh_sampler()

        # resume from previous run
        if resume and all([self.saves[ii].exists() for ii in range(self.ntemps)]):
            self._pt_resume()
        elif not resume:
            pass
        else:
            logger.exception('One or more chain files were not found. Starting from scratch.')


    def _init_mh_save(self):
        self.save = SaveData(outdir=self.outdir, filename='/chain_1.txt')


    def _init_mh_jumps(self):
        self.mix = JumpProposals(self.ndim, buf_size=self.buf_size)
        self.mix.add_jump(am, self.amweight)
        self.mix.add_jump(scam, self.scamweight)
        self.mix.add_jump(de, self.deweight)


    def _init_mh_sampler(self):
        if self.ret_chain:
            self.full_chain = np.zeros((self.num_samples, self.ndim))
        self.sampler = MHSampler(self.x0, self.lnlike, self.lnprior, self.mix, iterations=self.loop_iterations)


    def _mh_sampler_step(self):
        self.chain, self.like, self.prob, self.accept = self.sampler.sample()
        self.save(self.chain, self.like, self.prob, self.accept)
        if self.ret_chain:
            self.full_chain[self.sample_count:self.sample_count + self.loop_iterations, :] = self.chain
        self.mix.recursive_update(self.sample_count, self.chain)
        self.sample_count += self.loop_iterations


    def _init_ptswap(self):
        """
        Initialize PTSwap
        """
        self.ptswap = PTSwap(self.ndim, self.ntemps, tmin=self.tmin, tmax=self.tmax, tstep=self.tstep,
                             tinf=self.tinf, adapt_t0=self.adapt_t0, adapt_nu=self.adapt_nu, ladder=self.ladder)


    def _init_pt_saves(self):
        """
        Initialize save objects
        """
        self.filenames = ['/chain_{}.txt'.format(ii + 1) for ii in range(self.ntemps)]  # temps change (label by chain number)
        self.saves = [SaveData(outdir=self.outdir, filename=self.filenames[ii], resume=self.resume, thin=self.thin) for ii in range(self.ntemps)]


    def _init_pt_proposals(self):
        """
        Initialize JumpProposals (one for each temperature)
        """
        self.mixes = []
        for ii in range(self.ntemps):
            self.mixes.append(JumpProposals(self.ndim, buf_size=self.buf_size, groups=self.groups, cov=self.cov))
            self.mixes[ii].add_jump(am, self.amweight)
            self.mixes[ii].add_jump(scam, self.scamweight)
            self.mixes[ii].add_jump(de, self.deweight)


    def _init_pt_sampler(self):
        """
        Initialize sampler (one for each temperature) and chains
        """
        if self.ret_chain:
            self.full_chain = np.zeros((self.num_samples, self.ndim, self.ntemps))
        self.chain = np.zeros((self.loop_iterations, self.ndim, self.ntemps))
        self.lnlike_arr = np.zeros((self.loop_iterations, self.ntemps))
        self.lnprob_arr = np.zeros((self.loop_iterations, self.ntemps))
        self.accept_arr = np.zeros((self.loop_iterations, self.ntemps))
        if self.ncores > 1:
            # global_functions = GlobalFunctionActor.remote(self.lnlike, self.lnprior, self.x0[0])
            # self.samplers = [RayMHSampler.remote(self.x0[ii], global_functions, self.mixes[ii],
            #                  iterations=self.swap_count, init_temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]
            ss = SeedSequence()

            # Spawn off ncores child SeedSequences to pass to child processes.
            child_seeds = ss.spawn(self.ncores)
            streams = [default_rng(s) for s in child_seeds]
            self.samplers = [RayMHSampler.remote(self.x0[ii], self.lnlike, self.lnprior, self.mixes[ii], streams[ii],
                             iterations=self.swap_count, init_temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]
        else:
            self.samplers = [MHSampler(self.x0[ii], self.lnlike, self.lnprior, self.mixes[ii], rng,
                             iterations=self.swap_count, init_temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]


    def _pt_resume(self):
        """
        Resume from previous run
        """
        logger.info("Resuming from previous run.")
        with open(self.outdir + '/chain_0.txt', 'r') as f:
            full_chain = np.loadtxt(f)
        self.sample_count = full_chain.shape[0]
        self.sample_count = self.sample_count - self.sample_count % self.loop_iterations
        full_chain_cut = full_chain[self.sample_count:, :, :]

        for ii in range(self.ntemps):
            self.mixes[ii].recursive_update(self.sample_count, full_chain_cut[:, :, ii])


    def _ptstep(self):
        """
        Perform PT step
        """
        if self.ncores > 1:
            # res = self.pool.map(call_step, self.samplers)
            res = ray.get([sampler.sample.remote() for sampler in self.samplers])
            # logger.debug(len(res[0]))
        else:
            res = list(map(lambda sampler: sampler.sample(), self.samplers))
        low_idx = self.swap_tot
        high_idx = self.swap_tot + self.swap_count
        self.chain, self.lnlike_arr, self.lnprob_arr, self.accept_arr = update_chains(res, self.chain, self.lnlike_arr,
                                                                                      self.lnprob_arr, self.accept_arr,
                                                                                      low_idx, high_idx)
        self.sample_count += self.swap_count
        swap_idx = self.sample_count % self.loop_iterations - 1

        self.chain, self.lnlike_arr, self.logprob_arr = self.ptswap(self.chain, self.lnlike_arr, self.lnprob_arr, swap_idx)
        if self.adapt:
            self.ptswap.adapt_ladder()

        if self.ncores > 1:
            [self.samplers[ii].set_x0.remote(self.chain[swap_idx, :, ii], self.logprob_arr[swap_idx, ii], temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]
        else:
            [self.samplers[ii].set_x0(self.chain[swap_idx, :, ii], self.logprob_arr[swap_idx, ii], temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]


    def _save_step(self):
        for ii in range(self.ntemps):
            length = self.chain.shape[0]
            self.saves[ii](self.chain[:, :, ii], self.lnlike_arr[:, ii], self.lnprob_arr[:, ii],
                           np.repeat(self.ptswap.ladder[ii], length),
                           np.repeat(np.hstack([self.ptswap.compute_accept_ratio(), 0])[ii], length))
            self.mixes[ii].recursive_update(self.sample_count, self.chain[:, :, ii])


    def add_jump(self, jump, weight):
        try:
            for ii in range(self.ntemps):
                self.mixes[ii].add_jump(jump, weight)
        except AttributeError:
            self.mix.add_jump(jump, weight)


    def set_ntemps(self, ntemps):
        self.ntemps = ntemps
        self._init_ptswap()
        self._init_pt_saves()
        self._init_pt_proposals()
        self._init_pt_sampler()


    def sample(self):
        # set up count and iterations between loops
        if self.ntemps > 1:
            # PTSampler
            for _ in tqdm(range(0, int(self.num_samples), self.loop_iterations)):
                self.swap_tot = 0  # count the samples between swaps
                for _ in range(self.loop_iterations // self.swap_count):  # swap loops
                    self._ptstep()
                    self.swap_tot += self.swap_count
                self._save_step()
                if self.ret_chain:
                    self.full_chain[self.sample_count - self.loop_iterations:self.sample_count, :, :] = self.chain
                # self.sample_count += self.loop_iterations
        else:
            # MHSampler
            for _ in tqdm(range(0, int(self.num_samples), self.loop_iterations)):
                self._mh_sampler_step()

        if ray.is_initialized():
            ray.shutdown()
        if self.ret_chain:
            return self.full_chain

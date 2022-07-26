import numpy as np
from tqdm import tqdm

from loguru import logger
import ray

from multiprocessing import Pool
# from pathos.pools import ProcessPool as Pool
# from ray.util.multiprocessing import Pool

from impulse.mhsampler import MHSampler, mh_sample_step  #  parallel_mh_sample_step
from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData

from numba import njit


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

        if self.ncores > 1 and not ray.is_initialized():
            ray.init(num_cpus=self.ncores)

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
        self.samplers = [MHSampler(self.x0[ii], self.lnlike, self.lnprior, self.mixes[ii],
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
            res = ray.get([parallel_mh_sample_step.remote(sampler.lnlike_fn, sampler.lnprior_fn, sampler.prop_fn, sampler.x0,
                            sampler.temp, sampler.iterations, sampler.lnprob0,
                            sampler.chain, sampler.lnlike, sampler.lnprob, sampler.accept_rate) for sampler in self.samplers])
        else:
            # res = list(map(lambda sampler: sampler.sample(), self.samplers))
            res = list(map(lambda sampler: sampler.sample(), self.samplers))
        low_idx = self.swap_tot
        high_idx = self.swap_tot + self.swap_count
        self.chain, self.lnlike_arr, self.lnprob_arr, self.accept_arr = update_chains(res, self.chain, self.lnlike_arr,
                                                                                      self.lnprob_arr, self.accept_arr,
                                                                                      low_idx, high_idx)

        swap_idx = self.samplers[0].num_samples % self.loop_iterations - 1
        self.chain, self.lnlike_arr, self.logprob_arr = self.ptswap(self.chain, self.lnlike_arr, self.lnprob_arr, swap_idx)
        if self.adapt:
            self.ptswap.adapt_ladder()
        self.saves[0].save_swap_data(self.ptswap)
        [self.samplers[ii].set_x0(self.chain[swap_idx, :, ii], self.logprob_arr[swap_idx, ii], temp=self.ptswap.ladder[ii]) for ii in range(self.ntemps)]


    def _save_step(self):
        for ii in range(self.ntemps):
            self.saves[ii](self.chain[:, :, ii], self.lnlike_arr[:, ii], self.lnprob_arr[:, ii], self.accept_arr[:, ii])
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
                    self.full_chain[self.sample_count:self.sample_count + self.loop_iterations, :, :] = self.chain
                self.sample_count += self.loop_iterations
        else:
            # MHSampler
            for _ in tqdm(range(0, int(self.num_samples), self.loop_iterations)):
                self._mh_sampler_step()

        ray.shutdown()
        if self.ret_chain:
            return self.full_chain


# # TODO: make this a class and combine these two functions with cleaner counting built in
# def pt_sample(lnlike, lnprior, ndim, x0, num_samples=1_000_000, buf_size=50_000,
#               amweight=30, scamweight=15, deweight=50, ntemps=2, ncores=1, tmin=1, tmax=None, tstep=None,
#               swap_count=100, ladder=None, tinf=False, adapt=True, adapt_t0=100, adapt_nu=10,
#               loop_iterations=1000, outdir='./chains', temp_dir=None, ret_chain=False):

#     ptswap = PTSwap(ndim, ntemps, tmin=tmin, tmax=tmax, tstep=tstep,
#                     tinf=tinf, adapt_t0=adapt_t0,
#                     adapt_nu=adapt_nu, ladder=ladder)
#     saves = [SaveData(outdir=outdir, filename='/chain_{}.txt'.format(ptswap.ladder[ii])) for ii in range(ntemps)]

#     # make empty full chain
#     if ret_chain:
#         full_chain = np.zeros((num_samples, ndim, ntemps))
#     chain = np.zeros((loop_iterations, ndim, ntemps))
#     lnlike_arr = np.zeros((loop_iterations, ntemps))
#     lnprob_arr = np.zeros((loop_iterations, ntemps))
#     accept_arr = np.zeros((loop_iterations, ntemps))
#     # set up proposals
#     mixes = []
#     for ii in range(ntemps):
#         mixes.append(JumpProposals(ndim, buf_size=buf_size))
#         mixes[ii].add_jump(am, amweight)
#         mixes[ii].add_jump(scam, scamweight)
#         mixes[ii].add_jump(de, deweight)

#     # make set of samplers (1 for each temp)
#     samplers = [MHSampler(x0[ii], lnlike, lnprior, mixes[ii], iterations=swap_count, init_temp=ptswap.ladder[ii]) for ii in range(ntemps)]

#     # set up count and iterations between loops
#     count = 0
#     for _ in tqdm(range(0, int(num_samples), loop_iterations)):
#         swap_tot = 0
#         for _ in range(loop_iterations // swap_count):
#             with Pool(nodes=min(ntemps, ncores)) as p:
#                 res = p.map(lambda sampler: sampler.sample(), samplers)
#             for ii in range(len(res)):
#                 (chain[swap_tot:swap_tot + swap_count, :, ii],
#                  lnlike_arr[swap_tot:swap_tot + swap_count, ii],
#                  lnprob_arr[swap_tot:swap_tot + swap_count, ii],
#                  accept_arr[swap_tot:swap_tot + swap_count, ii]) = res[ii]
#             swap_idx = samplers[0].num_samples % loop_iterations - 1
#             chain, lnlike_arr, logprob_arr = ptswap(chain, lnlike_arr, lnprob_arr, swap_idx)
#             if adapt:
#                 ptswap.adapt_ladder(temp_dir=temp_dir)
#             [samplers[ii].set_x0(chain[swap_idx, :, ii], logprob_arr[swap_idx, ii], temp=ptswap.ladder[ii]) for ii in range(ntemps)]
#             # print(samplers[0].x0)
#             swap_tot += swap_count
#         for ii in range(ntemps):
#             saves[ii](chain[:, :, ii], lnlike_arr[:, ii], lnprob_arr[:, ii], accept_arr[:, ii])
#             mixes[ii].recursive_update(count, chain[:, :, ii])
#         if ret_chain:
#             full_chain[count:count + loop_iterations, :, :] = chain
#         count += loop_iterations
#     if ret_chain:
#         return full_chain

# class Sampler():
#     def __init__(self, lnlike, lnprior, x0, num_samples=1_000_000, buf_size=50_000,
#                  amweight=30, scamweight=15, deweight=50, loop_iterations=1000,
#                  outdir='./chains', filename='/chain_1.txt', ret_chain=False):
#         self.lnlike = lnlike
#         self.lnprior = lnprior
#         self.x0 = x0 if type(x0) != list else x0[0]
#         self.num_samples = int(num_samples)
#         self.ndim = len(x0) if type(x0) != list else len(x0[0])
#         self.buf_size = buf_size
#         self.amweight = amweight
#         self.scamweight = scamweight
#         self.deweight = deweight
#         self.loop_iterations = loop_iterations
#         self.outdir = outdir
#         self.filename = filename
#         self.ret_chain = ret_chain

#         self._init_save()
#         self._init_jumps()
#         self._init_sampler()

#         if ret_chain:
#             self.full_chain = np.zeros((self.num_samples, self.ndim))

#         self.count = 0  # num of samples so far

        

#     def _init_save(self):
#         self.save = SaveData(outdir=self.outdir, filename=self.filename)

#     def _init_jumps(self):
#         self.mix = JumpProposals(self.ndim, buf_size=self.buf_size)
#         self.mix.add_jump(am, self.amweight)
#         self.mix.add_jump(scam, self.scamweight)
#         self.mix.add_jump(de, self.deweight)

#     def _init_sampler(self):
#         self.sampler = MHSampler(self.x0, self.lnlike, self.lnprior, self.mix, iterations=self.loop_iterations)
    
#     def _sampler_step(self):
#         self.chain, self.like, self.prob, self.accept = self.sampler.sample()
#         self.save(self.chain, self.like, self.prob, self.accept)
#         if self.ret_chain:
#             self.full_chain[self.count:self.count + self.loop_iterations, :] = self.chain
#         self.mix.recursive_update(self.count, self.chain)
#         self.count += self.loop_iterations

#     def sample(self):
#         for _ in tqdm(range(0, int(self.num_samples), self.loop_iterations)):
#             self._sampler_step()
#         if self.ret_chain:
#             return self.full_chain


# Methods borrowed from BilbyMCMC to aid in parallelization

def call_step(sampler):
    sampler = sampler.sample()
    return sampler


_likelihood = None
_priors = None
_search_parameter_keys = None
_use_ratio = False


def _initialize_global_variables(
    likelihood,
    priors,
    search_parameter_keys,
    use_ratio,
    ):
    """
    Store a global copy of the likelihood, priors, and search keys for
    multiprocessing.
    """
    global _likelihood
    global _priors
    global _search_parameter_keys
    global _use_ratio
    _likelihood = likelihood
    _priors = priors
    _search_parameter_keys = search_parameter_keys
    _use_ratio = use_ratio
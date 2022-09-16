import os
import numpy as np
from impulse.random_nums import rng
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm

from loguru import logger

from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData
from impulse.mhsampler import mh_step

# from multiprocessing.pool import Pool

# from numba.typed import List


def update_chains(res, chain, lnlike_arr, lnprob_arr, accept_arr, low_idx, high_idx):
    for ii in np.arange(len(res)):
        (chain[low_idx:high_idx, :, ii],
         lnlike_arr[low_idx:high_idx, ii],
         lnprob_arr[low_idx:high_idx, ii],
         accept_arr[low_idx:high_idx, ii]) = res[ii]
    return chain, lnlike_arr, lnprob_arr, accept_arr


class PTSampler():
    def __init__(self,
                ndim,
                logl,
                logp,
                ncores=1,
                ntemps=1,
                swap_steps=100,
                tmin=1,
                tmax=None,
                tstep=None,
                tinf=False,
                adapt=False,
                adapt_t0=100,
                adapt_nu=10,
                ladder=None,
                buf_size=50_000,
                mean=None,
                cov=None,
                groups=None,
                loglargs=[],
                loglkwargs={},
                logpargs=[],
                logpkwargs={},
                cov_update=1000,
                save_freq=1000,
                SCAMweight=30,
                AMweight=15,
                DEweight=50,
                outdir="./chains",
                resume=False,
                seed=None):

        # setup loglikelihood and logprior functions
        self.logl = _function_wrapper(logl, loglargs, loglkwargs)
        self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        # PTSwap parameters
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.tinf = tinf
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.ladder = ladder
        self.adapt = adapt
        self.outdir = outdir
        self.resume = resume
        self.ncores = ncores
        self.ndim = ndim
        self.ntemps = ntemps
        self.cov_update = cov_update
        self.save_freq = save_freq
        self.swap_steps = swap_steps

        # sample counter
        self.counter = 0

        # setup standard jump proposals
        self.mixes = []
        for ii in range(self.ntemps):
            self.mixes.append(JumpProposals(self.ndim, buf_size=buf_size, groups=groups, cov=cov, mean=mean))
            self.mixes[ii].add_jump(am, AMweight)
            self.mixes[ii].add_jump(scam, SCAMweight)
            self.mixes[ii].add_jump(de, DEweight)

        # setup random number generator
        stream = SeedSequence(seed)
        seeds = stream.generate_state(ntemps + 1)
        self.rng = [default_rng(s) for s in seeds]

        # setup parallel tempering
        self.ptswap = PTSwap(self.ndim, self.ntemps, self.rng[-1], tmin=self.tmin, tmax=self.tmax, tstep=self.tstep,
                             tinf=self.tinf, adapt_t0=self.adapt_t0, adapt_nu=self.adapt_nu, ladder=self.ladder)

    def save_state(self):
        pass

    def load_state(self):
        pass

    def sample(self, x0, num_samples, thin=1, ret_chain=False):
        if ret_chain:
            full_chain = np.zeros((num_samples, self.ndim, self.ntemps))

        # initial sample
        self.x0 = np.array(x0)  # (ntemps, ndim)
        self.lnlike0 = np.array([self.logl(self.x0[jj]) for jj in range(self.ntemps)])
        self.lnprob0 = np.array([self.logp(self.x0[jj]) + self.lnlike0[jj] for jj in range(self.ntemps)])
        self.accept = np.zeros(self.ntemps)

        # setup saves
        self.filenames = ['/chain_{}.txt'.format(ii + 1) for ii in range(self.ntemps)]  # temps change (label by chain number)
        self.saves = [SaveData(outdir=self.outdir, filename=self.filenames[ii], resume=self.resume, thin=thin) for ii in range(self.ntemps)]

        # setup arrays
        self.chain = np.zeros((self.save_freq, self.ndim, self.ntemps))
        self.lnlike_arr = np.zeros((self.save_freq, self.ntemps))
        self.lnprob_arr = np.zeros((self.save_freq, self.ntemps))
        self.accept_arr = np.zeros((self.save_freq, self.ntemps))

        for ii in tqdm(range(num_samples)):
            kk = ii % self.save_freq
            self.counter += 1

            # metropolis hastings step + update chains
            for jj in range(self.ntemps):
                self.x0[jj], self.lnlike0[jj], self.lnprob0[jj], self.accept[jj] = mh_step(self.x0[jj], self.lnlike0[jj], self.lnprob0[jj], self.logl,
                                                                                           self.logp, self.mixes[jj], self.rng[jj], self.ptswap.ladder[jj])
                self.chain[kk, :, jj] = self.x0[jj]
                self.lnlike_arr[kk, jj] = self.lnlike0[jj]
                self.lnprob_arr[kk, jj] = self.lnprob0[jj]
                self.accept_arr[kk, jj] = self.accept[jj]

            # swap sometimes
            if self.counter % self.swap_steps == 0 and self.counter > 1:
                self.x0s, self.lnlike0s = self.ptswap.swap(self.chain[kk, :, :], self.lnlike_arr[kk, :])
                for jj in range(self.ntemps):
                    self.x0[jj] = self.x0s[jj]
                    self.lnlike0[jj] = self.lnlike0s[jj]
                    self.lnprob0[jj] = self.logp(self.x0[jj]) + self.lnlike0[jj]

                    self.chain[kk, :, jj], self.lnlike_arr[kk, jj] = self.x0s[jj], self.lnlike0s[jj]
                    self.lnprob0[jj] = 1 / self.ptswap.ladder[jj] * self.lnlike0[jj] + self.logp(self.x0[jj])
                    self.lnprob_arr[kk, jj] = self.lnprob0[jj]

            # update covariance matrix
            if self.counter % self.cov_update == 0 and self.counter > 1:
                [self.mixes[jj].recursive_update(self.counter, self.chain[:, :, jj]) for jj in range(self.ntemps)]
                # adaptive temperature spacing
                if self.adapt:
                    self.ptswap.adapt_ladder()

            # save and save state of PTSampler
            if self.counter % self.save_freq == 0 and self.counter > 1:
                [self.saves[jj](self.chain[:, :, jj], self.lnlike_arr[:, jj],
                                self.lnprob_arr[:, jj], self.accept_arr[:, jj]) for jj in range(self.ntemps)]
                self.saves[0].save_swap_data(self.ptswap)
                # output a pickle to resume from
                self.save_state()

            if ret_chain:
                full_chain[ii, :, :] = self.chain[kk, :, :]

        if ret_chain:
            return full_chain


class _function_wrapper(object):

    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)

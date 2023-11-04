from dataclasses import dataclass
import os, pathlib
from typing import Callable
import numpy as np
# from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
# from loguru import logger
# import ray

# from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, ChainStats, am, scam, de
from impulse.mhsampler import MHState, mh_kernel

@dataclass
class ShortChain:
    """
    class to hold a short chain of save_freq iterations
    """
    ndim: int
    short_iters: int
    iteration: int = 1
    outdir: str = './chains/'
    filename: str = 'chain_.txt'
    resume: bool = False
    thin: int = 1
    temp: float = 1.0

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.short_iters))
        self.lnlike = np.zeros((self.short_iters))
        self.accept = np.zeros((self.short_iters))
        self.filepath = os.path.join(self.outdir, self.filename)
        self.filepath = os.path.join(self.filepath, str(self.temp))

        pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if self.exists(self.outdir, self.filename) and not self.resume:
            with open(self.filepath, 'w') as _:
                pass

    def add_state(self,
                  new_state: MHState):
        self.samples[self.iteration % self.short_iters] = new_state.position
        self.iteration += 1

    def set_filepath(self, outdir, filename):
        self.filepath = os.path.join(outdir, filename)

    def exists(self, outdir, filename):
        return pathlib.Path(os.path.join(outdir, filename)).exists()

    def save_chain(self):
        to_save = np.column_stack([self.samples, self.lnlike, self.lnprob, self.accept])[::self.thin]
        with open(self.filepath, 'a+') as f:
            np.savetxt(f, to_save)

# class TestSampler:
#     def __init__(self,
#                  ndim: int,
#                  lnlike: Callable,
#                  lnprior: Callable,
#                  buffer_size: int = 50_000,
#                  sample_mean: float = None,
#                  sample_cov: np.ndarray = None,
#                  groups: list = None,
#                  loglargs: list = [],
#                  loglkwargs: dict = {},
#                  logpargs: list = [],
#                  logpkwargs: dict = {},
#                  cov_update: int = 100,
#                  save_freq: int = 1000,
#                  scam_weight: float = 30,
#                  am_weight: float = 15,
#                  de_weight: float = 50,
#                  seed: int = None,
#                  outdir: str = './chains'
#                  ) -> None:

#         self.ndim = ndim
#         self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs)
#         self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs)
#         self.rng = np.random.default_rng(seed)  # change this to seedsequence later on!

#         self.chain_stats = ChainStats(ndim, self.rng, groups=groups, sample_cov=sample_cov,
#                                     sample_mean=sample_mean, buffer_size=buffer_size)
#         self.jumps = JumpProposals(self.chain_stats)
#         self.jumps.add_jump(am, am_weight)
#         self.jumps.add_jump(scam, scam_weight)
#         self.jumps.add_jump(de, de_weight)

#         self.cov_update = cov_update
#         self.save_freq = save_freq
#         self.outdir = outdir

#     def sample(self,
#                initial_sample: np.ndarray,
#                num_iterations: int,
#                thin: int = 1):

#         short_chain = ShortChain(self.ndim, self.save_freq, thin=thin)  # keep save_freq samples

#         # set up initial state here:
#         x0 = np.array(initial_sample, dtype=np.float64)
#         lnlike0 = self.lnlike(x0)
#         lnprior0 = self.lnprior(x0)
#         initial_state = MHState(x0,
#                                 lnlike0,
#                                 lnprior0,
#                                 lnlike0 + lnprior0
#                                 )

#         # initial sample and go!
#         state = mh_kernel(initial_state, self.jumps, self.lnlike,
#                           self.lnprior, self.rng)
#         short_chain.add_state(state)
#         for jj in tqdm(range(1, num_iterations), initial=1, total=num_iterations):
#             state = mh_kernel(state, self.jumps, self.lnlike, self.lnprior, self.rng)
#             short_chain.add_state(state)
#             if jj % self.cov_update == 0:
#                 self.chain_stats.recursive_update(self.chain_stats.sample_total, short_chain.samples)
#             if jj % self.save_freq == 0:
#                 short_chain.save_chain()

class PTTestSampler:
    def __init__(self,
                 ndim: int,
                 lnlike: Callable,
                 lnprior: Callable,
                 buffer_size: int = 50_000,
                 sample_mean: float = None,
                 sample_cov: np.ndarray = None,
                 groups: list = None,
                 loglargs: list = [],
                 loglkwargs: dict = {},
                 logpargs: list = [],
                 logpkwargs: dict = {},
                 cov_update: int = 100,
                 save_freq: int = 1000,
                 scam_weight: float = 30,
                 am_weight: float = 15,
                 de_weight: float = 50,
                 seed: int = None,
                 outdir: str = './chains',
                 num_temps: int = 2
                 ) -> None:

        self.ndim = ndim
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs)
        sequence = np.random.SeedSequence(seed)
        sequence.spawn(num_temps)
        self.rngs = [np.random.default_rng(s) for s in sequence]

        self.chain_stats = ChainStats(ndim, self.rng, groups=groups, sample_cov=sample_cov,
                                      sample_mean=sample_mean, buffer_size=buffer_size)
        self.jumps = JumpProposals(self.chain_stats)
        self.jumps.add_jump(am, am_weight)
        self.jumps.add_jump(scam, scam_weight)
        self.jumps.add_jump(de, de_weight)

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir

    def sample(self,
               initial_sample: np.ndarray,
               num_iterations: int,
               thin: int = 1):

        # set up temperatures
        short_chain = ShortChain(self.ndim, self.save_freq, thin=thin)  # keep save_freq samples

        # set up initial state here:
        x0 = np.array(initial_sample, dtype=np.float64)
        lnlike0 = self.lnlike(x0)
        lnprior0 = self.lnprior(x0)
        initial_state = MHState(x0,
                                lnlike0,
                                lnprior0,
                                lnlike0 + lnprior0
                                )

        # initial sample and go!
        state = mh_kernel(initial_state, self.jumps, self.lnlike,
                          self.lnprior, self.rng)
        short_chain.add_state(state)
        for jj in tqdm(range(1, num_iterations), initial=1, total=num_iterations):
            for ii in range(1, ntemps):
                state = mh_kernel(state, self.jumps, self.lnlike, self.lnprior, self.rng)
            short_chain.add_state(state)
            if jj % self.cov_update == 0:
                self.chain_stats.recursive_update(self.chain_stats.sample_total, short_chain.samples)
            if jj % self.save_freq == 0:
                short_chain.save_chain()

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

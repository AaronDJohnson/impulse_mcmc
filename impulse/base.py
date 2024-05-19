from typing import Callable
import numpy as np
# from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
# from loguru import logger
# import ray

# from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, ChainStats, am, scam, de
from impulse.mhsampler import MHState, mh_kernel
from impulse.ptsampler import PTState, pt_kernel

class TemperedLogPosterior:
    """
    Make the tempered logposterior
    """
    def __init__(self,
                 loglikelihood: Callable,
                 logprior: Callable,
                 loglargs: list = None,
                 loglkwargs: dict = None,
                 logpargs: list = None,
                 logpkwargs: dict = None,
                 temperature: float = 1.0
                 ):
        self.temperature = temperature

        if loglargs is None:
            loglargs = []
        if loglkwargs is None:
            loglkwargs = {}
        if logpargs is None:
            logpargs = []
        if logpkwargs is None:
            logpkwargs = {}

        self.loglikelihood = loglikelihood
        self.logprior = logprior

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def get_tempered_logposterior(self, params: np.ndarray) -> float:
        return 1 / self.temperature * self.loglikelihood(params, *self.loglargs, **self.loglkwargs) + self.logprior(params, *self.logpargs, **self.logpkwargs)
    
    def get_loglikelihood(self, params: np.ndarray) -> float:
        return self.loglikelihood(params, *self.loglargs, **self.loglkwargs)
    
    def get_logprior(self, params: np.ndarray) -> float:
        return self.logprior(params, *self.logpargs, **self.logpkwargs)


def make_rng_seeds(seed: int, ntemps: int):
    # set up pieces for each temperature
    sequence = np.random.SeedSequence(seed)
    seeds = sequence.spawn(ntemps + 1)  # extra one for the ptswaps
    rngs = [np.random.default_rng(s) for s in seeds]
    return rngs


class PTTestSampler:
    def __init__(self,
                 ndim: int,
                 tlposterior: TemperedLogPosterior,
                 buffer_size: int = 50_000,
                 sample_mean: float = None,
                 sample_cov: np.ndarray = None,
                 groups: list = None,
                 cov_update: int = 100,
                 save_freq: int = 1000,
                 scam_weight: float = 30,
                 am_weight: float = 15,
                 de_weight: float = 50,
                 seed: int = None,
                 outdir: str = './chains',
                 ntemps: int = 2,
                 swap_steps: int = 100,
                 min_temp: float = 1.0,
                 max_temp: float = None,
                 temp_step: float = None,
                 ladder: list = None,
                 inf_temp: bool = False,
                 adapt_t0: int = 100,
                 adapt_nu: int = 10
                 ) -> None:

        self.rngs = make_rng_seeds(seed, ntemps)

        self.ndim = ndim
        self.ntemps = ntemps
        self.swap_steps = swap_steps
        
        self.chain_stats = [ChainStats(ndim, self.rngs[ii], groups=groups, sample_cov=sample_cov,
                                       sample_mean=sample_mean, buffer_size=buffer_size) for ii in range(ntemps)]
        self.jumps = [JumpProposals(self.chain_stats[ii]) for ii in range(ntemps)]
        for ii in range(ntemps):
            self.jumps[ii].add_jump(am, am_weight)
            self.jumps[ii].add_jump(scam, scam_weight)
            self.jumps[ii].add_jump(de, de_weight)
        self.ptstate = PTState(self.ndim, ntemps, swap_steps=swap_steps, min_temp=min_temp, max_temp=max_temp,
                               temp_step=temp_step, ladder=ladder, inf_temp=inf_temp, adapt_t0=adapt_t0, adapt_nu=adapt_nu)

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir

    def sample(self,
               initial_sample: np.ndarray,
               num_iterations: int,
               thin: int = 1):

        # set up temperatures
        short_chains = [ShortChain(self.ndim, self.save_freq, thin=thin,
                        nchain=ii, outdir=self.outdir) for ii in range(self.ntemps)]  # keep save_freq samples

        # set up initial state here:
        x0 = np.array(initial_sample, dtype=np.float64)
        lnlike0 = self.lnlike(x0)
        lnprior0 = self.lnprior(x0)
        initial_states = [MHState(x0,
                                  lnlike0,
                                  lnprior0,
                                  1/self.ptstate.ladder[ii] * (lnlike0) + lnprior0,
                                  temp=self.ptstate.ladder[ii]
                                  ) for ii in range(self.ntemps)]

        # initial sample and go!
        states = [mh_kernel(initial_states[ii], self.jumps[ii], self.lnlike, self.lnprior, self.rngs[ii]) for ii in range(self.ntemps)]
        [short_chains[ii].add_state(states[ii]) for ii in range(self.ntemps)]

        for jj in tqdm(range(1, num_iterations), initial=1, total=num_iterations):
            states = [mh_kernel(states[ii], self.jumps[ii], self.lnlike, self.lnprior, self.rngs[ii]) for ii in range(self.ntemps)]
            [short_chains[ii].add_state(states[ii]) for ii in range(self.ntemps)]
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                states = pt_kernel(states, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1])
                [short_chains[ii].add_state(states[ii]) for ii in range(self.ntemps)]
                self.ptstate.adapt_ladder()
            if jj % self.cov_update == 0:
                [self.chain_stats[ii].recursive_update(self.chain_stats[ii].sample_total, short_chains[ii].samples) for ii in range(self.ntemps)]
            if jj % self.save_freq == 0:
                [short_chains[ii].save_chain() for ii in range(self.ntemps)]
        [short_chains[ii].save_chain() for ii in range(self.ntemps)]

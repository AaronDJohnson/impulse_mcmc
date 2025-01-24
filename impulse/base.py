from dataclasses import dataclass
import os
import pathlib
from typing import Callable
import numpy as np
from tqdm import tqdm

from impulse.proposals import JumpProposals, ChainStats, am, scam, de
from impulse.mhsampler import MHState, mh_step, vectorized_mh_step
from impulse.ptsampler import PTState, pt_kernel

@dataclass
class ShortChain:
    """
    class to hold a short chain of save_freq iterations
    """
    ndim: int
    short_iters: int
    iteration: int = 1
    outdir: str = './chains/'
    resume: bool = False
    thin: int = 1
    nchain: float = 1

    def __post_init__(self):
        if self.thin > self.short_iters:
            raise ValueError("There are not enough samples to thin. Increase save_freq.")
        self.samples = np.zeros((self.short_iters, self.ndim))
        self.lnprob = np.zeros((self.short_iters))
        self.lnlike = np.zeros((self.short_iters))
        self.accept = np.zeros((self.short_iters))
        self.var_temp = np.zeros((self.short_iters))
        self.filename = f'chain_{self.nchain}.txt'
        self.filepath = os.path.join(self.outdir, self.filename)

        pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if self.exists(self.outdir, self.filename) and not self.resume:
            with open(self.filepath, 'w') as _:
                pass

    def add_state(self,
                  new_state: MHState):
        self.samples[self.iteration % self.short_iters] = new_state.position
        self.lnprob[self.iteration % self.short_iters] = new_state.lnprob
        self.lnlike[self.iteration % self.short_iters] = new_state.lnlike
        self.accept[self.iteration % self.short_iters] = new_state.accepted
        self.var_temp[self.iteration % self.short_iters] = new_state.temp
        self.iteration += 1

    def set_filepath(self, outdir, filename):
        self.filepath = os.path.join(outdir, filename)

    def exists(self, outdir, filename):
        return pathlib.Path(os.path.join(outdir, filename)).exists()

    def save_chain(self):
        to_save = np.column_stack([self.samples, self.lnlike, self.lnprob, self.accept, self.var_temp])[::self.thin]
        with open(self.filepath, 'a+') as f:
            np.savetxt(f, to_save)

class PTSampler:
    def __init__(self,
                 ndim: int,
                 lnlike: Callable,
                 lnprior: Callable,
                 buffer_size: int = 50_000,
                 sample_mean: float = None,
                 sample_cov: np.ndarray = None,
                 groups: list = None,
                 loglargs: list = None,
                 loglkwargs: dict = None,
                 logpargs: list = None,
                 logpkwargs: dict = None,
                 cov_update: int = 100,
                 save_freq: int = 1000,
                 scam_weight: float = 30,
                 am_weight: float = 15,
                 de_weight: float = 50,
                 seed: int = None,
                 outdir: str = './chains',
                 ntemps: int = 2,
                 swap_steps: int = 10,
                 min_temp: float = 1.0,
                 max_temp: float = None,
                 temp_step: float = None,
                 ladder: list = None,
                 inf_temp: bool = False,
                 adapt_t0: int = 100,
                 adapt_nu: int = 10,
                 vectorized: bool = False,
                 ) -> None:

        if loglargs is None:
            loglargs = []
        if loglkwargs is None:
            loglkwargs = {}
        if logpargs is None:
            logpargs = []
        if logpkwargs is None:
            logpkwargs = {}

        self.ndim = ndim
        self.ntemps = ntemps
        self.swap_steps = swap_steps
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs)

        # set up pieces for each temperature
        sequence = np.random.SeedSequence(seed)
        # each chain needs its own random number generator with a seed
        seeds = sequence.spawn(ntemps + 1)  # extra seed for the ptswaps
        self.rngs = [np.random.default_rng(s) for s in seeds]
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
        self.vectorized = vectorized

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

        # check for bad initial samples
        try:  # non-vectorized
            if ~np.isfinite(lnlike0):
                raise ValueError("Initial sample likelihood value is not finite.")
            if ~np.isfinite(lnlike0):
                raise ValueError("Initial sample is outside the prior bounds.")
        except ValueError:  # vectorized
            if np.all(~np.isfinite(lnlike0)):
                raise ValueError("Some likelihood values are not finite.")
            if np.any(~np.isfinite(lnprior0)):
                raise ValueError("An initial value falls outside the prior bounds.")

        initial_states = [MHState(x0 if not self.vectorized else x0.reshape(-1, self.ndim)[ii],
                                lnlike0 if not self.vectorized else lnlike0[ii],
                                lnprior0 if not self.vectorized else lnprior0[ii],
                                1/self.ptstate.ladder[ii] * (lnlike0) + lnprior0 if not self.vectorized else 1/self.ptstate.ladder[ii] * (lnlike0[ii]) + lnprior0[ii],
                                temp=self.ptstate.ladder[ii]
                                ) for ii in range(self.ntemps)]

        # initial sample and go!
        if self.vectorized:
            states = vectorized_mh_step(initial_states, self.jumps, self.lnlike, self.lnprior, self.rngs[0], self.ndim)
        else:
            states = [mh_step(initial_states[ii], self.jumps[ii], self.lnlike, self.lnprior, self.rngs[ii]) for ii in range(self.ntemps)]
        [short_chains[ii].add_state(states[ii]) for ii in range(self.ntemps)]

        for jj in tqdm(range(1, num_iterations), initial=1, total=num_iterations):
            if self.vectorized:
                states = vectorized_mh_step(states, self.jumps, self.lnlike, self.lnprior, self.rngs[0], self.ndim)
            else:
                states = [mh_step(states[ii], self.jumps[ii], self.lnlike, self.lnprior, self.rngs[ii]) for ii in range(self.ntemps)]
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

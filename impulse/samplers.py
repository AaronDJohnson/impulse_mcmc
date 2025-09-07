from typing import Callable, Optional, List
import numpy as np
from tqdm import tqdm

from impulse.proposals import JumpProposals, ProposalBundle, am, scam, de
from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.input_function_wrapper import _function_wrapper
from impulse.sampler_state import SamplerState, PTState
from impulse.file_io import ShortChain
from impulse.sampler_step import vectorized_mh_step, pt_step
from impulse.resume import checkpoint_sampler, load_checkpoint, check_for_checkpoint

def setup_seeds(seed: Optional[int], ntemps: int) -> List[np.random.Generator]:
    # set up pieces for each temperature
    sequence = np.random.SeedSequence(seed)
    # each chain needs its own random number generator with a seed
    seeds = sequence.spawn(ntemps + 1)  # extra seed for the ptswaps
    rngs = [np.random.default_rng(s) for s in seeds]
    return rngs

def setup_chain_stats(ndim, ptstate, rngs, groups, sample_cov, sample_mean, buffer_size, temps) -> MultiChainStats:
    ntemps = len(temps)
    chain_stats = [ChainStats(ndim, ptstate, ii, rngs[ii], groups=groups, sample_cov=sample_cov,
                              sample_mean=sample_mean, buffer_size=buffer_size) for ii in range(ntemps)]
    multi_chain_stats = MultiChainStats(chain_stats)
    return multi_chain_stats

def setup_standard_jumps(multi_chain_stats: MultiChainStats, am_weight, scam_weight, de_weight) -> ProposalBundle:
    jumps = [JumpProposals(multi_chain_stats.chain_stats[ii]) for ii in range(multi_chain_stats.ntemps)]
    for ii in range(multi_chain_stats.ntemps):
        jumps[ii].add_jump(am, am_weight)
        jumps[ii].add_jump(scam, scam_weight)
        jumps[ii].add_jump(de, de_weight)
    return ProposalBundle(jumps)

def setup_initial_position(initial_position: np.ndarray, ntemps: int) -> np.ndarray:
    # set up initial state here: normalize incoming initial_position to (ntemps, ndim)
    _x0 = np.asarray(initial_position, dtype=np.float64)
    if _x0.ndim == 1:
        positions = np.tile(_x0.reshape(1, -1), (ntemps, 1))
    elif _x0.ndim == 2:
        if _x0.shape[0] == ntemps:
            positions = _x0
        elif _x0.shape[0] == 1:
            positions = np.tile(_x0, (ntemps, 1))
        else:
            raise ValueError(f"initial_position has { _x0.shape[0] } rows but expected 1 or {self.ntemps}")
    else:
        raise ValueError("initial_position must be 1-D (ndim,) or 2-D (ntemps, ndim)")
    return positions

class PTSampler:
    def __init__(self,
                 ndim: int,
                 lnlike: Callable,
                 lnprior: Callable,
                 buffer_size: int = 50_000,
                 sample_mean: Optional[np.ndarray] = None,
                 sample_cov: Optional[np.ndarray] = None,
                 groups: Optional[list] = None,
                 loglargs: Optional[tuple] = None,
                 loglkwargs: Optional[dict] = None,
                 logpargs: Optional[tuple] = None,
                 logpkwargs: Optional[dict] = None,
                 cov_update: int = 100,
                 save_freq: int = 1000,
                 scam_weight: float = 30,
                 am_weight: float = 15,
                 de_weight: float = 50,
                 seed: Optional[int] = None,
                 outdir: str = './chains',
                 ntemps: int = 21,
                 swap_steps: int = 1,
                 min_temp: float = 1.0,
                 max_temp: Optional[float] = None,
                 temp_step: Optional[float] = None,
                 ladder: Optional[np.ndarray] = None,
                 inf_temp: bool = False,
                 adapt_t0: int = 100,
                 adapt_nu: int = 10,
                 resume: bool = False,
                 vectorized: bool = False,
                 ) -> None:

        if loglargs is None:
            loglargs = ()
        if loglkwargs is None:
            loglkwargs = {}
        if logpargs is None:
            logpargs = ()
        if logpkwargs is None:
            logpkwargs = {}

        self.ndim = ndim
        self.ntemps = ntemps
        self.swap_steps = swap_steps
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs, vectorized=vectorized)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs, vectorized=vectorized)

        self.rngs = setup_seeds(seed, ntemps)

        self.ptstate = PTState(self.ndim, ntemps, swap_steps=swap_steps, min_temp=min_temp, max_temp=max_temp,
                               temp_step=temp_step, ladder=ladder, inf_temp=inf_temp, adapt_t0=adapt_t0, adapt_nu=adapt_nu)
        self.multi_chain_stats = setup_chain_stats(ndim, self.ptstate, self.rngs, groups, sample_cov, sample_mean, buffer_size, self.ptstate.ladder)
        self.proposal_bundle = setup_standard_jumps(self.multi_chain_stats, am_weight, scam_weight, de_weight)

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume

    def add_custom_jump(self, proposal, weight):
        self.proposal_bundle.add_jump(proposal, weight)

    def sample(self,
               initial_position: np.ndarray,
               num_iterations: int,
               thin: int = 1):

        if self.ptstate.ladder is None:  # this shouldn't happen!
            raise ValueError("PTState ladder is not initialized")
        # setup save chains
        self.short_chain = ShortChain(self.ndim, self.ntemps, self.save_freq,
                                 iteration=0, outdir=self.outdir, resume=self.resume,
                                 thin=thin)
        # set up initial state here:
        initial_position = setup_initial_position(initial_position, self.ntemps)
        # initial_position = np.tile(np.asarray(initial_position, dtype=np.float64), (self.ntemps, 1))

        lnlike0 = self.lnlike(initial_position)
        lnprior0 = self.lnprior(initial_position)
        lnprob0 = 1 / self.ptstate.ladder * lnlike0 + lnprior0
        initial_state = SamplerState(initial_position, lnlike0, lnprior0, lnprob0, accepted=np.ones(self.ntemps), temps=self.ptstate.ladder)

        # check for bad initial samples
        if np.all(~np.isfinite(lnlike0)):
            raise ValueError("Some likelihood values are not finite.")
        if np.any(~np.isfinite(lnprior0)):
            raise ValueError("An initial value falls outside the prior bounds.")

        self.state = vectorized_mh_step(initial_state, self.proposal_bundle, self.lnlike, self.lnprior, self.rngs[0])

        # look for checkpoint in outdir
        self.checkpoint_path = check_for_checkpoint(self.outdir)
        if self.resume and self.checkpoint_path is not None:
            print(f"Resuming from checkpoint: {self.checkpoint_path}")
            loaded = load_checkpoint(self.checkpoint_path, lnlike=self.lnlike, lnprior=self.lnprior)
            self.__dict__.update(loaded.__dict__)  # copy the state from the checkpointed sampler to this one

        for jj in tqdm(range(self.short_chain.iteration, num_iterations), initial=self.short_chain.iteration, total=num_iterations, desc="Sampling"):
            self.state = vectorized_mh_step(self.state, self.proposal_bundle, self.lnlike, self.lnprior, self.rngs[0])
            # save the samples
            self.short_chain.add_state(self.state)
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(self.state, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1])
                self.ptstate.adapt_ladder()
            if jj % self.cov_update == 0:
                self.multi_chain_stats.recursive_update(self.short_chain.samples)
            if jj % self.save_freq == 0 and jj > 0:  # don't save the initial state
                self.short_chain.save_chain()
                checkpoint_sampler(self, path=self.checkpoint_path)
        # save the final iteration too
        self.short_chain.save_chain()

from typing import Callable, Optional, List
import logging
import os
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from impulse.proposals import JumpProposals, ProposalBundle, am, scam, de
from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.input_function_wrapper import _function_wrapper
from impulse.sampler_state import SamplerState, PTState
from impulse.file_io import ShortChain
from impulse.sampler_step import vectorized_mh_step, pt_step
from impulse.resume import checkpoint_sampler, load_checkpoint, check_for_checkpoint

def setup_seeds(seed: Optional[int], ntemps: int) -> List[np.random.Generator]:
    """
    Initialize random number generators for parallel tempering chains.

    Creates independent random number generators for each temperature chain
    plus one additional generator for parallel tempering swaps.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility. If None, uses system entropy.
    ntemps : int
        Number of temperature chains.

    Returns
    -------
    list of np.random.Generator
        List of independent random number generators, length ntemps + 1.
        The last generator is reserved for PT swaps.

    Examples
    --------
    >>> rngs = setup_seeds(42, 5)
    >>> len(rngs)
    6
    >>> # Each generator produces independent sequences
    >>> sample1 = rngs[0].random()
    >>> sample2 = rngs[1].random()
    """
    # set up pieces for each temperature
    sequence = np.random.SeedSequence(seed)
    # each chain needs its own random number generator with a seed
    seeds = sequence.spawn(ntemps + 1)  # extra seed for the ptswaps
    rngs = [np.random.default_rng(s) for s in seeds]
    return rngs

def setup_chain_stats(ndim, ptstate, rngs, groups, sample_cov, sample_mean, buffer_size, temps) -> MultiChainStats:
    """
    Initialize chain statistics tracking for all temperature chains.

    Creates ChainStats objects for each temperature chain to track covariance,
    means, and other statistics needed for adaptive proposals.

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    ptstate : PTState
        Parallel tempering state object.
    rngs : list of np.random.Generator
        Random number generators for each chain.
    groups : list or None
        Parameter groups for block updates.
    sample_cov : np.ndarray or None
        Initial covariance estimate.
    sample_mean : np.ndarray or None
        Initial mean estimate.
    buffer_size : int
        Size of sample buffer for statistics.
    temps : np.ndarray
        Temperature ladder.

    Returns
    -------
    MultiChainStats
        Container managing statistics for all chains.

    Examples
    --------
    >>> import numpy as np
    >>> from impulse.sampler_state import PTState
    >>> ptstate = PTState(2, 3)
    >>> rngs = setup_seeds(42, 3)
    >>> temps = np.array([1.0, 2.0, 4.0])
    >>> stats = setup_chain_stats(2, ptstate, rngs, None, None, None, 1000, temps)
    >>> stats.ntemps
    3
    """
    ntemps = len(temps)
    chain_stats = [ChainStats(ndim, ptstate, ii, rngs[ii], groups=groups, sample_cov=sample_cov,
                              sample_mean=sample_mean, buffer_size=buffer_size) for ii in range(ntemps)]
    multi_chain_stats = MultiChainStats(chain_stats)
    return multi_chain_stats

def setup_standard_jumps(multi_chain_stats: MultiChainStats, am_weight, scam_weight, de_weight) -> ProposalBundle:
    """
    Configure standard MCMC proposal distributions with specified weights.

    Sets up adaptive Metropolis (AM), single-component adaptive Metropolis (SCAM),
    and differential evolution (DE) proposals for each temperature chain.

    Parameters
    ----------
    multi_chain_stats : MultiChainStats
        Statistics container for all chains.
    am_weight : float
        Relative weight for adaptive Metropolis proposals.
    scam_weight : float
        Relative weight for single-component adaptive Metropolis proposals.
    de_weight : float
        Relative weight for differential evolution proposals.

    Returns
    -------
    ProposalBundle
        Container with proposal distributions for all chains.

    Examples
    --------
    >>> # Typical weights favoring DE proposals
    >>> bundle = setup_standard_jumps(stats, am_weight=15, scam_weight=30, de_weight=50)
    >>> # Each chain now has three proposal types with specified weights
    """
    jumps = [JumpProposals(multi_chain_stats.chain_stats[ii]) for ii in range(multi_chain_stats.ntemps)]
    for ii in range(multi_chain_stats.ntemps):
        jumps[ii].add_jump(am, am_weight)
        jumps[ii].add_jump(scam, scam_weight)
        jumps[ii].add_jump(de, de_weight)
    return ProposalBundle(jumps)

def setup_initial_position(initial_position: np.ndarray, ntemps: int) -> np.ndarray:
    """
    Prepare initial positions for all temperature chains.

    Converts various input formats to a standardized (ntemps, ndim) array
    by replicating positions across temperature chains as needed.

    Parameters
    ----------
    initial_position : array_like
        Initial parameter values. Can be:
        - 1-D array of shape (ndim,): replicated for all chains
        - 2-D array of shape (1, ndim): replicated for all chains  
        - 2-D array of shape (ntemps, ndim): used directly
    ntemps : int
        Number of temperature chains.

    Returns
    -------
    np.ndarray
        Initial positions with shape (ntemps, ndim).

    Raises
    ------
    ValueError
        If input shape is incompatible with expected dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> # Single position replicated across chains
    >>> pos_1d = np.array([1.0, 2.0])
    >>> positions = setup_initial_position(pos_1d, ntemps=3)
    >>> positions.shape
    (3, 2)
    >>> np.all(positions == pos_1d)
    True

    >>> # Different positions for each chain
    >>> pos_2d = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
    >>> positions = setup_initial_position(pos_2d, ntemps=3)
    >>> positions.shape
    (3, 2)
    """
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
            raise ValueError(f"initial_position has { _x0.shape[0] } rows but expected 1 or {ntemps}")
    else:
        raise ValueError("initial_position must be 1-D (ndim,) or 2-D (ntemps, ndim)")
    return positions

class PTSampler:
    """
    Parallel Tempering Markov Chain Monte Carlo sampler.

    A sophisticated MCMC sampler that uses parallel tempering to improve mixing
    and exploration of complex posterior distributions. Supports adaptive proposals,
    checkpoint/resume functionality, and vectorized likelihood evaluations.

    Parameters
    ----------
    ndim : int
        Dimensionality of the parameter space.
    lnlike : callable
        Log-likelihood function that accepts parameter arrays.
    lnprior : callable  
        Log-prior function that accepts parameter arrays.
    buffer_size : int, default 50000
        Size of internal buffer for storing samples and computing statistics.
    sample_mean : np.ndarray, optional
        Initial estimate of parameter means for adaptive proposals.
    sample_cov : np.ndarray, optional
        Initial covariance matrix estimate for adaptive proposals.
    groups : list, optional
        Parameter groups for block updates. If None, treats all parameters as one group.
    loglargs : tuple, optional
        Additional positional arguments for likelihood function.
    loglkwargs : dict, optional
        Additional keyword arguments for likelihood function.
    logpargs : tuple, optional
        Additional positional arguments for prior function.
    logpkwargs : dict, optional
        Additional keyword arguments for prior function.
    cov_update : int, default 100
        Frequency of covariance matrix updates (in iterations).
    save_freq : int, default 1000
        Frequency of saving chains to disk (in iterations).
    scam_weight : float, default 30
        Relative weight for single-component adaptive Metropolis proposals.
    am_weight : float, default 15
        Relative weight for adaptive Metropolis proposals.
    de_weight : float, default 50
        Relative weight for differential evolution proposals.
    seed : int, optional
        Random seed for reproducible sampling.
    outdir : str, default './chains'
        Directory for saving chain files and checkpoints.
    ntemps : int, default 21
        Number of temperature chains.
    swap_steps : int, default 1
        Frequency of temperature swap attempts.
    min_temp : float, default 1.0
        Minimum (cold) temperature.
    max_temp : float, optional
        Maximum (hot) temperature. If None, determined automatically.
    temp_step : float, optional
        Temperature spacing parameter. If None, determined automatically.
    ladder : np.ndarray, optional
        Custom temperature ladder. Overrides automatic temperature selection.
    inf_temp : bool, default False
        Whether to include an infinite temperature chain.
    adapt_t0 : int, default 100
        Initial adaptation period for temperature ladder.
    adapt_nu : int, default 10
        Adaptation frequency for temperature ladder.
    resume : bool, default False
        Whether to resume from existing checkpoint.
    vectorized : bool, default False
        Whether likelihood and prior functions support vectorized evaluation.

    Attributes
    ----------
    state : SamplerState
        Current state of all temperature chains.
    ptstate : PTState
        Parallel tempering specific state (temperature ladder, swap statistics).
    multi_chain_stats : MultiChainStats
        Statistics tracking for adaptive proposals.
    proposal_bundle : ProposalBundle
        Collection of proposal distributions for all chains.

    Examples
    --------
    >>> import numpy as np
    >>> from impulse import PTSampler
    >>> 
    >>> # Define a simple 2D Gaussian likelihood
    >>> def log_likelihood(x):
    ...     return -0.5 * np.sum(x**2)
    >>> 
    >>> # Uniform prior on [-5, 5]^2
    >>> def log_prior(x):
    ...     return 0.0 if np.all(np.abs(x) <= 5) else -np.inf
    >>> 
    >>> # Create sampler
    >>> sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior,
    ...                    ntemps=10, seed=42)
    >>> 
    >>> # Run sampling
    >>> initial_pos = np.array([0.0, 0.0])
    >>> sampler.sample(initial_pos, num_iterations=10000)
    Sampling: 100%|██████████| 10000/10000 [00:45<00:00, 220.11it/s]

    >>> # Access results
    >>> print(f"Final acceptance rate: {sampler.state.accepted.mean():.3f}")
    >>> print(f"Temperature swaps accepted: {sampler.ptstate.swap_accept.sum()}")

    Notes
    -----
    - The sampler automatically saves chains and checkpoints during sampling
    - Temperature ladder adaptation helps optimize parallel tempering efficiency  
    - Adaptive proposals improve as the sampler learns the target distribution
    - Vectorized functions can significantly improve performance for expensive likelihoods
    """
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
                 threads: int = 1,
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
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs, vectorized=vectorized, threads=threads)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs, vectorized=vectorized, threads=threads)

        self.rngs = setup_seeds(seed, ntemps)

        self.ptstate = PTState(self.ndim, ntemps, swap_steps=swap_steps, min_temp=min_temp, max_temp=max_temp,
                               temp_step=temp_step, ladder=ladder, inf_temp=inf_temp, adapt_t0=adapt_t0, adapt_nu=adapt_nu)
        self.multi_chain_stats = setup_chain_stats(ndim, self.ptstate, self.rngs, groups, sample_cov, sample_mean, buffer_size, self.ptstate.ladder)
        self.proposal_bundle = setup_standard_jumps(self.multi_chain_stats, am_weight, scam_weight, de_weight)

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume

    @classmethod
    def from_rjmcmc(cls,
                    rjmcmc_space,
                    birth_weight: float = 15,
                    death_weight: float = 15,
                    nmodel_weight: float = 10,
                    swap_weight: float = 15,
                    am_weight: float = 15,
                    scam_weight: float = 15,
                    de_weight: float = 15,
                    **kwargs):
        """
        Construct a PTSampler pre-configured for RJMCMC model selection.

        Parameters
        ----------
        rjmcmc_space : RJMCMCProductSpace
            Configured RJMCMC product space object.
        birth_weight : float
            Relative weight for birth proposals.
        death_weight : float
            Relative weight for death proposals.
        nmodel_weight : float
            Relative weight for uniform model-index jumps.
        swap_weight : float
            Relative weight for source-swap proposals.
        am_weight : float
            Relative weight for adaptive Metropolis proposals.
        scam_weight : float
            Relative weight for single-component AM proposals.
        de_weight : float
            Relative weight for differential evolution proposals.
        **kwargs
            Additional keyword arguments passed to ``PTSampler.__init__``
            (e.g. ``ntemps``, ``seed``, ``outdir``).

        Returns
        -------
        PTSampler
            Sampler with birth, death, nmodel_jump, and source_swap
            proposals already registered.

        Examples
        --------
        >>> from impulse.rjmcmc import RJMCMCProductSpace
        >>> space = RJMCMCProductSpace(loglike, logprior, 3, 3, draw_fn)
        >>> sampler = PTSampler.from_rjmcmc(space, ntemps=15, seed=42)
        >>> x0 = space.draw_initial_position(np.random.default_rng(42))
        >>> sampler.sample(x0, num_iterations=50000)
        """
        sampler = cls(
            ndim=rjmcmc_space.ndim,
            lnlike=rjmcmc_space.get_loglikelihood,
            lnprior=rjmcmc_space.get_logprior,
            groups=rjmcmc_space.get_default_groups(),
            am_weight=am_weight,
            scam_weight=scam_weight,
            de_weight=de_weight,
            **kwargs,
        )
        sampler.add_custom_jump(rjmcmc_space.get_birth_proposal(), birth_weight)
        sampler.add_custom_jump(rjmcmc_space.get_death_proposal(), death_weight)
        sampler.add_custom_jump(rjmcmc_space.get_nmodel_jump(), nmodel_weight)
        sampler.add_custom_jump(rjmcmc_space.get_source_swap_proposal(), swap_weight)
        return sampler

    def add_custom_jump(self, proposal, weight):
        """
        Add a custom proposal distribution to all temperature chains.

        Parameters
        ----------
        proposal : callable
            A proposal function that takes ChainStats and returns (new_sample, qxy).
        weight : float
            Relative weight for this proposal type.

        Examples
        --------
        >>> def custom_proposal(chain_stats):
        ...     # Custom proposal logic here
        ...     return new_sample, log_proposal_ratio
        >>> sampler.add_custom_jump(custom_proposal, weight=25)
        """
        self.proposal_bundle.add_jump(proposal, weight)

    def sample(self,
               initial_position: np.ndarray,
               num_iterations: int,
               thin: int = 1):
        """
        Run parallel tempering MCMC sampling.

        Performs the main sampling loop, handling proposal generation, acceptance/rejection,
        temperature swaps, adaptive updates, and periodic saves.

        Parameters
        ----------
        initial_position : array_like
            Starting position(s) for the chains. See setup_initial_position for formats.
        num_iterations : int
            Total number of MCMC iterations to perform.
        thin : int, default 1
            Thinning factor for saved samples. Only every thin-th sample is saved.

        Raises
        ------
        ValueError
            If initial likelihood or prior values are not finite.

        Examples
        --------
        >>> sampler = PTSampler(2, log_likelihood, log_prior)
        >>> sampler.sample([0.0, 0.0], num_iterations=10000)
        >>> # Chains are automatically saved to ./chains/ directory

        Notes
        -----
        - Progress is displayed via tqdm progress bar
        - Checkpoints are saved periodically for resuming interrupted runs
        - Temperature swaps and covariance updates occur at specified intervals
        - All chains are saved to disk at save_freq intervals
        """

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
        if np.any(~np.isfinite(lnlike0)):
            raise ValueError("Some likelihood values are not finite.")
        if np.any(~np.isfinite(lnprior0)):
            raise ValueError("An initial value falls outside the prior bounds.")

        self.state = initial_state

        # look for checkpoint in outdir
        self.checkpoint_path = check_for_checkpoint(self.outdir)
        if self.resume and self.checkpoint_path is not None:
            logger.info("Resuming from checkpoint: %s", self.checkpoint_path)
            loaded = load_checkpoint(self.checkpoint_path, lnlike=self.lnlike, lnprior=self.lnprior)
            self.__dict__.update(loaded.__dict__)  # copy the state from the checkpointed sampler to this one

        _last_cov_iter = self.short_chain.iteration

        for jj in tqdm(range(self.short_chain.iteration, num_iterations), initial=self.short_chain.iteration, total=num_iterations, desc="Sampling"):
            self.state = vectorized_mh_step(self.state, self.proposal_bundle, self.lnlike, self.lnprior, self.rngs[0])
            # save before add_state to prevent overwriting unsaved data
            if jj > 0 and jj % self.save_freq == 0:
                self.short_chain.save_chain()
                checkpoint_sampler(self, path=self.checkpoint_path)
            self.short_chain.add_state(self.state)
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(self.state, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1])
                self.ptstate.adapt_ladder()
            if jj % self.cov_update == 0:
                new_count = self.short_chain.iteration - _last_cov_iter
                if new_count > 0:
                    new_samples = self.short_chain.get_recent_samples(new_count)
                    self.multi_chain_stats.recursive_update(new_samples)
                _last_cov_iter = self.short_chain.iteration
        # save the final iteration too
        self.short_chain.save_chain()

    def load_chain(self):
        """
        Load saved chain files from disk.

        Reads the chain files written by the sampler and returns them as
        a dictionary of arrays stacked across temperature chains.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - ``samples`` : np.ndarray, shape (ntemps, nsamples, ndim)
            - ``lnlike`` : np.ndarray, shape (ntemps, nsamples)
            - ``lnprob`` : np.ndarray, shape (ntemps, nsamples)
            - ``accepted`` : np.ndarray, shape (ntemps, nsamples)
            - ``temperature`` : np.ndarray, shape (ntemps, nsamples)

        Raises
        ------
        FileNotFoundError
            If any expected chain file does not exist.

        Examples
        --------
        >>> sampler = PTSampler(ndim=2, lnlike=log_likelihood, lnprior=log_prior)
        >>> sampler.sample([0.0, 0.0], num_iterations=10000)
        >>> chain = sampler.load_chain()
        >>> print(chain['samples'].shape)
        (21, 10000, 2)
        """
        samples, lnlike, lnprob, accepted, temperature = [], [], [], [], []
        for ii in range(self.ntemps):
            filepath = os.path.join(self.outdir, f'chain_{ii}.txt')
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Chain file not found: {filepath}")
            data = np.loadtxt(filepath)
            samples.append(data[:, :self.ndim])
            lnlike.append(data[:, self.ndim])
            lnprob.append(data[:, self.ndim + 1])
            accepted.append(data[:, self.ndim + 2])
            temperature.append(data[:, self.ndim + 3])

        return {
            'samples': np.array(samples),
            'lnlike': np.array(lnlike),
            'lnprob': np.array(lnprob),
            'accepted': np.array(accepted),
            'temperature': np.array(temperature),
        }

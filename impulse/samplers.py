from typing import Callable, Optional, List
import logging
import os
import warnings
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from impulse.proposals import JumpProposals, ProposalBundle, am, scam, de, make_early_de
from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.input_function_wrapper import _function_wrapper
from impulse.sampler_state import SamplerState, PTState, tempered_lnprobs
from impulse.file_io import ShortChain
from impulse.sampler_step import vectorized_mh_step, pt_step
from impulse.resume import checkpoint_sampler, load_checkpoint, check_for_checkpoint
from impulse.rjmcmc_proposals import migrate_legacy_birth_death
from impulse.wrapping import WrapSpec, PeriodicSpec

# Sentinel default for ``num_adapt``: distinguishes "not passed" (keep a
# checkpointed value on resume) from an explicitly passed value — including
# an explicit ``None`` (adapt forever), which must override a checkpointed
# freeze on purpose, not by accident.  Never stored on a sampler instance,
# so it can never end up inside a pickled checkpoint.
_UNSET = object()

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
    jax : bool, default False
        Set True when the likelihood is JAX-traced/JIT-compiled. The MH step
        will then evaluate the likelihood on the full proposal batch on every
        iteration (masking invalid rows afterwards) so the input shape stays
        constant and the JIT cache is reused instead of recompiling.

        This does *not* skip computation for rows that fall outside the prior
        — the likelihood is still computed for every row in the batch, and the
        invalid rows are zeroed out only after the call. Use this flag when
        the cost of JAX recompilation dominates the cost of evaluating a few
        extra rows (almost always true for JIT'd likelihoods). When the
        likelihood is plain vectorized NumPy and the prior-rejection rate is
        high, leave ``jax=False`` so the step can genuinely skip invalid
        rows.
    num_adapt : int, optional
        Number of iterations during which adaptation is allowed. Once the
        global iteration counter (which persists across checkpoint resume)
        reaches ``num_adapt``, all adaptation freezes: the covariance/mean/SVD
        recomputes feeding the AM/SCAM proposals, the DE sample buffer,
        temperature-ladder adaptation, and refits of adaptive custom
        proposals (e.g. normalizing flows). The DE buffer is frozen too —
        not just its covariance contribution — because a rolling buffer
        would keep the kernel history-dependent; DE continues proposing
        from the frozen buffer. The transition kernel is therefore fixed
        from iteration ``num_adapt`` on, so later samples are exactly
        Markovian; samples drawn before the freeze are warmup and should
        be discarded for strict asymptotic guarantees. ``None`` adapts
        forever, preserving historical behavior.

        Resume semantics: when ``num_adapt`` is not passed (the default),
        resuming keeps the checkpointed value — un-freezing on resume by
        default would produce a half-frozen kernel, because proposals
        whose frozen state is pickled (e.g. a frozen normalizing flow)
        stay frozen while everything else adapts again. An explicitly
        passed value — including an explicit ``None`` — overrides the
        checkpointed value, with a warning when they differ. Fresh (non
        -resumed) runs treat the default exactly like ``None``.

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
                 jax: bool = False,
                 threads: int = 1,
                 periodic: Optional[PeriodicSpec] = None,
                 num_adapt: Optional[int] = _UNSET,
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
        self.wrap = WrapSpec.from_dict(periodic)
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs, vectorized=vectorized, jax=jax, threads=threads)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs, vectorized=vectorized, jax=jax, threads=threads)

        self.rngs = setup_seeds(seed, ntemps)

        self.ptstate = PTState(self.ndim, ntemps, swap_steps=swap_steps, min_temp=min_temp, max_temp=max_temp,
                               temp_step=temp_step, ladder=ladder, inf_temp=inf_temp, adapt_t0=adapt_t0, adapt_nu=adapt_nu)
        self.multi_chain_stats = setup_chain_stats(ndim, self.ptstate, self.rngs, groups, sample_cov, sample_mean, buffer_size, self.ptstate.ladder)
        self.proposal_bundle = setup_standard_jumps(self.multi_chain_stats, am_weight, scam_weight, de_weight)

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume
        # NEVER store the _UNSET sentinel on self (it must not end up in
        # pickled checkpoints); remember instead whether the caller passed
        # num_adapt explicitly, which controls the resume semantics.
        self._num_adapt_explicit = num_adapt is not _UNSET
        self.num_adapt = None if num_adapt is _UNSET else num_adapt

    def _adaptation_active(self, iteration: int) -> bool:
        """True while adaptation may still run at this global iteration.

        The ``getattr`` guards the public
        ``load_checkpoint(...)`` -> ``.sample()`` path: checkpoints written
        before ``num_adapt`` existed produce samplers without the
        attribute (unpickling bypasses ``__init__``), and missing means
        adapt forever — the historical behavior.

        Parameters
        ----------
        iteration : int
            Global iteration counter (persists across checkpoint resume).

        Returns
        -------
        bool
            True if adaptation is still allowed at ``iteration``.
        """
        num_adapt = getattr(self, 'num_adapt', None)
        return num_adapt is None or iteration < num_adapt

    def _freeze_adaptive_proposals(self) -> None:
        """Permanently freeze registered proposals that adapt internal state.

        Duck-typed: any proposal exposing a callable ``freeze_adaptation``
        (e.g. :class:`~impulse.flow_proposals.NormalizingFlowProposal`)
        is told to stop refitting. Idempotent.
        """
        for jp in self.proposal_bundle.jump_proposals:
            for prop in jp.proposal_list:
                freeze = getattr(prop, 'freeze_adaptation', None)
                if callable(freeze):
                    freeze()

    def _migrate_or_warn_legacy_birth_death(self) -> None:
        """Migrate resumed pre-fix RJ birth/death wiring, or warn loudly.

        Checkpoints written before the detailed-balance fix register
        ``birth_proposal`` and ``death_proposal`` as SEPARATE
        constant-weight jumps. That wiring violates detailed balance and
        biases the model posterior toward fewer sources; resuming it
        unchanged reproduces the bias. Detection starts from the proposal
        ``__name__``\\ s over the restored proposal lists, then inspects the
        attribute layout: CURRENT-code standalone registrations carry the
        same ``__name__``\\ s, but the current ``DeathProposal`` stores
        ``draw_from_prior`` (it re-fills the vacated slot) while the legacy
        one never did.  A pair whose death proposals all carry
        ``draw_from_prior`` is therefore NOT migrated — it is not legacy —
        and an accurate warning is emitted instead (standalone birth/death
        registration violates detailed balance; use the combined kernel).

        When a true legacy pair is found, a best-effort migration
        (:func:`impulse.rjmcmc_proposals.migrate_legacy_birth_death`)
        reconstructs the combined ``birth_death`` kernel from the
        unpickled legacy birth proposal and replaces the pair in every
        chain with their summed selection weight, then warns that
        PRE-resume samples remain biased. If reconstruction fails the
        checkpoint is left untouched and the historical loud warning is
        emitted instead.
        """
        props = [
            prop
            for jp in self.proposal_bundle.jump_proposals
            for prop in jp.proposal_list
        ]
        names = {getattr(prop, '__name__', '') for prop in props}
        if 'birth_proposal' not in names and 'death_proposal' not in names:
            return
        deaths = [
            p for p in props
            if getattr(p, '__name__', '') == 'death_proposal'
        ]
        if deaths and all(
                callable(getattr(p, 'draw_from_prior', None))
                for p in deaths):
            warnings.warn(
                "Resumed checkpoint registers separate standalone "
                "'birth_proposal'/'death_proposal' jumps whose attribute "
                "layout matches current-code standalone registrations (the "
                "death proposal carries draw_from_prior), not a pre-fix "
                "legacy checkpoint; no migration was attempted. Standalone "
                "birth/death registration in a constant-weight mixture "
                "violates detailed balance and biases the model posterior "
                "toward fewer sources: register the ONE combined "
                "birth-death kernel (make_birth_death_proposal or "
                "from_rjmcmc) instead.",
                UserWarning,
            )
            return
        migrated = migrate_legacy_birth_death(
            self.proposal_bundle.jump_proposals)
        if migrated is not None:
            warnings.warn(
                "Resumed checkpoint registered separate 'birth_proposal'/"
                "'death_proposal' jumps (pre-detailed-balance-fix wiring). "
                "The checkpoint was migrated automatically: the pair was "
                "replaced by the combined 'birth_death' kernel with their "
                "summed selection weight, so sampling continues from a "
                "detailed-balance-correct kernel. Model posteriors built "
                "from PRE-resume samples remain biased toward fewer "
                "sources and should be discarded.",
                UserWarning,
            )
            return
        warnings.warn(
            "Resumed checkpoint registers separate 'birth_proposal'/"
            "'death_proposal' jumps: it predates the detailed-balance "
            "fix and carries the biased birth/death wiring, so model "
            "posteriors will remain biased toward fewer sources. Start "
            "a fresh run (or re-register the combined birth-death "
            "kernel) for correct model posteriors.",
            UserWarning,
        )

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
                    de_min_fill: int = 100,
                    **kwargs):
        """
        Construct a PTSampler pre-configured for RJMCMC model selection.

        Parameters
        ----------
        rjmcmc_space : RJMCMCProductSpace
            Configured RJMCMC product space object.
        birth_weight : float
            Contribution to the combined birth-death kernel's selection
            weight.  Birth and death are registered as ONE kernel whose
            selection weight is ``birth_weight + death_weight``; the split
            between birth and death is governed by the space's
            ``prob_schedule`` (registering them as separate constant-weight
            jumps violates detailed balance).
        death_weight : float
            Contribution to the combined birth-death kernel's selection
            weight; see ``birth_weight``.
        nmodel_weight : float
            Relative weight for uniform model-index jumps.
        swap_weight : float
            Relative weight for source-swap proposals.
        am_weight : float
            Relative weight for adaptive Metropolis proposals.
        scam_weight : float
            Relative weight for single-component AM proposals.
        de_weight : float
            Relative weight for the differential evolution move.  In RJ
            configurations this weight is given to the min-fill-gated
            :class:`~impulse.proposals.EarlyDE` variant rather than the
            stock ``de`` (see Notes).
        de_min_fill : int
            Minimum per-model buffer fill before the DE difference move
            activates; see :class:`~impulse.proposals.EarlyDE`.
        **kwargs
            Additional keyword arguments passed to ``PTSampler.__init__``
            (e.g. ``ntemps``, ``seed``, ``outdir``).

        Returns
        -------
        PTSampler
            Sampler with birth, death, nmodel_jump, source_swap, and
            early-DE proposals already registered (see Notes for when
            they are skipped).

        Notes
        -----
        For a single-model space (``rjmcmc_space.num_models == 1``) the
        birth-death kernel, the model-index jump, and the source-swap
        proposal are all skipped — none is meaningful with one model, and
        the birth-death kernel itself rejects ``max_sources < 2`` — so
        only the standard continuous jumps (AM, SCAM, early-DE) are
        registered.  The birth-death kernel is also skipped when
        ``birth_weight + death_weight == 0``.

        The stock ``de`` jump requires a completely FULL sample buffer
        (more than ``buffer_size`` samples in the current model's buffer,
        50,000 by default); with per-model statistics the run's samples
        are split across all model indices, so at realistic run lengths
        no model's buffer ever fills and ``JumpProposals`` silently
        substitutes ``gaussian`` for every ``de`` selection.  ``de`` is
        therefore registered with weight 0 (never selected) and
        ``de_weight`` goes to :class:`~impulse.proposals.EarlyDE`, which
        runs the identical difference move as soon as the current model's
        buffer holds ``de_min_fill`` samples.  This move is what diffuses
        along within-model degeneracy ridges (e.g. amplitude-splitting
        ridges in source-counting problems) that random-walk proposals
        traverse too slowly, and without it model posteriors can be
        metastably wrong at realistic run lengths.

        Examples
        --------
        >>> from impulse.rjmcmc import RJMCMCProductSpace
        >>> space = RJMCMCProductSpace(loglike, logprior, 3, 3, draw_fn)
        >>> sampler = PTSampler.from_rjmcmc(space, ntemps=15, seed=42)
        >>> x0 = space.draw_initial_position(np.random.default_rng(42))
        >>> sampler.sample(x0, num_iterations=50000)
        """
        # Expand per-source sample_cov / sample_mean to full product space
        sample_cov = kwargs.pop('sample_cov', None)
        if sample_cov is not None:
            sample_cov = np.asarray(sample_cov)
            if sample_cov.shape == (rjmcmc_space.num_params, rjmcmc_space.num_params):
                full_cov = np.zeros((rjmcmc_space.ndim, rjmcmc_space.ndim))
                for i in range(rjmcmc_space.num_models):
                    sl = slice(i * rjmcmc_space.num_params, (i + 1) * rjmcmc_space.num_params)
                    full_cov[sl, sl] = sample_cov
                full_cov[-1, -1] = 1.0  # model index
                sample_cov = full_cov
        sample_mean = kwargs.pop('sample_mean', None)
        if sample_mean is not None:
            sample_mean = np.asarray(sample_mean)
            if sample_mean.shape == (rjmcmc_space.num_params,):
                full_mean = np.zeros(rjmcmc_space.ndim)
                for i in range(rjmcmc_space.num_models):
                    sl = slice(i * rjmcmc_space.num_params, (i + 1) * rjmcmc_space.num_params)
                    full_mean[sl] = sample_mean
                sample_mean = full_mean

        sampler = cls(
            ndim=rjmcmc_space.ndim,
            lnlike=rjmcmc_space.get_loglikelihood,
            lnprior=rjmcmc_space.get_logprior,
            groups=rjmcmc_space.get_default_groups(),
            sample_cov=sample_cov,
            sample_mean=sample_mean,
            am_weight=am_weight,
            scam_weight=scam_weight,
            # stock de is gated on buffer_full, which per-model buffers
            # never reach at realistic run lengths; the min-fill-gated
            # EarlyDE registered below carries de_weight instead
            de_weight=0,
            **kwargs,
        )
        # Trans-dimensional and label-permuting jumps only exist for
        # multi-model spaces: with a single model there is no birth/death
        # move to make, no other model index to jump to, and no second
        # source slot to swap with (BirthDeathProposal itself rejects
        # max_sources < 2), so only the standard continuous jumps are
        # registered.
        if rjmcmc_space.num_models > 1:
            if birth_weight != death_weight:
                warnings.warn(
                    "birth_weight != death_weight has no effect on the birth/death "
                    "split: birth and death form one combined kernel selected with "
                    "weight birth_weight + death_weight, and the split is governed "
                    "by the space's prob_schedule.",
                    UserWarning,
                )
            # Birth and death must be one kernel with schedule-driven selection;
            # separate constant-weight jumps violate detailed balance (see
            # impulse.rjmcmc_proposals.BirthDeathProposal).
            if birth_weight + death_weight > 0:
                sampler.add_custom_jump(rjmcmc_space.get_birth_death_proposal(),
                                        birth_weight + death_weight)
            sampler.add_custom_jump(rjmcmc_space.get_nmodel_jump(), nmodel_weight)
            sampler.add_custom_jump(rjmcmc_space.get_source_swap_proposal(), swap_weight)
        if de_weight > 0:
            sampler.add_custom_jump(make_early_de(de_min_fill), de_weight)
        sampler.multi_chain_stats.enable_per_model(
            rjmcmc_space.num_models, rjmcmc_space.num_params,
        )
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
        - When ``num_adapt`` is set, all adaptation stops once the global
          iteration counter reaches it; samples before the freeze are warmup
          and should be discarded for strict asymptotic guarantees

        Resume semantics: ``num_iterations`` is a GLOBAL iteration target —
        a resumed run continues from the checkpointed iteration counter up
        to ``num_iterations``, so pass the total, not the increment.  A
        checkpoint is written at the END of every iteration ``jj`` with
        ``jj > 0`` and ``jj % save_freq == 0``, capturing the sampler after
        that iteration fully completed (post PT-swap, post adaptation),
        including every RNG stream.  On resume the chain files are truncated
        back to the checkpointed flushed-row count and all iterations after
        the checkpoint are re-generated bit-identically, so an interrupted
        (or prematurely stopped) run resumed to ``N`` total iterations
        produces chain files identical to a single uninterrupted ``N``
        -iteration run.
        """

        if self.ptstate.ladder is None:  # this shouldn't happen!
            raise ValueError("PTState ladder is not initialized")
        # setup save chains
        self.short_chain = ShortChain(self.ndim, self.ntemps, self.save_freq,
                                 iteration=0, outdir=self.outdir, resume=self.resume,
                                 thin=thin)
        # iteration of the last covariance refresh; kept on the instance so
        # it is pickled into checkpoints (a resume overwrites this fresh
        # value with the checkpointed one via __dict__.update below)
        self._last_cov_iter = self.short_chain.iteration
        # set up initial state here:
        initial_position = setup_initial_position(initial_position, self.ntemps)
        if self.wrap is not None:
            initial_position = self.wrap.apply(initial_position)

        lnlike0 = self.lnlike(initial_position)
        lnprior0 = self.lnprior(initial_position)
        lnprob0 = tempered_lnprobs(lnlike0, lnprior0, self.ptstate.ladder)
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
            # num_adapt resume semantics: an EXPLICITLY passed constructor
            # value (including an explicit None) wins over the checkpointed
            # value, with a warning when they differ; the default keeps the
            # checkpointed value — silently un-freezing a checkpointed
            # freeze would resume a half-frozen kernel (proposals whose
            # frozen state is pickled, e.g. a frozen normalizing flow, stay
            # frozen while everything else adapts again).  getattr guards
            # the load_checkpoint(...)->sample() path, where unpickling
            # bypasses __init__ (pre-num_adapt checkpoints lack both
            # attributes).
            constructor_num_adapt = getattr(self, 'num_adapt', None)
            num_adapt_explicit = getattr(self, '_num_adapt_explicit', False)
            constructor_resume = self.resume
            constructor_checkpoint_path = self.checkpoint_path
            self.__dict__.update(loaded.__dict__)  # copy the state from the checkpointed sampler to this one
            # the checkpoint carries the ORIGINAL run's resume flag (often
            # False) and checkpoint path (None until its first checkpoint);
            # keep this run's values or later file handling would truncate
            # instead of append
            self.resume = constructor_resume
            self.checkpoint_path = constructor_checkpoint_path
            checkpoint_num_adapt = getattr(loaded, 'num_adapt', None)
            if num_adapt_explicit:
                if checkpoint_num_adapt != constructor_num_adapt:
                    logger.warning(
                        "Resume: overriding checkpointed num_adapt=%s with "
                        "the resuming constructor's explicitly passed "
                        "num_adapt=%s. Proposals whose frozen state is "
                        "pickled (e.g. normalizing flows frozen by "
                        "freeze_adaptation) remain frozen regardless: their "
                        "freeze is irreversible and survives the "
                        "checkpoint, so removing or extending the freeze "
                        "only re-enables the other adaptive components.",
                        checkpoint_num_adapt, constructor_num_adapt,
                    )
                self.num_adapt = constructor_num_adapt
            else:
                self.num_adapt = checkpoint_num_adapt
            self._num_adapt_explicit = num_adapt_explicit
            self._migrate_or_warn_legacy_birth_death()
            # drop chain-file rows written after the checkpoint (e.g. by the
            # final flush of a run that completed normally): the loop below
            # re-generates those iterations bit-identically from the
            # checkpointed RNG streams, so stale rows would be duplicates
            self.short_chain.truncate_files_to_saved()

        _proposals_frozen = False

        for jj in tqdm(range(self.short_chain.iteration, num_iterations), initial=self.short_chain.iteration, total=num_iterations, desc="Sampling"):
            adapting = self._adaptation_active(jj)
            if not adapting and not _proposals_frozen:
                self._freeze_adaptive_proposals()
                _proposals_frozen = True
            self.state = vectorized_mh_step(self.state, self.proposal_bundle, self.lnlike, self.lnprior, self.rngs[0], wrap=self.wrap)
            self.proposal_bundle.report_accepts(self.state.accepted)
            # save before add_state to prevent overwriting unsaved data
            if jj > 0 and jj % self.save_freq == 0:
                self.short_chain.save_chain()
                self.save_chain_acceptance_rates()
            self.short_chain.add_state(self.state)
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(self.state, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1])
                if adapting:
                    self.ptstate.adapt_ladder()
                    # adapt_ladder mutates the ladder (aliased by state.temps) in
                    # place, so the tempered lnprobs must be recomputed for the
                    # new temperatures
                    self.state.lnprobs = tempered_lnprobs(
                        self.state.lnlikes, self.state.lnpriors, self.ptstate.ladder)
            # Adaptation gate: past num_adapt neither the covariance/mean/SVD
            # nor the DE buffer update, so the transition kernel is fixed (DE
            # keeps proposing from the frozen buffer).
            # _last_cov_iter is an instance attribute (not a loop local) so
            # the covariance-refresh cadence itself is checkpointed state and
            # survives a resume even when the checkpoint iteration is not a
            # covariance-update boundary.
            if adapting and jj % self.cov_update == 0:
                new_count = self.short_chain.iteration - self._last_cov_iter
                if new_count > 0:
                    new_samples = self.short_chain.get_recent_samples(new_count)
                    self.multi_chain_stats.recursive_update(new_samples)
                self._last_cov_iter = self.short_chain.iteration
            # checkpoint at the END of the iteration: the pickle then
            # captures a fully completed iteration (post PT-swap, post
            # adaptation) with every RNG stream at an iteration boundary,
            # so a resumed run continues at jj + 1 bit-identically
            if jj > 0 and jj % self.save_freq == 0:
                checkpoint_sampler(self, path=self.checkpoint_path)
        # save the final iteration too
        self.short_chain.save_chain()
        self.save_chain_acceptance_rates()

    def proposal_acceptance_rates(self) -> dict:
        """Per-proposal acceptance statistics aggregated across chains.

        Returns
        -------
        dict
            ``{name: {calls, accepts, rate, per_chain: [...]}}``
        """
        return self.proposal_bundle.acceptance_report()

    def chain_acceptance_rates(self) -> dict:
        """Per-chain MH acceptance summary, plus PT swap rates.

        Returns
        -------
        dict
            ``temperatures`` (list of T per chain),
            ``mh`` (list per chain: ``{calls, accepts, rate, per_proposal}``),
            ``pt_swap`` (np.ndarray of length ``ntemps - 1`` with the
            accept rate for each neighbour-pair swap, or empty array if
            ``ntemps == 1``).
        """
        ladder = self.ptstate.ladder
        return {
            'temperatures': [] if ladder is None else ladder.tolist(),
            'mh': self.proposal_bundle.chain_acceptance_rates(),
            'pt_swap': self.ptstate.compute_accept_ratio() if self.ntemps > 1
                       else np.array([]),
        }

    def save_chain_acceptance_rates(self, path: Optional[str] = None) -> str:
        """Write a JSON snapshot of chain acceptance rates to disk.

        Includes per-chain MH rate, per-proposal × per-chain rate,
        aggregate per-proposal rate, and PT swap acceptance per pair.

        Parameters
        ----------
        path : str, optional
            Output path. Defaults to ``<outdir>/chain_acceptance.json``.

        Returns
        -------
        str
            Resolved path written.
        """
        import json
        if path is None:
            path = os.path.join(self.outdir, "chain_acceptance.json")
        report = self.chain_acceptance_rates()
        swap = report['pt_swap']
        report['pt_swap'] = (swap.tolist() if hasattr(swap, 'tolist')
                             else list(swap))
        report['per_proposal'] = self.proposal_acceptance_rates()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(report, fp, indent=2)
        return path

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

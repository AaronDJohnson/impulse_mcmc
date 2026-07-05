"""Internal shared parallel-tempering engine (private module).

:class:`_PTSamplerBase` owns everything :class:`impulse.PTSampler` and
:class:`impulse.RJPTSampler` have in common: constructor wiring (function
wrappers, per-chain RNGs, PT state/ladder, chain statistics, proposal
bundle, ``num_adapt`` sentinel handling), the ``sample()`` loop skeleton
(checkpoint resume incl. legacy birth/death migration and ``num_adapt``
restore; per-iteration proposals -> MH step -> PT swap -> tempered-lnprob
recompute after ladder adaptation -> adaptation-gated stats/cov updates ->
buffer/save -> end-of-iteration checkpoint), acceptance-rate reporting,
and ``load_chain``. Subclasses specialize genuinely divergent behavior
through the small ``_``-prefixed hook methods (checkpoint load/write,
pre-loop preparation, the adaptation-freeze transition, the post-MH step,
and the save-time flush).

Everything here is internal API: the public classes remain
``impulse.PTSampler`` and ``impulse.RJPTSampler`` at their historical
module locations, and the module-level setup helpers keep their public
import paths via re-export from :mod:`impulse.samplers`.

Bit-exactness contract: the loop skeleton preserves the exact RNG call
order of the pre-refactor samplers — hooks fire at precisely the points
where the two implementations previously diverged, and nothing else may
consume random numbers.
"""

import logging
import os
import warnings
from typing import Any, Callable, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.file_io import ShortChain
from impulse.input_function_wrapper import _function_wrapper
from impulse.proposals import JumpProposals, ProposalBundle, am, de, make_early_de, scam
from impulse.resume import check_for_checkpoint, checkpoint_sampler
from impulse.rjmcmc_proposals import migrate_legacy_birth_death
from impulse.sampler_state import PTState, SamplerState, tempered_lnprobs
from impulse.sampler_step import pt_step, vectorized_mh_step
from impulse.wrapping import PeriodicSpec, WrapSpec

# Sentinel default for ``num_adapt``: distinguishes "not passed" (keep a
# checkpointed value on resume) from an explicitly passed value — including
# an explicit ``None`` (adapt forever), which must override a checkpointed
# freeze on purpose, not by accident.  Never stored on a sampler instance,
# so it can never end up inside a pickled checkpoint.  Typed ``Any`` so it
# can stand in as the default for ``Optional[int]`` parameters.
_UNSET: Any = object()


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


def setup_chain_stats(
    ndim, ptstate, rngs, groups, sample_cov, sample_mean, buffer_size, temps
) -> MultiChainStats:
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
    chain_stats = [
        ChainStats(
            ndim,
            ptstate,
            ii,
            rngs[ii],
            groups=groups,
            sample_cov=sample_cov,
            sample_mean=sample_mean,
            buffer_size=buffer_size,
        )
        for ii in range(ntemps)
    ]
    multi_chain_stats = MultiChainStats(chain_stats)
    return multi_chain_stats


def setup_standard_jumps(
    multi_chain_stats: MultiChainStats, am_weight, scam_weight, de_weight
) -> ProposalBundle:
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
    jumps = [
        JumpProposals(multi_chain_stats.chain_stats[ii]) for ii in range(multi_chain_stats.ntemps)
    ]
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
            raise ValueError(
                f"initial_position has { _x0.shape[0] } rows but expected 1 or {ntemps}"
            )
    else:
        raise ValueError("initial_position must be 1-D (ndim,) or 2-D (ntemps, ndim)")
    return positions


def _expand_rjmcmc_cov_mean(rjmcmc_space, kwargs: dict) -> tuple:
    """Expand per-source ``sample_cov`` / ``sample_mean`` to the product space.

    Shared by ``PTSampler.from_rjmcmc`` and ``RJPTSampler.from_rjmcmc``:
    pops ``sample_cov`` / ``sample_mean`` out of ``kwargs`` (so they are not
    forwarded twice) and, when they are shaped for a SINGLE source block,
    tiles them block-diagonally / block-wise across all model slots of the
    full product space (with unit variance for the trailing model index).
    Values already shaped for the full space pass through unchanged.

    Parameters
    ----------
    rjmcmc_space : RJMCMCProductSpace
        Configured RJMCMC product space.
    kwargs : dict
        Keyword arguments destined for the sampler constructor; mutated in
        place (``sample_cov`` / ``sample_mean`` are removed).

    Returns
    -------
    tuple
        ``(sample_cov, sample_mean)`` expanded (or passed through / None).
    """
    sample_cov = kwargs.pop("sample_cov", None)
    if sample_cov is not None:
        sample_cov = np.asarray(sample_cov)
        if sample_cov.shape == (rjmcmc_space.num_params, rjmcmc_space.num_params):
            full_cov = np.zeros((rjmcmc_space.ndim, rjmcmc_space.ndim))
            for i in range(rjmcmc_space.num_models):
                sl = slice(i * rjmcmc_space.num_params, (i + 1) * rjmcmc_space.num_params)
                full_cov[sl, sl] = sample_cov
            full_cov[-1, -1] = 1.0  # model index
            sample_cov = full_cov
    sample_mean = kwargs.pop("sample_mean", None)
    if sample_mean is not None:
        sample_mean = np.asarray(sample_mean)
        if sample_mean.shape == (rjmcmc_space.num_params,):
            full_mean = np.zeros(rjmcmc_space.ndim)
            for i in range(rjmcmc_space.num_models):
                sl = slice(i * rjmcmc_space.num_params, (i + 1) * rjmcmc_space.num_params)
                full_mean[sl] = sample_mean
            sample_mean = full_mean
    return sample_cov, sample_mean


def _register_rjmcmc_jumps(
    sampler,
    rjmcmc_space,
    *,
    birth_weight: float,
    death_weight: float,
    nmodel_weight: float,
    swap_weight: float,
    de_weight: float,
    de_min_fill: int,
) -> None:
    """Register the RJ jump set on a freshly constructed sampler.

    Shared tail of both ``from_rjmcmc`` classmethods: registers the ONE
    combined birth-death kernel (separate constant-weight birth/death jumps
    violate detailed balance), the model-index jump, and the source-swap
    proposal for multi-model spaces, the min-fill-gated
    :class:`~impulse.proposals.EarlyDE` difference move, and enables
    per-model chain statistics.
    """
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
            sampler.add_custom_jump(
                rjmcmc_space.get_birth_death_proposal(), birth_weight + death_weight
            )
        sampler.add_custom_jump(rjmcmc_space.get_nmodel_jump(), nmodel_weight)
        sampler.add_custom_jump(rjmcmc_space.get_source_swap_proposal(), swap_weight)
    if de_weight > 0:
        sampler.add_custom_jump(make_early_de(de_min_fill), de_weight)
    sampler.multi_chain_stats.enable_per_model(
        rjmcmc_space.num_models,
        rjmcmc_space.num_params,
    )


class _PTSamplerBase:
    """Shared engine behind :class:`impulse.PTSampler` and :class:`impulse.RJPTSampler`.

    Internal — instantiate one of the public subclasses instead. Subclass
    hook points (all with PTSampler-appropriate defaults):

    ``_load_checkpoint(path)``
        Return the unpickled sampler to resume from (must rebind callables).
    ``_write_checkpoint()``
        Write the end-of-iteration checkpoint.
    ``_prepare_run(resumed)``
        Runs after resume handling, before the sampling loop (e.g. legacy
        attribute back-fill, NUTS diagnostics setup / warmup).
    ``_on_adaptation_freeze()``
        Runs exactly once at the ``num_adapt`` freeze transition.
    ``_post_mh_step(adapting)``
        Runs after the MH step / acceptance reporting, before the save
        block (e.g. interleaved NUTS transitions).
    ``_save_flush()``
        Flushes buffered outputs at every save boundary and at the end.
    """

    # Subclasses override so log records keep their historical logger
    # names ("impulse.samplers" / "impulse.rjpt_sampler"); class attribute,
    # so it is never pickled into checkpoints.
    _logger = logger

    def __init__(
        self,
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
        outdir: str = "./chains",
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
        self.lnlike = _function_wrapper(
            lnlike, loglargs, loglkwargs, vectorized=vectorized, jax=jax, threads=threads
        )
        self.lnprior = _function_wrapper(
            lnprior, logpargs, logpkwargs, vectorized=vectorized, jax=jax, threads=threads
        )

        self.rngs = setup_seeds(seed, ntemps)

        self.ptstate = PTState(
            self.ndim,
            ntemps,
            swap_steps=swap_steps,
            min_temp=min_temp,
            max_temp=max_temp,
            temp_step=temp_step,
            ladder=ladder,
            inf_temp=inf_temp,
            adapt_t0=adapt_t0,
            adapt_nu=adapt_nu,
        )
        self.multi_chain_stats = setup_chain_stats(
            ndim,
            self.ptstate,
            self.rngs,
            groups,
            sample_cov,
            sample_mean,
            buffer_size,
            self.ptstate.ladder,
        )
        self.proposal_bundle = setup_standard_jumps(
            self.multi_chain_stats, am_weight, scam_weight, de_weight
        )

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume
        # NEVER store the _UNSET sentinel on self (it must not end up in
        # pickled checkpoints); remember instead whether the caller passed
        # num_adapt explicitly, which controls the resume semantics.
        self._num_adapt_explicit = num_adapt is not _UNSET
        self.num_adapt = None if num_adapt is _UNSET else num_adapt

    # ------------------------------------------------------------------
    # Adaptation freeze
    # ------------------------------------------------------------------

    def _adaptation_active(self, iteration: int) -> bool:
        """True while adaptation may still run at this global iteration.

        The ``getattr`` guards the public ``load_checkpoint(...)`` /
        ``load_rjpt_checkpoint(...)`` -> ``.sample()`` paths: checkpoints
        written before ``num_adapt`` existed produce samplers without the
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
        num_adapt = getattr(self, "num_adapt", None)
        return num_adapt is None or iteration < num_adapt

    def _freeze_adaptive_proposals(self) -> None:
        """Permanently freeze registered proposals that adapt internal state.

        Duck-typed: any proposal exposing a callable ``freeze_adaptation``
        (e.g. :class:`~impulse.flow_proposals.NormalizingFlowProposal`)
        is told to stop refitting. Idempotent.
        """
        for jp in self.proposal_bundle.jump_proposals:
            for prop in jp.proposal_list:
                freeze = getattr(prop, "freeze_adaptation", None)
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
        props = [prop for jp in self.proposal_bundle.jump_proposals for prop in jp.proposal_list]
        names = {getattr(prop, "__name__", "") for prop in props}
        if "birth_proposal" not in names and "death_proposal" not in names:
            return
        deaths = [p for p in props if getattr(p, "__name__", "") == "death_proposal"]
        if deaths and all(callable(getattr(p, "draw_from_prior", None)) for p in deaths):
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
        migrated = migrate_legacy_birth_death(self.proposal_bundle.jump_proposals)
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

    # ------------------------------------------------------------------
    # Custom jumps
    # ------------------------------------------------------------------

    def add_custom_jump(self, proposal, weight):
        """Add a custom proposal distribution to all temperature chains.

        See the public subclasses for the full proposal interface
        documentation.
        """
        self.proposal_bundle.add_jump(proposal, weight)

    # ------------------------------------------------------------------
    # Subclass hook points
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        """Load and return the checkpointed sampler to resume from.

        Subclasses must override with the matching
        :mod:`impulse.resume` loader (rebinding their unpicklable
        callables).
        """
        raise NotImplementedError

    def _write_checkpoint(self) -> None:
        """Write the end-of-iteration checkpoint pickle."""
        checkpoint_sampler(self, path=self.checkpoint_path)

    def _prepare_run(self, resumed: bool) -> None:
        """Hook between resume handling and the sampling loop (default: no-op).

        Parameters
        ----------
        resumed : bool
            True when a checkpoint was found and restored.
        """

    def _on_adaptation_freeze(self) -> None:
        """Runs exactly once when the ``num_adapt`` freeze takes effect."""
        self._freeze_adaptive_proposals()

    def _post_mh_step(self, adapting: bool) -> None:
        """Hook after the MH step, before the save block (default: no-op).

        Parameters
        ----------
        adapting : bool
            Whether adaptation is still active at this iteration.
        """

    def _save_flush(self) -> None:
        """Flush chain files and the acceptance-rate snapshot to disk."""
        self.short_chain.save_chain()
        self.save_chain_acceptance_rates()

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(self, initial_position: np.ndarray, num_iterations: int, thin: int = 1):
        """Run the parallel-tempering sampling loop.

        See the public subclasses for full parameter and resume-semantics
        documentation.
        """

        if self.ptstate.ladder is None:  # this shouldn't happen!
            raise ValueError("PTState ladder is not initialized")
        # setup save chains
        self.short_chain = ShortChain(
            self.ndim,
            self.ntemps,
            self.save_freq,
            iteration=0,
            outdir=self.outdir,
            resume=self.resume,
            thin=thin,
        )
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
        initial_state = SamplerState(
            initial_position,
            lnlike0,
            lnprior0,
            lnprob0,
            accepted=np.ones(self.ntemps),
            temps=self.ptstate.ladder,
        )

        # check for bad initial samples
        if np.any(~np.isfinite(lnlike0)):
            raise ValueError("Some likelihood values are not finite.")
        if np.any(~np.isfinite(lnprior0)):
            raise ValueError("An initial value falls outside the prior bounds.")

        self.state = initial_state

        # look for checkpoint in outdir
        _resumed_from_checkpoint = False
        self.checkpoint_path = check_for_checkpoint(self.outdir)
        if self.resume and self.checkpoint_path is not None:
            _resumed_from_checkpoint = True
            self._logger.info("Resuming from checkpoint: %s", self.checkpoint_path)
            loaded = self._load_checkpoint(self.checkpoint_path)
            # num_adapt resume semantics: an EXPLICITLY passed constructor
            # value (including an explicit None) wins over the checkpointed
            # value, with a warning when they differ; the default keeps the
            # checkpointed value — silently un-freezing a checkpointed
            # freeze would resume a half-frozen kernel (proposals whose
            # frozen state is pickled, e.g. a frozen normalizing flow, stay
            # frozen while everything else adapts again).  getattr guards
            # the public checkpoint-loader->sample() path, where unpickling
            # bypasses __init__ (pre-num_adapt checkpoints lack both
            # attributes).
            constructor_num_adapt = getattr(self, "num_adapt", None)
            num_adapt_explicit = getattr(self, "_num_adapt_explicit", False)
            constructor_resume = self.resume
            constructor_checkpoint_path = self.checkpoint_path
            self.__dict__.update(
                loaded.__dict__
            )  # copy the state from the checkpointed sampler to this one
            # the checkpoint carries the ORIGINAL run's resume flag (often
            # False) and checkpoint path (None until its first checkpoint);
            # keep this run's values or later file handling would truncate
            # instead of append
            self.resume = constructor_resume
            self.checkpoint_path = constructor_checkpoint_path
            checkpoint_num_adapt = getattr(loaded, "num_adapt", None)
            if num_adapt_explicit:
                if checkpoint_num_adapt != constructor_num_adapt:
                    self._logger.warning(
                        "Resume: overriding checkpointed num_adapt=%s with "
                        "the resuming constructor's explicitly passed "
                        "num_adapt=%s. Proposals whose frozen state is "
                        "pickled (e.g. normalizing flows frozen by "
                        "freeze_adaptation) remain frozen regardless: their "
                        "freeze is irreversible and survives the "
                        "checkpoint, so removing or extending the freeze "
                        "only re-enables the other adaptive components.",
                        checkpoint_num_adapt,
                        constructor_num_adapt,
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

        self._prepare_run(_resumed_from_checkpoint)

        _proposals_frozen = False

        for jj in tqdm(
            range(self.short_chain.iteration, num_iterations),
            initial=self.short_chain.iteration,
            total=num_iterations,
            desc="Sampling",
        ):
            adapting = self._adaptation_active(jj)
            if not adapting and not _proposals_frozen:
                self._on_adaptation_freeze()
                _proposals_frozen = True
            self.state = vectorized_mh_step(
                self.state,
                self.proposal_bundle,
                self.lnlike,
                self.lnprior,
                self.rngs[0],
                wrap=self.wrap,
            )
            self.proposal_bundle.report_accepts(self.state.accepted)
            self._post_mh_step(adapting)
            # save before add_state to prevent overwriting unsaved data
            if jj > 0 and jj % self.save_freq == 0:
                self._save_flush()
            self.short_chain.add_state(self.state)
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(
                    self.state, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1]
                )
                if adapting:
                    self.ptstate.adapt_ladder()
                    # adapt_ladder mutates the ladder (aliased by state.temps) in
                    # place, so the tempered lnprobs must be recomputed for the
                    # new temperatures
                    self.state.lnprobs = tempered_lnprobs(
                        self.state.lnlikes, self.state.lnpriors, self.ptstate.ladder
                    )
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
                self._write_checkpoint()
        # save the final iteration too
        self._save_flush()

    # ------------------------------------------------------------------
    # Acceptance-rate reporting
    # ------------------------------------------------------------------

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
            "temperatures": [] if ladder is None else ladder.tolist(),
            "mh": self.proposal_bundle.chain_acceptance_rates(),
            "pt_swap": self.ptstate.compute_accept_ratio() if self.ntemps > 1 else np.array([]),
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
        swap = report["pt_swap"]
        report["pt_swap"] = swap.tolist() if hasattr(swap, "tolist") else list(swap)
        report["per_proposal"] = self.proposal_acceptance_rates()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)
        return path

    # ------------------------------------------------------------------
    # load_chain
    # ------------------------------------------------------------------

    def load_chain(self) -> dict:
        """Load saved chain files from disk.

        Returns
        -------
        dict
            ``samples``, ``lnlike``, ``lnprob``, ``accepted``,
            ``temperature`` arrays with shape ``(ntemps, nsamples, ...)``.
        """
        samples, lnlike, lnprob, accepted, temperature = [], [], [], [], []
        for ii in range(self.ntemps):
            filepath = os.path.join(self.outdir, f"chain_{ii}.txt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Chain file not found: {filepath}")
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            samples.append(data[:, : self.ndim])
            lnlike.append(data[:, self.ndim])
            lnprob.append(data[:, self.ndim + 1])
            accepted.append(data[:, self.ndim + 2])
            temperature.append(data[:, self.ndim + 3])

        return {
            "samples": np.array(samples),
            "lnlike": np.array(lnlike),
            "lnprob": np.array(lnprob),
            "accepted": np.array(accepted),
            "temperature": np.array(temperature),
        }

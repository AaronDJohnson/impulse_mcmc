"""HybridPTSampler — Parallel Tempering with optional NUTS and birth-death moves.

A peer of PTSampler that interleaves MH, NUTS, and PT steps. Both samplers
share the internal parallel-tempering engine in :mod:`impulse._pt_base`
(constructor wiring, sample-loop skeleton, resume handling, acceptance
reporting); this module adds the NUTS transition machinery, per-model
step-size/mass-matrix adaptation, and NUTS diagnostics I/O.
"""

import logging
import os
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

from impulse._pt_base import (
    _UNSET,
    _expand_product_space_cov_mean,
    _PTSamplerBase,
    _register_model_selection_jumps,
)
from impulse.nuts.adapter import PerModelNUTSAdapter
from impulse.nuts.core import NUTSState, nuts_step
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.proposals import DE_MIN_FILL
from impulse.resume import CheckpointMismatchError, checkpoint_sampler, load_hybrid_checkpoint
from impulse.sampler_state import SamplerState, tempered_lnprobs
from impulse.utils import prepare_files
from impulse.wrapping import PeriodicSpec


def _adapter_view(field: str) -> property:
    """Compat property exposing a ``PerModelNUTSAdapter`` field under its 2.0 name.

    impulse 2.0 kept the per-model NUTS adaptation caches as raw private
    attributes on the HybridPTSampler instance; tests and diagnostics poke
    them.  These class-level properties keep that surface readable AND
    writable while the state lives on the adapter — and, being class-level,
    they never enter ``__dict__``, so new checkpoints record only
    ``_nuts_adapter``.  Every access routes through
    :meth:`HybridPTSampler._ensure_nuts_adapter`, which transparently migrates
    2.0-era raw attributes restored by unpickling into an adapter.
    """

    def fget(self: "HybridPTSampler"):
        return getattr(self._ensure_nuts_adapter(), field)

    def fset(self: "HybridPTSampler", value) -> None:
        setattr(self._ensure_nuts_adapter(), field, value)

    return property(fget, fset)


class HybridPTSampler(_PTSamplerBase):
    """Parallel Tempering sampler with optional NUTS and birth-death moves.

    Interleaves Metropolis-Hastings proposals (including product-space
    birth/death model moves), NUTS transitions on active continuous
    parameters, and parallel tempering swaps.

    Parameters
    ----------
    ndim : int
        Dimensionality of the parameter space.
    lnlike : callable
        Log-likelihood function.
    lnprior : callable
        Log-prior function.
    lnlike_grad : callable, optional
        Gradient of the log-likelihood on active continuous parameters.
        Signature: ``(active_params) -> (loglike_value, gradient_array)``.
        If provided, NUTS steps are interleaved after each MH step.
    max_tree_depth : int
        Maximum NUTS tree depth for the cold chain.
    hot_chain_max_depth : int
        Maximum NUTS tree depth for hot (T > 1) chains.
    num_warmup : int
        Number of step-size-finding iterations during NUTS initialisation.
    mass_matrix_type : str
        Mass matrix type for NUTS: ``'unit'``, ``'diagonal'``, or ``'dense'``.
    target_accept : float
        Target acceptance probability for NUTS step-size tuning.
    mass_matrix_adapt_interval : int
        Number of NUTS steps between mass matrix re-estimation.
    mass_matrix_min_samples : int
        Minimum non-divergent cold-chain samples before updating mass matrix.
    step_size_min : float
        Lower bound for adapted NUTS step size (default 1e-4).
    step_size_max : float
        Upper bound for adapted NUTS step size (default 5.0).
    num_adapt : int, optional
        Number of iterations during which adaptation is allowed. Once the
        global iteration counter (which persists across checkpoint resume)
        reaches ``num_adapt``, all adaptation freezes: the
        covariance/mean/SVD recomputes feeding AM/SCAM, the DE sample
        buffer (frozen entirely — a rolling buffer would keep the kernel
        history-dependent; DE keeps proposing from the frozen buffer),
        temperature-ladder adaptation, NUTS dual-averaging step-size
        updates (step sizes stay at their current adapted values), NUTS
        mass-matrix re-estimation, and refits of adaptive custom proposals
        (e.g. normalizing flows). One exception: a model dimension visited
        for the first time after the freeze still gets a one-time lazy
        step-size / unit-mass-matrix initialization, since no adapted
        value exists for it yet. Samples drawn before the freeze are
        warmup and should be discarded for strict asymptotic guarantees.
        ``None`` adapts forever, preserving historical behavior.

        Resume semantics: when ``num_adapt`` is not passed (the default),
        resuming keeps the checkpointed value — un-freezing on resume by
        default would produce a half-frozen kernel, because proposals
        whose frozen state is pickled (e.g. a frozen normalizing flow)
        stay frozen while everything else adapts again. An explicitly
        passed value — including an explicit ``None`` — overrides the
        checkpointed value, with a warning when they differ. Fresh (non
        -resumed) runs treat the default exactly like ``None``.
    buffer_size, groups, sample_mean, sample_cov, loglargs, loglkwargs,
    logpargs, logpkwargs, cov_update, save_freq, scam_weight, am_weight,
    de_weight, de_min_fill, seed, outdir, ntemps, swap_steps, min_temp,
    max_temp, temp_step, ladder, inf_temp, adapt_t0, adapt_nu, resume,
    vectorized
        Same as :class:`PTSampler`.
    """

    _logger = logger

    def __init__(
        self,
        ndim: int,
        lnlike: Callable,
        lnprior: Callable,
        # NUTS (optional)
        lnlike_grad: Optional[Callable] = None,
        max_tree_depth: int = 10,
        hot_chain_max_depth: int = 5,
        num_warmup: int = 200,
        mass_matrix_type: str = "diagonal",
        target_accept: float = 0.8,
        mass_matrix_adapt_interval: int = 200,
        mass_matrix_min_samples: int = 50,
        step_size_min: float = 1e-4,
        step_size_max: float = 5.0,
        # Standard PTSampler args
        buffer_size: int = 50_000,
        groups: Optional[list] = None,
        unmanaged_indices: Optional[list] = None,
        sample_mean: Optional[np.ndarray] = None,
        sample_cov: Optional[np.ndarray] = None,
        loglargs: Optional[tuple] = None,
        loglkwargs: Optional[dict] = None,
        logpargs: Optional[tuple] = None,
        logpkwargs: Optional[dict] = None,
        cov_update: int = 100,
        save_freq: int = 1000,
        scam_weight: float = 30,
        am_weight: float = 15,
        de_weight: float = 50,
        de_min_fill: int = DE_MIN_FILL,
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
        # Shared PT wiring (function wrappers, RNGs, PT state, chain stats,
        # proposal bundle, num_adapt sentinel handling)
        super().__init__(
            ndim,
            lnlike,
            lnprior,
            buffer_size=buffer_size,
            sample_mean=sample_mean,
            sample_cov=sample_cov,
            groups=groups,
            unmanaged_indices=unmanaged_indices,
            loglargs=loglargs,
            loglkwargs=loglkwargs,
            logpargs=logpargs,
            logpkwargs=logpkwargs,
            cov_update=cov_update,
            save_freq=save_freq,
            scam_weight=scam_weight,
            am_weight=am_weight,
            de_weight=de_weight,
            de_min_fill=de_min_fill,
            seed=seed,
            outdir=outdir,
            ntemps=ntemps,
            swap_steps=swap_steps,
            min_temp=min_temp,
            max_temp=max_temp,
            temp_step=temp_step,
            ladder=ladder,
            inf_temp=inf_temp,
            adapt_t0=adapt_t0,
            adapt_nu=adapt_nu,
            resume=resume,
            vectorized=vectorized,
            jax=jax,
            threads=threads,
            periodic=periodic,
            num_adapt=num_adapt,
        )

        # Keep raw references for NUTS gradient building
        self._raw_lnlike = lnlike
        self._raw_lnprior = lnprior

        # NUTS configuration
        self.lnlike_grad = lnlike_grad
        self.nuts_enabled = lnlike_grad is not None
        self.max_tree_depth = max_tree_depth
        self.hot_chain_max_depth = hot_chain_max_depth
        self.num_warmup = num_warmup
        self.target_accept = target_accept

        if isinstance(mass_matrix_type, str):
            self._mass_matrix_type = MassMatrixType(mass_matrix_type)
        else:
            self._mass_matrix_type = mass_matrix_type

        # Per-model NUTS adaptation state (picklable component; keyed by
        # n_active). 2.0 checkpoints carried this state as raw instance
        # attributes instead — see _ensure_nuts_adapter for the migration.
        self._nuts_adapter = PerModelNUTSAdapter(
            mass_matrix_adapt_interval=mass_matrix_adapt_interval,
            mass_matrix_min_samples=mass_matrix_min_samples,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
        )

        # product-space model selection (set by from_product_space)
        self._product_space = None

        # NUTS diagnostics buffer (populated during sampling)
        self._nuts_diag_data: Optional[list] = None

    # ------------------------------------------------------------------
    # NUTS adapter access and 2.0-checkpoint migration
    # ------------------------------------------------------------------

    def _ensure_nuts_adapter(self) -> PerModelNUTSAdapter:
        """Return the NUTS adapter, migrating 2.0-era raw attributes if present.

        impulse 2.0 pickled the per-model NUTS adaptation caches as raw
        dict/set attributes directly on the HybridPTSampler instance.
        Unpickling such a checkpoint (both the ``resume=True``
        ``__dict__.update`` path in ``sample()`` and the public
        ``load_hybrid_checkpoint(...)`` path) leaves those raw entries in
        ``self.__dict__``, where the class-level compat properties shadow
        them; they are consumed here to rebuild the adapter with identical
        state.  Like the legacy birth/death migration this runs on resume,
        but silently — it is an internal representation change with
        identical behavior, so there is nothing to warn about.  A sampler
        with neither the adapter nor raw attributes (pre-NUTS-adaptation
        checkpoints) gets a fresh adapter with default configuration,
        mirroring the historical ``_prepare_run`` back-fill.
        """
        state = self.__dict__
        if PerModelNUTSAdapter.has_legacy_state(state):
            # Raw 2.0 attributes win over any constructor-fresh adapter:
            # they carry the checkpointed adaptation state. Config scalars
            # MISSING from the checkpoint fall back to the resuming
            # constructor's values (the fresh adapter), matching the old
            # __dict__.update-then-backfill semantics.
            state["_nuts_adapter"] = PerModelNUTSAdapter.from_legacy_state(
                state, defaults=state.get("_nuts_adapter")
            )
        adapter = state.get("_nuts_adapter")
        if adapter is None:
            adapter = PerModelNUTSAdapter()
            state["_nuts_adapter"] = adapter
        return adapter

    # Compat views of the adapter state under the historical private names
    # (see _adapter_view). Read/write; never pickled.
    _step_sizes = _adapter_view("step_sizes")
    _mass_matrices = _adapter_view("mass_matrices")
    _dual_averagers = _adapter_view("dual_averagers")
    _nuts_sample_buffers = _adapter_view("sample_buffers")
    _mass_matrix_injected = _adapter_view("mass_matrix_injected")
    _nuts_steps_since_mm_update = _adapter_view("steps_since_mm_update")
    _mass_matrix_adapt_interval = _adapter_view("mass_matrix_adapt_interval")
    _mass_matrix_min_samples = _adapter_view("mass_matrix_min_samples")
    _step_size_min = _adapter_view("step_size_min")
    _step_size_max = _adapter_view("step_size_max")

    # ------------------------------------------------------------------
    # Adaptation freeze
    # ------------------------------------------------------------------

    def _finalize_step_sizes(self) -> None:
        """Replace primal dual-averaging iterates with smoothed step sizes.

        Called once at the ``num_adapt`` freeze transition; see
        :meth:`impulse.nuts.adapter.PerModelNUTSAdapter.finalize_step_sizes`
        for the full rationale (the smoothed ``DualAveraging.finalize()``
        value is pinned, not the noisy primal iterate).
        """
        self._ensure_nuts_adapter().finalize_step_sizes()

    # ------------------------------------------------------------------
    # Shared-engine hook points
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        """Load a hybrid-sampler checkpoint, rebinding the unpicklable callables."""
        return load_hybrid_checkpoint(
            path,
            lnlike=self.lnlike,
            lnprior=self.lnprior,
            raw_lnlike=self._raw_lnlike,
            raw_lnprior=self._raw_lnprior,
            lnlike_grad=self.lnlike_grad,
        )

    def _write_checkpoint(self) -> None:
        """Checkpoint, omitting the raw/gradient callables (unpicklable)."""
        checkpoint_sampler(
            self,
            path=self.checkpoint_path,
            omit=("lnlike", "lnprior", "_raw_lnlike", "_raw_lnprior", "lnlike_grad"),
        )

    def _capture_subclass_state(self, arrays: dict, meta: dict) -> None:
        """Add hybrid-sampler-specific state (NUTS adapter, diagnostics row count).

        The product space and gradient callables are NOT captured — resume
        reconstructs them via ``from_product_space`` / the constructor.
        """
        adapter = self._ensure_nuts_adapter()
        ad_arrays, ad_meta = adapter.get_checkpoint_state()
        for k, v in ad_arrays.items():
            arrays[f"nuts.{k}"] = v
        meta["nuts_adapter"] = ad_meta
        meta["nuts_enabled"] = bool(self.nuts_enabled)
        if hasattr(self, "_nuts_diag_rows_written"):
            meta["nuts_diag_rows_written"] = int(self._nuts_diag_rows_written)

    def _restore_subclass_state(self, arrays: dict, meta: dict) -> None:
        """Restore hybrid-sampler-specific state into the freshly constructed adapter."""
        # NUTS on/off is derived from lnlike_grad, not a registered proposal,
        # so the proposal-name guard cannot catch its absence. Verify it
        # here: a checkpoint written with NUTS but reconstructed without
        # lnlike_grad (or vice versa) would silently run a different
        # RNG-consumption path and diverge from the checkpoint.
        ck_nuts = meta.get("nuts_enabled")
        if ck_nuts is not None and bool(ck_nuts) != bool(self.nuts_enabled):
            raise CheckpointMismatchError(
                f"nuts_enabled mismatch: checkpoint has nuts_enabled={bool(ck_nuts)} "
                f"but the reconstructed sampler has nuts_enabled={bool(self.nuts_enabled)}. "
                "Reconstruct with the same lnlike_grad argument used for the original run."
            )
        ad_arrays = {k[len("nuts.") :]: v for k, v in arrays.items() if k.startswith("nuts.")}
        self._ensure_nuts_adapter().set_checkpoint_state(ad_arrays, meta["nuts_adapter"])
        if "nuts_diag_rows_written" in meta:
            self._nuts_diag_rows_written = int(meta["nuts_diag_rows_written"])

    def _prepare_run(self, resumed: bool) -> None:
        """Legacy attribute back-fill, NUTS diagnostics setup, NUTS warmup.

        Runs after resume handling, before the sampling loop — exactly
        where the pre-refactor ``sample()`` performed these steps.
        """
        # Backward-compat: 2.0 checkpoints carry the NUTS adaptation caches
        # as raw instance attributes (migrated into the adapter here), and
        # old checkpoints may predate num_adapt entirely.
        self._ensure_nuts_adapter()
        if not hasattr(self, "num_adapt"):
            self.num_adapt = None

        # NUTS diagnostics file
        if self.nuts_enabled:
            self._nuts_diag_data = []
            self._nuts_diag_path = os.path.join(self.outdir, "nuts_diagnostics.txt")
            prepare_files([self._nuts_diag_path], resume=self.resume)
            if resumed:
                # mirror the chain-file truncation: drop diagnostics rows
                # written after the checkpoint (the resumed iterations
                # re-generate them)
                self._truncate_nuts_diagnostics()
            else:
                self._nuts_diag_rows_written = 0

        # NUTS warmup: find initial step sizes
        if self.nuts_enabled:
            for k in range(self.ntemps):
                logp_and_grad, active_idx = self._make_tempered_logp_grad(k, self.state)
                active_params = self.state.positions[k][active_idx]
                if len(active_params) > 0:
                    self._nuts_adapter.get_or_find_step_size(
                        k,
                        active_params,
                        logp_and_grad,
                        self.rngs[k],
                        self.target_accept,
                    )

    def _on_adaptation_freeze(self) -> None:
        """Freeze adaptive proposals and the NUTS step sizes."""
        super()._on_adaptation_freeze()
        # Freeze the smoothed dual-averaging step sizes, not the
        # noisy primal iterates tracked during adaptation
        self._finalize_step_sizes()

    def _post_mh_step(self, adapting: bool) -> None:
        """Step B: NUTS transition on active continuous params (if enabled)."""
        if self.nuts_enabled:
            self.state, cold_diag = self._nuts_step_all_chains(self.state, adapt=adapting)
            if cold_diag is not None:
                # initialized in _prepare_run whenever nuts_enabled
                assert self._nuts_diag_data is not None
                self._nuts_diag_data.append(cold_diag)
            if adapting:
                self._maybe_adapt_mass_matrices()

    def _save_flush(self) -> None:
        """Flush chain files, acceptance rates, and NUTS diagnostics."""
        super()._save_flush()
        if self.nuts_enabled:
            self._flush_nuts_diagnostics()

    # ------------------------------------------------------------------
    # from_product_space classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_product_space(
        cls,
        product_space,
        lnlike_grad: Optional[Callable] = None,
        birth_weight: float = 15,
        death_weight: float = 15,
        nmodel_weight: float = 10,
        swap_weight: float = 15,
        am_weight: float = 15,
        scam_weight: float = 15,
        de_weight: float = 15,
        de_min_fill: int = 100,
        **kwargs,
    ) -> "HybridPTSampler":
        """Construct a HybridPTSampler pre-configured for product-space model selection.

        Parameters
        ----------
        product_space : BirthDeathProductSpace
            Configured birth-death product space.
        lnlike_grad : callable, optional
            Gradient function for NUTS on active continuous params.
        birth_weight, death_weight, nmodel_weight, swap_weight : float
            Relative weights for the model-move proposals.  Birth and death are
            registered as ONE combined kernel selected with weight
            ``birth_weight + death_weight``; the birth/death split is
            governed by the space's ``prob_schedule`` (registering them as
            separate constant-weight jumps violates detailed balance).
        am_weight, scam_weight, de_weight : float
            Relative weights for standard MH proposals.
        de_min_fill : int
            Minimum per-model buffer fill before the DE difference move
            activates; see :func:`impulse.proposals.de`.
        **kwargs
            Additional keyword arguments forwarded to ``__init__``.

        Notes
        -----
        For a single-model space (``product_space.num_models == 1``) the
        birth-death kernel, the model-index jump, and the source-swap
        proposal are all skipped — none is meaningful with one model, and
        the birth-death kernel itself rejects ``max_sources < 2`` — so
        only the standard continuous jumps (AM, SCAM, DE) are
        registered.  The birth-death kernel is also skipped when
        ``birth_weight + death_weight == 0``.

        The ``de`` move is min-fill-gated: it activates as soon as the
        current model's buffer holds ``de_min_fill`` samples, returning
        the current position unchanged below the threshold; see
        ``PTSampler.from_product_space`` for the full rationale.
        """
        # Expand per-source sample_cov / sample_mean to full product space
        sample_cov, sample_mean = _expand_product_space_cov_mean(product_space, kwargs)

        sampler = cls(
            ndim=product_space.ndim,
            lnlike=product_space.get_loglikelihood,
            lnprior=product_space.get_logprior,
            lnlike_grad=lnlike_grad,
            groups=product_space.get_default_groups(),
            # get_default_groups deliberately omits the model index: it is moved
            # by the birth/death kernel and nmodel_jump, never by am/scam/de.
            # Declare that so ChainStats does not warn about an uncovered index.
            unmanaged_indices=[product_space.ndim - 1],
            sample_cov=sample_cov,
            sample_mean=sample_mean,
            am_weight=am_weight,
            scam_weight=scam_weight,
            de_weight=de_weight,
            de_min_fill=de_min_fill,
            **kwargs,
        )
        sampler._product_space = product_space
        _register_model_selection_jumps(
            sampler,
            product_space,
            birth_weight=birth_weight,
            death_weight=death_weight,
            nmodel_weight=nmodel_weight,
            swap_weight=swap_weight,
        )
        return sampler

    # ------------------------------------------------------------------
    # add_custom_jump
    # ------------------------------------------------------------------

    def add_custom_jump(self, proposal, weight):
        """Add a custom proposal distribution to all temperature chains.

        Parameters
        ----------
        proposal : callable
            Proposal with signature ``proposal(chain_stats: ChainStats) ->
            (new_sample: np.ndarray, qxy: float)``, where ``qxy`` is the
            log proposal-density ratio

                ``qxy = log q(x | y) - log q(y | x)``,

            with ``x`` the CURRENT sample, ``y`` the PROPOSED sample, and
            ``q(a | b)`` the density of proposing ``a`` from ``b``.
            ``qxy`` is ADDED to the log-posterior ratio in the
            Metropolis-Hastings acceptance, so positive ``qxy`` favors
            acceptance. Symmetric proposals (``q(y|x) == q(x|y)``, e.g. a
            Gaussian random walk) must return ``qxy = 0.0``; for an
            asymmetric example (a multiplicative random walk whose ``qxy``
            is the log-Jacobian of the rescaling) see the "Custom
            proposals" section of the README and docs.

            Checkpoints store no code, so the proposal is not serialized.
            To resume, re-register the same proposals in the same order
            with the same weights; the sampler verifies this against the
            checkpoint and raises ``CheckpointMismatchError`` otherwise.
            Callable classes must define a ``__name__`` attribute; it keys
            acceptance-rate reports and is what the resume check matches
            on. A proposal that adapts internal state can persist it by
            implementing ``get_checkpoint_state`` / ``set_checkpoint_state``.
        weight : float
            Relative weight for this proposal (normalized against all
            registered proposals).
        """
        super().add_custom_jump(proposal, weight)

    # ------------------------------------------------------------------
    # Public: mass matrix injection
    # ------------------------------------------------------------------

    def set_mass_matrix(self, n_active: int, mass_matrix: MassMatrix):
        """Inject an external mass matrix (e.g., Fisher-based) for a given dimension.

        The injected matrix is preserved for the entire run: online
        mass-matrix adaptation never overwrites an injected entry (only the
        dual-averaging step size continues to re-tune against it).

        Parameters
        ----------
        n_active : int
            Number of active continuous parameters this matrix applies to.
        mass_matrix : MassMatrix
            Mass matrix to use for NUTS proposals.

        Notes
        -----
        A Fisher information matrix approximates the posterior PRECISION,
        which under the Stan convention (inverse metric = posterior
        covariance) is exactly the mass matrix. Build it with
        :meth:`MassMatrix.from_precision` (or the raw ``MassMatrix``
        constructor) — never :meth:`MassMatrix.from_covariance`, which
        inverts its argument and would silently install the inverse of
        the intended metric.
        """
        self._ensure_nuts_adapter().set_mass_matrix(n_active, mass_matrix, self.target_accept)

    # ------------------------------------------------------------------
    # Internal: NUTS helpers
    # ------------------------------------------------------------------

    def _get_active_indices(self, params):
        """Return indices of active continuous parameters.

        For product-space models, active params = first
        ``(nmodel+1)*num_params``.  For fixed-dim models, active = all params.
        """
        if self._product_space is not None:
            layout = self._product_space.layout
            return layout.active_indices(layout.model_index_of(params))
        return np.arange(self.ndim)

    def _make_tempered_logp_grad(self, chain_idx, state):
        """Build a tempered ``(x_active) -> (logp, grad)`` for one chain.

        Embeds the active params back into the full vector for
        likelihood/prior evaluation and scales by ``1/T``.
        """
        full_params = state.positions[chain_idx].copy()
        T = state.temps[chain_idx]
        active_idx = self._get_active_indices(full_params)
        lnlike_grad = self.lnlike_grad
        # only reachable on the NUTS path, which requires lnlike_grad
        assert lnlike_grad is not None
        raw_lnprior = self._raw_lnprior

        def logp_and_grad(x_active):
            trial = full_params.copy()
            trial[active_idx] = x_active

            # Check prior FIRST — cheap and catches out-of-bounds before
            # potentially expensive/unstable gradient computation.
            if self._product_space is not None:
                # All source blocks (active and inactive), model index excluded.
                lp = raw_lnprior(trial[: self._product_space.layout.nmodel_index])
            else:
                lp = raw_lnprior(trial)

            if not np.isfinite(lp):
                return -np.inf, np.zeros_like(x_active)

            ll, grad_ll = lnlike_grad(x_active)

            if not np.isfinite(ll) or not np.all(np.isfinite(grad_ll)):
                return -np.inf, np.zeros_like(x_active)

            logp = ll / T + lp
            grad = grad_ll / T
            return logp, grad

        return logp_and_grad, active_idx

    def _nuts_step_all_chains(self, state, adapt: bool = True):
        """Run one NUTS transition on each temperature chain.

        Parameters
        ----------
        state : SamplerState
            Current state of all chains.
        adapt : bool
            When False (past the ``num_adapt`` freeze), dual-averaging
            step-size updates are skipped — step sizes stay at their
            current adapted values — and cold-chain samples are no longer
            buffered for mass-matrix re-estimation.

        Returns updated SamplerState and cold-chain diagnostics dict.
        """
        new_positions = state.positions.copy()
        new_lnlikes = state.lnlikes.copy()
        new_lnpriors = state.lnpriors.copy()
        cold_diag = None

        for k in range(self.ntemps):
            params = state.positions[k]
            active_idx = self._get_active_indices(params)
            active_params = params[active_idx].copy()
            n_active = len(active_params)

            if n_active == 0:
                continue

            logp_and_grad, _ = self._make_tempered_logp_grad(k, state)
            rng = self.rngs[k]

            step_size = self._nuts_adapter.get_or_find_step_size(
                k,
                active_params,
                logp_and_grad,
                rng,
                self.target_accept,
            )

            mass_matrix = self._nuts_adapter.mass_matrix_for(n_active)

            logp_val, grad_val = logp_and_grad(active_params)
            if not np.isfinite(logp_val) or not np.all(np.isfinite(grad_val)):
                continue

            nuts_state = NUTSState(
                position=active_params,
                logp=logp_val,
                grad=grad_val,
                step_size=step_size,
                mass_matrix=mass_matrix,
            )

            depth = self.max_tree_depth if k == 0 else self.hot_chain_max_depth
            nuts_state = nuts_step(nuts_state, logp_and_grad, rng, max_tree_depth=depth)

            # Write active params back
            new_params = new_positions[k].copy()
            new_params[active_idx] = nuts_state.position
            if self.wrap is not None:
                new_params = self.wrap.apply(new_params)
            new_positions[k] = new_params

            # Recompute untempered lnlike and lnprior
            new_lnlikes[k] = (
                self._raw_lnlike(nuts_state.position)
                if self._product_space is None
                else self.lnlike(new_positions[k : k + 1])[0]
            )
            new_lnpriors[k] = (
                self._raw_lnprior(new_params)
                if self._product_space is None
                else self.lnprior(new_positions[k : k + 1])[0]
            )

            # Online step size adaptation via dual averaging (adapter-owned).
            # Past the num_adapt freeze (adapt=False) DA is not updated and
            # the step size keeps its last adapted value, so the kernel is
            # fixed.
            if adapt:
                self._nuts_adapter.update_step_size(
                    k,
                    n_active,
                    nuts_state.mean_accept_prob,
                    nuts_state.step_size,
                )

            # Cold chain: collect non-divergent samples for mass matrix
            # adaptation (frozen along with mass-matrix re-estimation)
            if adapt and k == 0 and not nuts_state.divergent:
                self._nuts_adapter.buffer_cold_sample(n_active, nuts_state.position)

            # Cold chain diagnostics
            if k == 0:
                cold_diag = {
                    "tree_depth": nuts_state.tree_depth,
                    "divergent": int(nuts_state.divergent),
                    "energy_error": nuts_state.energy_error,
                    "step_size": self._nuts_adapter.current_step_size(k, n_active),
                    "mean_accept_prob": nuts_state.mean_accept_prob,
                    "n_active": n_active,
                }

        new_lnprobs = tempered_lnprobs(new_lnlikes, new_lnpriors, state.temps)
        new_state = SamplerState(
            new_positions,
            new_lnlikes,
            new_lnpriors,
            new_lnprobs,
            state.accepted,
            state.temps,
        )
        return new_state, cold_diag

    def _cold_active_params(self):
        """Fresh copy of the cold chain's active continuous parameters."""
        cold_params = self.state.positions[0]
        active_idx = self._get_active_indices(cold_params)
        return cold_params[active_idx].copy()

    def _cold_logp_and_grad(self):
        """The cold chain's tempered ``(x_active) -> (logp, grad)``."""
        return self._make_tempered_logp_grad(0, self.state)[0]

    def _maybe_adapt_mass_matrices(self):
        """Periodically re-estimate NUTS mass matrices from cold-chain samples.

        Called once per iteration from the sample loop; delegates to
        :meth:`impulse.nuts.adapter.PerModelNUTSAdapter.maybe_adapt_mass_matrices`
        (interval/min-sample gating, injected-matrix protection, step-size
        recalibration, and trial-step validation), passing the cold chain's
        position, tempered gradient, and RNG stream. Kept as a sampler
        method so the ``_post_mh_step`` call site stays patchable.
        """
        self._nuts_adapter.maybe_adapt_mass_matrices(
            self._mass_matrix_type,
            self.target_accept,
            self._cold_active_params,
            self._cold_logp_and_grad,
            self.rngs[0],
        )

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(
        self,
        initial_position: np.ndarray,
        num_iterations: int,
        thin: int = 1,
    ):
        """Run the hybrid MH + NUTS + PT sampling loop.

        Parameters
        ----------
        initial_position : array_like
            Starting position(s). See :func:`setup_initial_position`.
        num_iterations : int
            Total number of MCMC iterations.
        thin : int
            Thinning factor for saved samples.

        Notes
        -----
        When ``num_adapt`` is set, all adaptation stops once the global
        iteration counter reaches it; samples before the freeze are warmup
        and should be discarded for strict asymptotic guarantees.

        Resume semantics: ``num_iterations`` is a GLOBAL iteration target —
        a resumed run continues from the checkpointed iteration counter up
        to ``num_iterations``, so pass the total, not the increment.  A
        checkpoint is written at the END of every iteration ``jj`` with
        ``jj > 0`` and ``jj % save_freq == 0``, capturing the sampler after
        that iteration fully completed (post NUTS, post PT-swap, post
        adaptation), including every RNG stream.  On resume the chain files
        (and the NUTS diagnostics file) are truncated back to the
        checkpointed flushed-row count and all iterations after the
        checkpoint are re-generated bit-identically, so an interrupted run
        resumed to ``N`` total iterations produces chain files identical to
        a single uninterrupted ``N``-iteration run.
        """
        return super().sample(initial_position, num_iterations, thin)

    # ------------------------------------------------------------------
    # NUTS diagnostics I/O
    # ------------------------------------------------------------------

    def _ensure_nuts_diag_rows_written(self):
        """Lazily initialize ``_nuts_diag_rows_written`` for legacy checkpoints.

        Samplers unpickled from checkpoints written before row tracking
        existed lack the attribute.  Mirroring
        :meth:`impulse.file_io.ShortChain._ensure_rows_written`, the counter
        must never restart at 0 — an undercount pickled forward would make
        the NEXT resume's :meth:`_truncate_nuts_diagnostics` rewrite the
        diagnostics file as a tiny prefix — so it is re-seeded from the
        current on-disk line count.  Called before every read/append of the
        counter so the value is correct regardless of whether a flush or a
        truncation runs first after unpickling.
        """
        if hasattr(self, "_nuts_diag_rows_written"):
            return
        path = getattr(self, "_nuts_diag_path", None)
        if path is not None and os.path.exists(path):
            with open(path, "r") as fp:
                self._nuts_diag_rows_written = sum(1 for _ in fp)
        else:
            self._nuts_diag_rows_written = 0

    def _flush_nuts_diagnostics(self):
        """Write buffered NUTS diagnostics to disk."""
        if not self._nuts_diag_data:
            return
        # Seed the row counter from disk BEFORE appending (legacy
        # pre-row-tracking checkpoints lack the attribute).
        self._ensure_nuts_diag_rows_written()
        rows = np.array(
            [
                [
                    d["tree_depth"],
                    d["divergent"],
                    d["energy_error"],
                    d["step_size"],
                    d["mean_accept_prob"],
                    d.get("n_active", 0),
                ]
                for d in self._nuts_diag_data
            ]
        )
        with open(self._nuts_diag_path, "a") as fp:
            np.savetxt(fp, rows, fmt="%.18e")
        self._nuts_diag_data = []
        self._nuts_diag_rows_written += len(rows)

    def _truncate_nuts_diagnostics(self):
        """Truncate the NUTS diagnostics file to the checkpointed row count.

        Resume counterpart of
        :meth:`impulse.file_io.ShortChain.truncate_files_to_saved`:
        diagnostics rows written after the checkpoint was taken belong to
        iterations the resumed run re-generates, so they are dropped to
        avoid duplication.  Checkpoints that predate row tracking carry no
        ``_nuts_diag_rows_written``; for those the counter is re-seeded from
        the current on-disk line count (see
        :meth:`_ensure_nuts_diag_rows_written`), making this call a no-op —
        the historical append-only behavior.
        """
        if not os.path.exists(self._nuts_diag_path):
            return
        self._ensure_nuts_diag_rows_written()
        rows = self._nuts_diag_rows_written
        with open(self._nuts_diag_path, "r") as fp:
            lines = fp.readlines()
        if len(lines) > rows:
            with open(self._nuts_diag_path, "w") as fp:
                fp.writelines(lines[:rows])

    # ------------------------------------------------------------------
    # proposal_acceptance_rates
    # ------------------------------------------------------------------

    def proposal_acceptance_rates(self) -> dict:
        """Per-proposal acceptance statistics aggregated across chains.

        Returns
        -------
        dict
            ``{name: {calls, accepts, rate, per_chain: [...]}}``
        """
        return super().proposal_acceptance_rates()

    def chain_acceptance_rates(self) -> dict:
        """Per-chain MH acceptance summary, plus PT swap rates.

        Returns
        -------
        dict
            ``temperatures`` (list of T per chain),
            ``mh`` (list per chain: ``{calls, accepts, rate, per_proposal}``),
            ``pt_swap`` (np.ndarray of length ``ntemps - 1`` with the
            accept rate for each neighbour-pair swap, or empty if
            ``ntemps == 1``).
        """
        return super().chain_acceptance_rates()

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
        return super().save_chain_acceptance_rates(path)

    # ------------------------------------------------------------------
    # load_chain
    # ------------------------------------------------------------------

    def load_chain(self) -> dict:
        """Load saved chain files from disk.

        Returns
        -------
        dict
            ``samples``, ``lnlike``, ``lnprob``, ``accepted``, ``temperature``
            arrays with shape ``(ntemps, nsamples, ...)``.
            If NUTS was enabled, also includes cold-chain ``tree_depth``,
            ``divergent``, ``energy_error``, ``step_size``, ``mean_accept_prob``.
        """
        result = super().load_chain()

        # NUTS extras
        nuts_path = os.path.join(self.outdir, "nuts_diagnostics.txt")
        if os.path.exists(nuts_path):
            nuts_data = np.loadtxt(nuts_path)
            if nuts_data.ndim == 1:
                nuts_data = nuts_data.reshape(1, -1)
            result["tree_depth"] = nuts_data[:, 0].astype(int)
            result["divergent"] = nuts_data[:, 1].astype(bool)
            result["energy_error"] = nuts_data[:, 2]
            result["step_size"] = nuts_data[:, 3]
            result["mean_accept_prob"] = nuts_data[:, 4]
            if nuts_data.shape[1] >= 6:
                result["n_active"] = nuts_data[:, 5].astype(int)

        return result

    # ------------------------------------------------------------------
    # get_diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Summary diagnostics from the sampling run.

        Returns
        -------
        dict
            Keys: ``num_divergent``, ``num_max_depth``, ``mean_tree_depth``,
            ``mean_accept_prob``, ``final_step_size``, ``pt_swap_accept``.
        """
        diag: dict = {}

        # PT swap info
        if self.ptstate is not None:
            diag["pt_swap_accept"] = self.ptstate.compute_accept_ratio().tolist()

        # NUTS diagnostics from disk
        nuts_path = os.path.join(self.outdir, "nuts_diagnostics.txt")
        if os.path.exists(nuts_path):
            data = np.loadtxt(nuts_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            tree_depth = data[:, 0].astype(int)
            divergent = data[:, 1].astype(bool)
            accept_prob = data[:, 4]
            diag["num_divergent"] = int(np.sum(divergent))
            diag["num_max_depth"] = int(np.sum(tree_depth >= self.max_tree_depth))
            diag["mean_tree_depth"] = float(np.mean(tree_depth))
            diag["mean_accept_prob"] = float(np.mean(accept_prob))
            diag["final_step_size"] = float(data[-1, 3])

        # Per-proposal acceptance rates
        diag["proposal_acceptance"] = self.proposal_acceptance_rates()

        return diag

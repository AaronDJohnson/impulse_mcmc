"""Per-model NUTS adaptation state for the hybrid sampler (internal module).

:class:`PerModelNUTSAdapter` owns the per-model NUTS adaptation caches and
logic that :class:`impulse.experimental.HybridPTSampler` historically kept as raw instance
attributes: per-``(chain, dimension)`` step sizes and dual averagers,
per-dimension mass matrices, cold-chain sample buffers, injected
(Fisher) mass-matrix bookkeeping, the periodic mass-matrix re-estimation
with trial-step validation, and the freeze-time step-size finalization.
"Per model" means keyed by the number of ACTIVE continuous parameters
(``n_active``), which in RJ runs varies with the model index.

The adapter is a plain picklable component: ``HybridPTSampler`` constructs one
in ``__init__`` and pickles it inside checkpoints.  Checkpoints written by
impulse 2.0 carry the raw attributes on the sampler instance instead;
``HybridPTSampler`` rebuilds the adapter from them on resume via
:meth:`PerModelNUTSAdapter.from_legacy_state`.

Everything here is internal API — nothing is exported publicly, and the
adapter consumes no random numbers beyond those the pre-extraction sampler
code consumed (the RNG streams are passed in at call time and never
stored).
"""

from typing import Callable, Optional

import numpy as np

from impulse.nuts.core import NUTSState, nuts_step
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.warmup import DualAveraging, find_reasonable_step_size, regularized_mass_matrix


class PerModelNUTSAdapter:
    """Per-model NUTS step-size and mass-matrix adaptation state.

    Parameters
    ----------
    mass_matrix_adapt_interval : int
        Number of NUTS steps between mass matrix re-estimation.
    mass_matrix_min_samples : int
        Minimum non-divergent cold-chain samples before updating the mass
        matrix.
    step_size_min : float
        Lower bound for adapted NUTS step sizes.
    step_size_max : float
        Upper bound for adapted NUTS step sizes.

    Notes
    -----
    All state is picklable (floats, ``MassMatrix``, ``DualAveraging``,
    lists of arrays).  The 2.0-era raw attribute names on ``HybridPTSampler``
    map onto the adapter fields as follows:

    ======================================  ===========================
    2.0 HybridPTSampler instance attribute      adapter field
    ======================================  ===========================
    ``_step_sizes``                         ``step_sizes``
    ``_mass_matrices``                      ``mass_matrices``
    ``_dual_averagers``                     ``dual_averagers``
    ``_nuts_sample_buffers``                ``sample_buffers``
    ``_mass_matrix_injected``               ``mass_matrix_injected``
    ``_nuts_steps_since_mm_update``         ``steps_since_mm_update``
    ``_mass_matrix_adapt_interval``         ``mass_matrix_adapt_interval``
    ``_mass_matrix_min_samples``            ``mass_matrix_min_samples``
    ``_step_size_min``                      ``step_size_min``
    ``_step_size_max``                      ``step_size_max``
    ======================================  ===========================
    """

    # 2.0-era raw instance attribute names (see from_legacy_state)
    _LEGACY_ATTRS = (
        "_step_sizes",
        "_mass_matrices",
        "_dual_averagers",
        "_nuts_sample_buffers",
        "_mass_matrix_injected",
        "_nuts_steps_since_mm_update",
        "_mass_matrix_adapt_interval",
        "_mass_matrix_min_samples",
        "_step_size_min",
        "_step_size_max",
    )

    def __init__(
        self,
        mass_matrix_adapt_interval: int = 200,
        mass_matrix_min_samples: int = 50,
        step_size_min: float = 1e-4,
        step_size_max: float = 5.0,
    ) -> None:
        self.mass_matrix_adapt_interval = mass_matrix_adapt_interval
        self.mass_matrix_min_samples = mass_matrix_min_samples
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max

        self.step_sizes: dict = {}  # (chain_idx, n_active) -> float
        self.mass_matrices: dict = {}  # n_active -> MassMatrix
        self.dual_averagers: dict = {}  # (chain_idx, n_active) -> DualAveraging
        self.sample_buffers: dict = {}  # n_active -> list[np.ndarray]
        self.mass_matrix_injected: set = set()  # n_active with externally set matrices
        self.steps_since_mm_update: dict = {}  # n_active -> int counter

    # ------------------------------------------------------------------
    # 2.0-checkpoint migration
    # ------------------------------------------------------------------

    @classmethod
    def has_legacy_state(cls, state: dict) -> bool:
        """True if ``state`` (an instance ``__dict__``) carries 2.0-era raw caches."""
        return any(key in state for key in cls._LEGACY_ATTRS)

    @classmethod
    def from_legacy_state(
        cls, state: dict, defaults: Optional["PerModelNUTSAdapter"] = None
    ) -> "PerModelNUTSAdapter":
        """Build an adapter from 2.0-era raw HybridPTSampler attributes, consuming them.

        impulse 2.0 pickled the adaptation caches as raw attributes
        directly on the sampler instance.  This constructor adopts those
        objects (not copies — DualAveraging instances, mass matrices, and
        sample buffers continue exactly where the checkpoint left them)
        and POPS the raw entries out of ``state`` so a re-checkpoint
        pickles the adapter instead.  Missing entries fall back to the
        same defaults the historical ``_prepare_run`` back-fill used for
        checkpoints predating each attribute.

        Parameters
        ----------
        state : dict
            The resumed sampler's ``__dict__``; mutated in place.

        Returns
        -------
        PerModelNUTSAdapter
            Adapter carrying the checkpointed adaptation state.
        """
        fallback = defaults if defaults is not None else cls()
        adapter = cls(
            mass_matrix_adapt_interval=state.pop(
                "_mass_matrix_adapt_interval", fallback.mass_matrix_adapt_interval
            ),
            mass_matrix_min_samples=state.pop(
                "_mass_matrix_min_samples", fallback.mass_matrix_min_samples
            ),
            step_size_min=state.pop("_step_size_min", fallback.step_size_min),
            step_size_max=state.pop("_step_size_max", fallback.step_size_max),
        )
        adapter.step_sizes = state.pop("_step_sizes", {})
        adapter.mass_matrices = state.pop("_mass_matrices", {})
        adapter.dual_averagers = state.pop("_dual_averagers", {})
        adapter.sample_buffers = state.pop("_nuts_sample_buffers", {})
        adapter.mass_matrix_injected = state.pop("_mass_matrix_injected", set())
        adapter.steps_since_mm_update = state.pop("_nuts_steps_since_mm_update", {})
        return adapter

    # ------------------------------------------------------------------
    # No-code-execution checkpoint (npz arrays + JSON metadata)
    # ------------------------------------------------------------------

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        """Serialize the adapter to ``(arrays, meta)`` for the new checkpoint.

        Tuple-keyed caches (keyed by ``(chain, n_active)`` or ``n_active``)
        are flattened into JSON-friendly lists; mass matrices and cold-sample
        buffers contribute ``np.ndarray`` entries to ``arrays``.  Array keys
        are local (the sampler prefixes them, e.g. ``nuts.mm.na2.inv_diag``).
        """
        arrays: dict = {}
        meta: dict = {
            "mass_matrix_adapt_interval": int(self.mass_matrix_adapt_interval),
            "mass_matrix_min_samples": int(self.mass_matrix_min_samples),
            "step_size_min": float(self.step_size_min),
            "step_size_max": float(self.step_size_max),
            "step_sizes": [[int(c), int(na), float(v)] for (c, na), v in self.step_sizes.items()],
            "dual_averagers": [
                [int(c), int(na), da.get_checkpoint_state()]
                for (c, na), da in self.dual_averagers.items()
            ],
            "mass_matrices": [],
            "sample_buffers": [],
            "mass_matrix_injected": sorted(int(na) for na in self.mass_matrix_injected),
            "steps_since_mm_update": [
                [int(na), int(v)] for na, v in self.steps_since_mm_update.items()
            ],
        }
        for na, mm in self.mass_matrices.items():
            mm_arrays, mm_meta = mm.get_checkpoint_state()
            for k, v in mm_arrays.items():
                arrays[f"mm.na{int(na)}.{k}"] = v
            meta["mass_matrices"].append([int(na), mm_meta])
        for na, buf in self.sample_buffers.items():
            n = len(buf)
            if n > 0:
                arrays[f"buf.na{int(na)}"] = np.asarray(buf, dtype=np.float64)
            meta["sample_buffers"].append([int(na), int(n)])
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        """Restore adapter state from :meth:`get_checkpoint_state` output.

        Adapter config scalars are overwritten with the checkpointed values,
        then every cache is rebuilt with tuple keys restored.
        """
        self.mass_matrix_adapt_interval = meta["mass_matrix_adapt_interval"]
        self.mass_matrix_min_samples = meta["mass_matrix_min_samples"]
        self.step_size_min = meta["step_size_min"]
        self.step_size_max = meta["step_size_max"]
        self.step_sizes = {(int(c), int(na)): float(v) for c, na, v in meta["step_sizes"]}
        self.dual_averagers = {
            (int(c), int(na)): DualAveraging.from_checkpoint_state(s)
            for c, na, s in meta["dual_averagers"]
        }
        self.mass_matrices = {}
        for na, mm_meta in meta["mass_matrices"]:
            na = int(na)
            prefix = f"mm.na{na}."
            mm_arrays = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            self.mass_matrices[na] = MassMatrix.from_checkpoint_state(mm_arrays, mm_meta)
        self.sample_buffers = {}
        for na, n in meta["sample_buffers"]:
            na = int(na)
            if n > 0:
                buf_arr = np.asarray(arrays[f"buf.na{na}"])
                self.sample_buffers[na] = [np.array(row, dtype=float) for row in buf_arr]
            else:
                self.sample_buffers[na] = []
        self.mass_matrix_injected = {int(na) for na in meta["mass_matrix_injected"]}
        self.steps_since_mm_update = {int(na): int(v) for na, v in meta["steps_since_mm_update"]}

    # ------------------------------------------------------------------
    # Step-size / mass-matrix bookkeeping
    # ------------------------------------------------------------------

    def mass_matrix_for(self, n_active: int) -> MassMatrix:
        """Return the mass matrix for ``n_active``, creating a UNIT one if absent."""
        if n_active not in self.mass_matrices:
            self.mass_matrices[n_active] = MassMatrix(n_active, MassMatrixType.UNIT)
        return self.mass_matrices[n_active]

    def get_or_find_step_size(
        self,
        chain_idx: int,
        active_params: np.ndarray,
        logp_and_grad: Callable,
        rng: np.random.Generator,
        target_accept: float,
    ) -> float:
        """Look up or compute the step size for this chain/model dimension.

        Also ensures a DualAveraging instance exists for online adaptation.
        """
        n_active = len(active_params)
        cache_key = (chain_idx, n_active)

        if cache_key not in self.step_sizes:
            # Create mass matrix for this dimension if needed
            mass_matrix = self.mass_matrix_for(n_active)
            logp, grad = logp_and_grad(active_params)
            if not np.isfinite(logp):
                step_size = 0.1  # fallback
            else:
                step_size = find_reasonable_step_size(
                    active_params,
                    logp,
                    grad,
                    logp_and_grad,
                    mass_matrix,
                    rng,
                )
            self.step_sizes[cache_key] = step_size

        # Ensure a DualAveraging instance exists for this (chain, n_active)
        if cache_key not in self.dual_averagers:
            self.dual_averagers[cache_key] = DualAveraging(
                target_accept=target_accept,
                initial_step_size=self.step_sizes[cache_key],
            )

        return self.step_sizes[cache_key]

    def current_step_size(self, chain_idx: int, n_active: int) -> float:
        """Current step size for ``(chain_idx, n_active)`` (must already exist)."""
        return self.step_sizes[(chain_idx, n_active)]

    def update_step_size(
        self,
        chain_idx: int,
        n_active: int,
        mean_accept_prob: float,
        fallback_step: float,
    ) -> None:
        """Online step-size adaptation via dual averaging (one NUTS transition).

        ALL steps (divergent or not) feed DA.  Divergent steps push the
        step size down (``accept_prob`` near 0), non-divergent push up;
        ``np.clip`` prevents catastrophic collapse or explosion.  A missing
        dual averager (never created for this key) stores ``fallback_step``
        unchanged.
        """
        da_key = (chain_idx, n_active)
        if da_key in self.dual_averagers:
            adapted_step = self.dual_averagers[da_key].update(mean_accept_prob)
            adapted_step = np.clip(adapted_step, self.step_size_min, self.step_size_max)
            self.step_sizes[da_key] = adapted_step
        else:
            self.step_sizes[da_key] = fallback_step

    def buffer_cold_sample(self, n_active: int, position: np.ndarray) -> None:
        """Buffer a non-divergent cold-chain sample for mass-matrix re-estimation."""
        if n_active not in self.sample_buffers:
            self.sample_buffers[n_active] = []
        self.sample_buffers[n_active].append(position.copy())

    # ------------------------------------------------------------------
    # Adaptation freeze
    # ------------------------------------------------------------------

    def finalize_step_sizes(self) -> None:
        """Replace primal dual-averaging iterates with smoothed step sizes.

        Called once at the ``num_adapt`` freeze transition.  During
        adaptation ``step_sizes`` tracks the noisy PRIMAL dual-averaging
        iterate ``exp(log_step)``, which deliberately overshoots (its
        anchor is ``mu = log(10 * step)``); the converged estimate is the
        smoothed ``DualAveraging.finalize()`` value ``exp(log_step_bar)``.
        Freezing the primal iterate — especially in the transient right
        after a mass-matrix commit resets dual averaging — can pin a step
        size several times the converged value with no way to correct it.

        A dual averager that never received an update (or yields a
        non-finite/non-positive value) leaves the current step size
        untouched.  Idempotent: finalize() is a pure read of DA state.
        """
        for key, da in self.dual_averagers.items():
            if getattr(da, "count", 0) == 0:
                continue  # never updated; nothing smoothed to freeze to
            finalized = da.finalize()
            if not np.isfinite(finalized) or finalized <= 0:
                continue  # keep the current step size
            self.step_sizes[key] = float(np.clip(finalized, self.step_size_min, self.step_size_max))

    # ------------------------------------------------------------------
    # Mass matrix injection
    # ------------------------------------------------------------------

    def set_mass_matrix(self, n_active: int, mass_matrix: MassMatrix, target_accept: float) -> None:
        """Inject an external mass matrix (e.g., Fisher-based) for a given dimension.

        The injected matrix is preserved for the entire run: online
        mass-matrix adaptation never overwrites an injected entry (only the
        dual-averaging step size continues to re-tune against it).
        """
        self.mass_matrices[n_active] = mass_matrix
        self.mass_matrix_injected.add(n_active)
        # Reset any DualAveraging instances for this n_active so step sizes
        # re-tune to the new mass matrix.
        for key in list(self.dual_averagers.keys()):
            if key[1] == n_active:
                current_step = self.step_sizes.get(key, 0.1)
                self.dual_averagers[key] = DualAveraging(
                    target_accept=target_accept,
                    initial_step_size=current_step,
                )

    # ------------------------------------------------------------------
    # Mass matrix adaptation from samples
    # ------------------------------------------------------------------

    @staticmethod
    def adapt_mass_matrix_from_samples(samples, n_active, mass_matrix_type) -> MassMatrix:
        """Compute regularized covariance from samples and return a MassMatrix.

        Parameters
        ----------
        samples : list of np.ndarray
            Position samples, each of shape ``(n_active,)``.
        n_active : int
            Dimensionality.
        mass_matrix_type : MassMatrixType
            Desired mass matrix type.

        Returns
        -------
        MassMatrix
            New mass matrix M = (regularized covariance)^{-1}.
        """
        return regularized_mass_matrix(samples, n_active, mass_matrix_type)

    def maybe_adapt_mass_matrices(
        self,
        mass_matrix_type: MassMatrixType,
        target_accept: float,
        get_cold_active_params: Callable[[], np.ndarray],
        make_logp_and_grad: Callable[[], Callable],
        rng: np.random.Generator,
    ) -> None:
        """Periodically re-estimate mass matrices from cold-chain samples.

        Called once per iteration from the sample loop.  For each
        ``n_active`` with buffered samples, checks whether enough steps and
        samples have accumulated to warrant a mass matrix update.

        Three protections are applied:

        1. Injected (Fisher) mass matrices are never overwritten.
        2. When a mass matrix changes, step sizes are recalibrated via
           ``find_reasonable_step_size`` instead of inheriting the old value.
        3. Candidate mass matrices are validated with trial NUTS steps;
           if a majority diverge the candidate is rejected.

        Parameters
        ----------
        mass_matrix_type : MassMatrixType
            Desired mass matrix type for re-estimated matrices.
        target_accept : float
            Target acceptance probability for the reset dual averagers.
        get_cold_active_params : callable
            Zero-argument callable returning a fresh copy of the cold
            chain's active continuous parameters.
        make_logp_and_grad : callable
            Zero-argument callable returning the cold chain's tempered
            ``(x_active) -> (logp, grad)`` function.
        rng : np.random.Generator
            The cold chain's RNG stream (consumed by step-size search and
            trial NUTS steps, exactly as the pre-extraction sampler did).
        """
        for n_active in list(self.sample_buffers.keys()):
            # Increment step counter
            self.steps_since_mm_update.setdefault(n_active, 0)
            self.steps_since_mm_update[n_active] += 1

            # Check interval
            if self.steps_since_mm_update[n_active] < self.mass_matrix_adapt_interval:
                continue

            samples = self.sample_buffers.get(n_active, [])
            n_samples = len(samples)

            # Need minimum samples
            if n_samples < self.mass_matrix_min_samples:
                continue

            # FIX 1: Never overwrite injected (Fisher) mass matrices.
            # These are analytically computed and far superior to sample estimates.
            if n_active in self.mass_matrix_injected:
                self.steps_since_mm_update[n_active] = 0
                continue

            # Compute candidate mass matrix from samples
            candidate_mm = self.adapt_mass_matrix_from_samples(
                samples,
                n_active,
                mass_matrix_type,
            )

            # FIX 2: Find step size appropriate for the NEW mass matrix
            # (old code inherited the old step size, causing immediate divergences)
            active_params = get_cold_active_params()

            if len(active_params) != n_active:
                # Model dimension changed since buffer was filled; skip
                self.steps_since_mm_update[n_active] = 0
                continue

            logp_and_grad = make_logp_and_grad()

            logp_val, grad_val = logp_and_grad(active_params)
            if np.isfinite(logp_val):
                candidate_step = find_reasonable_step_size(
                    active_params,
                    logp_val,
                    grad_val,
                    logp_and_grad,
                    candidate_mm,
                    rng,
                )
                candidate_step = np.clip(candidate_step, self.step_size_min, self.step_size_max)
            else:
                candidate_step = 0.1

            # FIX 3: Validate candidate with trial NUTS steps.
            # Run a few short-tree NUTS steps; reject if majority diverge.
            n_trial = 5
            n_divergent = 0
            trial_pos = active_params.copy()
            trial_logp, trial_grad = logp_val, grad_val

            for _ in range(n_trial):
                if not np.isfinite(trial_logp):
                    n_divergent += 1
                    break
                trial_state = NUTSState(
                    position=trial_pos,
                    logp=trial_logp,
                    grad=trial_grad,
                    step_size=candidate_step,
                    mass_matrix=candidate_mm,
                )
                trial_result = nuts_step(trial_state, logp_and_grad, rng, max_tree_depth=3)
                if trial_result.divergent:
                    n_divergent += 1
                trial_pos = trial_result.position
                trial_logp = trial_result.logp
                trial_grad = trial_result.grad

            if n_divergent > n_trial // 2:
                # Reject candidate — too many divergences. Reset counter, try later.
                self.steps_since_mm_update[n_active] = 0
                continue

            # Commit the validated mass matrix
            self.mass_matrices[n_active] = candidate_mm

            # Reset step counter; keep recent half of sample buffer
            self.steps_since_mm_update[n_active] = 0
            half = n_samples // 2
            self.sample_buffers[n_active] = samples[-half:]

            # Reset DA with the step size calibrated to the new mass matrix
            for key in list(self.dual_averagers.keys()):
                if key[1] == n_active:
                    self.step_sizes[key] = candidate_step
                    self.dual_averagers[key] = DualAveraging(
                        target_accept=target_accept,
                        initial_step_size=candidate_step,
                    )

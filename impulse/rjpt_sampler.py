"""RJPTSampler — Parallel Tempering with optional NUTS and RJMCMC.

A peer of PTSampler that interleaves MH, NUTS, and PT steps.
Reuses existing building blocks without subclassing PTSampler.
"""

import logging
import os
import numpy as np
from typing import Callable, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

from impulse.proposals import JumpProposals, ProposalBundle, am, scam, de
from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.input_function_wrapper import _function_wrapper
from impulse.sampler_state import SamplerState, PTState
from impulse.file_io import ShortChain
from impulse.sampler_step import vectorized_mh_step, pt_step
from impulse.wrapping import WrapSpec, PeriodicSpec
from impulse.resume import checkpoint_sampler, check_for_checkpoint, load_rjpt_checkpoint
from impulse.samplers import (
    setup_seeds, setup_chain_stats, setup_standard_jumps, setup_initial_position,
)
from impulse.nuts.core import NUTSState, nuts_step
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.warmup import find_reasonable_step_size, DualAveraging
from impulse.utils import prepare_files


class RJPTSampler:
    """Parallel Tempering sampler with optional NUTS and RJMCMC.

    Interleaves Metropolis-Hastings proposals (including RJ birth/death),
    NUTS transitions on active continuous parameters, and parallel
    tempering swaps.

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
    buffer_size, groups, sample_mean, sample_cov, loglargs, loglkwargs,
    logpargs, logpkwargs, cov_update, save_freq, scam_weight, am_weight,
    de_weight, seed, outdir, ntemps, swap_steps, min_temp, max_temp,
    temp_step, ladder, inf_temp, adapt_t0, adapt_nu, resume, vectorized
        Same as :class:`PTSampler`.
    """

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
        threads: int = 1,
        periodic: Optional[PeriodicSpec] = None,
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
        self.lnlike = _function_wrapper(lnlike, loglargs, loglkwargs, vectorized=vectorized, threads=threads)
        self.lnprior = _function_wrapper(lnprior, logpargs, logpkwargs, vectorized=vectorized, threads=threads)

        # Keep raw references for NUTS gradient building
        self._raw_lnlike = lnlike
        self._raw_lnprior = lnprior

        self.rngs = setup_seeds(seed, ntemps)

        self.ptstate = PTState(
            self.ndim, ntemps, swap_steps=swap_steps, min_temp=min_temp,
            max_temp=max_temp, temp_step=temp_step, ladder=ladder,
            inf_temp=inf_temp, adapt_t0=adapt_t0, adapt_nu=adapt_nu,
        )
        self.multi_chain_stats = setup_chain_stats(
            ndim, self.ptstate, self.rngs, groups, sample_cov, sample_mean,
            buffer_size, self.ptstate.ladder,
        )
        self.proposal_bundle = setup_standard_jumps(
            self.multi_chain_stats, am_weight, scam_weight, de_weight,
        )

        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume

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

        # NUTS caches (picklable)
        self._step_sizes: dict = {}      # (chain_idx, nmodel_or_ndim) -> float
        self._mass_matrices: dict = {}   # nmodel_or_ndim -> MassMatrix

        # Online NUTS adaptation state (picklable)
        self._dual_averagers: dict = {}           # (chain_idx, n_active) -> DualAveraging
        self._nuts_sample_buffers: dict = {}      # n_active -> list[np.ndarray]
        self._mass_matrix_injected: set = set()   # n_active values with externally set mass matrices
        self._nuts_steps_since_mm_update: dict = {}  # n_active -> int counter
        self._mass_matrix_adapt_interval = mass_matrix_adapt_interval
        self._mass_matrix_min_samples = mass_matrix_min_samples
        self._step_size_min = step_size_min
        self._step_size_max = step_size_max

        # RJ-specific (set by from_rjmcmc)
        self._rjmcmc_space = None

        # NUTS diagnostics buffer (populated during sampling)
        self._nuts_diag_data = None

    # ------------------------------------------------------------------
    # from_rjmcmc classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_rjmcmc(
        cls,
        rjmcmc_space,
        lnlike_grad: Optional[Callable] = None,
        birth_weight: float = 15,
        death_weight: float = 15,
        nmodel_weight: float = 10,
        swap_weight: float = 15,
        am_weight: float = 15,
        scam_weight: float = 15,
        de_weight: float = 15,
        **kwargs,
    ) -> "RJPTSampler":
        """Construct an RJPTSampler pre-configured for RJMCMC model selection.

        Parameters
        ----------
        rjmcmc_space : RJMCMCProductSpace
            Configured RJMCMC product space.
        lnlike_grad : callable, optional
            Gradient function for NUTS on active continuous params.
        birth_weight, death_weight, nmodel_weight, swap_weight : float
            Relative weights for RJ proposals.
        am_weight, scam_weight, de_weight : float
            Relative weights for standard MH proposals.
        **kwargs
            Additional keyword arguments forwarded to ``__init__``.
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
            lnlike_grad=lnlike_grad,
            groups=rjmcmc_space.get_default_groups(),
            sample_cov=sample_cov,
            sample_mean=sample_mean,
            am_weight=am_weight,
            scam_weight=scam_weight,
            de_weight=de_weight,
            **kwargs,
        )
        sampler._rjmcmc_space = rjmcmc_space
        sampler.add_custom_jump(rjmcmc_space.get_birth_proposal(), birth_weight)
        sampler.add_custom_jump(rjmcmc_space.get_death_proposal(), death_weight)
        sampler.add_custom_jump(rjmcmc_space.get_nmodel_jump(), nmodel_weight)
        sampler.add_custom_jump(rjmcmc_space.get_source_swap_proposal(), swap_weight)
        sampler.multi_chain_stats.enable_per_model(
            rjmcmc_space.num_models, rjmcmc_space.num_params,
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
            Proposal function ``(ChainStats) -> (sample, qxy)``.
        weight : float
            Relative weight for this proposal.
        """
        self.proposal_bundle.add_jump(proposal, weight)

    # ------------------------------------------------------------------
    # Public: mass matrix injection
    # ------------------------------------------------------------------

    def set_mass_matrix(self, n_active: int, mass_matrix: MassMatrix):
        """Inject an external mass matrix (e.g., Fisher-based) for a given dimension.

        The matrix is preserved until at least ``2 * mass_matrix_min_samples``
        non-divergent cold-chain samples accumulate, after which online
        adaptation may overwrite it.

        Parameters
        ----------
        n_active : int
            Number of active continuous parameters this matrix applies to.
        mass_matrix : MassMatrix
            Mass matrix to use for NUTS proposals.
        """
        self._mass_matrices[n_active] = mass_matrix
        self._mass_matrix_injected.add(n_active)
        # Reset any DualAveraging instances for this n_active so step sizes
        # re-tune to the new mass matrix.
        for key in list(self._dual_averagers.keys()):
            if key[1] == n_active:
                current_step = self._step_sizes.get(key, 0.1)
                self._dual_averagers[key] = DualAveraging(
                    target_accept=self.target_accept,
                    initial_step_size=current_step,
                )

    # ------------------------------------------------------------------
    # Internal: mass matrix adaptation from samples
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_mass_matrix_from_samples(samples, n_active, mass_matrix_type):
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
            New mass matrix estimated from samples.
        """
        arr = np.array(samples)
        n = len(arr)
        if n < 2:
            return MassMatrix(n_active, MassMatrixType.UNIT)

        sample_cov = np.cov(arr, rowvar=False)
        if sample_cov.ndim == 0:
            sample_cov = sample_cov.reshape(1, 1)

        # Regularization: shrink toward diagonal (Stan's approach)
        shrinkage = 5.0 / (n + 5.0)
        reg_cov = (1 - shrinkage) * sample_cov + shrinkage * np.diag(np.diag(sample_cov) + 1e-3)

        if mass_matrix_type == MassMatrixType.DIAGONAL:
            diag = np.maximum(np.diag(reg_cov), 1e-10)
            return MassMatrix(n_active, MassMatrixType.DIAGONAL, diagonal=diag)
        elif mass_matrix_type == MassMatrixType.DENSE:
            reg_cov += 1e-8 * np.eye(n_active)
            try:
                np.linalg.cholesky(reg_cov)
                return MassMatrix(n_active, MassMatrixType.DENSE, dense=reg_cov)
            except np.linalg.LinAlgError:
                diag = np.maximum(np.diag(reg_cov), 1e-10)
                return MassMatrix(n_active, MassMatrixType.DIAGONAL, diagonal=diag)
        else:
            return MassMatrix(n_active, MassMatrixType.UNIT)

    # ------------------------------------------------------------------
    # Internal: NUTS helpers
    # ------------------------------------------------------------------

    def _get_active_indices(self, params):
        """Return indices of active continuous parameters.

        For RJ models, active params = first ``(nmodel+1)*num_params``.
        For fixed-dim models, active = all params.
        """
        if self._rjmcmc_space is not None:
            nmodel = int(np.rint(params[-1]))
            num_params = self._rjmcmc_space.num_params
            return np.arange((nmodel + 1) * num_params)
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
        raw_lnprior = self._raw_lnprior

        def logp_and_grad(x_active):
            trial = full_params.copy()
            trial[active_idx] = x_active

            # Check prior FIRST — cheap and catches out-of-bounds before
            # potentially expensive/unstable gradient computation.
            if self._rjmcmc_space is not None:
                lp = raw_lnprior(trial[:self._rjmcmc_space.num_models * self._rjmcmc_space.num_params])
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

    def _get_nuts_cache_key(self, params):
        """Cache key for step sizes: ``(chain_idx, n_active)``."""
        if self._rjmcmc_space is not None:
            return int(np.rint(params[-1]))
        return self.ndim

    def _get_or_find_step_size(self, chain_idx, active_params, logp_and_grad, rng):
        """Look up or compute step size for this chain/model dimension.

        Also ensures a DualAveraging instance exists for online adaptation.
        """
        n_active = len(active_params)
        cache_key = (chain_idx, n_active)

        if cache_key not in self._step_sizes:
            # Create mass matrix for this dimension if needed
            if n_active not in self._mass_matrices:
                self._mass_matrices[n_active] = MassMatrix(n_active, MassMatrixType.UNIT)

            mass_matrix = self._mass_matrices[n_active]
            logp, grad = logp_and_grad(active_params)
            if not np.isfinite(logp):
                step_size = 0.1  # fallback
            else:
                step_size = find_reasonable_step_size(
                    active_params, logp, grad, logp_and_grad, mass_matrix, rng,
                )
            self._step_sizes[cache_key] = step_size

        # Ensure a DualAveraging instance exists for this (chain, n_active)
        if cache_key not in self._dual_averagers:
            self._dual_averagers[cache_key] = DualAveraging(
                target_accept=self.target_accept,
                initial_step_size=self._step_sizes[cache_key],
            )

        return self._step_sizes[cache_key]

    def _nuts_step_all_chains(self, state):
        """Run one NUTS transition on each temperature chain.

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

            step_size = self._get_or_find_step_size(
                k, active_params, logp_and_grad, rng,
            )

            if n_active not in self._mass_matrices:
                self._mass_matrices[n_active] = MassMatrix(n_active, MassMatrixType.UNIT)
            mass_matrix = self._mass_matrices[n_active]

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
            new_lnlikes[k] = self._raw_lnlike(nuts_state.position) if self._rjmcmc_space is None else self.lnlike(new_positions[k:k+1])[0]
            new_lnpriors[k] = self._raw_lnprior(new_params) if self._rjmcmc_space is None else self.lnprior(new_positions[k:k+1])[0]

            # Online step size adaptation via dual averaging.
            # ALL steps (divergent or not) feed DA. Divergent steps push
            # step size down (accept_prob ≈ 0), non-divergent push up.
            # np.clip prevents catastrophic collapse or explosion.
            da_key = (k, n_active)
            if da_key in self._dual_averagers:
                adapted_step = self._dual_averagers[da_key].update(
                    nuts_state.mean_accept_prob,
                )
                adapted_step = np.clip(adapted_step, self._step_size_min, self._step_size_max)
                self._step_sizes[da_key] = adapted_step
            else:
                self._step_sizes[da_key] = nuts_state.step_size

            # Cold chain: collect non-divergent samples for mass matrix adaptation
            if k == 0 and not nuts_state.divergent:
                if n_active not in self._nuts_sample_buffers:
                    self._nuts_sample_buffers[n_active] = []
                self._nuts_sample_buffers[n_active].append(
                    nuts_state.position.copy(),
                )

            # Cold chain diagnostics
            if k == 0:
                cold_diag = {
                    "tree_depth": nuts_state.tree_depth,
                    "divergent": int(nuts_state.divergent),
                    "energy_error": nuts_state.energy_error,
                    "step_size": self._step_sizes[da_key],
                    "mean_accept_prob": nuts_state.mean_accept_prob,
                    "n_active": n_active,
                }

        new_lnprobs = 1.0 / state.temps * new_lnlikes + new_lnpriors
        new_state = SamplerState(
            new_positions, new_lnlikes, new_lnpriors, new_lnprobs,
            state.accepted, state.temps,
        )
        return new_state, cold_diag

    def _maybe_adapt_mass_matrices(self):
        """Periodically re-estimate mass matrices from cold-chain samples.

        Called once per iteration from the sample loop. For each ``n_active``
        with buffered samples, checks whether enough steps and samples have
        accumulated to warrant a mass matrix update.

        Three protections are applied:
        1. Injected (Fisher) mass matrices are never overwritten.
        2. When a mass matrix changes, step sizes are recalibrated via
           ``find_reasonable_step_size`` instead of inheriting the old value.
        3. Candidate mass matrices are validated with trial NUTS steps;
           if a majority diverge the candidate is rejected.
        """
        for n_active in list(self._nuts_sample_buffers.keys()):
            # Increment step counter
            self._nuts_steps_since_mm_update.setdefault(n_active, 0)
            self._nuts_steps_since_mm_update[n_active] += 1

            # Check interval
            if self._nuts_steps_since_mm_update[n_active] < self._mass_matrix_adapt_interval:
                continue

            samples = self._nuts_sample_buffers.get(n_active, [])
            n_samples = len(samples)

            # Need minimum samples
            if n_samples < self._mass_matrix_min_samples:
                continue

            # FIX 1: Never overwrite injected (Fisher) mass matrices.
            # These are analytically computed and far superior to sample estimates.
            if n_active in self._mass_matrix_injected:
                self._nuts_steps_since_mm_update[n_active] = 0
                continue

            # Compute candidate mass matrix from samples
            candidate_mm = self._adapt_mass_matrix_from_samples(
                samples, n_active, self._mass_matrix_type,
            )

            # FIX 2: Find step size appropriate for the NEW mass matrix
            # (old code inherited the old step size, causing immediate divergences)
            cold_params = self.state.positions[0]
            active_idx = self._get_active_indices(cold_params)
            active_params = cold_params[active_idx].copy()

            if len(active_params) != n_active:
                # Model dimension changed since buffer was filled; skip
                self._nuts_steps_since_mm_update[n_active] = 0
                continue

            logp_and_grad, _ = self._make_tempered_logp_grad(0, self.state)
            rng = self.rngs[0]

            logp_val, grad_val = logp_and_grad(active_params)
            if np.isfinite(logp_val):
                candidate_step = find_reasonable_step_size(
                    active_params, logp_val, grad_val, logp_and_grad,
                    candidate_mm, rng,
                )
                candidate_step = np.clip(candidate_step, self._step_size_min, self._step_size_max)
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
                    position=trial_pos, logp=trial_logp, grad=trial_grad,
                    step_size=candidate_step, mass_matrix=candidate_mm,
                )
                trial_result = nuts_step(trial_state, logp_and_grad, rng, max_tree_depth=3)
                if trial_result.divergent:
                    n_divergent += 1
                trial_pos = trial_result.position
                trial_logp = trial_result.logp
                trial_grad = trial_result.grad

            if n_divergent > n_trial // 2:
                # Reject candidate — too many divergences. Reset counter, try later.
                self._nuts_steps_since_mm_update[n_active] = 0
                continue

            # Commit the validated mass matrix
            self._mass_matrices[n_active] = candidate_mm

            # Reset step counter; keep recent half of sample buffer
            self._nuts_steps_since_mm_update[n_active] = 0
            half = n_samples // 2
            self._nuts_sample_buffers[n_active] = samples[-half:]

            # Reset DA with the step size calibrated to the new mass matrix
            for key in list(self._dual_averagers.keys()):
                if key[1] == n_active:
                    self._step_sizes[key] = candidate_step
                    self._dual_averagers[key] = DualAveraging(
                        target_accept=self.target_accept,
                        initial_step_size=candidate_step,
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
        """
        if self.ptstate.ladder is None:
            raise ValueError("PTState ladder is not initialized")

        # Chain storage
        self.short_chain = ShortChain(
            self.ndim, self.ntemps, self.save_freq,
            iteration=0, outdir=self.outdir, resume=self.resume, thin=thin,
        )

        # Initial state
        initial_position = setup_initial_position(initial_position, self.ntemps)
        if self.wrap is not None:
            initial_position = self.wrap.apply(initial_position)
        lnlike0 = self.lnlike(initial_position)
        lnprior0 = self.lnprior(initial_position)
        lnprob0 = 1.0 / self.ptstate.ladder * lnlike0 + lnprior0
        initial_state = SamplerState(
            initial_position, lnlike0, lnprior0, lnprob0,
            accepted=np.ones(self.ntemps), temps=self.ptstate.ladder,
        )

        if np.any(~np.isfinite(lnlike0)):
            raise ValueError("Some likelihood values are not finite.")
        if np.any(~np.isfinite(lnprior0)):
            raise ValueError("An initial value falls outside the prior bounds.")

        self.state = initial_state

        # Checkpoint / resume
        self.checkpoint_path = check_for_checkpoint(self.outdir)
        if self.resume and self.checkpoint_path is not None:
            logger.info("Resuming from checkpoint: %s", self.checkpoint_path)
            loaded = load_rjpt_checkpoint(
                self.checkpoint_path,
                lnlike=self.lnlike, lnprior=self.lnprior,
                raw_lnlike=self._raw_lnlike, raw_lnprior=self._raw_lnprior,
                lnlike_grad=self.lnlike_grad,
            )
            self.__dict__.update(loaded.__dict__)

        # Backward-compat: old checkpoints may lack new adaptation attributes
        if not hasattr(self, '_dual_averagers'):
            self._dual_averagers = {}
        if not hasattr(self, '_nuts_sample_buffers'):
            self._nuts_sample_buffers = {}
        if not hasattr(self, '_mass_matrix_injected'):
            self._mass_matrix_injected = set()
        if not hasattr(self, '_nuts_steps_since_mm_update'):
            self._nuts_steps_since_mm_update = {}
        if not hasattr(self, '_mass_matrix_adapt_interval'):
            self._mass_matrix_adapt_interval = 200
        if not hasattr(self, '_mass_matrix_min_samples'):
            self._mass_matrix_min_samples = 50
        if not hasattr(self, '_step_size_min'):
            self._step_size_min = 1e-4
        if not hasattr(self, '_step_size_max'):
            self._step_size_max = 5.0

        # NUTS diagnostics file
        if self.nuts_enabled:
            self._nuts_diag_data = []
            self._nuts_diag_path = os.path.join(self.outdir, "nuts_diagnostics.txt")
            prepare_files([self._nuts_diag_path], resume=self.resume)

        # NUTS warmup: find initial step sizes
        if self.nuts_enabled:
            for k in range(self.ntemps):
                logp_and_grad, active_idx = self._make_tempered_logp_grad(k, self.state)
                active_params = self.state.positions[k][active_idx]
                if len(active_params) > 0:
                    self._get_or_find_step_size(k, active_params, logp_and_grad, self.rngs[k])

        _last_cov_iter = self.short_chain.iteration

        for jj in tqdm(
            range(self.short_chain.iteration, num_iterations),
            initial=self.short_chain.iteration,
            total=num_iterations,
            desc="Sampling",
        ):
            # Step A: MH step (includes RJ proposals if registered)
            self.state = vectorized_mh_step(
                self.state, self.proposal_bundle, self.lnlike, self.lnprior, self.rngs[0],
                wrap=self.wrap,
            )
            self.proposal_bundle.report_accepts(self.state.accepted)

            # Step B: NUTS step on active continuous params
            if self.nuts_enabled:
                self.state, cold_diag = self._nuts_step_all_chains(self.state)
                if cold_diag is not None:
                    self._nuts_diag_data.append(cold_diag)
                self._maybe_adapt_mass_matrices()

            # Step C: Save / checkpoint
            if jj > 0 and jj % self.save_freq == 0:
                self.short_chain.save_chain()
                self.save_chain_acceptance_rates()
                if self.nuts_enabled:
                    self._flush_nuts_diagnostics()
                checkpoint_sampler(
                    self, path=self.checkpoint_path,
                    omit=("lnlike", "lnprior", "_raw_lnlike", "_raw_lnprior", "lnlike_grad"),
                )
            self.short_chain.add_state(self.state)

            # Step D: PT swap
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(
                    self.state, self.ptstate, self.lnlike, self.lnprior, self.rngs[-1],
                )
                self.ptstate.adapt_ladder()

            # Step E: Covariance update
            if jj % self.cov_update == 0:
                new_count = self.short_chain.iteration - _last_cov_iter
                if new_count > 0:
                    new_samples = self.short_chain.get_recent_samples(new_count)
                    self.multi_chain_stats.recursive_update(new_samples)
                _last_cov_iter = self.short_chain.iteration

        # Final save
        self.short_chain.save_chain()
        self.save_chain_acceptance_rates()
        if self.nuts_enabled:
            self._flush_nuts_diagnostics()

    # ------------------------------------------------------------------
    # NUTS diagnostics I/O
    # ------------------------------------------------------------------

    def _flush_nuts_diagnostics(self):
        """Write buffered NUTS diagnostics to disk."""
        if not self._nuts_diag_data:
            return
        rows = np.array([
            [d["tree_depth"], d["divergent"], d["energy_error"],
             d["step_size"], d["mean_accept_prob"],
             d.get("n_active", 0)]
            for d in self._nuts_diag_data
        ])
        with open(self._nuts_diag_path, "a") as fp:
            np.savetxt(fp, rows, fmt="%.18e")
        self._nuts_diag_data = []

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
        return self.proposal_bundle.acceptance_report()

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
        samples, lnlike, lnprob, accepted, temperature = [], [], [], [], []
        for ii in range(self.ntemps):
            filepath = os.path.join(self.outdir, f"chain_{ii}.txt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Chain file not found: {filepath}")
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            samples.append(data[:, :self.ndim])
            lnlike.append(data[:, self.ndim])
            lnprob.append(data[:, self.ndim + 1])
            accepted.append(data[:, self.ndim + 2])
            temperature.append(data[:, self.ndim + 3])

        result = {
            "samples": np.array(samples),
            "lnlike": np.array(lnlike),
            "lnprob": np.array(lnprob),
            "accepted": np.array(accepted),
            "temperature": np.array(temperature),
        }

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



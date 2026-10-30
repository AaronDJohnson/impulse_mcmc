"""Parallel Tempering MCMC sampler, state containers, and transition steps."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from impulse import chain_io
from impulse.chain_io import ShortChain
from impulse.chain_stats import ChainStats, MultiChainStats
from impulse.proposals import DE_MIN_FILL, DEProposal, JumpProposals, ProposalBundle, am, de, scam
from impulse.resume import (
    _jsonable_rng_state,
    _rng_state_from_json,
    check_for_checkpoint,
    checkpoint_sampler,
    restore_state_checkpoint,
)

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_UNSET: Any = object()

PeriodicSpec = Dict[int, Union[float, Tuple[float, float]]]


@dataclass
class WrapSpec:
    """Wrap selected columns of a position array into ``[low, high)``."""

    indices: np.ndarray
    lows: np.ndarray
    periods: np.ndarray

    @classmethod
    def from_dict(cls, periodic: Optional[PeriodicSpec]) -> Optional[WrapSpec]:
        if not periodic:
            return None
        idx = sorted(periodic.keys())
        lows, highs = [], []
        for i in idx:
            spec = periodic[i]
            if isinstance(spec, (tuple, list)):
                if len(spec) != 2:
                    raise ValueError(
                        f"periodic[{i}] must be a scalar period or a (low, high) pair; got {spec}"
                    )
                lo, hi = float(spec[0]), float(spec[1])
            else:
                lo, hi = 0.0, float(spec)
            if not (hi > lo):
                raise ValueError(f"periodic[{i}]: high ({hi}) must be > low ({lo})")
            lows.append(lo)
            highs.append(hi)
        lows_arr = np.asarray(lows, dtype=float)
        highs_arr = np.asarray(highs, dtype=float)
        return cls(
            indices=np.asarray(idx, dtype=int),
            lows=lows_arr,
            periods=highs_arr - lows_arr,
        )

    def apply(self, positions: np.ndarray) -> np.ndarray:
        x = np.array(positions, copy=True)
        self.apply_inplace(x)
        return x

    def apply_inplace(self, positions: np.ndarray) -> None:
        if positions.ndim == 1:
            positions[self.indices] = (
                positions[self.indices] - self.lows
            ) % self.periods + self.lows
        else:
            positions[..., self.indices] = (
                positions[..., self.indices] - self.lows
            ) % self.periods + self.lows


class _function_wrapper:
    """Batched wrapper around user likelihood and prior callables."""

    def __init__(
        self,
        f: Callable,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        *,
        vectorized: bool = False,
        jax: bool = False,
        zero_copy: bool = True,
        threads: int = 1,
    ):
        self.f = f
        self.args = tuple(args) if args else ()
        self.kwargs = dict(kwargs) if kwargs else {}
        self.vectorized = bool(vectorized)
        self.jax = bool(jax)
        self.zero_copy = bool(zero_copy)
        self.threads = int(threads)
        self._executor: Optional[ThreadPoolExecutor] = None

    def __call__(self, x: Any):
        x_arr = np.asanyarray(x) if self.zero_copy else np.asarray(x)
        if x_arr.ndim != 2:
            raise ValueError("Input to _function_wrapper must be 2-D")
        n = x_arr.shape[0]
        if n == 0:
            raise ValueError("Input to _function_wrapper must have nonzero leading dimension")
        if self.zero_copy and not x_arr.flags.c_contiguous:
            x_arr = np.ascontiguousarray(x_arr)

        if self.vectorized:
            if not self.args and not self.kwargs:
                out = self.f(x_arr)
            else:
                out = self.f(x_arr, *self.args, **self.kwargs)
            out_arr = np.asanyarray(out) if self.zero_copy else np.asarray(out)
            if out_arr.shape[0] != n:
                raise ValueError(
                    "Vectorized function returned array with incorrect leading dimension"
                )
            return out_arr

        if self.threads > 1 and n > 1:
            return self._threaded_eval(x_arr, n)

        f = self.f
        args = self.args
        kwargs = self.kwargs
        if self.zero_copy:
            if not args and not kwargs:
                first_arr = np.asanyarray(f(x_arr[0]))
                out_dtype = np.result_type(first_arr.dtype, np.float64)
                shape = (n,) if first_arr.ndim == 0 else (n,) + first_arr.shape
                results = np.empty(shape, dtype=out_dtype)
                results[0] = first_arr
                for i in range(1, n):
                    results[i] = f(x_arr[i])
                return results
            first_arr = np.asanyarray(f(x_arr[0], *args, **kwargs))
            out_dtype = np.result_type(first_arr.dtype, np.float64)
            shape = (n,) if first_arr.ndim == 0 else (n,) + first_arr.shape
            results = np.empty(shape, dtype=out_dtype)
            results[0] = first_arr
            for i in range(1, n):
                results[i] = f(x_arr[i], *args, **kwargs)
            return results
        if not args and not kwargs:
            return np.asarray([f(x_arr[i]) for i in range(n)])
        return np.asarray([f(x_arr[i], *args, **kwargs) for i in range(n)])

    def _get_executor(self):
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=self.threads)
        return self._executor

    def _threaded_eval(self, x_arr, n):
        executor = self._get_executor()
        futures = [executor.submit(self.f, x_arr[i], *self.args, **self.kwargs) for i in range(n)]
        results = np.empty(n, dtype=np.float64)
        for i, fut in enumerate(futures):
            results[i] = fut.result()
        return results

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def __repr__(self):
        name = getattr(self.f, "__name__", repr(self.f))
        return f"<_function_wrapper {name} vectorized={self.vectorized} jax={self.jax} zero_copy={self.zero_copy}>"


def tempered_lnprobs(lnlikes: np.ndarray, lnpriors: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Compute tempered log-posteriors ``lnprior + lnlike / T``, handling ``T = inf``."""
    if len(temps) > 0 and not math.isinf(float(temps[-1])):
        return (1.0 / temps) * lnlikes + lnpriors
    with np.errstate(invalid="ignore"):
        beta = np.where(np.isinf(temps), 0.0, 1.0 / temps)
        return np.where(beta == 0.0, lnpriors, beta * lnlikes + lnpriors)


@dataclass
class SamplerState:
    """Positions, log-densities, acceptance flags, and temperatures across all chains."""

    positions: np.ndarray
    lnlikes: np.ndarray
    lnpriors: np.ndarray
    lnprobs: np.ndarray
    accepted: np.ndarray
    temps: np.ndarray

    _CHECKPOINT_FIELDS = ("positions", "lnlikes", "lnpriors", "lnprobs", "accepted", "temps")

    @property
    def ntemps(self) -> int:
        return int(self.positions.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.positions.shape[1])

    def get_checkpoint_state(self) -> dict:
        return {name: np.asarray(getattr(self, name)) for name in self._CHECKPOINT_FIELDS}

    @classmethod
    def from_checkpoint_state(cls, arrays: dict) -> SamplerState:
        return cls(*(np.array(arrays[name]) for name in cls._CHECKPOINT_FIELDS))


@dataclass
class PTState:
    """Parallel tempering temperature ladder and swap statistics."""

    ndim: int
    ntemps: int
    swap_steps: int = 1
    min_temp: float = 1.0
    max_temp: float | None = None
    temp_step: float | None = None
    nswaps: int = 1
    ladder: np.ndarray | None = None
    inf_temp: bool = False
    adapt_t0: float = 100
    adapt_nu: float = 10

    def __post_init__(self):
        if self.ladder is None:
            self.ladder = self.compute_temp_ladder()
        self.swap_accept = np.zeros(self.ntemps - 1)

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        arrays = {
            "ladder": np.asarray(self.ladder),
            "swap_accept": np.asarray(self.swap_accept),
        }
        meta = {
            "nswaps": int(self.nswaps),
            "temp_step": None if self.temp_step is None else float(self.temp_step),
        }
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        self.ladder = np.array(arrays["ladder"], dtype=float)
        self.swap_accept = np.array(arrays["swap_accept"], dtype=float)
        self.nswaps = int(meta["nswaps"])
        if meta["temp_step"] is not None:
            self.temp_step = meta["temp_step"]

    def compute_accept_ratio(self) -> np.ndarray:
        return self.swap_accept / self.nswaps

    def compute_temp_ladder(self) -> np.ndarray:
        if self.inf_temp:
            self.ntemps -= 1
        if self.temp_step is None and self.max_temp is None:
            self.temp_step = 1 + np.sqrt(2 / self.ndim)
        elif self.temp_step is None and self.max_temp is not None:
            if self.ntemps > 1:
                self.temp_step = np.exp(np.log(self.max_temp / self.min_temp) / (self.ntemps - 1))
            else:
                self.temp_step = 1.0
        temp_idxs = np.arange(self.ntemps)
        if self.temp_step is None:
            raise ValueError("temp_step is not initialized")
        ladder = self.min_temp * self.temp_step**temp_idxs
        if self.inf_temp:
            self.ntemps += 1
            ladder = np.concatenate([ladder, [np.inf]])
        return ladder

    def adapt_ladder(self) -> None:
        """Adapt intermediate temperatures using Vousden et al. (2016) hyperbolic decay."""
        if self.ladder is None:
            raise ValueError("PTState ladder is not initialized")
        if len(self.ladder) <= 2:
            return
        decay = self.adapt_t0 / (self.nswaps + self.adapt_t0)
        kappa = decay / self.adapt_nu
        accept_ratio = self.compute_accept_ratio()
        dscaled_accept = kappa * (accept_ratio[:-1] - accept_ratio[1:])
        delta_temps = (self.ladder[1:-1] - self.ladder[:-2]) * np.exp(dscaled_accept)
        self.ladder[1:-1] = np.cumsum(delta_temps) + self.ladder[0]


def vectorized_mh_step(
    state: SamplerState,
    prop_fn: ProposalBundle,
    lnlike_fn: Callable,
    lnprior_fn: Callable,
    rng: np.random.Generator,
    wrap: Optional[WrapSpec] = None,
) -> SamplerState:
    """Execute one Metropolis-Hastings step across all temperature chains."""
    x_stars, qxys = prop_fn(state)
    if wrap is not None:
        x_stars = wrap.apply(x_stars)

    # Every row is evaluated, including chains whose proposal returned the
    # current point, so vectorized user functions always get the full batch.
    lnprior_stars = lnprior_fn(x_stars)
    finite = np.isfinite(lnprior_stars)
    lnprior_stars = np.where(finite, lnprior_stars, -np.inf)

    if getattr(lnlike_fn, "jax", False):
        lnlike_stars = np.where(finite, np.asarray(lnlike_fn(x_stars)), -np.inf)
    else:
        lnlike_stars = np.full(len(x_stars), -np.inf)
        if np.any(finite):
            lnlike_stars[finite] = lnlike_fn(x_stars[finite])

    lnprob_stars = tempered_lnprobs(lnlike_stars, lnprior_stars, state.temps)

    probability_ratios = lnprob_stars - state.lnprobs + qxys
    accepts = np.log(rng.random(len(state.temps))) < probability_ratios

    return SamplerState(
        np.where(accepts[:, None], x_stars, state.positions),
        np.where(accepts, lnlike_stars, state.lnlikes),
        np.where(accepts, lnprior_stars, state.lnpriors),
        np.where(accepts, lnprob_stars, state.lnprobs),
        accepts.astype(int),
        state.temps,
    )


def pt_step(
    state: SamplerState,
    ptstate: PTState,
    rng: np.random.Generator,
    *,
    recompute_lnprobs: bool = True,
) -> SamplerState:
    """Perform parallel-tempering swap attempts from hottest to coldest chain."""
    ladder = ptstate.ladder
    if ladder is None:
        raise ValueError("PTState ladder is not initialized")
    n = len(ladder)
    swap_map = list(range(n))
    log_likes = state.lnlikes
    ll_list = log_likes.tolist()
    ladder_list = ladder.tolist()
    swap_accept = ptstate.swap_accept
    any_swapped = False
    # One draw per pair, hottest first: the same values as n - 1 scalar draws.
    uniforms = rng.random(n - 1).tolist()

    for swap_chain, u in zip(range(n - 2, -1, -1), uniforms):
        i_cold = swap_map[swap_chain]
        i_hot = swap_map[swap_chain + 1]
        t_cold = ladder_list[swap_chain]
        t_hot = ladder_list[swap_chain + 1]
        ll_cold = ll_list[i_cold]
        ll_hot = ll_list[i_hot]
        if math.isinf(t_hot):
            log_acc_ratio = -ll_cold / t_cold + ll_hot / t_cold
        else:
            log_acc_ratio = -ll_cold / t_cold - ll_hot / t_hot + ll_hot / t_cold + ll_cold / t_hot
        # log(0) is -inf (always accept); math.log raises on it instead.
        if (math.log(u) if u > 0.0 else -math.inf) <= log_acc_ratio:
            swap_map[swap_chain] = i_hot
            swap_map[swap_chain + 1] = i_cold
            swap_accept[swap_chain] += 1
            any_swapped = True

    ptstate.nswaps += 1
    if any_swapped:
        new_positions = state.positions[swap_map]
        new_loglikes = log_likes[swap_map]
        new_logpriors = state.lnpriors[swap_map]
    else:
        new_positions = state.positions
        new_loglikes = log_likes
        new_logpriors = state.lnpriors
    new_lnprobs = (
        tempered_lnprobs(new_loglikes, new_logpriors, ladder)
        if recompute_lnprobs
        else state.lnprobs
    )
    return SamplerState(
        new_positions,
        new_loglikes,
        new_logpriors,
        new_lnprobs,
        np.ones(n, dtype=int),
        ladder,
    )


def setup_seeds(seed: Optional[int], ntemps: int) -> List[np.random.Generator]:
    """Spawn ``ntemps + 1`` independent generators (one per chain plus PT swaps)."""
    seeds = np.random.SeedSequence(seed).spawn(ntemps + 1)
    return [np.random.default_rng(s) for s in seeds]


def setup_chain_stats(
    ndim,
    ptstate,
    rngs,
    groups,
    sample_cov,
    sample_mean,
    buffer_size,
    temps,
    buffer_thin: int = 1,
) -> MultiChainStats:
    """Create ``ChainStats`` for each temperature chain."""
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
            buffer_thin=buffer_thin,
        )
        for ii in range(len(temps))
    ]
    return MultiChainStats(chain_stats)


def setup_standard_jumps(
    multi_chain_stats: MultiChainStats,
    am_weight,
    scam_weight,
    de_weight,
    de_min_fill: int = DE_MIN_FILL,
) -> ProposalBundle:
    """Configure AM, SCAM, and DE proposals with specified weights for each chain."""
    de_jump = de if de_min_fill == DE_MIN_FILL else DEProposal(de_min_fill)
    jumps = [JumpProposals(cs) for cs in multi_chain_stats.chain_stats]
    for jp in jumps:
        jp.add_jump(am, am_weight)
        jp.add_jump(scam, scam_weight)
        jp.add_jump(de_jump, de_weight)
    return ProposalBundle(jumps)


def setup_initial_position(initial_position: np.ndarray, ntemps: int) -> np.ndarray:
    """Normalize ``initial_position`` to shape ``(ntemps, ndim)``."""
    x0 = np.asarray(initial_position, dtype=np.float64)
    if x0.ndim == 1:
        return np.tile(x0.reshape(1, -1), (ntemps, 1))
    if x0.ndim == 2:
        if x0.shape[0] == ntemps:
            return x0
        if x0.shape[0] == 1:
            return np.tile(x0, (ntemps, 1))
        raise ValueError(f"initial_position has {x0.shape[0]} rows but expected 1 or {ntemps}")
    raise ValueError("initial_position must be 1-D (ndim,) or 2-D (ntemps, ndim)")


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
    buffer_size : int, default 2000
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
    de_min_fill : int, default 100
        Minimum number of history-buffer samples before the differential
        evolution move activates; below the threshold ``de`` returns the
        current position unchanged. See :func:`impulse.de`.
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
        whose frozen state is checkpointed (e.g. a frozen normalizing flow)
        stay frozen while everything else adapts again. An explicitly
        passed value — including an explicit ``None`` — overrides the
        checkpointed value, with a warning when they differ. Fresh (non
        -resumed) runs treat the default exactly like ``None``.
    verbose : bool, default True
        Show the tqdm sampling progress bar. Set ``False`` to silence it
        (batch/cluster jobs, nested loops, notebooks). Presentation
        only: it does not affect the chain, is never checkpointed, and is not
        verified on resume -- a run resumed with ``verbose=False`` stays quiet
        even if the checkpoint came from a verbose run.
    chain_format : {'binary', 'text'}, default 'binary'
        On-disk encoding for the chain files. ``'binary'`` writes
        ``chain_<i>.bin``: raw ``float64`` records, ``ndim + 4`` values per row,
        no header or delimiters. ``'text'`` writes the historical
        ``chain_<i>.txt`` with ``%.18e`` columns, readable by ``np.loadtxt`` and
        greppable on a cluster, but slower to write and ~3x larger. Both store
        identical values (``%.18e`` round-trips a ``float64`` exactly), so the
        choice affects only speed, size, and readability.
        :meth:`load_chain` detects and reads either, so code that goes through
        it is unaffected by the choice. Keep it the same across a resume: the
        encoding is part of the chain files on disk, not of the checkpoint.
    periodic : dict, optional
        Circular parameters, mapping parameter index to either a period
        (``{2: 2 * np.pi}``) or an explicit ``(low, high)`` interval
        (``{2: (-np.pi, np.pi)}``). Declared parameters are wrapped back into
        their interval after every proposal, so a phase cannot random-walk to
        :math:`\\pm\\infty` and inflate the adaptive covariance. Undeclared
        parameters are untouched.
    buffer_thin : int, default 25
        Retain one sample per ``buffer_thin`` iterations in the history buffer.
        The buffer therefore spans ``buffer_size * buffer_thin`` iterations
        (50,000 at the defaults) while holding only ``buffer_size`` rows, which
        keeps the DE difference vectors spread over a long horizon instead of
        being packed into the most recent few thousand iterations. Raising it
        lengthens the horizon; lowering it packs the window tighter. Verified on
        resume — changing it raises ``CheckpointMismatchError``. The
        parallel-tempering user guide explains what the finite window does and
        does not guarantee.
    threads : int, default 1
        Evaluate a NON-vectorized ``lnlike``/``lnprior`` across the temperature
        chains on a ``ThreadPoolExecutor`` of this size. Ignored when
        ``vectorized=True`` (the batched call already covers all rows) and when
        there is only one row to evaluate. Because the workers are threads,
        this only helps if the function releases the GIL — NumPy/SciPy-heavy or
        compiled likelihoods do; pure-Python ones do not. Prefer ``vectorized``
        where the model supports it.

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

    _logger = logger

    def __init__(
        self,
        ndim: int,
        lnlike: Callable,
        lnprior: Callable,
        buffer_size: int = 2_000,
        buffer_thin: int = 25,
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
        verbose: bool = True,
        chain_format: str = "binary",
    ) -> None:
        self.ndim = ndim
        self.ntemps = ntemps
        self.swap_steps = swap_steps
        self.wrap = WrapSpec.from_dict(periodic)
        self.lnlike = _function_wrapper(
            lnlike,
            loglargs or (),
            loglkwargs or {},
            vectorized=vectorized,
            jax=jax,
            threads=threads,
        )
        self.lnprior = _function_wrapper(
            lnprior,
            logpargs or (),
            logpkwargs or {},
            vectorized=vectorized,
            jax=jax,
            threads=threads,
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
            buffer_thin,
        )
        self.proposal_bundle = setup_standard_jumps(
            self.multi_chain_stats, am_weight, scam_weight, de_weight, de_min_fill=de_min_fill
        )
        self.cov_update = cov_update
        self.save_freq = save_freq
        self.outdir = outdir
        self.resume = resume
        self.chain_format = chain_io.validate_format(chain_format)
        self.verbose = verbose
        self._num_adapt_explicit = num_adapt is not _UNSET
        self.num_adapt = None if num_adapt is _UNSET else num_adapt

    def _adaptation_active(self, iteration: int) -> bool:
        return self.num_adapt is None or iteration < self.num_adapt

    def _freeze_adaptive_proposals(self) -> None:
        for jp in self.proposal_bundle.jump_proposals:
            for prop in jp.proposal_list:
                freeze = getattr(prop, "freeze_adaptation", None)
                if callable(freeze):
                    freeze()

    def add_custom_jump(self, proposal, weight) -> None:
        """Add a custom proposal distribution to all temperature chains."""
        self.proposal_bundle.add_jump(proposal, weight)

    def _capture_checkpoint_state(self) -> tuple[dict, dict]:
        from impulse import __version__

        arrays: dict = {}
        meta: dict = {
            "sampler_class": type(self).__name__,
            "impulse_version": __version__,
            "ndim": int(self.ndim),
            "ntemps": int(self.ntemps),
            "swap_steps": int(self.swap_steps),
            "cov_update": int(self.cov_update),
            "save_freq": int(self.save_freq),
            "buffer_size": int(self.multi_chain_stats.chain_stats[0].buffer_size),
            "buffer_thin": int(self.multi_chain_stats.chain_stats[0].buffer_thin),
            "chain_format": str(self.chain_format),
            "num_adapt": None if self.num_adapt is None else int(self.num_adapt),
            "num_adapt_explicit": bool(self._num_adapt_explicit),
            "last_cov_iter": int(self._last_cov_iter),
            "rngs": [_jsonable_rng_state(rng.bit_generator.state) for rng in self.rngs],
        }
        for name, arr in self.state.get_checkpoint_state().items():
            arrays[f"state.{name}"] = arr

        pt_arrays, pt_meta = self.ptstate.get_checkpoint_state()
        for k, v in pt_arrays.items():
            arrays[f"ptstate.{k}"] = v
        meta["ptstate"] = pt_meta

        meta["chain_stats"] = []
        for i, cs in enumerate(self.multi_chain_stats.chain_stats):
            cs_arrays, cs_meta = cs.get_checkpoint_state()
            for k, v in cs_arrays.items():
                arrays[f"cs.c{i}.{k}"] = v
            meta["chain_stats"].append(cs_meta)

        sc_arrays, sc_meta = self.short_chain.get_checkpoint_state()
        for k, v in sc_arrays.items():
            arrays[f"sc.{k}"] = v
        meta["short_chain"] = sc_meta

        meta["proposal_bundle"] = [
            {
                "names": [getattr(p, "__name__", type(p).__name__) for p in jp.proposal_list],
                "weights": [float(w) for w in jp.proposal_weights],
                "probs": (
                    None if jp.proposal_probs is None else [float(x) for x in jp.proposal_probs]
                ),
                "calls": [int(x) for x in jp._proposal_calls],
                "accepts": [int(x) for x in jp._proposal_accepts],
                "last_idx": int(jp._last_proposal_idx),
            }
            for jp in self.proposal_bundle.jump_proposals
        ]

        stateful = []
        for idx, prop in enumerate(self.proposal_bundle.jump_proposals[0].proposal_list):
            getter = getattr(prop, "get_checkpoint_state", None)
            if callable(getter):
                stateful.append(
                    {
                        "idx": idx,
                        "name": getattr(prop, "__name__", type(prop).__name__),
                        "state": getter(),
                    }
                )
        meta["stateful_proposals"] = stateful
        return arrays, meta

    def _restore_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        for rng, state in zip(self.rngs, meta["rngs"]):
            rng.bit_generator.state = _rng_state_from_json(state)

        self.ptstate.set_checkpoint_state(
            {k[len("ptstate.") :]: v for k, v in arrays.items() if k.startswith("ptstate.")},
            meta["ptstate"],
        )
        state_arrays = {k[len("state.") :]: v for k, v in arrays.items() if k.startswith("state.")}
        self.state = SamplerState.from_checkpoint_state(state_arrays)
        assert self.ptstate.ladder is not None
        self.state.temps = self.ptstate.ladder

        for i, cs in enumerate(self.multi_chain_stats.chain_stats):
            prefix = f"cs.c{i}."
            cs_arrays = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}
            cs.set_checkpoint_state(cs_arrays, meta["chain_stats"][i])

        if not hasattr(self, "short_chain"):
            self.short_chain = ShortChain(
                self.ndim,
                self.ntemps,
                max(self.save_freq, self.cov_update),
                iteration=0,
                outdir=self.outdir,
                resume=True,
                thin=int(meta["short_chain"].get("thin", 1)),
                chain_format=self.chain_format,
            )
        sc_arrays = {k[len("sc.") :]: v for k, v in arrays.items() if k.startswith("sc.")}
        self.short_chain.set_checkpoint_state(sc_arrays, meta["short_chain"])

        for jp, jp_meta in zip(self.proposal_bundle.jump_proposals, meta["proposal_bundle"]):
            jp.proposal_weights = [float(w) for w in jp_meta["weights"]]
            jp.proposal_probs = (
                None if jp_meta["probs"] is None else np.array(jp_meta["probs"], dtype=float)
            )
            jp._refresh_cumprobs()
            jp._proposal_calls = np.array(jp_meta["calls"], dtype=np.int64)
            jp._proposal_accepts = np.array(jp_meta["accepts"], dtype=np.int64)
            jp._last_proposal_idx = int(jp_meta["last_idx"])

        for entry in meta.get("stateful_proposals", []):
            prop = self.proposal_bundle.jump_proposals[0].proposal_list[entry["idx"]]
            setter = getattr(prop, "set_checkpoint_state", None)
            if callable(setter):
                setter(entry["state"])

        self._last_cov_iter = int(meta["last_cov_iter"])

    def _open_short_chain(self, thin: int) -> None:
        self.short_chain = ShortChain(
            self.ndim,
            self.ntemps,
            max(self.save_freq, self.cov_update),
            iteration=0,
            outdir=self.outdir,
            resume=self.resume,
            thin=thin,
            chain_format=self.chain_format,
        )
        self._last_cov_iter = self.short_chain.iteration

    def _set_initial_state(self, initial_position: np.ndarray) -> None:
        ladder = self.ptstate.ladder
        assert ladder is not None
        initial_position = setup_initial_position(initial_position, self.ntemps)
        if self.wrap is not None:
            initial_position = self.wrap.apply(initial_position)
        lnlike0 = self.lnlike(initial_position)
        lnprior0 = self.lnprior(initial_position)
        lnprob0 = tempered_lnprobs(lnlike0, lnprior0, ladder)
        if np.any(~np.isfinite(lnlike0)):
            raise ValueError("Some likelihood values are not finite.")
        if np.any(~np.isfinite(lnprior0)):
            raise ValueError("An initial value falls outside the prior bounds.")
        self.state = SamplerState(
            initial_position,
            lnlike0,
            lnprior0,
            lnprob0,
            accepted=np.ones(self.ntemps),
            temps=ladder,
        )

    def _restore_from_checkpoint(self) -> bool:
        self.checkpoint_path = check_for_checkpoint(self.outdir)
        if self.resume and self.checkpoint_path is None:
            paths = getattr(self.short_chain, "filepaths", [])
            non_empty = [p for p in paths if os.path.exists(p) and os.path.getsize(p) > 0]
            if non_empty:
                legacy = os.path.join(self.outdir, "sampler_checkpoint.pkl")
                legacy_note = (
                    f" Found a legacy pickle checkpoint ({os.path.basename(legacy)}); this "
                    "version only resumes from sampler_checkpoint.npz + .json, so finish "
                    "that run with the impulse version that wrote it."
                    if os.path.exists(legacy)
                    else ""
                )
                raise RuntimeError(
                    f"resume=True but no usable checkpoint was found in {self.outdir!r}, "
                    f"while chain files already contain data "
                    f"({os.path.basename(non_empty[0])} and {len(non_empty) - 1} other(s)). "
                    "Continuing would append a fresh cold-start run onto the existing "
                    "chain, splicing an unconverged transient into the middle of the "
                    "file. Either restore the checkpoint, or start a clean run with "
                    "resume=False or a new outdir (note resume=False DELETES the "
                    f"existing chain files in that directory).{legacy_note}"
                )
        if not (self.resume and self.checkpoint_path is not None):
            return False

        self._logger.info("Resuming from checkpoint: %s", self.checkpoint_path)
        constructor_num_adapt = self.num_adapt
        num_adapt_explicit = self._num_adapt_explicit
        meta = restore_state_checkpoint(self, self.checkpoint_path)
        checkpoint_num_adapt = meta.get("num_adapt", None)
        if num_adapt_explicit:
            if checkpoint_num_adapt != constructor_num_adapt:
                self._logger.warning(
                    "Resume: overriding checkpointed num_adapt=%s with "
                    "the resuming constructor's explicitly passed "
                    "num_adapt=%s. Proposals whose frozen state is "
                    "serialized (e.g. normalizing flows frozen by "
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
        self.short_chain.truncate_files_to_saved()
        return True

    def _save_flush(self) -> None:
        self.short_chain.save_chain()
        self.save_chain_acceptance_rates()

    def sample(self, initial_position: np.ndarray, num_iterations: int, thin: int = 1):
        """Run the parallel-tempering MCMC sampling loop."""
        if self.ptstate.ladder is None:
            raise ValueError("PTState ladder is not initialized")
        self._open_short_chain(thin)
        self._set_initial_state(initial_position)
        self._restore_from_checkpoint()

        proposals_frozen = False
        for jj in tqdm(
            range(self.short_chain.iteration, num_iterations),
            initial=self.short_chain.iteration,
            total=num_iterations,
            desc="Sampling",
            disable=not self.verbose,
        ):
            adapting = self._adaptation_active(jj)
            if not adapting and not proposals_frozen:
                self._freeze_adaptive_proposals()
                proposals_frozen = True
            self.state = vectorized_mh_step(
                self.state,
                self.proposal_bundle,
                self.lnlike,
                self.lnprior,
                self.rngs[0],
                wrap=self.wrap,
            )
            self.proposal_bundle.report_accepts(self.state.accepted)
            if jj > 0 and jj % self.save_freq == 0:
                self._save_flush()
            self.short_chain.add_state(self.state)
            if jj % self.swap_steps == 0 and self.ntemps > 1:
                self.state = pt_step(
                    self.state, self.ptstate, self.rngs[-1], recompute_lnprobs=not adapting
                )
                if adapting:
                    self.ptstate.adapt_ladder()
                    self.state.lnprobs = tempered_lnprobs(
                        self.state.lnlikes, self.state.lnpriors, self.ptstate.ladder
                    )
            if adapting and jj % self.cov_update == 0:
                new_count = self.short_chain.iteration - self._last_cov_iter
                if new_count > 0:
                    new_samples = self.short_chain.get_recent_samples(new_count)
                    self.multi_chain_stats.update_from_window(new_samples)
                self._last_cov_iter = self.short_chain.iteration
            if jj > 0 and jj % self.save_freq == 0:
                checkpoint_sampler(self, path=self.checkpoint_path)
        self._save_flush()

    def proposal_acceptance_rates(self) -> dict:
        return self.proposal_bundle.acceptance_report()

    def chain_acceptance_rates(self) -> dict:
        ladder = self.ptstate.ladder
        return {
            "temperatures": [] if ladder is None else ladder.tolist(),
            "mh": self.proposal_bundle.chain_acceptance_rates(),
            "pt_swap": self.ptstate.compute_accept_ratio() if self.ntemps > 1 else np.array([]),
        }

    def save_chain_acceptance_rates(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.outdir, "chain_acceptance.json")
        report = self.chain_acceptance_rates()
        swap = report["pt_swap"]
        report["pt_swap"] = swap.tolist() if hasattr(swap, "tolist") else list(swap)
        report["per_proposal"] = self.proposal_acceptance_rates()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        return path

    def load_chain(self) -> dict:
        """Load saved chain files from disk across all temperature chains."""
        samples, lnlike, lnprob, accepted, temperature = [], [], [], [], []
        ncols = self.ndim + 4
        for ii in range(self.ntemps):
            base = os.path.join(self.outdir, f"chain_{ii}")
            fmt = chain_io.detect_format(base)
            if fmt is None:
                raise FileNotFoundError(f"Chain file not found: {base}.bin or {base}.txt")
            data = chain_io.read_rows(base + chain_io.chain_suffix(fmt), ncols, fmt)
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

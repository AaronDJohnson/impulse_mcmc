"""Reversible-jump parallel tempering for a variable number of components.

The state is ``(k, theta_k)``: ``n_shared`` parameters every model has, then
``k`` components of ``n_component`` parameters each, with
``min_components <= k <= max_components``. It is stored as one padded row of
width ``D + 1``, ``D = n_shared + max_components * n_component``: the shared
parameters, then ``max_components`` component slots (slots past ``k`` hold
NaN), then ``k``. :class:`RJLayout` is the only code that knows this
convention.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from impulse.chain_stats import (
    COV_RIDGE_REL,
    ChainStats,
    MultiChainStats,
    _HistoryBuffer,
    svd_groups,
)
from impulse.proposals import (
    _SQRT2_INV,
    DE_MIN_FILL,
    JumpProposals,
    ProposalBundle,
    _draw_group_index,
    _draw_jump_scale,
)
from impulse.resume import CheckpointMismatchError
from impulse.samplers import PeriodicSpec, PTSampler


@dataclass(frozen=True)
class RJLayout:
    """Slot convention for padded reversible-jump rows.

    A row has ``width = D + 1`` entries: ``n_shared`` shared parameters,
    ``max_components`` slots of ``n_component`` parameters (NaN past ``k``),
    then ``k``. A parameter vector is the same without the trailing ``k``.
    The active parameters at model ``k`` are exactly the first
    ``n_active(k) = n_shared + k * n_component`` entries.

    Components are numbered ``0 .. k - 1``. :meth:`insert` and :meth:`remove`
    shift the later components and update ``k``; both return new arrays and
    never modify their input. Custom moves reach the layout as
    ``chain_stats.layout``.

    Parameters
    ----------
    n_shared : int
        Parameters every model has (noise level, background); may be 0.
    n_component : int
        Parameters per component, at least 1.
    max_components : int
        Largest allowed ``k``, at least 1.
    min_components : int, default 0
        Smallest allowed ``k``.
    """

    n_shared: int
    n_component: int
    max_components: int
    min_components: int = 0
    _masks: Dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("n_shared", "n_component", "max_components", "min_components"):
            value = getattr(self, name)
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.n_shared < 0:
            raise ValueError(f"n_shared must be >= 0, got {self.n_shared}")
        if self.n_component < 1:
            raise ValueError(f"n_component must be >= 1, got {self.n_component}")
        if self.max_components < 1:
            raise ValueError(f"max_components must be >= 1, got {self.max_components}")
        if not 0 <= self.min_components <= self.max_components:
            raise ValueError(
                "min_components must satisfy 0 <= min_components <= max_components "
                f"({self.max_components}), got {self.min_components}"
            )

    @property
    def ndim(self) -> int:
        """Length ``D`` of the padded parameter vector (without ``k``)."""
        return self.n_shared + self.max_components * self.n_component

    @property
    def width(self) -> int:
        """Length ``D + 1`` of a stored row."""
        return self.ndim + 1

    def n_active(self, k: int) -> int:
        """Number of active parameters at model ``k``."""
        return self.n_shared + int(k) * self.n_component

    def check_k(self, k: Any) -> int:
        """Return ``k`` as an int; ``ValueError`` unless it is a valid model index."""
        kf = float(k)
        if not kf.is_integer() or not self.min_components <= kf <= self.max_components:
            raise ValueError(
                f"k must be an integer in [{self.min_components}, {self.max_components}], "
                f"got {k!r}"
            )
        return int(kf)

    def k(self, row: np.ndarray) -> int:
        """Model index stored in the last entry of ``row``.

        Assumes a valid row; use :meth:`check_k` to validate untrusted input.
        """
        return int(row[-1])

    def params(self, row: np.ndarray) -> np.ndarray:
        """Padded parameter vector(s): ``row`` without the trailing ``k`` (a view)."""
        return row[..., : self.ndim]

    def shared(self, x: np.ndarray) -> np.ndarray:
        """Shared parameters of row(s) or parameter vector(s) (a view)."""
        return x[..., : self.n_shared]

    def components(self, x: np.ndarray) -> np.ndarray:
        """Component slots of row(s) or parameter vector(s).

        Shape ``(..., max_components, n_component)``.
        """
        x = np.asarray(x)
        return x[..., self.n_shared : self.ndim].reshape(
            x.shape[:-1] + (self.max_components, self.n_component)
        )

    def slot(self, j: int) -> slice:
        """Slice of the parameters of component ``j``."""
        start = self.n_shared + int(j) * self.n_component
        return slice(start, start + self.n_component)

    def active_mask(self, k: int) -> np.ndarray:
        """Read-only ``(D,)`` mask of the parameters active at model ``k``."""
        k = int(k)
        mask = self._masks.get(k)
        if mask is None:
            base = np.arange(self.ndim) < self.n_active(k)
            base.setflags(write=False)
            mask = base.view()
            self._masks[k] = mask
        return mask

    def active_mask_rows(self, ks: Any) -> np.ndarray:
        """``(n, D)`` mask for a batch of model indices."""
        n_act = self.n_shared + np.asarray(ks).astype(np.intp) * self.n_component
        return np.arange(self.ndim)[None, :] < n_act[:, None]

    def make_row(self, x: Any, k: Any) -> np.ndarray:
        """Build a stored row from a ``(D,)`` parameter vector and ``k``.

        Entries past ``k`` become NaN; the shared parameters and the first
        ``k`` components must be finite.
        """
        k = self.check_k(k)
        x = np.asarray(x, dtype=float)
        if x.shape != (self.ndim,):
            raise ValueError(f"expected a parameter vector of shape ({self.ndim},), got {x.shape}")
        n_act = self.n_active(k)
        if not np.all(np.isfinite(x[:n_act])):
            raise ValueError(f"the shared parameters and the first k={k} components must be finite")
        row = np.full(self.width, np.nan)
        row[:n_act] = x[:n_act]
        row[-1] = k
        return row

    def _check_row(self, row: Any) -> None:
        if np.shape(row) != (self.width,):
            raise ValueError(
                f"row must have shape ({self.width},) (parameters then k); got {np.shape(row)}"
            )

    def insert(self, row: np.ndarray, j: int, u: Any) -> np.ndarray:
        """Copy of ``row`` with component ``u`` inserted at position ``j`` (``0 <= j <= k``).

        Checks the row width, ``j``, ``u`` and that ``k < max_components``; it does
        not enforce ``min_components``. Never modifies ``row``.
        """
        self._check_row(row)
        j = int(j)
        k = self.k(row)
        p = self.n_component
        if k >= self.max_components:
            raise ValueError(f"cannot insert: k={k} is already max_components")
        if not 0 <= j <= k:
            raise ValueError(f"insert position j={j} must be in [0, {k}]")
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape != (p,):
            raise ValueError(f"a component has {p} parameters, got {u.size}")
        if not np.all(np.isfinite(u)):
            raise ValueError("inserted component must be finite")
        row = np.asarray(row, dtype=float)
        out = row.copy()
        lo = self.n_shared + j * p
        hi = self.n_active(k)
        out[lo + p : hi + p] = row[lo:hi]
        out[lo : lo + p] = u
        out[-1] = k + 1
        return out

    def remove(self, row: np.ndarray, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(row without component j, the removed component)`` for ``0 <= j < k``.

        Checks the row width and ``j``; it does not enforce ``min_components``
        (``k`` may drop below it). Never modifies ``row``.
        """
        self._check_row(row)
        j = int(j)
        k = self.k(row)
        p = self.n_component
        if not 0 <= j < k:
            raise ValueError(f"remove position j={j} must be in [0, {k - 1}]")
        row = np.asarray(row, dtype=float)
        out = row.copy()
        lo = self.n_shared + j * p
        hi = self.n_active(k)
        u = row[lo : lo + p].copy()
        out[lo : hi - p] = row[lo + p : hi]
        out[hi - p : hi] = np.nan
        out[-1] = k - 1
        return out, u


_MISSING: Any = object()

ActiveBlock = Tuple[np.ndarray, np.ndarray, int, float]


def pairwise_covariance(params: np.ndarray, fallback_cov: np.ndarray, min_rows: int) -> np.ndarray:
    """Pairwise-complete covariance of NaN-padded rows.

    Entry ``(a, b)`` is the sample covariance over the rows where both ``a``
    and ``b`` are present (not NaN); entries with fewer than ``min_rows`` such
    rows take ``fallback_cov[a, b]``. The result is symmetric but not
    necessarily positive semi-definite.
    """
    if min_rows < 2:
        raise ValueError(f"min_rows must be >= 2, got {min_rows}")
    params = np.asarray(params, dtype=np.float64)
    present = ~np.isnan(params)
    weights = present.astype(np.float64)
    counts = weights.sum(axis=0)
    filled = np.where(present, params, 0.0)
    # Centre each column first: covariance is shift-invariant, and centring
    # avoids cancellation when a parameter's mean is large next to its spread.
    centre = filled.sum(axis=0) / np.maximum(counts, 1.0)
    centred = np.where(present, params - centre, 0.0)
    n_ab = weights.T @ weights
    s_ab = centred.T @ centred
    t_ab = centred.T @ weights  # [a, b]: sum of column a over the rows where b is present
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (s_ab - t_ab * t_ab.T / n_ab) / (n_ab - 1.0)
    cov = np.where(n_ab >= min_rows, cov, fallback_cov)
    return 0.5 * (cov + cov.T)


def clip_to_psd(cov: np.ndarray, floor: float) -> np.ndarray:
    """``cov`` if its smallest eigenvalue is at least ``floor``, else clipped at ``floor``."""
    try:
        w, v = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return cov
    if w[0] >= floor:
        return cov
    fixed = (v * np.maximum(w, floor)) @ v.T
    return 0.5 * (fixed + fixed.T)


def masked_moments(
    params: np.ndarray,
    fallback_mean: np.ndarray,
    fallback_cov: np.ndarray,
    min_rows: int,
    floor: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-column mean and repaired pairwise-complete covariance, or None if not finite."""
    params = np.asarray(params, dtype=np.float64)
    present = ~np.isnan(params)
    counts = present.sum(axis=0)
    sums = np.where(present, params, 0.0).sum(axis=0)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), fallback_mean)
    cov = pairwise_covariance(params, fallback_cov, min_rows)
    if not np.all(np.isfinite(cov)):
        return None
    return mean, clip_to_psd(cov, floor)


@dataclass
class MaskedChainStats(ChainStats):
    """Adaptive statistics over NaN-padded reversible-jump rows.

    ``ndim`` is the padded parameter count ``D``: history rows store the
    parameters with the ``k`` column dropped and NaN in inactive slots. The
    covariance is pairwise-complete over co-active rows
    (:func:`pairwise_covariance`), falls back to the initial covariance for
    pairs seen together fewer than ``min_pair_rows`` times, and is clipped to
    be positive semi-definite (``sample_cov``, which is pooled over models).

    Within-model moves use :meth:`active_block`: the factor of one group's
    active coordinates at one ``k``. The block comes from
    ``(n_k * S_k + n0 * P_k) / (n_k + n0)``: ``S_k`` is the sample covariance
    of the ``n_k`` buffered rows of model exactly ``k`` (needs ``n_k >= 2``),
    ``P_k`` the active block of the pooled ``sample_cov``, and
    ``n0 = max(min_pair_rows, 2 * d_k)`` (``d_k`` active parameters) is the
    shrinkage weight, so a model with few or degenerate rows leans on the
    pooled block instead of a rank-deficient estimate. Per-model covariances
    and factors are computed on first use and cached until the next update,
    so they cost nothing to checkpoint and rebuild identically. The initial
    covariance is kept as the fallback for rarely co-active pairs and is
    checkpointed with the other state.
    """

    layout: Optional[RJLayout] = None
    min_pair_rows: int = 10

    def __post_init__(self) -> None:
        if self.layout is None:
            raise ValueError("MaskedChainStats needs an RJLayout (layout=...)")
        if self.ndim != self.layout.ndim:
            raise ValueError(
                f"ndim={self.ndim} does not match the layout's padded dimension "
                f"{self.layout.ndim}"
            )
        if self.min_pair_rows < 2:
            raise ValueError(f"min_pair_rows must be >= 2, got {self.min_pair_rows}")
        super().__post_init__()
        assert self.sample_cov is not None and self.sample_mean is not None
        self._fallback_cov = np.array(self.sample_cov, dtype=float)
        self._fallback_mean = np.array(self.sample_mean, dtype=float)
        self._factors: Dict[Tuple[int, int], Optional[ActiveBlock]] = {}
        self._model_cov: Dict[int, Optional[np.ndarray]] = {}
        self._eligible: Dict[int, List[int]] = {}

    def update_from_window(self, sample_num: int, new_samples: np.ndarray) -> None:
        """Store new ``(n, D + 1)`` rows without ``k`` and recompute the masked moments."""
        layout = self.layout
        assert layout is not None and self.groups is not None
        assert self.svd_U is not None and self.svd_S is not None
        new_samples = np.asarray(new_samples, dtype=float)
        ks = new_samples[:, self.ndim]
        if not (
            np.all(np.isfinite(ks))
            and np.all(ks == np.floor(ks))
            and np.all((ks >= 0) & (ks <= layout.max_components))
        ):
            return
        active = layout.active_mask_rows(ks)
        params = new_samples[:, : self.ndim]
        if not np.all(np.isfinite(params[active])):
            return
        # Inactive slots are NaN by convention; never store stale values there.
        params = np.where(active, params, np.nan)
        stored = _HistoryBuffer.select_for_storage(params, self._seen_raw, self.buffer_thin)
        self._seen_raw += len(params)
        if len(stored) == 0:
            return
        self.sample_total += len(stored)
        self.update_buffer(stored)
        self._factors.clear()
        self._model_cov.clear()
        if sample_num + len(params) < 2:
            return
        n_valid = min(self.sample_total, self.buffer_size)
        moments = masked_moments(
            self._buffer[-n_valid:],
            self._fallback_mean,
            self._fallback_cov,
            self.min_pair_rows,
            self._cov_ridge,
        )
        if moments is None:
            return
        self.sample_mean, self.sample_cov = moments
        self.svd_U, self.svd_S, self.proposal_L = svd_groups(
            self.svd_U,
            self.svd_S,
            self.groups,
            self.sample_cov,
            self.proposal_L,
            ridge=self._cov_ridge,
        )

    recursive_update = update_from_window

    def get_checkpoint_state(self) -> Tuple[dict, dict]:
        arrays, meta = super().get_checkpoint_state()
        arrays["fallback_cov"] = np.asarray(self._fallback_cov)
        arrays["fallback_mean"] = np.asarray(self._fallback_mean)
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        super().set_checkpoint_state(arrays, meta)
        if "fallback_cov" in arrays:
            self._fallback_cov = np.array(arrays["fallback_cov"], dtype=float)
            self._fallback_mean = np.array(arrays["fallback_mean"], dtype=float)
            scale = float(np.mean(np.diag(self._fallback_cov)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            self._cov_ridge = COV_RIDGE_REL * scale
        self._factors.clear()
        self._model_cov.clear()
        self._eligible.clear()

    def eligible_groups(self, k: int) -> List[int]:
        """Indices of the groups with at least one coordinate active at model ``k``."""
        cached = self._eligible.get(k)
        if cached is None:
            assert self.layout is not None
            n_act = self.layout.n_active(k)
            cached = [
                gi
                for gi, g in enumerate(self.group_index_arrays())
                if g.size and int(g.min()) < n_act
            ]
            self._eligible[k] = cached
        return cached

    def _model_covariance(self, k: int) -> Optional[np.ndarray]:
        """Model-``k`` covariance over its ``n_active(k)`` coordinates, shrunk to the pooled block.

        ``(n_k * S_k + n0 * P_k) / (n_k + n0)`` with ``n_k`` the buffered rows
        of model exactly ``k``; just ``P_k`` when ``n_k < 2``. None when the
        model has no active parameter.
        """
        if k in self._model_cov:
            return self._model_cov[k]
        layout = self.layout
        assert layout is not None and self.sample_cov is not None
        d_k = layout.n_active(k)
        cov: Optional[np.ndarray] = None
        if d_k > 0:
            cov = np.array(self.sample_cov[:d_k, :d_k], dtype=float)
            n_valid = min(self.sample_total, self.buffer_size)
            if n_valid > 0:
                rows = self._buffer[-n_valid:]
                n_comp = np.count_nonzero(np.isfinite(rows[:, layout.n_shared :]), axis=1)
                mine = rows[n_comp // layout.n_component == k][:, :d_k]
                n_k = len(mine)
                if n_k >= 2:
                    own = np.atleast_2d(np.cov(mine, rowvar=False))
                    if np.all(np.isfinite(own)):
                        n0 = max(self.min_pair_rows, 2 * d_k)
                        cov = (n_k * own + n0 * cov) / (n_k + n0)
        self._model_cov[k] = cov
        return cov

    def active_block(self, group_index: int, k: int) -> Optional[ActiveBlock]:
        """``(indices, L, n_eff, sqrt(2 n_eff))`` for one group's active coordinates at ``k``.

        ``L`` is the scaled eigen-factor (``L @ L.T`` is the covariance block
        plus the ridge). The block is the model-``k`` covariance shrunk toward
        the pooled ``sample_cov`` (see the class docstring). Returns None when the group
        has no active coordinate.
        """
        key = (int(group_index), int(k))
        entry = self._factors.get(key, _MISSING)
        if entry is not _MISSING:
            return entry
        assert self.layout is not None and self.sample_cov is not None
        group = self.group_index_arrays()[key[0]]
        idx = group[group < self.layout.n_active(k)]
        if idx.size == 0:
            entry = None
        else:
            source = self._model_covariance(key[1])
            assert source is not None
            block = np.ascontiguousarray(source[np.ix_(idx, idx)])
            _, _, factors = svd_groups(
                [None], [None], [np.arange(idx.size)], block, ridge=self._cov_ridge
            )
            entry = (idx, factors[0], int(idx.size), math.sqrt(2 * idx.size))
        self._factors[key] = entry
        return entry

    def draw_active_block(self, rng: np.random.Generator, k: int) -> Optional[ActiveBlock]:
        """Pick a group with active coordinates at ``k``; return its :meth:`active_block`."""
        eligible = self.eligible_groups(k)
        if not eligible:
            return None
        return self.active_block(eligible[_draw_group_index(rng, len(eligible))], k)


def masked_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a - b`` with 0 wherever either entry is NaN (an inactive slot)."""
    d = np.subtract(a, b, dtype=float)
    d[np.isnan(d)] = 0.0
    return d


def _layout(chain_stats: MaskedChainStats) -> RJLayout:
    layout = chain_stats.layout
    assert layout is not None
    return layout


def rj_am(chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
    """Adaptive Metropolis on the active coordinates of one group.

    As :func:`impulse.am`, with ``n_eff`` the number of active coordinates in
    the group and the covariance factor of that active block. Returns the
    current state when no group has an active coordinate.
    """
    rng = chain_stats.rng
    x = chain_stats.current_sample
    assert x is not None
    block = chain_stats.draw_active_block(rng, _layout(chain_stats).k(x))
    if block is None:
        return x.copy(), 0.0
    idx, factor, neff, sqrt_2neff = block
    scale = _draw_jump_scale(rng)
    q = x.copy()
    q[idx] += 2.4 / sqrt_2neff * scale * (factor @ rng.standard_normal(neff))
    return q, 0.0


def rj_scam(chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
    """Single-component Adaptive Metropolis along one principal axis of the active block."""
    rng = chain_stats.rng
    x = chain_stats.current_sample
    assert x is not None
    block = chain_stats.draw_active_block(rng, _layout(chain_stats).k(x))
    if block is None:
        return x.copy(), 0.0
    idx, factor, neff, _ = block
    scale = _draw_jump_scale(rng)
    ind = rng.integers(0, neff)
    q = x.copy()
    q[idx] += rng.standard_normal() * _SQRT2_INV * scale * factor[:, ind]
    return q, 0.0


class RJDEProposal:
    """Differential evolution on the active coordinates of one group.

    The difference of two stored rows is taken on the group's active
    coordinates, with 0 where either row lacks the slot. Swapping the two rows
    negates the difference with the same mask, so the move is symmetric. If
    the difference is zero on every active coordinate (either row lacks all of
    them), the row pair is redrawn, up to ``max_pair_draws`` draws in total;
    this depends only on the buffer and the group, never on the current point,
    so symmetry holds. If every draw is zero the current state is returned.
    The current state is also returned until ``min_fill`` rows are stored, or
    when no group has an active coordinate.
    """

    __name__ = "rj_de"
    max_pair_draws = 16

    def __init__(self, min_fill: int = DE_MIN_FILL):
        if min_fill < 2:
            raise ValueError(
                f"min_fill must be >= 2 (two distinct buffer rows are needed for a "
                f"difference move), got {min_fill}"
            )
        self.min_fill = int(min_fill)

    def __call__(self, chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
        rng = chain_stats.rng
        x = chain_stats.current_sample
        assert x is not None
        buf = chain_stats._buffer
        n_filled = min(chain_stats.sample_total, len(buf))
        if n_filled < max(self.min_fill, 2):
            return x.copy(), 0.0
        block = chain_stats.draw_active_block(rng, _layout(chain_stats).k(x))
        if block is None:
            return x.copy(), 0.0
        idx, _, _, sqrt_2neff = block
        offset = len(buf) - n_filled
        for _ in range(self.max_pair_draws):
            mm = rng.integers(0, n_filled)
            nn = rng.integers(0, n_filled)
            while mm == nn:
                nn = rng.integers(0, n_filled)
            diff = masked_difference(buf[offset + mm, idx], buf[offset + nn, idx])
            if diff.any():
                scale = 1.0 if rng.random() > 0.5 else rng.random() * 2.4 / sqrt_2neff
                q = x.copy()
                q[idx] += scale * diff
                return q, 0.0
        return x.copy(), 0.0


rj_de = RJDEProposal()


class IndependentComponentProposal:
    """Component proposal that ignores the current state.

    Parameters
    ----------
    draw : callable
        ``draw(rng) -> u``: one component (``n_component`` values).
    logpdf : callable
        ``logpdf(u) -> float``: the normalized log density of ``draw``'s
        output; finite wherever the component prior is.
    """

    def __init__(self, draw: Callable, logpdf: Callable):
        if not callable(draw) or not callable(logpdf):
            raise TypeError("IndependentComponentProposal needs callable draw and logpdf")
        self._draw = draw
        self._logpdf = logpdf

    def draw(self, rng: np.random.Generator, x: np.ndarray, k: int) -> np.ndarray:
        """One new component; ``x`` and ``k`` are ignored."""
        return self._draw(rng)

    def logpdf(self, u: np.ndarray, x: np.ndarray, k: int) -> float:
        """Log density of component ``u``; ``x`` and ``k`` are ignored."""
        return self._logpdf(u)


def _check_component_proposal(proposal: Any) -> None:
    missing = [name for name in ("draw", "logpdf") if not callable(getattr(proposal, name, None))]
    if missing:
        raise ValueError(
            "component_proposal must have callable draw(rng, x, k) and logpdf(u, x, k); "
            f"{type(proposal).__name__} is missing {missing}"
        )


def _readonly(a: np.ndarray) -> np.ndarray:
    """A read-only view, so a component proposal cannot write into the live state."""
    v = a.view()
    v.flags.writeable = False
    return v


def _draw_component(
    proposal: Any, rng: np.random.Generator, params: np.ndarray, k: int, layout: RJLayout
) -> np.ndarray:
    u = np.asarray(proposal.draw(rng, params, k), dtype=float)
    if u.shape != (layout.n_component,):
        raise ValueError(
            f"{type(proposal).__name__}.draw returned shape {u.shape}; expected "
            f"({layout.n_component},): one component"
        )
    return u


class BirthDeath:
    """Birth or death of one component: the built-in move between models.

    Birth inserts ``u ~ component_proposal.draw`` at a uniform position;
    death removes a uniformly chosen component. ``b_k = d_k = 1/2`` inside
    the allowed range, ``b_min = d_max = 1``. The map is a slot insertion,
    so the Jacobian is exactly 1. One instance serves every chain.

    A death whose component has zero (or non-finite) proposal density is
    rejected with ``qxy = -inf`` and counted in ``rejected_deaths``. The
    methods ``_choose_death_slot(rng, k)`` and ``_selection_log_ratio(k)``
    are overridable hooks (used by the statistical power tests).
    """

    __name__ = "birth_death"
    _BRANCHES = ("birth", "death")

    def __init__(self, layout: RJLayout, component_proposal: Any, ntemps: int):
        if layout.min_components == layout.max_components:
            raise ValueError(
                "BirthDeath needs min_components < max_components; "
                f"both are {layout.max_components}"
            )
        self.layout = layout
        self.component_proposal = component_proposal
        self._calls = np.zeros((ntemps, 2), dtype=np.int64)
        self._accepts = np.zeros((ntemps, 2), dtype=np.int64)
        self._last_branch = np.full(ntemps, -1, dtype=np.int64)
        self.rejected_deaths = 0

    def _birth_probability(self, k: int) -> float:
        if k >= self.layout.max_components:
            return 0.0
        if k <= self.layout.min_components:
            return 1.0
        return 0.5

    def _log_b(self, k: int) -> float:
        p = self._birth_probability(k)
        return math.log(p) if p > 0.0 else -math.inf

    def _log_d(self, k: int) -> float:
        p = 1.0 - self._birth_probability(k)
        return math.log(p) if p > 0.0 else -math.inf

    def _selection_log_ratio(self, k: int) -> float:
        """``log d_{k+1} - log b_k``: the move-choice term of a birth from ``k``."""
        return self._log_d(k + 1) - self._log_b(k)

    def _choose_death_slot(self, rng: np.random.Generator, k: int) -> int:
        return int(rng.integers(0, k))

    def __call__(self, chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
        layout = self.layout
        proposal = self.component_proposal
        rng = chain_stats.rng
        x = chain_stats.current_sample
        assert x is not None
        k = layout.k(x)
        chain = chain_stats.chain_index
        if rng.random() < self._birth_probability(k):
            self._last_branch[chain] = 0
            self._calls[chain, 0] += 1
            j = int(rng.integers(0, k + 1))
            params = _readonly(layout.params(x))
            u = _draw_component(proposal, rng, params, k, layout)
            log_g = float(proposal.logpdf(u, params, k))
            if not math.isfinite(log_g):
                return x.copy(), -math.inf
            return layout.insert(x, j, u), self._selection_log_ratio(k) - log_g
        self._last_branch[chain] = 1
        self._calls[chain, 1] += 1
        j = self._choose_death_slot(rng, k)
        y, u = layout.remove(x, j)
        log_g = float(proposal.logpdf(u, _readonly(layout.params(y)), k - 1))
        if not math.isfinite(log_g):
            self.rejected_deaths += 1
            return x.copy(), -math.inf
        return y, log_g - self._selection_log_ratio(k - 1)

    def report_branch_accept(self, chain_index: int, accepted: bool) -> None:
        """Count an accepted birth or death for the branch ``chain_index`` last took."""
        branch = int(self._last_branch[chain_index])
        if accepted and branch >= 0:
            self._accepts[chain_index, branch] += 1

    def branch_counts(self, chain_index: int) -> Dict[str, Tuple[int, int]]:
        """``{"birth": (calls, accepts), "death": (calls, accepts)}`` for one chain."""
        return {
            name: (int(self._calls[chain_index, b]), int(self._accepts[chain_index, b]))
            for b, name in enumerate(self._BRANCHES)
        }

    def get_checkpoint_state(self) -> dict:
        """Per-chain branch counts and ``rejected_deaths`` as JSON-able lists."""
        return {
            "calls": self._calls.tolist(),
            "accepts": self._accepts.tolist(),
            "last_branch": self._last_branch.tolist(),
            "rejected_deaths": int(self.rejected_deaths),
        }

    def set_checkpoint_state(self, state: dict) -> None:
        """Restore the state written by :meth:`get_checkpoint_state`."""
        self._calls = np.array(state["calls"], dtype=np.int64)
        self._accepts = np.array(state["accepts"], dtype=np.int64)
        self._last_branch = np.array(state["last_branch"], dtype=np.int64)
        self.rejected_deaths = int(state["rejected_deaths"])


class ComponentRedraw:
    """Replace one component with a fresh draw from the component proposal (same ``k``)."""

    __name__ = "component_redraw"

    def __init__(self, layout: RJLayout, component_proposal: Any):
        self.layout = layout
        self.component_proposal = component_proposal

    def __call__(self, chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
        layout = self.layout
        proposal = self.component_proposal
        rng = chain_stats.rng
        x = chain_stats.current_sample
        assert x is not None
        k = layout.k(x)
        if k == 0:
            return x.copy(), 0.0
        j = int(rng.integers(0, k))
        rest, old = layout.remove(x, j)
        params = _readonly(layout.params(rest))
        u = _draw_component(proposal, rng, params, k - 1, layout)
        log_new = float(proposal.logpdf(u, params, k - 1))
        log_old = float(proposal.logpdf(old, params, k - 1))
        if not (math.isfinite(log_new) and math.isfinite(log_old)):
            return x.copy(), -math.inf
        return layout.insert(rest, j, u), log_old - log_new


_FORWARDED_HOOKS = ("get_checkpoint_state", "set_checkpoint_state", "freeze_adaptation")


def _jump_name(move: Any) -> str:
    return str(getattr(move, "__name__", type(move).__name__))


class _WrappedJump:
    """A registered user move: keeps its name and forwards the hooks it has."""

    def __init__(self, move: Callable, layout: RJLayout):
        if not callable(move):
            raise TypeError(f"jump {_jump_name(move)!r} is not callable")
        self.move = move
        self.layout = layout
        self.__name__ = _jump_name(move)
        for hook in _FORWARDED_HOOKS:
            fn = getattr(move, hook, None)
            if callable(fn):
                setattr(self, hook, fn)

    def _result(self, chain_stats: MaskedChainStats, x: np.ndarray) -> Tuple[np.ndarray, float]:
        row, qxy = self.move(chain_stats)
        row = np.asarray(row, dtype=float)
        if np.may_share_memory(row, x):
            raise ValueError(
                f"jump {self.__name__!r} returned an array sharing memory with "
                "chain_stats.current_sample; moves must return a new array and not "
                "modify current_sample"
            )
        if row.shape != (self.layout.width,):
            raise ValueError(
                f"jump {self.__name__!r} returned a row of shape {row.shape}; expected "
                f"({self.layout.width},): the padded parameters followed by k"
            )
        return row, float(qxy)


class _CheckedWithinModelJump(_WrappedJump):
    """A custom within-model move: must keep ``k`` and leave inactive slots NaN."""

    def __call__(self, chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
        x = chain_stats.current_sample
        assert x is not None
        k = self.layout.k(x)
        row, qxy = self._result(chain_stats, x)
        if row[-1] != k:
            raise ValueError(
                f"within-model jump {self.__name__!r} changed k from {k} to {row[-1]}; "
                "register moves that change k with add_model_jump"
            )
        if not np.all(np.isnan(row[self.layout.n_active(k) : self.layout.ndim])):
            raise ValueError(
                f"within-model jump {self.__name__!r} wrote to an inactive slot "
                f"(entries past k={k} must stay NaN)"
            )
        if not np.all(np.isfinite(row[: self.layout.n_active(k)])):
            raise ValueError(
                f"within-model jump {self.__name__!r} returned a non-finite value among "
                f"the active parameters at k={k}"
            )
        return row, qxy


class _ModelJump(_WrappedJump):
    """A custom between-model move; the sampler adds ``±log_jacobian`` when ``k`` changes."""

    def __init__(self, move: Callable, layout: RJLayout, log_jacobian: float):
        super().__init__(move, layout)
        self.log_jacobian = float(log_jacobian)

    def __call__(self, chain_stats: MaskedChainStats) -> Tuple[np.ndarray, float]:
        layout = self.layout
        x = chain_stats.current_sample
        assert x is not None
        k_old = layout.k(x)
        row, qxy = self._result(chain_stats, x)
        try:
            k_new = layout.check_k(row[-1])
        except ValueError as err:
            raise ValueError(f"model jump {self.__name__!r} returned an invalid k: {err}") from None
        n_act = layout.n_active(k_new)
        if not np.all(np.isfinite(row[:n_act])):
            raise ValueError(
                f"model jump {self.__name__!r} returned a non-finite value among the active "
                f"parameters at k={k_new}"
            )
        if not np.all(np.isnan(row[n_act : layout.ndim])):
            raise ValueError(
                f"model jump {self.__name__!r} returned values in slots past k={k_new}; "
                "they must be NaN"
            )
        if k_new > k_old:
            qxy += self.log_jacobian
        elif k_new < k_old:
            qxy -= self.log_jacobian
        return row, qxy


class _RJJumpProposals(JumpProposals):
    """JumpProposals that also reports moves with branches (birth/death) per branch."""

    def report_accept(self, accepted: bool) -> None:
        super().report_accept(accepted)
        idx = self._last_proposal_idx
        if idx >= 0:
            hook = getattr(self.proposal_list[idx], "report_branch_accept", None)
            if hook is not None:
                hook(self.chain_stats.chain_index, accepted)

    def acceptance_rates(self) -> dict:
        result = {}
        for i, proposal in enumerate(self.proposal_list):
            branch_counts = getattr(proposal, "branch_counts", None)
            if branch_counts is None:
                counts = {
                    proposal.__name__: (
                        int(self._proposal_calls[i]),
                        int(self._proposal_accepts[i]),
                    )
                }
            else:
                counts = branch_counts(self.chain_stats.chain_index)
            for name, (calls, accepts) in counts.items():
                rate = accepts / calls if calls > 0 else 0.0
                result[name] = {"calls": calls, "accepts": accepts, "rate": rate}
        return result


class _RJCallable:
    """Adapt ``f(x, k)`` to engine rows, which carry ``k`` as the last column."""

    def __init__(self, f: Callable, ndim: int, vectorized: bool):
        if not callable(f):
            raise TypeError(f"{f!r} is not callable")
        self.f = f
        self.ndim = int(ndim)
        self.vectorized = bool(vectorized)
        self.__name__ = _jump_name(f)

    def __call__(self, x: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        d = self.ndim
        if self.vectorized:
            return self.f(np.ascontiguousarray(x[:, :d]), x[:, d].astype(np.intp), *args, **kwargs)
        return self.f(x[:d], int(x[d]), *args, **kwargs)


def _merge_periodic(
    layout: RJLayout,
    periodic: Optional[PeriodicSpec],
    component_periodic: Optional[PeriodicSpec],
) -> Optional[PeriodicSpec]:
    merged: PeriodicSpec = {}
    for i, spec in (periodic or {}).items():
        if not 0 <= int(i) < layout.ndim:
            raise ValueError(
                f"periodic index {i} is outside the padded parameter vector 0..{layout.ndim - 1}"
            )
        merged[int(i)] = spec
    for c, spec in (component_periodic or {}).items():
        if not 0 <= int(c) < layout.n_component:
            raise ValueError(
                f"component_periodic index {c} is outside a component's parameters "
                f"0..{layout.n_component - 1}"
            )
        for j in range(layout.max_components):
            i = layout.slot(j).start + int(c)
            if i in merged:
                raise ValueError(
                    f"parameter {i} (component slot {j}, entry {c}) is declared in both "
                    "periodic and component_periodic"
                )
            merged[i] = spec
    return merged or None


_RESERVED_JUMP_NAMES = BirthDeath._BRANCHES


def _check_jump_name(move: Any) -> None:
    name = _jump_name(move)
    if name in _RESERVED_JUMP_NAMES:
        raise ValueError(
            f"cannot register a jump named {name!r}: the built-in birth/death move reports "
            f"under {list(_RESERVED_JUMP_NAMES)} in proposal_acceptance_rates(); give the "
            "move another __name__"
        )


class RJPTSampler(PTSampler):
    """Reversible-jump parallel-tempering sampler for a variable number of components.

    Samples ``(k, theta_k)``: ``n_shared`` parameters every model has, plus
    ``k`` components of ``n_component`` parameters each, with
    ``min_components <= k <= max_components``. Moves that change ``k`` are
    accepted with Green's reversible-jump ratio; tempering, swaps, adaptive
    within-model proposals, chain files and checkpoints work as in
    :class:`PTSampler`. See the *Reversible jump* user guide page.

    Parameters
    ----------
    n_shared : int
        Parameters every model has; may be 0.
    n_component : int
        Parameters per component.
    max_components : int
        Largest ``k``. Every row is padded to
        ``D = n_shared + max_components * n_component`` parameters; slots past
        ``k`` hold NaN.
    lnlike, lnprior : callable
        ``f(x, k)``. Per-row mode: ``x`` is the padded ``(D,)`` vector and
        ``k`` an int. With ``vectorized=True``: ``x`` is ``(n, D)`` and ``k``
        an ``(n,)`` integer array. Entries past ``k`` are NaN. ``lnprior`` is
        the joint prior over numbered slots including ``log p(k)``; for
        exchangeable components it is ``log p(k) + sum_{i<k} log pi(theta_i)``
        plus the shared prior, with no ``k!``. The likelihood is tempered and
        the prior is not.
    component_proposal : object
        Birth density for new components, with ``draw(rng, x, k) -> u``
        (``n_component`` values) and ``logpdf(u, x, k) -> float``, where
        ``(x, k)`` is always the smaller model's state. ``logpdf`` must be the
        normalized density of what ``draw`` returns and finite wherever the
        component prior is. Optional ``get_checkpoint_state`` /
        ``set_checkpoint_state`` and ``freeze_adaptation`` hooks are honoured
        and called once by the sampler. :class:`IndependentComponentProposal`
        wraps state-free functions.
    min_components : int, default 0
        Smallest ``k``. With ``min_components == max_components`` the built-in
        birth/death move is not registered.
    birth_death_weight : float, default 40
        Weight of the built-in birth/death move (about 30% of moves with the
        default ``am``/``scam``/``de`` weights); 0 disables it.
    component_redraw_weight : float, default 0
        Weight of ``component_redraw``, which replaces one component with a
        fresh draw from ``component_proposal`` at the same ``k``.
    component_periodic : dict, optional
        Periodic component parameters, ``{index within a component: period or
        (low, high)}``, applied to every slot.
    periodic : dict, optional
        Periodic parameters by index into the padded vector ``0 .. D - 1``
        (normally the shared ones). May not overlap ``component_periodic``.
    vectorized : bool, default False
        As in :class:`PTSampler`; selects the calling convention above.
    **kwargs
        Any other :class:`PTSampler` keyword (``ntemps``, ``outdir``,
        ``seed``, ``groups``, ``sample_cov``, ...). ``groups`` and
        ``sample_cov`` index the padded vector. ``ndim`` is set by the sampler
        and may not be passed.

    Attributes
    ----------
    layout : RJLayout
        The padded-row convention, also available to moves as
        ``chain_stats.layout``.
    birth_death : BirthDeath or None
        The built-in birth/death move, if registered.

    Notes
    -----
    Rows seen by the parallel-tempering engine have width ``D + 1`` with ``k``
    last, so the default temperature-ladder step ``1 + sqrt(2 / ndim)`` uses
    ``ndim = D + 1``; pass ``temp_step`` or ``max_temp`` to set it directly.

    The within-model moves are ``rj_am``, ``rj_scam`` and ``rj_de`` (weights
    ``am_weight``, ``scam_weight``, ``de_weight``), restricted to the active
    coordinates. Add more with :meth:`add_custom_jump`, and moves between
    models with :meth:`add_model_jump`.

    Components are exchangeable, so a slot does not follow one physical
    component through the run (label switching): summarize ``k``, the shared
    parameters and label-free quantities rather than per-slot values.
    """

    def __init__(
        self,
        *,
        n_shared: int,
        n_component: int,
        max_components: int,
        lnlike: Callable,
        lnprior: Callable,
        component_proposal: Any,
        min_components: int = 0,
        birth_death_weight: float = 40.0,
        component_redraw_weight: float = 0.0,
        component_periodic: Optional[PeriodicSpec] = None,
        periodic: Optional[PeriodicSpec] = None,
        vectorized: bool = False,
        **kwargs: Any,
    ) -> None:
        if "ndim" in kwargs:
            raise TypeError(
                "RJPTSampler sets ndim itself (n_shared + max_components * n_component, "
                "plus the k column); do not pass ndim"
            )
        self.layout = RJLayout(n_shared, n_component, max_components, min_components)
        _check_component_proposal(component_proposal)
        for name, weight in (
            ("birth_death_weight", birth_death_weight),
            ("component_redraw_weight", component_redraw_weight),
        ):
            if not (math.isfinite(weight) and weight >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {weight!r}")
        self.component_proposal = component_proposal
        self._wrapped_jumps: Dict[int, _WrappedJump] = {}
        self._model_jumps: List[_ModelJump] = []
        d = self.layout.ndim
        super().__init__(
            ndim=d + 1,
            lnlike=_RJCallable(lnlike, d, vectorized),
            lnprior=_RJCallable(lnprior, d, vectorized),
            periodic=_merge_periodic(self.layout, periodic, component_periodic),
            vectorized=vectorized,
            **kwargs,
        )
        self.birth_death: Optional[BirthDeath] = None
        if birth_death_weight > 0 and self.layout.max_components > self.layout.min_components:
            self.birth_death = BirthDeath(self.layout, component_proposal, self.ntemps)
            self.proposal_bundle.add_jump(self.birth_death, birth_death_weight)
        self.component_redraw: Optional[ComponentRedraw] = None
        if component_redraw_weight > 0:
            self.component_redraw = ComponentRedraw(self.layout, component_proposal)
            self.proposal_bundle.add_jump(self.component_redraw, component_redraw_weight)

    def _build_chain_stats(
        self,
        groups: Optional[list],
        sample_cov: Optional[np.ndarray],
        sample_mean: Optional[np.ndarray],
        buffer_size: int,
        buffer_thin: int,
    ) -> MultiChainStats:
        chain_stats: List[ChainStats] = [
            MaskedChainStats(
                self.layout.ndim,
                self.ptstate,
                ii,
                self.rngs[ii],
                groups=groups,
                sample_cov=sample_cov,
                sample_mean=sample_mean,
                buffer_size=buffer_size,
                buffer_thin=buffer_thin,
                layout=self.layout,
            )
            for ii in range(self.ntemps)
        ]
        return MultiChainStats(chain_stats)

    def _build_proposal_bundle(
        self, am_weight: float, scam_weight: float, de_weight: float, de_min_fill: int
    ) -> ProposalBundle:
        de_jump = RJDEProposal(de_min_fill)
        jumps: List[JumpProposals] = [
            _RJJumpProposals(cs) for cs in self.multi_chain_stats.chain_stats
        ]
        for jp in jumps:
            jp.add_jump(rj_am, am_weight)
            jp.add_jump(rj_scam, scam_weight)
            jp.add_jump(de_jump, de_weight)
        return ProposalBundle(jumps)

    def add_custom_jump(self, proposal: Any, weight: float) -> None:
        """Register a within-model move on every chain.

        ``proposal(chain_stats) -> (row, qxy)``. ``chain_stats.current_sample``
        is the current full row, shape ``(D + 1,)`` with ``k`` last, and
        ``chain_stats.layout`` is the sampler's :class:`RJLayout`
        (``layout.k(row)``, ``layout.params(row)``, ``layout.slot(j)``). The
        move must return a NEW ``(D + 1,)`` array (for example
        ``chain_stats.current_sample.copy()``, then modified) and must not
        modify ``chain_stats.current_sample``, which is a view of the live
        chain state; a returned array that shares memory with it raises
        ``ValueError``. The returned row must keep ``k`` (``row[-1]``), leave
        the slots past ``k`` NaN and have finite active values; each is
        checked and raises ``ValueError`` naming the move. A move named
        ``"birth"`` or ``"death"`` is refused: those names are the built-in
        birth/death entries of :meth:`proposal_acceptance_rates`.
        """
        _check_jump_name(proposal)
        wrapped = self._wrapped_jumps.get(id(proposal))
        if wrapped is None:
            wrapped = _CheckedWithinModelJump(proposal, self.layout)
            self._wrapped_jumps[id(proposal)] = wrapped
        elif not isinstance(wrapped, _CheckedWithinModelJump):
            raise ValueError(f"{wrapped.__name__!r} is already registered as a model jump")
        self.proposal_bundle.add_jump(wrapped, weight)

    def add_model_jump(self, move: Any, weight: float, log_jacobian: float = 0.0) -> None:
        """Register a move between models on every chain.

        ``move(chain_stats) -> (row, qxy)``, with ``row[-1]`` the new ``k``
        and ``qxy`` Green's ratio without the Jacobian:
        ``log[j(y->x) g'(u') / (j(x->y) g(u))]``, plus ``log|det J|`` only if
        it depends on the state. The sampler adds ``log_jacobian`` when ``k``
        increases and subtracts it when ``k`` decreases. Register a move and
        its reverse as one proposal, which computes ``j(y->x)`` itself.

        ``chain_stats.current_sample`` is the current full row, shape
        ``(D + 1,)`` with ``k`` last, and ``chain_stats.layout`` is the
        sampler's :class:`RJLayout`; :meth:`RJLayout.insert` and
        :meth:`RJLayout.remove` build the new row as a copy. The move must
        return a NEW array and must not modify ``chain_stats.current_sample``,
        which is a view of the live chain state; a returned array that shares
        memory with it raises ``ValueError``. Returned rows are checked
        (shape ``(D + 1,)``, valid ``k``, finite active values, NaN past
        ``k``); a violation raises ``ValueError`` naming the move. A move
        named ``"birth"`` or ``"death"`` is refused: those names are the
        built-in birth/death entries of :meth:`proposal_acceptance_rates`.
        """
        _check_jump_name(move)
        log_jacobian = float(log_jacobian)
        if not math.isfinite(log_jacobian):
            raise ValueError(
                f"log_jacobian for model jump {_jump_name(move)!r} must be finite, "
                f"got {log_jacobian}"
            )
        wrapped = self._wrapped_jumps.get(id(move))
        if wrapped is None:
            wrapped = _ModelJump(move, self.layout, log_jacobian)
            self._wrapped_jumps[id(move)] = wrapped
            self._model_jumps.append(wrapped)
        elif isinstance(wrapped, _ModelJump):
            wrapped.log_jacobian = log_jacobian
        else:
            raise ValueError(f"{wrapped.__name__!r} is already registered as a within-model jump")
        self.proposal_bundle.add_jump(wrapped, weight)

    def _freeze_adaptive_proposals(self) -> None:
        super()._freeze_adaptive_proposals()
        freeze = getattr(self.component_proposal, "freeze_adaptation", None)
        if callable(freeze):
            freeze()

    def sample(
        self,
        initial_position: np.ndarray,
        num_iterations: int,
        thin: int = 1,
        initial_k: Any = None,
    ) -> None:
        """Run the sampler.

        Parameters
        ----------
        initial_position : np.ndarray
            Padded parameters without ``k``: ``(D,)`` for every temperature or
            ``(ntemps, D)``. Entries past ``initial_k`` are ignored (set to NaN).
        num_iterations : int
            Total iterations, including any already completed on resume.
        thin : int, default 1
            Save every ``thin``-th iteration.
        initial_k : int or sequence of int, optional
            Starting number of components, one value or one per temperature;
            defaults to ``min_components``.
        """
        layout = self.layout
        x0 = np.asarray(initial_position, dtype=float)
        if x0.ndim == 1 or (x0.ndim == 2 and x0.shape[0] == 1):
            x0 = np.broadcast_to(x0.reshape(1, -1), (self.ntemps, x0.shape[-1]))
        if x0.shape != (self.ntemps, layout.ndim):
            raise ValueError(
                f"initial_position must have shape ({layout.ndim},) or "
                f"({self.ntemps}, {layout.ndim}): the padded parameters without k; "
                f"got {np.shape(initial_position)}"
            )
        k0 = layout.min_components if initial_k is None else initial_k
        try:
            ks = np.broadcast_to(np.asarray(k0), (self.ntemps,))
        except ValueError:
            raise ValueError(
                f"initial_k must be one integer or {self.ntemps} of them, got {initial_k!r}"
            ) from None
        rows = np.stack([layout.make_row(x0[i], ks[i]) for i in range(self.ntemps)])
        super().sample(rows, num_iterations, thin=thin)
        if self.birth_death is not None and self.birth_death.rejected_deaths:
            warnings.warn(
                f"birth_death: {self.birth_death.rejected_deaths} death proposal(s) were "
                "rejected because component_proposal.logpdf returned -inf for the component "
                "being removed. The proposal must be finite wherever the component prior "
                "is; otherwise those deaths can never happen and the posterior on k is "
                "biased.",
                RuntimeWarning,
                stacklevel=2,
            )

    def load_chain(self) -> dict:
        """Load the saved chains.

        Returns the :meth:`PTSampler.load_chain` keys, with ``samples`` the
        NaN-padded parameters ``(ntemps, n, D)`` (no ``k`` column), plus
        ``k`` ``(ntemps, n)`` integers and the views ``shared``
        ``(ntemps, n, n_shared)`` and ``components``
        ``(ntemps, n, max_components, n_component)``.
        """
        chain = super().load_chain()
        rows = chain["samples"]
        d = self.layout.ndim
        samples = np.ascontiguousarray(rows[..., :d])
        chain["samples"] = samples
        chain["k"] = rows[..., d].astype(np.int64)
        chain["shared"] = self.layout.shared(samples)
        chain["components"] = self.layout.components(samples)
        return chain

    def _rj_settings(self) -> dict:
        return {
            "n_shared": self.layout.n_shared,
            "n_component": self.layout.n_component,
            "max_components": self.layout.max_components,
            "min_components": self.layout.min_components,
            "component_proposal": type(self.component_proposal).__name__,
            "model_jumps": [[j.__name__, j.log_jacobian] for j in self._model_jumps],
        }

    def _capture_checkpoint_state(self) -> Tuple[dict, dict]:
        arrays, meta = super()._capture_checkpoint_state()
        getter = getattr(self.component_proposal, "get_checkpoint_state", None)
        rj = self._rj_settings()
        rj["component_proposal_state"] = getter() if callable(getter) else None
        meta["reversible_jump"] = rj
        return arrays, meta

    def _restore_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        super()._restore_checkpoint_state(arrays, meta)
        state = meta["reversible_jump"]["component_proposal_state"]
        setter = getattr(self.component_proposal, "set_checkpoint_state", None)
        if state is not None and callable(setter):
            setter(state)

    def _verify_extra_metadata(self, meta: dict) -> None:
        rj = meta.get("reversible_jump")
        if rj is None:
            raise CheckpointMismatchError(
                "checkpoint has no reversible-jump settings; it was not written by RJPTSampler"
            )
        fresh = self._rj_settings()
        for key in ("n_shared", "n_component", "max_components", "min_components"):
            expected, actual = int(rj[key]), int(fresh[key])
            if expected != actual:
                raise CheckpointMismatchError(
                    f"{key} mismatch: checkpoint has {key}={expected} but the reconstructed "
                    f"sampler has {key}={actual}; rebuild the sampler with {key}={expected} "
                    "to resume this run, or start a fresh run (resume=False, or a new "
                    "outdir) to change it."
                )
        if rj["component_proposal"] != fresh["component_proposal"]:
            raise CheckpointMismatchError(
                f"component_proposal mismatch: checkpoint used {rj['component_proposal']!r} "
                f"but the reconstructed sampler has {fresh['component_proposal']!r}."
            )
        checkpointed = [[str(name), float(c)] for name, c in rj["model_jumps"]]
        if checkpointed != fresh["model_jumps"]:
            raise CheckpointMismatchError(
                f"model jump mismatch: checkpoint registered {checkpointed} "
                f"([name, log_jacobian]) but the reconstructed sampler has "
                f"{fresh['model_jumps']}; register the same model jumps with the same "
                "log_jacobian before resuming."
            )

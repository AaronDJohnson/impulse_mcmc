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

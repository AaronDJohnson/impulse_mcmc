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

from impulse.chain_stats import ChainStats, MultiChainStats, _HistoryBuffer, svd_groups
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
        """Model index stored in the last entry of ``row``."""
        return int(row[-1])

    def params(self, row: np.ndarray) -> np.ndarray:
        """Padded parameter vector(s): ``row`` without the trailing ``k`` (a view)."""
        return row[..., : self.ndim]

    def shared(self, x: np.ndarray) -> np.ndarray:
        """Shared parameters of row(s) or parameter vector(s) (a view)."""
        return x[..., : self.n_shared]

    def components(self, x: np.ndarray) -> np.ndarray:
        """Component slots of row(s) or parameter vector(s), ``(..., max_components, n_component)``."""
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
            mask = np.arange(self.ndim) < self.n_active(k)
            mask.setflags(write=False)
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

    def insert(self, row: np.ndarray, j: int, u: Any) -> np.ndarray:
        """Copy of ``row`` with component ``u`` inserted at position ``j`` (``0 <= j <= k``)."""
        k = self.k(row)
        p = self.n_component
        if k >= self.max_components:
            raise ValueError(f"cannot insert: k={k} is already max_components")
        if not 0 <= j <= k:
            raise ValueError(f"insert position j={j} must be in [0, {k}]")
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.shape != (p,):
            raise ValueError(f"a component has {p} parameters, got {u.size}")
        row = np.asarray(row, dtype=float)
        out = row.copy()
        lo = self.n_shared + j * p
        hi = self.n_active(k)
        out[lo + p : hi + p] = row[lo:hi]
        out[lo : lo + p] = u
        out[-1] = k + 1
        return out

    def remove(self, row: np.ndarray, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(row without component j, the removed component)`` for ``0 <= j < k``."""
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

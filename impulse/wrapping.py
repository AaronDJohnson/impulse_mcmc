"""Periodic-parameter wrapping.

Some parameters (e.g. phases) are intrinsically circular: any real value
maps to a unique point on a period [low, high). Without explicit handling,
a Markov chain on a parameter with a flat prior (over the period) will
random-walk to ±∞ and the AM/SCAM/DE adaptive proposal scale will grow
without bound.

A ``WrapSpec`` declares which dimensions are periodic and applies modular
reduction to keep stored positions inside their stated period. The user's
log-prior and log-likelihood are assumed to be invariant under shifts of
the period, so wrapping does not affect target densities — and because
``period``-shifted Gaussian proposals remain symmetric on the torus, the
proposal ratio ``qxy`` is unchanged (still 0 for AM/SCAM/DE/gaussian).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np


PeriodicSpec = Dict[int, Union[float, Tuple[float, float]]]


@dataclass
class WrapSpec:
    """Wrap selected columns of a position array into ``[low, high)``."""

    indices: np.ndarray  # int array, shape (k,)
    lows: np.ndarray     # float array, shape (k,)
    periods: np.ndarray  # float array, shape (k,) — high - low

    @classmethod
    def from_dict(cls, periodic: Optional[PeriodicSpec]) -> Optional["WrapSpec"]:
        """Build a WrapSpec from a user-facing dict.

        The dict maps parameter index → either a scalar period (low=0) or a
        ``(low, high)`` tuple.

        Examples
        --------
        ``{2: 2*np.pi}``                  → wrap dim 2 into [0, 2π)
        ``{2: (-np.pi, np.pi)}``          → wrap dim 2 into [-π, π)
        ``{0: 1.0, 3: (-0.5, 0.5)}``      → multiple periodic dims
        """
        if not periodic:
            return None
        idx = sorted(periodic.keys())
        lows, highs = [], []
        for i in idx:
            spec = periodic[i]
            if isinstance(spec, (tuple, list)):
                if len(spec) != 2:
                    raise ValueError(
                        f"periodic[{i}] must be a scalar period or a "
                        f"(low, high) pair; got {spec}")
                lo, hi = float(spec[0]), float(spec[1])
            else:
                lo, hi = 0.0, float(spec)
            if not (hi > lo):
                raise ValueError(
                    f"periodic[{i}]: high ({hi}) must be > low ({lo})")
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
        """Return a copy of ``positions`` with periodic columns wrapped.

        Accepts shape ``(ndim,)`` or ``(n_chains, ndim)``.
        """
        x = np.array(positions, copy=True)
        if x.ndim == 1:
            x[self.indices] = (
                (x[self.indices] - self.lows) % self.periods + self.lows
            )
        else:
            x[..., self.indices] = (
                (x[..., self.indices] - self.lows) % self.periods + self.lows
            )
        return x

    def apply_inplace(self, positions: np.ndarray) -> None:
        """In-place version of :meth:`apply`. Mutates ``positions``."""
        if positions.ndim == 1:
            positions[self.indices] = (
                (positions[self.indices] - self.lows) % self.periods + self.lows
            )
        else:
            positions[..., self.indices] = (
                (positions[..., self.indices] - self.lows) % self.periods
                + self.lows
            )

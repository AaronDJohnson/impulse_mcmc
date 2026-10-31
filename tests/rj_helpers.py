"""Shared pieces for the reversible-jump tests."""

import numpy as np

from impulse.reversible_jump import MaskedChainStats
from impulse.samplers import PTState


def padded_rows(layout, n, seed=0):
    """``n`` random stored rows, k uniform over the layout's range, columns offset from 0."""
    rng = np.random.default_rng(seed)
    ks = rng.integers(layout.min_components, layout.max_components + 1, size=n)
    x = rng.normal(size=(n, layout.ndim)) * np.arange(1, layout.ndim + 1) + 3.0
    return np.stack([layout.make_row(x[i], ks[i]) for i in range(n)])


def make_stats(layout, seed=0, **kwargs):
    """A one-chain MaskedChainStats for ``layout``."""
    ptstate = PTState(ndim=layout.width, ntemps=1)
    return MaskedChainStats(
        layout.ndim, ptstate, 0, np.random.default_rng(seed), layout=layout, **kwargs
    )

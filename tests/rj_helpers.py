"""Shared pieces for the reversible-jump tests."""

import math

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


LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def normal_logpdf(z):
    """Sum of standard-normal log densities."""
    z = np.asarray(z, dtype=float)
    return float(np.sum(-0.5 * z * z - LOG_SQRT_2PI))


def birth_probability(layout, k):
    if k >= layout.max_components:
        return 0.0
    if k <= layout.min_components:
        return 1.0
    return 0.5


class AffineBirthDeath:
    """Birth/death pair with theta = mu + sigma * u, u ~ N(0, I).

    Registered with add_model_jump(..., log_jacobian=sum(log sigma)): qxy here
    omits the Jacobian, which the sampler adds (birth) or subtracts (death).
    Births whose theta leaves the unit-cube prior support are rejected (qxy = -inf).
    """

    __name__ = "affine_birth_death"

    def __init__(self, mu, sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)

    def __call__(self, cs):
        layout, rng, x = cs.layout, cs.rng, cs.current_sample
        k = layout.k(x)
        b = birth_probability(layout, k)
        if rng.random() < b:
            j = int(rng.integers(0, k + 1))
            u = rng.standard_normal(layout.n_component)
            theta = self.mu + self.sigma * u
            if np.any(theta < 0.0) or np.any(theta > 1.0):
                return x.copy(), -np.inf
            y = layout.insert(x, j, theta)
            d_next = 1.0 - birth_probability(layout, k + 1)
            return y, math.log(d_next) - math.log(b) - normal_logpdf(u)
        j = int(rng.integers(0, k))
        y, theta = layout.remove(x, j)
        u = (theta - self.mu) / self.sigma
        b_prev = birth_probability(layout, k - 1)
        return y, math.log(b_prev) - math.log(1.0 - b) + normal_logpdf(u)


def assert_rj_chains_identical(
    a, b, keys=("samples", "k", "lnlike", "lnprob", "accepted", "temperature")
):
    for key in keys:
        assert a[key].shape == b[key].shape, f"{key}: shape {a[key].shape} vs {b[key].shape}"
        assert np.array_equal(a[key], b[key], equal_nan=True), f"{key} differs"


def batch_mean_z(series, target, n_batches=40):
    """(mean - target) / batch-means standard error, on the last multiple of n_batches."""
    series = np.asarray(series, dtype=float)
    n = len(series) // n_batches * n_batches
    batches = series[len(series) - n :].reshape(n_batches, -1).mean(axis=1)
    se = batches.std(ddof=1) / math.sqrt(n_batches)
    return (batches.mean() - target) / max(se, 1e-12)

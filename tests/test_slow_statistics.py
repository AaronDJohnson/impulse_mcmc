"""Long-run statistical checks of PTSampler against targets with known moments.

Marked slow: these run tens of thousands of iterations so the Monte Carlo error
is small enough for tight tolerances. Seeds are fixed, so they are deterministic.
"""

import numpy as np
import pytest

from impulse import PTSampler, effective_sample_size

pytestmark = pytest.mark.slow

MU = np.array([1.0, -2.0, 0.5])
COV = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, -0.1], [0.0, -0.1, 0.25]])
PREC = np.linalg.inv(COV)


def _gauss_lnlike(x):
    d = x - MU
    return -0.5 * d @ PREC @ d


def _bimodal_lnlike(x):
    # two unit-width modes 8 sigma apart along x[0]; x[1] is a plain N(0, 1)
    return np.logaddexp(-0.5 * (x[0] - 4.0) ** 2, -0.5 * (x[0] + 4.0) ** 2) - 0.5 * x[1] ** 2


def _box_lnprior(x):
    return 0.0 if np.all(np.abs(x) < 20) else -np.inf


def _cold_samples(sampler, burn):
    return sampler.load_chain()["samples"][0, burn:]


def test_correlated_gaussian_moments(tmp_path):
    sampler = PTSampler(
        3,
        _gauss_lnlike,
        _box_lnprior,
        ntemps=4,
        seed=2024,
        outdir=str(tmp_path),
        verbose=False,
    )
    sampler.sample(np.zeros(3), 60_000)
    x = _cold_samples(sampler, burn=10_000)

    ess = np.array([effective_sample_size(x[:, [k]])[0] for k in range(3)])
    assert np.all(ess > 1_000), f"cold-chain ESS too low to test against: {ess}"
    se = np.sqrt(np.diag(COV) / ess)
    z = (x.mean(axis=0) - MU) / se
    assert np.all(np.abs(z) < 5), f"mean off by {z} standard errors"
    np.testing.assert_allclose(np.cov(x, rowvar=False), COV, atol=0.05)


def test_tempering_mixes_between_separated_modes(tmp_path):
    sampler = PTSampler(
        2,
        _bimodal_lnlike,
        _box_lnprior,
        ntemps=8,
        max_temp=50.0,
        seed=7,
        outdir=str(tmp_path),
        verbose=False,
    )
    sampler.sample(np.array([4.0, 0.0]), 40_000)
    x0 = _cold_samples(sampler, burn=5_000)[:, 0]

    right = np.mean(x0 > 0)
    assert 0.35 < right < 0.65, f"cold chain spent {right:.2f} of its time in one mode"
    crossings = np.count_nonzero(np.diff(np.sign(x0)) != 0)
    assert crossings > 1_000, f"only {crossings} mode crossings"
    # within each mode the cold chain must still be a unit-width Gaussian
    for sign in (1, -1):
        mode = x0[np.sign(x0) == sign]
        assert abs(mode.mean() - 4.0 * sign) < 0.15
        assert abs(mode.std() - 1.0) < 0.1

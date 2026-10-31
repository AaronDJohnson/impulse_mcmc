"""RJPTSampler evaluation modes: per-row, vectorized, threads and JAX."""

import numpy as np
import pytest
from rj_helpers import assert_rj_chains_identical

from impulse import IndependentComponentProposal, RJPTSampler

K = 3  # one shared parameter, up to 3 one-parameter components
LOG4 = np.log(4.0)


def _draw(rng):
    return rng.uniform(0.0, 4.0, 1)


def _logpdf(u):
    u = np.asarray(u)
    return -LOG4 if np.all((u >= 0) & (u <= 4)) else -np.inf


UNIFORM = IndependentComponentProposal(_draw, _logpdf)


# Per-row and batched versions do the same floating-point operations in the same
# order (squares written as products; adding 0.0 for inactive slots is exact),
# so they agree bit for bit and the chains can be compared exactly.
def _row_lnlike(x, k):
    d = x[0] - 1.0
    total = d * d
    for i in range(K):
        if i < k:
            r = (x[1 + i] - 2.0) / 0.5
            total = total + r * r
    return -0.5 * total


def _row_lnprior(x, k):
    active = x[: 1 + k]
    return -k * LOG4 if np.all((active >= 0.0) & (active <= 4.0)) else -np.inf


def _batch_lnlike(x, k):
    d = x[:, 0] - 1.0
    total = d * d
    for i in range(K):
        r = (x[:, 1 + i] - 2.0) / 0.5
        total = total + np.where(i < k, r * r, 0.0)
    return -0.5 * total


def _batch_lnprior(x, k):
    active = np.arange(1 + K)[None, :] < (1 + k)[:, None]
    inside = np.where(active, (x >= 0.0) & (x <= 4.0), True).all(axis=1)
    return np.where(inside, -k * LOG4, -np.inf)


def _make(outdir, **kwargs):
    options = dict(
        n_shared=1,
        n_component=1,
        max_components=K,
        lnlike=_row_lnlike,
        lnprior=_row_lnprior,
        component_proposal=UNIFORM,
        ntemps=3,
        seed=21,
        outdir=str(outdir),
        verbose=False,
    )
    options.update(kwargs)
    return RJPTSampler(**options)


X0 = np.array([1.0, 2.0, 2.0, 2.0])


@pytest.mark.parametrize("mode", ["vectorized", "threads"])
def test_modes_reproduce_the_per_row_chain(tmp_path, mode):
    reference = _make(tmp_path / "rows")
    reference.sample(X0, 400, initial_k=1)
    if mode == "vectorized":
        other = _make(
            tmp_path / mode, lnlike=_batch_lnlike, lnprior=_batch_lnprior, vectorized=True
        )
    else:
        other = _make(tmp_path / mode, threads=2)
    other.sample(X0, 400, initial_k=1)
    chain = reference.load_chain()
    assert_rj_chains_identical(chain, other.load_chain())
    assert len(np.unique(chain["k"][0])) > 1


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def test_jax_masked_likelihood(tmp_path, jax_x64):
    jax = jax_x64
    jnp = jax.numpy

    @jax.jit
    def lnlike(x, k):
        active = jnp.arange(K)[None, :] < k[:, None]
        resid = jnp.where(active, (x[:, 1:] - 2.0) / 0.5, 0.0)  # 0 * NaN would be NaN
        return -0.5 * ((x[:, 0] - 1.0) ** 2 + jnp.sum(resid**2, axis=1))

    ks = np.array([0, 1, 2, 3])
    rows = np.tile([1.3, 2.1, 1.7, 2.4], (4, 1))
    padded = np.where(np.arange(1 + K)[None, :] < (1 + ks)[:, None], rows, np.nan)
    np.testing.assert_allclose(
        np.asarray(lnlike(padded, ks)), _batch_lnlike(padded, ks), rtol=1e-13
    )

    sampler = _make(tmp_path, lnlike=lnlike, lnprior=_batch_lnprior, vectorized=True, jax=True)
    sampler.sample(X0, 300, initial_k=1)
    chain = sampler.load_chain()
    assert np.all(np.isfinite(chain["lnlike"]))
    assert len(np.unique(chain["k"][0])) > 1

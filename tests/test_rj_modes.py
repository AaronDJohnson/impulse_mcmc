"""RJPTSampler evaluation modes: per-row, vectorized, threads and JAX."""

import threading

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


def _sampler(outdir, **kwargs):
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
    reference = _sampler(tmp_path / "rows")
    reference.sample(X0, 400, initial_k=1)
    workers = set()
    if mode == "vectorized":
        other = _sampler(
            tmp_path / mode, lnlike=_batch_lnlike, lnprior=_batch_lnprior, vectorized=True
        )
    else:

        def lnlike(x, k):
            workers.add(threading.current_thread() is not threading.main_thread())
            return _row_lnlike(x, k)

        other = _sampler(tmp_path / mode, lnlike=lnlike, threads=2)
    other.sample(X0, 400, initial_k=1)
    chain = reference.load_chain()
    assert_rj_chains_identical(chain, other.load_chain())
    assert len(np.unique(chain["k"][0])) > 1
    if mode == "threads":
        assert True in workers  # some rows were evaluated on the pool, not serially


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
    traced = []

    def lnlike_impl(x, k):
        traced.append(x.shape)  # runs once per compilation, not per call
        active = jnp.arange(K)[None, :] < k[:, None]
        comp = jnp.where(active, x[:, 1:], 2.0)  # no NaN reaches the arithmetic (or its grad)
        resid = jnp.where(active, (comp - 2.0) / 0.5, 0.0)
        return -0.5 * ((x[:, 0] - 1.0) ** 2 + jnp.sum(resid**2, axis=1))

    lnlike = jax.jit(lnlike_impl)
    ks = np.array([0, 1, 2, 3])
    rows = np.tile([1.3, 2.1, 1.7, 2.4], (4, 1))
    padded = np.where(np.arange(1 + K)[None, :] < (1 + ks)[:, None], rows, np.nan)
    expected = _batch_lnlike(padded, ks)
    np.testing.assert_allclose(np.asarray(lnlike(padded, ks)), expected, rtol=1e-13)
    with jax.disable_jit():  # XLA can rewrite mask * x into a select; eager mode cannot
        np.testing.assert_allclose(np.asarray(lnlike_impl(padded, ks)), expected, rtol=1e-13)

    # Gradients: the two-where pattern keeps NaN out of f AND f'. A nonlinear slot
    # function (log) is NaN in the gradient if inactive NaN slots reach it unsanitized.
    def log_slots(x, k):
        active = jnp.arange(K)[None, :] < k[:, None]
        safe = jnp.where(active, x[:, 1:], 1.0)  # in-domain value where f and f' are finite
        return jnp.sum(jnp.where(active, jnp.log(safe), 0.0), axis=1)

    grad = jax.grad(lambda x: jnp.sum(log_slots(x, ks)))(jnp.asarray(padded))
    assert np.all(np.isfinite(np.asarray(grad)[~np.isnan(padded)]))
    assert np.all(np.asarray(grad)[np.isnan(padded)] == 0.0)

    shapes = set()

    def recorded(x, k):
        shapes.add(x.shape)
        return lnlike(x, k)

    del traced[:]
    sampler = _sampler(
        tmp_path / "jax", lnlike=recorded, lnprior=_batch_lnprior, vectorized=True, jax=True
    )
    sampler.sample(X0, 300, initial_k=1)
    chain = sampler.load_chain()
    # jax=True evaluates every row every step: one static shape, compiled once for all k
    # Exact trace list (not just len == 1) is deliberate: a retrace per k would show here.
    # If a future JAX version legitimately traces twice, relax to len(traced) == 1.
    assert shapes == {(3, 1 + K)}
    assert traced == [(3, 1 + K)]

    reference = _sampler(
        tmp_path / "numpy", lnlike=_batch_lnlike, lnprior=_batch_lnprior, vectorized=True
    )
    reference.sample(X0, 300, initial_k=1)
    expected_chain = reference.load_chain()
    np.testing.assert_array_equal(chain["k"], expected_chain["k"])
    np.testing.assert_allclose(chain["lnlike"], expected_chain["lnlike"], rtol=1e-12)
    assert len(np.unique(chain["k"][0])) > 1

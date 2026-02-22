"""
End-to-end integration tests for the RJMCMC pipeline.

Uses a simple toy problem where the correct model posterior is
analytically tractable (or at least strongly peaked) to verify
that the sampler selects the right model.
"""

import pytest
import numpy as np
import tempfile
import shutil

from impulse.rjmcmc import RJMCMCProductSpace
from impulse.samplers import PTSampler


# -----------------------------------------------------------------------
# Toy problem: 1-D mean estimation with 1-3 identical components.
# Data generated from a single source; the sampler should strongly
# prefer nmodel=0.
# -----------------------------------------------------------------------

NUM_PARAMS = 2   # (amplitude, frequency) per source
MAX_SOURCES = 3
NDIM = MAX_SOURCES * NUM_PARAMS + 1

# prior bounds
LO = np.array([0.0, 0.0])
HI = np.array([5.0, 3.0])

# synthetic data: single sinusoid
RNG_DATA = np.random.default_rng(0)
N_PTS = 100
T_GRID = np.linspace(0, 2 * np.pi, N_PTS)
SIGMA = 1.0
TRUE_A, TRUE_F = 2.0, 1.0
SIGNAL = TRUE_A * np.sin(2 * np.pi * TRUE_F * T_GRID)
DATA = SIGNAL + SIGMA * RNG_DATA.standard_normal(N_PTS)


def _source_draw(rng):
    return rng.uniform(LO, HI)


def _logprior(params):
    n = len(params)
    for i in range(n // NUM_PARAMS):
        p = params[i * NUM_PARAMS:(i + 1) * NUM_PARAMS]
        if np.any(p < LO) or np.any(p > HI):
            return -np.inf
    return 0.0


def _loglike(params):
    n_sources = len(params) // NUM_PARAMS
    model = np.zeros(N_PTS)
    for i in range(n_sources):
        a = params[i * NUM_PARAMS]
        f = params[i * NUM_PARAMS + 1]
        model += a * np.sin(2 * np.pi * f * T_GRID)
    return -0.5 * np.sum(((DATA - model) / SIGMA) ** 2)


@pytest.fixture
def rjmcmc_space():
    return RJMCMCProductSpace(
        loglikelihood=_loglike,
        logprior=_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_source_draw,
    )


@pytest.fixture
def outdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestFromRJMCMC:
    def test_construction(self, rjmcmc_space, outdir):
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
        )
        assert sampler.ndim == NDIM
        # standard (am, scam, de) + birth + death + nmodel + swap = 7
        n_proposals = len(sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert n_proposals == 7

    def test_initial_position(self, rjmcmc_space):
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        assert x0.shape == (NDIM,)
        assert int(np.rint(x0[-1])) == 0
        # should be within prior
        assert np.isfinite(_logprior(x0[:NUM_PARAMS]))

    def test_short_run(self, rjmcmc_space, outdir):
        """Smoke test: sampler runs without error for a few iterations."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
            save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)
        chain = sampler.load_chain()
        assert chain['samples'].shape == (3, 500, NDIM)


class TestModelRecovery:
    """Run long enough to check that the preferred model is correct."""

    @pytest.mark.slow
    def test_prefers_one_source(self, rjmcmc_space, outdir):
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space,
            ntemps=5,
            seed=42,
            outdir=outdir,
            save_freq=5000,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=20_000)

        chain = sampler.load_chain()
        cold = chain['samples'][0]
        probs = rjmcmc_space.model_posterior_probs(cold, burn=5000)

        # 1 source should be strongly preferred
        assert np.argmax(probs) == 0, f"Expected nmodel=0, got argmax={np.argmax(probs)}, probs={probs}"
        assert probs[0] > 0.5, f"P(1 source) = {probs[0]:.3f}, expected > 0.5"


class TestPriorEnforcement:
    """Verify that ALL source parameters (active + inactive) stay within prior."""

    def test_all_params_in_bounds(self, rjmcmc_space, outdir):
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
            save_freq=1000,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=2000)

        chain = sampler.load_chain()
        cold = chain['samples'][0]

        for i in range(MAX_SOURCES):
            block = cold[:, i * NUM_PARAMS:(i + 1) * NUM_PARAMS]
            assert np.all(block >= LO - 1e-10), f"Source {i} below lower bound"
            assert np.all(block <= HI + 1e-10), f"Source {i} above upper bound"

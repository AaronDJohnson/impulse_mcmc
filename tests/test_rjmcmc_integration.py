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
import pickle

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


class TestSampleCovExpansion:
    """Verify that from_rjmcmc expands per-source sample_cov to full product space."""

    def test_per_source_cov_expanded(self, rjmcmc_space, outdir):
        per_source_cov = np.array([[4.0, 0.5], [0.5, 1.0]])
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
            sample_cov=per_source_cov,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert cs.sample_cov.shape == (NDIM, NDIM)
        # Each source block should match the per-source cov
        for i in range(MAX_SOURCES):
            sl = slice(i * NUM_PARAMS, (i + 1) * NUM_PARAMS)
            np.testing.assert_array_equal(cs.sample_cov[sl, sl], per_source_cov)
        # Model index slot should be 1
        assert cs.sample_cov[-1, -1] == 1.0

    def test_full_cov_passed_through(self, rjmcmc_space, outdir):
        full_cov = np.eye(NDIM) * 2.0
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
            sample_cov=full_cov,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        np.testing.assert_array_equal(cs.sample_cov, full_cov)

    def test_per_source_mean_expanded(self, rjmcmc_space, outdir):
        per_source_mean = np.array([2.5, 1.5])
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
            sample_mean=per_source_mean,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert cs.sample_mean.shape == (NDIM,)
        for i in range(MAX_SOURCES):
            sl = slice(i * NUM_PARAMS, (i + 1) * NUM_PARAMS)
            np.testing.assert_array_equal(cs.sample_mean[sl], per_source_mean)
        assert cs.sample_mean[-1] == 0.0

    def test_short_run_with_per_source_cov(self, rjmcmc_space, outdir):
        """Smoke test: sampler runs with per-source covariance."""
        per_source_cov = np.diag([1.0, 0.5])
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
            sample_cov=per_source_cov, save_freq=200,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)
        chain = sampler.load_chain()
        assert chain['samples'].shape == (2, 200, NDIM)


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


class TestPerModelStats:
    """Verify per-model adaptive proposal statistics."""

    def test_per_model_state_initialized(self, rjmcmc_space, outdir):
        """Per-model state has correct groups for each model."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert hasattr(cs, '_per_model')
        assert len(cs._per_model) == MAX_SOURCES
        # nmodel=0 -> 1 active source group
        assert len(cs._per_model[0].groups) == 1
        assert cs._per_model[0].groups[0] == list(range(NUM_PARAMS))
        # nmodel=1 -> 2 active source groups
        assert len(cs._per_model[1].groups) == 2
        # nmodel=2 -> 3 active source groups
        assert len(cs._per_model[2].groups) == 3

    def test_update_sample_swaps_groups(self, rjmcmc_space, outdir):
        """update_sample swaps in model-specific groups."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]

        pos = np.zeros(NDIM)
        pos[-1] = 0  # nmodel=0
        cs.update_sample(pos)
        assert len(cs.groups) == 1

        pos[-1] = 1  # nmodel=1
        cs.update_sample(pos)
        assert len(cs.groups) == 2

        pos[-1] = 2  # nmodel=2
        cs.update_sample(pos)
        assert len(cs.groups) == 3

    def test_update_sample_swaps_buffer(self, rjmcmc_space, outdir):
        """update_sample swaps in model-specific buffer and sample_total."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]

        # Manually set per-model sample_total to different values
        cs._per_model[0].sample_total = 10
        cs._per_model[1].sample_total = 20

        pos = np.zeros(NDIM)
        pos[-1] = 0
        cs.update_sample(pos)
        assert cs.sample_total == 10

        pos[-1] = 1
        cs.update_sample(pos)
        assert cs.sample_total == 20

    def test_recursive_update_routes_samples(self, rjmcmc_space, outdir):
        """recursive_update partitions samples by nmodel."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]

        rng = np.random.default_rng(99)
        n_samples = 50
        samples = rng.uniform(LO.min(), HI.max(), size=(n_samples, NDIM))
        # 30 samples with nmodel=0, 20 with nmodel=1
        samples[:30, -1] = 0
        samples[30:, -1] = 1

        cs.recursive_update(0, samples)

        assert cs._per_model[0].sample_total == 30
        assert cs._per_model[1].sample_total == 20
        assert cs._per_model[2].sample_total == 0

    def test_proposal_L_diverges(self, rjmcmc_space, outdir):
        """proposal_L diverges between models after model-specific samples."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]

        # Initial proposal_L[0] should be identical (both from same cov)
        L0_before = cs._per_model[0].proposal_L[0].copy()
        L1_before = cs._per_model[1].proposal_L[0].copy()
        np.testing.assert_array_equal(L0_before, L1_before)

        # Feed model-specific samples with different distributions
        rng = np.random.default_rng(99)
        n = 200
        samples_m0 = np.zeros((n, NDIM))
        samples_m0[:, 0] = rng.normal(1.0, 0.1, n)  # tight
        samples_m0[:, 1] = rng.normal(1.0, 0.1, n)
        samples_m0[:, -1] = 0

        samples_m1 = np.zeros((n, NDIM))
        samples_m1[:, 0] = rng.normal(2.5, 2.0, n)  # wide
        samples_m1[:, 1] = rng.normal(1.5, 2.0, n)
        samples_m1[:, -1] = 1

        all_samples = np.vstack([samples_m0, samples_m1])
        cs.recursive_update(0, all_samples)

        L0_after = cs._per_model[0].proposal_L[0]
        L1_after = cs._per_model[1].proposal_L[0]
        # They should now differ
        assert not np.allclose(L0_after, L1_after), \
            "proposal_L should diverge after model-specific updates"

    def test_short_run_with_per_model(self, rjmcmc_space, outdir):
        """Smoke test: RJMCMC sampling runs correctly with per-model stats."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=3, seed=42, outdir=outdir, save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)
        chain = sampler.load_chain()
        assert chain['samples'].shape == (3, 500, NDIM)

    def test_pickle_roundtrip(self, rjmcmc_space, outdir):
        """Pickle round-trip preserves per-model state."""
        sampler = PTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=2, seed=42, outdir=outdir, save_freq=200,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)

        cs_before = sampler.multi_chain_stats.chain_stats[0]
        data = pickle.dumps(cs_before)
        cs_after = pickle.loads(data)

        assert hasattr(cs_after, '_per_model')
        assert len(cs_after._per_model) == MAX_SOURCES
        for k in range(MAX_SOURCES):
            pm_before = cs_before._per_model[k]
            pm_after = cs_after._per_model[k]
            assert len(pm_after.groups) == len(pm_before.groups)
            assert pm_after.sample_total == pm_before.sample_total
            for i in range(len(pm_after.groups)):
                if pm_after.proposal_L[i] is not None:
                    np.testing.assert_array_equal(
                        pm_after.proposal_L[i], pm_before.proposal_L[i],
                    )

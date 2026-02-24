"""Tests for RJPTSampler — hybrid MH + NUTS + PT sampler."""

import pytest
import numpy as np
import tempfile
import shutil
import os
import pickle

from impulse.rjpt_sampler import RJPTSampler, load_rjpt_checkpoint
from impulse.rjmcmc import RJMCMCProductSpace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _simple_lnlike(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return -0.5 * np.sum(x**2)
    return -0.5 * np.sum(x**2, axis=1)


def _simple_lnprior(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return 0.0 if np.all(np.abs(x) <= 10) else -np.inf
    result = np.zeros(x.shape[0])
    mask = np.any(np.abs(x) > 10, axis=1)
    result[mask] = -np.inf
    return result


def _simple_lnlike_grad(x):
    """(active_params) -> (loglike, grad) for standard Gaussian."""
    x = np.asarray(x, dtype=np.float64)
    ll = -0.5 * np.sum(x**2)
    grad = -x
    return ll, grad


# RJMCMC fixtures
NUM_PARAMS = 2
MAX_SOURCES = 3
LO = np.array([0.0, 0.0])
HI = np.array([5.0, 3.0])


def _rj_source_draw(rng):
    return rng.uniform(LO, HI)


def _rj_logprior(params):
    n = len(params)
    for i in range(n // NUM_PARAMS):
        p = params[i * NUM_PARAMS:(i + 1) * NUM_PARAMS]
        if np.any(p < LO) or np.any(p > HI):
            return -np.inf
    return 0.0


RNG_DATA = np.random.default_rng(0)
N_PTS = 50
T_GRID = np.linspace(0, 2 * np.pi, N_PTS)
SIGMA = 1.0
TRUE_A, TRUE_F = 2.0, 1.0
SIGNAL = TRUE_A * np.sin(2 * np.pi * TRUE_F * T_GRID)
DATA = SIGNAL + SIGMA * RNG_DATA.standard_normal(N_PTS)


def _rj_loglike(params):
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
        loglikelihood=_rj_loglike,
        logprior=_rj_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_rj_source_draw,
    )


# ---------------------------------------------------------------------------
# TestRJPTSamplerBasic
# ---------------------------------------------------------------------------

class TestRJPTSamplerBasic:

    def test_init_mh_only(self, temp_dir):
        """MH+PT, no NUTS/RJ."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=3, seed=42, outdir=temp_dir,
        )
        assert sampler.ndim == 2
        assert sampler.ntemps == 3
        assert sampler.nuts_enabled is False
        assert sampler._rjmcmc_space is None

    def test_init_with_nuts(self, temp_dir):
        """lnlike_grad provided enables NUTS."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3, seed=42, outdir=temp_dir,
        )
        assert sampler.nuts_enabled is True
        assert sampler.max_tree_depth == 10
        assert sampler.hot_chain_max_depth == 5

    def test_from_rjmcmc(self, rjmcmc_space, temp_dir):
        """RJ via classmethod."""
        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=3, seed=42, outdir=temp_dir,
        )
        assert sampler.ndim == rjmcmc_space.ndim
        assert sampler._rjmcmc_space is rjmcmc_space
        # 3 standard + 4 RJ = 7 proposals
        n_proposals = len(sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert n_proposals == 7

    def test_from_rjmcmc_with_nuts(self, rjmcmc_space, temp_dir):
        """Both RJ and NUTS."""
        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3, seed=42, outdir=temp_dir,
        )
        assert sampler.nuts_enabled is True
        assert sampler._rjmcmc_space is rjmcmc_space

    def test_threads_param(self, temp_dir):
        """RJPTSampler(threads=2) initializes and forwards to wrappers."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=2, seed=42, outdir=temp_dir, threads=2,
        )
        assert sampler.lnlike.threads == 2
        assert sampler.lnprior.threads == 2

    def test_add_custom_jump(self, temp_dir):
        """Proposal added to all chains."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=2, seed=42, outdir=temp_dir,
        )

        def custom(chain_stats):
            return chain_stats.current_sample.copy(), 0.0

        sampler.add_custom_jump(custom, weight=25)
        for jp in sampler.proposal_bundle.jump_proposals:
            assert len(jp.proposal_list) == 4  # 3 standard + 1 custom


# ---------------------------------------------------------------------------
# TestRJPTSamplerMHPT
# ---------------------------------------------------------------------------

class TestRJPTSamplerMHPT:

    def test_sample_gaussian_2d(self, temp_dir):
        """Basic MH+PT sampling on 2D Gaussian."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=3, seed=42, outdir=temp_dir, save_freq=500,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=500)
        assert sampler.short_chain.iteration == 500

    def test_load_chain_format(self, temp_dir):
        """Correct dict keys and shapes."""
        ntemps = 3
        n_iter = 200
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=ntemps, seed=42, outdir=temp_dir, save_freq=200,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=n_iter)

        chain = sampler.load_chain()
        assert "samples" in chain
        assert "lnlike" in chain
        assert "lnprob" in chain
        assert "accepted" in chain
        assert "temperature" in chain
        assert chain["samples"].shape == (ntemps, n_iter, 2)
        assert chain["lnlike"].shape == (ntemps, n_iter)

    def test_checkpoint_resume(self, temp_dir):
        """Round-trip pickle checkpoint."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            ntemps=2, seed=42, outdir=temp_dir, save_freq=100,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=200)

        # Checkpoint should exist
        ckpt_path = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        assert os.path.exists(ckpt_path)

        # Load checkpoint
        loaded = load_rjpt_checkpoint(
            ckpt_path,
            lnlike=sampler.lnlike,
            lnprior=sampler.lnprior,
            raw_lnlike=_simple_lnlike,
            raw_lnprior=_simple_lnprior,
        )
        assert loaded.ndim == 2
        assert loaded.ntemps == 2


# ---------------------------------------------------------------------------
# TestRJPTSamplerNUTS
# ---------------------------------------------------------------------------

class TestRJPTSamplerNUTS:

    def test_sample_gaussian_with_nuts(self, temp_dir):
        """PT+NUTS on 2D Gaussian, verify mean and variance."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3, seed=42, outdir=temp_dir, save_freq=1000,
            max_tree_depth=6, hot_chain_max_depth=3,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=2000)

        chain = sampler.load_chain()
        cold_samples = chain["samples"][0, 500:]  # burn-in
        mean = np.mean(cold_samples, axis=0)
        var = np.var(cold_samples, axis=0)
        np.testing.assert_allclose(mean, [0.0, 0.0], atol=0.3)
        np.testing.assert_allclose(var, [1.0, 1.0], atol=0.5)

    def test_tempered_gradient(self, temp_dir):
        """Gradient scaled by 1/T."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3, seed=42, outdir=temp_dir,
        )
        # Build initial state to test gradient
        x0 = setup_initial_position_for_test(sampler)
        sampler.state = x0

        # Build tempered logp_and_grad for chain 0 (T=1) and chain 2 (T>1)
        logp_grad_cold, _ = sampler._make_tempered_logp_grad(0, sampler.state)
        logp_grad_hot, _ = sampler._make_tempered_logp_grad(2, sampler.state)

        test_x = np.array([1.0, 1.0])
        _, grad_cold = logp_grad_cold(test_x)
        _, grad_hot = logp_grad_hot(test_x)

        T_hot = sampler.ptstate.ladder[2]
        # grad_hot should be grad_cold / T_hot (prior grad is 0 for uniform)
        np.testing.assert_allclose(grad_hot, grad_cold / T_hot, atol=1e-10)

    def test_step_size_cached(self, temp_dir):
        """Step size found and reused."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2, seed=42, outdir=temp_dir, save_freq=50,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=50)
        # Step sizes should be cached
        assert len(sampler._step_sizes) > 0

    def test_diagnostics(self, temp_dir):
        """tree_depth, divergent, etc. present in diagnostics."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2, seed=42, outdir=temp_dir, save_freq=100,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=100)

        diag = sampler.get_diagnostics()
        assert "num_divergent" in diag
        assert "num_max_depth" in diag
        assert "mean_tree_depth" in diag
        assert "mean_accept_prob" in diag
        assert "final_step_size" in diag
        assert "pt_swap_accept" in diag

    def test_nuts_diagnostics_in_chain(self, temp_dir):
        """NUTS diagnostic columns appear in load_chain output."""
        sampler = RJPTSampler(
            ndim=2, lnlike=_simple_lnlike, lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2, seed=42, outdir=temp_dir, save_freq=100,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=100)

        chain = sampler.load_chain()
        assert "tree_depth" in chain
        assert "divergent" in chain
        assert "energy_error" in chain
        assert "step_size" in chain
        assert "mean_accept_prob" in chain


# ---------------------------------------------------------------------------
# TestRJPTSamplerRJMCMC
# ---------------------------------------------------------------------------

class TestRJPTSamplerRJMCMC:

    def test_rjmcmc_short_run(self, rjmcmc_space, temp_dir):
        """Smoke test: RJ sampler runs without error."""
        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=3, seed=42, outdir=temp_dir, save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)

        chain = sampler.load_chain()
        assert chain["samples"].shape == (3, 500, rjmcmc_space.ndim)

    @pytest.mark.slow
    def test_rjmcmc_model_selection(self, rjmcmc_space, temp_dir):
        """Recover correct model count."""
        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space, ntemps=5, seed=42, outdir=temp_dir, save_freq=5000,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=20_000)

        chain = sampler.load_chain()
        cold = chain["samples"][0]
        probs = rjmcmc_space.model_posterior_probs(cold, burn=5000)
        assert np.argmax(probs) == 0
        assert probs[0] > 0.5

    def test_rjmcmc_with_nuts_smoke(self, rjmcmc_space, temp_dir):
        """RJ+NUTS+PT smoke test."""

        def rj_lnlike_grad(active_params):
            """Gradient of the sinusoid log-likelihood on active params."""
            n_sources = len(active_params) // NUM_PARAMS
            model = np.zeros(N_PTS)
            for i in range(n_sources):
                a = active_params[i * NUM_PARAMS]
                f = active_params[i * NUM_PARAMS + 1]
                model += a * np.sin(2 * np.pi * f * T_GRID)
            residual = DATA - model
            ll = -0.5 * np.sum((residual / SIGMA) ** 2)

            grad = np.zeros_like(active_params)
            for i in range(n_sources):
                a = active_params[i * NUM_PARAMS]
                f = active_params[i * NUM_PARAMS + 1]
                sin_term = np.sin(2 * np.pi * f * T_GRID)
                cos_term = np.cos(2 * np.pi * f * T_GRID)
                grad[i * NUM_PARAMS] = np.sum(residual * sin_term) / SIGMA**2
                grad[i * NUM_PARAMS + 1] = np.sum(residual * a * 2 * np.pi * T_GRID * cos_term) / SIGMA**2
            return ll, grad

        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space,
            lnlike_grad=rj_lnlike_grad,
            ntemps=3, seed=42, outdir=temp_dir, save_freq=200,
            max_tree_depth=4, hot_chain_max_depth=2,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)

        chain = sampler.load_chain()
        assert chain["samples"].shape[0] == 3
        assert "tree_depth" in chain

    def test_lazy_step_size_on_birth(self, rjmcmc_space, temp_dir):
        """New nmodel after birth triggers step size search."""
        sampler = RJPTSampler.from_rjmcmc(
            rjmcmc_space,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2, seed=42, outdir=temp_dir, save_freq=300,
            max_tree_depth=3,
        )
        rng = np.random.default_rng(42)
        x0 = rjmcmc_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=300)
        # After sampling, some step sizes should be cached
        assert len(sampler._step_sizes) > 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def setup_initial_position_for_test(sampler):
    """Create a valid SamplerState for testing internal methods."""
    from impulse.samplers import setup_initial_position
    from impulse.sampler_state import SamplerState

    positions = setup_initial_position(np.zeros(sampler.ndim), sampler.ntemps)
    lnlike0 = sampler.lnlike(positions)
    lnprior0 = sampler.lnprior(positions)
    lnprob0 = 1.0 / sampler.ptstate.ladder * lnlike0 + lnprior0
    return SamplerState(
        positions, lnlike0, lnprior0, lnprob0,
        accepted=np.ones(sampler.ntemps), temps=sampler.ptstate.ladder,
    )

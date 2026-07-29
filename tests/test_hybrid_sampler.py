"""Tests for HybridPTSampler — hybrid MH + NUTS + PT sampler."""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

from impulse.birth_death import BirthDeathProductSpace
from impulse.hybrid_sampler import HybridPTSampler, load_hybrid_checkpoint

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


# Product-space model-selection fixtures
NUM_PARAMS = 2
MAX_SOURCES = 3
LO = np.array([0.0, 0.0])
HI = np.array([5.0, 3.0])


def _rj_source_draw(rng):
    return rng.uniform(LO, HI)


def _rj_logprior(params):
    n = len(params)
    for i in range(n // NUM_PARAMS):
        p = params[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
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
def product_space():
    return BirthDeathProductSpace(
        loglikelihood=_rj_loglike,
        logprior=_rj_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_rj_source_draw,
    )


# ---------------------------------------------------------------------------
# TestHybridPTSamplerBasic
# ---------------------------------------------------------------------------


class TestHybridPTSamplerBasic:

    def test_init_mh_only(self, temp_dir):
        """MH+PT, no NUTS/RJ."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        assert sampler.ndim == 2
        assert sampler.ntemps == 3
        assert sampler.nuts_enabled is False
        assert sampler._product_space is None

    def test_init_with_nuts(self, temp_dir):
        """lnlike_grad provided enables NUTS."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        assert sampler.nuts_enabled is True
        assert sampler.max_tree_depth == 10
        assert sampler.hot_chain_max_depth == 5

    def test_from_product_space(self, product_space, temp_dir):
        """RJ via classmethod."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        assert sampler.ndim == product_space.ndim
        assert sampler._product_space is product_space
        # 3 standard + combined birth-death + nmodel + swap = 6 (the
        # unified min-fill-gated de carries de_weight directly; no
        # early_de registration, no weight-0 stock-de placeholder)
        n_proposals = len(sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert n_proposals == 6
        names = [p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list]
        assert "birth_death" in names
        assert "early_de" not in names
        assert "birth_proposal" not in names
        assert "death_proposal" not in names

    def test_from_product_space_single_model_space(self, temp_dir):
        """Regression: a single-model space must construct successfully.

        BirthDeathProposal rejects max_sources < 2, so from_product_space must
        skip the trans-dimensional jumps (all meaningless with one model)
        and register only the standard continuous jumps.
        """
        space = BirthDeathProductSpace(
            loglikelihood=_rj_loglike,
            logprior=_rj_logprior,
            num_sources=1,
            num_params=NUM_PARAMS,
            source_prior_draw=_rj_source_draw,
        )
        sampler = HybridPTSampler.from_product_space(
            space,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        assert sampler.ndim == NUM_PARAMS + 1
        names = sorted(p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert names == ["am", "de", "scam"]

    def test_from_product_space_per_source_cov(self, product_space, temp_dir):
        """Per-source sample_cov is expanded to full product space."""
        per_source_cov = np.array([[4.0, 0.5], [0.5, 1.0]])
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            sample_cov=per_source_cov,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert cs.sample_cov.shape == (product_space.ndim, product_space.ndim)
        for i in range(MAX_SOURCES):
            sl = slice(i * NUM_PARAMS, (i + 1) * NUM_PARAMS)
            np.testing.assert_array_equal(cs.sample_cov[sl, sl], per_source_cov)
        assert cs.sample_cov[-1, -1] == 1.0

    def test_from_product_space_per_source_mean(self, product_space, temp_dir):
        """Per-source sample_mean is expanded to full product space."""
        per_source_mean = np.array([2.5, 1.5])
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            sample_mean=per_source_mean,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert cs.sample_mean.shape == (product_space.ndim,)
        for i in range(MAX_SOURCES):
            sl = slice(i * NUM_PARAMS, (i + 1) * NUM_PARAMS)
            np.testing.assert_array_equal(cs.sample_mean[sl], per_source_mean)

    def test_from_product_space_with_nuts(self, product_space, temp_dir):
        """Both RJ and NUTS."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        assert sampler.nuts_enabled is True
        assert sampler._product_space is product_space

    def test_threads_param(self, temp_dir):
        """HybridPTSampler(threads=2) initializes and forwards to wrappers."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            threads=2,
        )
        assert sampler.lnlike.threads == 2
        assert sampler.lnprior.threads == 2

    def test_add_custom_jump(self, temp_dir):
        """Proposal added to all chains."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
        )

        def custom(chain_stats):
            return chain_stats.current_sample.copy(), 0.0

        sampler.add_custom_jump(custom, weight=25)
        for jp in sampler.proposal_bundle.jump_proposals:
            assert len(jp.proposal_list) == 4  # 3 standard + 1 custom


# ---------------------------------------------------------------------------
# TestHybridPTSamplerMHPT
# ---------------------------------------------------------------------------


class TestHybridPTSamplerMHPT:

    def test_sample_gaussian_2d(self, temp_dir):
        """Basic MH+PT sampling on 2D Gaussian."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=500)
        assert sampler.short_chain.iteration == 500

    def test_proposal_acceptance_rates(self, temp_dir):
        """Total calls equals num_iterations * ntemps, all proposals represented."""
        n_iter = 100
        ntemps = 3
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=ntemps,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=n_iter)

        rates = sampler.proposal_acceptance_rates()
        assert set(rates.keys()) == {"am", "scam", "de"}

        total_calls = sum(info["calls"] for info in rates.values())
        assert total_calls == n_iter * ntemps

        total_accepts = sum(info["accepts"] for info in rates.values())
        assert total_accepts <= total_calls
        assert total_accepts > 0

        for info in rates.values():
            assert len(info["per_chain"]) == ntemps
            if info["calls"] > 0:
                assert info["rate"] == pytest.approx(info["accepts"] / info["calls"], abs=1e-12)

    def test_diagnostics_includes_proposal_acceptance(self, temp_dir):
        """get_diagnostics includes proposal_acceptance with correct structure."""
        ntemps = 3
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=ntemps,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=100)

        diag = sampler.get_diagnostics()
        assert "proposal_acceptance" in diag
        pa = diag["proposal_acceptance"]
        assert set(pa.keys()) == {"am", "scam", "de"}
        for info in pa.values():
            assert "calls" in info and "accepts" in info and "rate" in info
            assert len(info["per_chain"]) == ntemps

    def test_load_chain_format(self, temp_dir):
        """Correct dict keys and shapes."""
        ntemps = 3
        n_iter = 200
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=ntemps,
            seed=42,
            outdir=temp_dir,
            save_freq=200,
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

    def test_lnprobs_consistent_after_ladder_adaptation(self, temp_dir):
        """lnprobs stay in sync with the adapted temperature ladder."""
        from impulse.sampler_state import tempered_lnprobs

        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=4,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        initial_ladder = sampler.ptstate.ladder.copy()
        sampler.sample(np.array([0.1, 0.1]), num_iterations=200)

        # adaptation must actually have moved the interior rungs
        assert not np.allclose(sampler.ptstate.ladder, initial_ladder)
        expected = tempered_lnprobs(
            sampler.state.lnlikes,
            sampler.state.lnpriors,
            sampler.ptstate.ladder,
        )
        np.testing.assert_allclose(sampler.state.lnprobs, expected, atol=1e-12)

    def test_checkpoint_resume(self, temp_dir):
        """Round-trip a LEGACY pickle checkpoint via load_hybrid_checkpoint.

        The default format is now the no-code-execution .npz/.json pair;
        ``load_hybrid_checkpoint`` is the legacy pickle loader, so this test
        writes a pickle explicitly to exercise it.
        """
        from impulse.resume import checkpoint_sampler

        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=100,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=200)

        # Write a legacy pickle checkpoint explicitly.
        ckpt_path = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        checkpoint_sampler(
            sampler,
            path=ckpt_path,
            format="pickle",
            omit=("lnlike", "lnprior", "_raw_lnlike", "_raw_lnprior", "lnlike_grad"),
        )
        assert os.path.exists(ckpt_path)

        # Load checkpoint (emits the legacy-pickle security warning)
        loaded = load_hybrid_checkpoint(
            ckpt_path,
            lnlike=sampler.lnlike,
            lnprior=sampler.lnprior,
            raw_lnlike=_simple_lnlike,
            raw_lnprior=_simple_lnprior,
        )
        assert loaded.ndim == 2
        assert loaded.ntemps == 2


# ---------------------------------------------------------------------------
# TestHybridPTSamplerNUTS
# ---------------------------------------------------------------------------


class TestHybridPTSamplerNUTS:

    def test_sample_gaussian_with_nuts(self, temp_dir):
        """PT+NUTS on 2D Gaussian, verify mean and variance."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
            save_freq=1000,
            max_tree_depth=6,
            hot_chain_max_depth=3,
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
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
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
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=50,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=50)
        # Step sizes should be cached
        assert len(sampler._step_sizes) > 0

    def test_diagnostics(self, temp_dir):
        """tree_depth, divergent, etc. present in diagnostics."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=100,
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
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=100,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=100)

        chain = sampler.load_chain()
        assert "tree_depth" in chain
        assert "divergent" in chain
        assert "energy_error" in chain
        assert "step_size" in chain
        assert "mean_accept_prob" in chain


# ---------------------------------------------------------------------------
# TestHybridPTSamplerModelSelection
# ---------------------------------------------------------------------------


class TestHybridPTSamplerModelSelection:

    def test_model_selection_short_run(self, product_space, temp_dir):
        """Smoke test: RJ sampler runs without error."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)

        chain = sampler.load_chain()
        assert chain["samples"].shape == (3, 500, product_space.ndim)

    @pytest.mark.slow
    def test_model_selection(self, product_space, temp_dir):
        """Recover correct model count.

        This was xfailed after the birth/death constant-weight-selection fix,
        with the failure attributed to continuous-space mixing.  That partly
        mis-attributed a second trans-dimensional defect: the value-preserving
        death left posterior-distributed parameters in the inactive slot, and
        ``nmodel_jump`` re-activated them with ``qxy = 0``, biasing the model
        posterior toward MORE sources.  With the death move now refreshing
        the vacated slot from the prior (see
        tests/test_birth_death_detailed_balance.py), this recovers the preferred
        model reliably (checked with seeds 42, 43, and 7)."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=5,
            seed=42,
            outdir=temp_dir,
            save_freq=5000,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=20_000)

        chain = sampler.load_chain()
        cold = chain["samples"][0]
        probs = product_space.model_posterior_probs(cold, burn=5000)
        assert np.argmax(probs) == 0
        assert probs[0] > 0.5

    def test_model_selection_with_nuts_smoke(self, product_space, temp_dir):
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
                grad[i * NUM_PARAMS + 1] = (
                    np.sum(residual * a * 2 * np.pi * T_GRID * cos_term) / SIGMA**2
                )
            return ll, grad

        sampler = HybridPTSampler.from_product_space(
            product_space,
            lnlike_grad=rj_lnlike_grad,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
            save_freq=200,
            max_tree_depth=4,
            hot_chain_max_depth=2,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)

        chain = sampler.load_chain()
        assert chain["samples"].shape[0] == 3
        assert "tree_depth" in chain

    def test_lazy_step_size_on_birth(self, product_space, temp_dir):
        """New nmodel after birth triggers step size search."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=300,
            max_tree_depth=3,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=300)
        # After sampling, some step sizes should be cached
        assert len(sampler._step_sizes) > 0


# ---------------------------------------------------------------------------
# TestPerModelStats
# ---------------------------------------------------------------------------


class TestPerModelStats:
    """Per-model adaptive statistics for HybridPTSampler."""

    def test_per_model_enabled_from_product_space(self, product_space, temp_dir):
        """from_product_space enables per-model stats on all chains."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
        )
        for cs in sampler.multi_chain_stats.chain_stats:
            assert hasattr(cs, "_per_model")
            assert len(cs._per_model) == MAX_SOURCES
            assert len(cs._per_model[0].groups) == 1
            assert len(cs._per_model[2].groups) == 3

    def test_model_selection_smoke_with_per_model(self, product_space, temp_dir):
        """Smoke test: HybridPTSampler runs with per-model stats."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)
        chain = sampler.load_chain()
        assert chain["samples"].shape == (3, 500, product_space.ndim)

    def test_pickle_roundtrip_hybrid(self, product_space, temp_dir):
        """Pickle round-trip preserves per-model state."""
        sampler = HybridPTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=200,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)

        cs_before = sampler.multi_chain_stats.chain_stats[0]
        data = pickle.dumps(cs_before)
        cs_after = pickle.loads(data)

        assert hasattr(cs_after, "_per_model")
        assert len(cs_after._per_model) == MAX_SOURCES
        for k in range(MAX_SOURCES):
            assert cs_after._per_model[k].sample_total == cs_before._per_model[k].sample_total


# ---------------------------------------------------------------------------
# TestNUTSAdapterComponent
# ---------------------------------------------------------------------------


class TestNUTSAdapterComponent:
    """The per-model NUTS adaptation state lives in a PerModelNUTSAdapter
    component (checkpointed as ``_nuts_adapter``); the historical private
    attribute names remain readable/writable views of it."""

    def _make(self, temp_dir, **kwargs):
        return HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            **kwargs,
        )

    def test_checkpoint_pickles_adapter_not_raw_attributes(self, temp_dir):
        """A LEGACY pickle checkpoint serializes the adapter object; none of
        the 2.0-era raw attribute names appear in the pickled instance dict
        (the compat views are class-level properties, never pickled).

        The default format is now the safe .npz/.json pair; this test writes
        a pickle explicitly to check the still-supported pickle path.
        """
        from impulse.nuts.adapter import PerModelNUTSAdapter
        from impulse.resume import checkpoint_sampler

        sampler = self._make(temp_dir, save_freq=10)
        sampler.sample(np.array([0.1, 0.1]), num_iterations=25)

        pkl = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        checkpoint_sampler(
            sampler,
            path=pkl,
            format="pickle",
            omit=("lnlike", "lnprior", "_raw_lnlike", "_raw_lnprior", "lnlike_grad"),
        )
        with open(pkl, "rb") as fp:
            loaded = pickle.load(fp)
        d = loaded.__dict__
        assert isinstance(d.get("_nuts_adapter"), PerModelNUTSAdapter)
        assert not PerModelNUTSAdapter.has_legacy_state(d)
        # compat views work on the unpickled sampler and mirror the adapter
        assert loaded._step_sizes is d["_nuts_adapter"].step_sizes
        assert len(loaded._step_sizes) > 0

    def test_adapter_config_from_constructor(self, temp_dir):
        """Constructor NUTS-adaptation arguments land on the adapter and
        stay visible under the historical private names."""
        sampler = self._make(
            temp_dir,
            mass_matrix_adapt_interval=17,
            mass_matrix_min_samples=6,
            step_size_min=1e-3,
            step_size_max=2.5,
        )
        adapter = sampler._nuts_adapter
        assert adapter.mass_matrix_adapt_interval == 17
        assert adapter.mass_matrix_min_samples == 6
        assert adapter.step_size_min == 1e-3
        assert adapter.step_size_max == 2.5
        assert sampler._mass_matrix_adapt_interval == 17
        assert sampler._mass_matrix_min_samples == 6
        assert sampler._step_size_min == 1e-3
        assert sampler._step_size_max == 2.5

    def test_injected_mass_matrix_never_overwritten(self, temp_dir):
        """set_mass_matrix installs an injected matrix that online
        mass-matrix adaptation must never replace, across several elapsed
        adaptation intervals."""
        from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType

        sampler = self._make(
            temp_dir,
            save_freq=10_000,
            mass_matrix_adapt_interval=10,
            mass_matrix_min_samples=5,
        )
        injected = MassMatrix.from_precision(np.diag([2.0, 4.0]), MassMatrixType.DIAGONAL)
        sampler.set_mass_matrix(2, injected)
        assert sampler._mass_matrix_injected == {2}

        sampler.sample(np.array([0.5, -0.5]), num_iterations=60)

        # several intervals elapsed, matrix still the injected OBJECT
        assert sampler._mass_matrices[2] is injected
        # and step sizes kept re-tuning against it
        assert any(da.count > 0 for da in sampler._dual_averagers.values())


# ---------------------------------------------------------------------------
# TestHybridNumAdapt
# ---------------------------------------------------------------------------


class TestHybridNumAdapt:
    """Adaptation-freeze semantics of HybridPTSampler(num_adapt=...)."""

    def _make(self, temp_dir, num_adapt):
        return HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=10_000,
            cov_update=10,
            mass_matrix_adapt_interval=10,
            mass_matrix_min_samples=5,
            num_adapt=num_adapt,
        )

    @staticmethod
    def _snapshot_at(sampler, iteration):
        """Capture NUTS adaptation state at the start of loop `iteration`.

        Hooks report_accepts (called once per loop iteration, before the
        NUTS step) so the snapshot reflects all adaptation through
        `iteration - 1` — exactly the frozen values when num_adapt equals
        `iteration`.
        """
        snap = {}
        counter = {"jj": -1}
        orig = sampler.proposal_bundle.report_accepts

        def spy(accepts):
            counter["jj"] += 1
            if counter["jj"] == iteration:
                snap["step_sizes"] = dict(sampler._step_sizes)
                snap["mass_matrices"] = dict(sampler._mass_matrices)
                snap["da_objects"] = dict(sampler._dual_averagers)
                snap["da_counts"] = {k: da.count for k, da in sampler._dual_averagers.items()}
            orig(accepts)

        sampler.proposal_bundle.report_accepts = spy
        return snap

    def test_num_adapt_default_none(self, temp_dir):
        """Default num_adapt=None adapts forever (historical behavior)."""
        sampler = HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            ntemps=2,
            outdir=temp_dir,
        )
        assert sampler.num_adapt is None
        assert sampler._adaptation_active(10**9) is True

    def test_num_adapt_missing_attribute_defaults_to_adapt_forever(self, temp_dir):
        """Resume-safety: checkpoints predating num_adapt keep adapting."""
        sampler = self._make(temp_dir, 10)
        # emulate resume from a checkpoint written before num_adapt existed
        del sampler.num_adapt
        assert sampler._adaptation_active(10**9) is True

    def test_num_adapt_freezes_nuts_step_sizes_and_mass_matrices(self, temp_dir):
        """With num_adapt=N, NUTS step sizes and mass matrices are unchanged
        after iteration N while sampling continues."""
        n_adapt, n_iter = 30, 80
        sampler = self._make(temp_dir, n_adapt)
        snap = self._snapshot_at(sampler, n_adapt)
        sampler.sample(np.array([0.5, -0.5]), num_iterations=n_iter)

        # sampling continued past the freeze
        assert sampler.short_chain.iteration == n_iter

        # step sizes unchanged after iteration n_adapt
        assert snap  # snapshot actually taken
        assert sampler._step_sizes == snap["step_sizes"]

        # mass matrices are the same objects (never re-estimated post-freeze)
        assert set(sampler._mass_matrices) == set(snap["mass_matrices"])
        for k, mm in sampler._mass_matrices.items():
            assert mm is snap["mass_matrices"][k]

        # dual averaging received no further updates (same objects, same counts)
        for k, da in sampler._dual_averagers.items():
            assert da is snap["da_objects"][k]
            assert da.count == snap["da_counts"][k]

    def test_num_adapt_none_keeps_nuts_adapting(self, temp_dir):
        """num_adapt=None must not accidentally freeze NUTS adaptation."""
        n_mark, n_iter = 30, 80
        sampler = self._make(temp_dir, None)
        snap = self._snapshot_at(sampler, n_mark)
        sampler.sample(np.array([0.5, -0.5]), num_iterations=n_iter)

        # dual averaging kept adapting after the marker iteration: either the
        # count advanced past the snapshot or the instance was replaced by a
        # mass-matrix commit (which resets DualAveraging). Final step-size
        # float comparison is not a reliable signal here because commits
        # reset step sizes to quantized find_reasonable_step_size values.
        assert any(
            da is not snap["da_objects"][k] or da.count > snap["da_counts"][k]
            for k, da in sampler._dual_averagers.items()
        )
        # mass matrices kept being re-estimated
        assert any(sampler._mass_matrices[k] is not mm for k, mm in snap["mass_matrices"].items())


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def setup_initial_position_for_test(sampler):
    """Create a valid SamplerState for testing internal methods."""
    from impulse.sampler_state import SamplerState
    from impulse.samplers import setup_initial_position

    positions = setup_initial_position(np.zeros(sampler.ndim), sampler.ntemps)
    lnlike0 = sampler.lnlike(positions)
    lnprior0 = sampler.lnprior(positions)
    lnprob0 = 1.0 / sampler.ptstate.ladder * lnlike0 + lnprior0
    return SamplerState(
        positions,
        lnlike0,
        lnprior0,
        lnprob0,
        accepted=np.ones(sampler.ntemps),
        temps=sampler.ptstate.ladder,
    )


# ---------------------------------------------------------------------------
# TestStepSizeFreezeFinalization
# ---------------------------------------------------------------------------


class TestStepSizeFreezeFinalization:
    """The num_adapt freeze must pin the SMOOTHED dual-averaging step size
    (``DualAveraging.finalize()``, exp(log_step_bar)), not the last noisy
    primal iterate exp(log_step) that ``_step_sizes`` tracks during
    adaptation. The primal iterate deliberately overshoots (mu anchors at
    log(10 * step)), so a freeze landing in the transient right after a
    mass-matrix commit resets DA would otherwise pin a step size several
    times the converged value forever.
    """

    def _make(self, temp_dir, **kwargs):
        return HybridPTSampler(
            ndim=2,
            lnlike=_simple_lnlike,
            lnprior=_simple_lnprior,
            lnlike_grad=_simple_lnlike_grad,
            ntemps=2,
            seed=42,
            outdir=temp_dir,
            save_freq=10_000,
            **kwargs,
        )

    def test_finalize_step_sizes_uses_smoothed_value(self, temp_dir):
        """_finalize_step_sizes replaces the primal iterate with the
        clipped finalize() value."""
        from impulse.nuts.warmup import DualAveraging

        sampler = self._make(temp_dir)
        da = DualAveraging(target_accept=0.8, initial_step_size=0.2)
        for accept_prob in [0.1, 0.3, 0.6, 0.9, 0.5]:
            primal = da.update(accept_prob)
        primal = float(np.clip(primal, sampler._step_size_min, sampler._step_size_max))
        smoothed = float(np.clip(da.finalize(), sampler._step_size_min, sampler._step_size_max))
        assert primal != pytest.approx(smoothed)  # meaningful distinction

        key = (0, 2)
        sampler._dual_averagers = {key: da}
        sampler._step_sizes = {key: primal}
        sampler._finalize_step_sizes()
        assert sampler._step_sizes[key] == pytest.approx(smoothed)

    def test_finalize_keeps_current_step_when_da_never_updated(self, temp_dir):
        from impulse.nuts.warmup import DualAveraging

        sampler = self._make(temp_dir)
        key = (0, 2)
        sampler._dual_averagers = {
            key: DualAveraging(
                target_accept=0.8,
                initial_step_size=0.7,
            )
        }
        sampler._step_sizes = {key: 0.123}
        sampler._finalize_step_sizes()
        assert sampler._step_sizes[key] == 0.123

    def test_finalize_keeps_current_step_when_finalize_nonfinite(self, temp_dir):
        from impulse.nuts.warmup import DualAveraging

        sampler = self._make(temp_dir)
        key = (0, 2)
        da = DualAveraging(target_accept=0.8, initial_step_size=0.2)
        da.update(0.5)
        da.log_step_bar = np.nan  # corrupted / undefined smoothed state
        sampler._dual_averagers = {key: da}
        sampler._step_sizes = {key: 0.456}
        sampler._finalize_step_sizes()
        assert sampler._step_sizes[key] == 0.456

    def test_freeze_after_mass_matrix_commit_pins_smoothed_step(self, temp_dir):
        """Freeze landing shortly after a mass-matrix commit (DA reset,
        post-reset transient): frozen step sizes must equal the finalized
        (smoothed) values, not the last primal iterates."""
        n_adapt, n_iter = 38, 60
        sampler = self._make(
            temp_dir,
            cov_update=10,
            mass_matrix_adapt_interval=10,
            mass_matrix_min_samples=5,
            num_adapt=n_adapt,
        )

        # Track the loop iteration (report_accepts runs once per iteration,
        # before the NUTS step) and mass-matrix commit iterations.
        it = {"jj": -1}
        orig_report = sampler.proposal_bundle.report_accepts

        def spy_report(accepts):
            it["jj"] += 1
            orig_report(accepts)

        sampler.proposal_bundle.report_accepts = spy_report

        commits = []
        orig_mm_adapt = sampler._maybe_adapt_mass_matrices

        def spy_mm_adapt():
            before = dict(sampler._mass_matrices)
            orig_mm_adapt()
            if any(sampler._mass_matrices.get(k) is not v for k, v in before.items()):
                commits.append(it["jj"])

        sampler._maybe_adapt_mass_matrices = spy_mm_adapt

        sampler.sample(np.array([0.5, -0.5]), num_iterations=n_iter)

        # The scenario is real: a commit happened before the freeze, close
        # enough that DA was still in its post-reset transient at freeze.
        assert commits, "no mass-matrix commit occurred before the freeze"
        assert commits[-1] < n_adapt
        assert n_adapt - commits[-1] <= 10

        # Post-freeze, DA state is untouched, so finalize() still returns
        # the smoothed value as of the freeze transition.
        checked = 0
        for key, da in sampler._dual_averagers.items():
            if da.count == 0:
                continue
            expected = float(
                np.clip(
                    da.finalize(),
                    sampler._step_size_min,
                    sampler._step_size_max,
                )
            )
            assert sampler._step_sizes[key] == pytest.approx(expected)
            checked += 1
        assert checked > 0

        # And it is genuinely the smoothed value, not the primal iterate:
        # for at least one chain the two differ.
        primals = {
            key: float(np.clip(np.exp(da.log_step), sampler._step_size_min, sampler._step_size_max))
            for key, da in sampler._dual_averagers.items()
            if da.count > 0
        }
        assert any(not np.isclose(primals[key], sampler._step_sizes[key]) for key in primals)


class TestNUTSBindsExtraArgs:
    """loglargs/logpargs must reach the NUTS path, not just the MH path.

    Regression test: the NUTS transition called the RAW user callables, which is
    where loglargs/loglkwargs get bound, so the extras were silently dropped.
    The MH step then targeted the user's intended density while NUTS targeted
    the function's DEFAULTS and the chain converged to neither -- measured sd
    5.05 against a true 4.0. With a REQUIRED extra argument the run instead died
    mid-sampling with a TypeError.
    """

    @staticmethod
    def _lnprior(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return 0.0 if np.all(np.abs(x) <= 40) else -np.inf
        out = np.zeros(x.shape[0])
        out[np.any(np.abs(x) > 40, axis=1)] = -np.inf
        return out

    @staticmethod
    def _grad(a, sigma=1.0):
        """Default 1.0 deliberately DIFFERS from the bound loglargs value of 4.0.

        With the gradient's default equal to the bound value the test passed
        whether or not the binding reached lnlike_grad -- it was blind to half
        its own bug. The gradient must receive the bound extras too.
        """
        a = np.asarray(a, dtype=float)
        return float(-0.5 * np.sum((a / sigma) ** 2)), -a / sigma**2

    def test_defaulted_extra_arg_targets_the_bound_density(self, temp_dir):
        """The silent case: without binding this sampled the wrong distribution."""

        def lnlike(x, sigma=1.0):  # default 1.0 differs from the bound 4.0
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(-0.5 * np.sum((x / sigma) ** 2))
            return -0.5 * np.sum((x / sigma) ** 2, axis=1)

        s = HybridPTSampler(
            ndim=2,
            lnlike=lnlike,
            lnprior=self._lnprior,
            lnlike_grad=self._grad,
            loglargs=(4.0,),
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=15000,
        )
        s.sample(np.zeros(2), num_iterations=15000)
        sd = s.load_chain()["samples"][0][3000:].std(axis=0)
        assert np.all(np.abs(sd - 4.0) < 0.6), f"targeted the wrong density: sd={sd}"

    def test_required_extra_arg_does_not_crash(self, temp_dir):
        """The loud case: a required extra arg used to TypeError mid-sampling."""

        def lnlike(x, sigma):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return float(-0.5 * np.sum((x / sigma) ** 2))
            return -0.5 * np.sum((x / sigma) ** 2, axis=1)

        s = HybridPTSampler(
            ndim=2,
            lnlike=lnlike,
            lnprior=self._lnprior,
            lnlike_grad=self._grad,
            loglargs=(4.0,),
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=3000,
        )
        s.sample(np.zeros(2), num_iterations=3000)  # must not raise
        assert s.load_chain()["samples"].shape[1] == 3000


class TestProductSpaceNUTSActuallyRuns:
    """The product-space NUTS transition must take steps, not silently no-op.

    Regression test: logp_and_grad handed product_space.get_logprior a TRUNCATED
    vector with the model index stripped. get_logprior reads that index off the
    END of the vector itself, so it rinted the last SOURCE parameter as a model
    index, failed the validity check and returned -inf for every proposal --
    NUTS was skipped on every iteration while nuts_enabled reported True.
    Measured 0 of 800 nuts_step calls; 800 of 800 after the fix.

    The failure only shows when the last source parameter is far from a valid
    model index, so the priors here deliberately live in [8, 14].
    """

    NP, NS = 2, 3
    LO = np.array([8.0, 8.0])
    HI = np.array([14.0, 14.0])
    _T = np.linspace(0, 1, 20)

    @classmethod
    def _ll(cls, p):
        p = np.asarray(p, dtype=float)
        m = np.zeros(20)
        for i in range(len(p) // cls.NP):
            m += p[i * cls.NP] * cls._T
        return -0.5 * float(np.sum(m**2))

    @classmethod
    def _lp(cls, p):
        p = np.asarray(p, dtype=float).reshape(-1, cls.NP)
        return 0.0 if np.all((p >= cls.LO) & (p <= cls.HI)) else -np.inf

    @classmethod
    def _grad(cls, a):
        a = np.asarray(a, dtype=float)
        m = np.zeros(20)
        for i in range(len(a) // cls.NP):
            m += a[i * cls.NP] * cls._T
        g = np.zeros_like(a)
        for i in range(len(a) // cls.NP):
            g[i * cls.NP] = float(np.sum(-m * cls._T))
        return -0.5 * float(np.sum(m**2)), g

    def test_nuts_step_is_actually_called(self, temp_dir, monkeypatch):
        import impulse.hybrid_sampler as hs
        from impulse.birth_death import BirthDeathProductSpace

        space = BirthDeathProductSpace(
            loglikelihood=self._ll,
            logprior=self._lp,
            num_sources=self.NS,
            num_params=self.NP,
            source_prior_draw=lambda r: r.uniform(self.LO, self.HI),
        )
        calls = {"n": 0}
        original = hs.nuts_step

        def counting_nuts_step(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        monkeypatch.setattr(hs, "nuts_step", counting_nuts_step)

        s = hs.HybridPTSampler.from_product_space(
            space,
            lnlike_grad=self._grad,
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=300,
        )
        s.sample(space.draw_initial_position(np.random.default_rng(1)), num_iterations=300)

        assert s.nuts_enabled
        assert calls["n"] > 0, "product-space NUTS silently took zero steps"

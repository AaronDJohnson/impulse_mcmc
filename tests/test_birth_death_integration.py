"""
End-to-end integration tests for the product-space model-selection pipeline.

Uses a simple toy problem where the correct model posterior is
analytically tractable (or at least strongly peaked) to verify
that the sampler selects the right model.
"""

import pickle
import shutil
import tempfile

import numpy as np
import pytest

from impulse.birth_death import BirthDeathProductSpace
from impulse.samplers import PTSampler

# -----------------------------------------------------------------------
# Toy problem: 1-D mean estimation with 1-3 identical components.
# Data generated from a single source; the sampler should strongly
# prefer nmodel=0.
# -----------------------------------------------------------------------

NUM_PARAMS = 2  # (amplitude, frequency) per source
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
        p = params[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
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
def product_space():
    return BirthDeathProductSpace(
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
    def test_construction(self, product_space, outdir):
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
        )
        assert sampler.ndim == NDIM
        # standard (am, scam, de) + combined birth-death + nmodel + swap
        # = 6 (birth and death are ONE kernel: schedule-driven selection
        # inside the kernel is what keeps detailed balance under constant
        # weights; de is the unified min-fill-gated DE and carries
        # de_weight directly — no weight-0 placeholder, no early_de)
        n_proposals = len(sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert n_proposals == 6
        names = [p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list]
        assert "birth_death" in names
        assert "early_de" not in names
        assert "birth_proposal" not in names
        assert "death_proposal" not in names
        # the unified de is selectable: it carries de_weight, activating
        # its difference move once the per-model buffer holds de_min_fill
        # samples (no hidden gaussian substitution exists anymore)
        jp = sampler.proposal_bundle.jump_proposals[0]
        de_idx = names.index("de")
        assert jp.proposal_probs[de_idx] > 0.0

    def test_single_model_space_constructs(self, outdir):
        """Regression: a single-model space must construct successfully.

        BirthDeathProposal rejects max_sources < 2, so from_product_space must
        skip the trans-dimensional jumps (birth-death, nmodel, source
        swap — all meaningless with one model) instead of building them
        and raising.  Only the standard continuous jumps are registered.
        """
        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=1,
            num_params=NUM_PARAMS,
            source_prior_draw=_source_draw,
        )
        sampler = PTSampler.from_product_space(space, ntemps=3, seed=42, outdir=outdir)
        assert sampler.ndim == NUM_PARAMS + 1
        names = sorted(p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list)
        assert names == ["am", "de", "scam"]

    def test_zero_birth_death_weight_skips_kernel(self, product_space, outdir):
        """birth_weight + death_weight == 0: the birth-death kernel is
        never selected, so it must not be constructed or registered (the
        other RJ jumps stay)."""
        sampler = PTSampler.from_product_space(
            product_space,
            birth_weight=0,
            death_weight=0,
            ntemps=3,
            seed=42,
            outdir=outdir,
        )
        names = [p.__name__ for p in sampler.proposal_bundle.jump_proposals[0].proposal_list]
        assert "birth_death" not in names
        assert "nmodel_jump" in names
        assert "source_swap_proposal" in names

    def test_initial_position(self, product_space):
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        assert x0.shape == (NDIM,)
        assert int(np.rint(x0[-1])) == 0
        # should be within prior
        assert np.isfinite(_logprior(x0[:NUM_PARAMS]))

    def test_short_run(self, product_space, outdir):
        """Smoke test: sampler runs without error for a few iterations."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
            save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)
        chain = sampler.load_chain()
        assert chain["samples"].shape == (3, 500, NDIM)


class TestSourcePriorLogpdfResolution:
    """The per-source prior-density fallback probe in BirthDeathProductSpace."""

    def test_additive_prior_passes_probe(self):
        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=_source_draw,
        )
        assert space._resolve_source_prior_logpdf() is _logprior

    def test_non_additive_prior_raises(self):
        """A cross-slot coupling makes logprior non-additive: the fallback
        per-source density would be provably wrong, so this must raise
        rather than warn and proceed."""

        def coupled_logprior(params):
            p = np.asarray(params, float)
            return -0.5 * float(np.sum(p)) ** 2  # not additive across slots

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=coupled_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=_source_draw,
        )
        with pytest.raises(ValueError, match="not additive"):
            space.get_birth_death_proposal()

    def test_probe_call_failure_raises_typeerror(self):
        """A logprior that cannot handle a single source's vector gets the
        curated TypeError (including when np.concatenate is what fails)."""

        def strict_logprior(params):
            if len(params) != MAX_SOURCES * NUM_PARAMS:
                raise ValueError("expected the full parameter vector")
            return 0.0

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=strict_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=_source_draw,
        )
        with pytest.raises(TypeError, match="source_prior_logpdf"):
            space.get_birth_death_proposal()

    def test_draws_outside_prior_support_raise(self):
        """A probe draw outside the prior support must raise.

        The fallback probe only runs when neither source_prior_logpdf nor
        source_proposal_logpdf was supplied — a configuration in which
        source_prior_draw is assumed to BE the prior.  An out-of-support
        draw proves it is not, so the birth/death draw-density wiring
        (prior density used as the draw density) is provably wrong and
        silently redrawing would hide the misconfiguration.
        """

        def broad_draw(rng):
            return rng.uniform(LO - 2.0, HI + 2.0)  # mostly out of bounds

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=broad_draw,
        )
        with pytest.raises(ValueError, match="outside the prior support"):
            space._resolve_source_prior_logpdf()

    def test_broad_draw_with_proposal_logpdf_skips_probe(self):
        """The same broader-than-prior draw is fine when its density is
        declared: source_proposal_logpdf supplies the draw density, so the
        prior-density probe (and its support check) is skipped."""

        def broad_draw(rng):
            return rng.uniform(LO - 2.0, HI + 2.0)

        def broad_logpdf(params):
            return float(-np.sum(np.log((HI + 2.0) - (LO - 2.0))))

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=broad_draw,
            source_proposal_logpdf=broad_logpdf,
        )
        log_proposal, log_prior = space._source_draw_density_args()
        assert log_proposal is broad_logpdf
        assert log_prior is None
        # and the kernel builds without probing
        space.get_birth_death_proposal()

    def test_single_model_skips_probe(self):
        """num_sources=1: the full prior IS the per-source prior, so no
        probe draws should be made at all."""

        def raising_draw(rng):
            raise AssertionError("probe must not draw for num_sources=1")

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=1,
            num_params=NUM_PARAMS,
            source_prior_draw=raising_draw,
        )
        assert space._resolve_source_prior_logpdf() is _logprior

    def test_explicit_source_prior_logpdf_bypasses_probe(self):
        def per_source(params):
            return 0.0

        space = BirthDeathProductSpace(
            loglikelihood=_loglike,
            logprior=_logprior,
            num_sources=MAX_SOURCES,
            num_params=NUM_PARAMS,
            source_prior_draw=_source_draw,
            source_prior_logpdf=per_source,
        )
        assert space._resolve_source_prior_logpdf() is per_source


class TestModelRecovery:
    """Run long enough to check that the preferred model is correct."""

    @pytest.mark.slow
    def test_prefers_one_source(self, product_space, outdir):
        """Model recovery on the sinusoid problem with the default mixture.

        Fix history (three formerly xfailing defects, all now covered by
        dedicated regressions in tests/test_birth_death_detailed_balance.py):

        1. Constant-weight birth/death selection violated detailed balance;
           birth and death are now ONE kernel with schedule-driven internal
           selection (``BirthDeathProposal``).
        2. The value-preserving death left posterior-distributed parameters
           in the vacated inactive slot, which ``nmodel_jump`` re-activated
           with ``qxy = 0`` (valid only for prior-distributed slots),
           pushing toward MORE sources; death now refreshes the slot from
           the prior, and the kill-last death replaced the uniform-kill
           variant whose missing kill-choice ``qxy`` factor biased toward
           FEWER sources.
        3. With trans-dimensional moves enumeration-exact, this test STILL
           failed at the pinned seed 42 (P ~ [0.43-0.32, 0.47-0.52, 0.10])
           with a continuous-space mixing failure: the k=2 conditional
           posterior has an amplitude-splitting degenerate ridge (both
           sources at f ~ 1, a1 + a2 ~ 2.3, likelihood >= k=1's best), and
           exiting it to the death gateway (min amplitude ~ 0) requires
           1-D diffusion along the ridge that am/scam traverse at 4-8%
           acceptance while the stock ``de`` — the move designed for
           exactly such ridge jumps — NEVER ran: it is gated on a FULL
           per-model buffer (> 50,000 samples in the current model), which
           a 20k-iteration run split across 3 models cannot reach, so
           ``JumpProposals`` silently substituted ``gaussian``.
           ``from_product_space`` now registers the min-fill-gated ``EarlyDE``
           (active after 100 within-model samples, drawing from the tail
           of the partially filled buffer), which restores the ridge move;
           it accepts at ~40% here and this test passes at seeds 42, 7,
           43, 101, and 202 (P(1 source) = 0.54-0.63 vs the prior-MC gold
           standard ~[0.64, 0.29, 0.07]).
        """
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=5,
            seed=42,
            outdir=outdir,
            save_freq=5000,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=20_000)

        chain = sampler.load_chain()
        cold = chain["samples"][0]
        probs = product_space.model_posterior_probs(cold, burn=5000)

        # 1 source should be strongly preferred
        assert (
            np.argmax(probs) == 0
        ), f"Expected nmodel=0, got argmax={np.argmax(probs)}, probs={probs}"
        assert probs[0] > 0.5, f"P(1 source) = {probs[0]:.3f}, expected > 0.5"


class TestSampleCovExpansion:
    """Verify that from_product_space expands per-source sample_cov to full product space."""

    def test_per_source_cov_expanded(self, product_space, outdir):
        per_source_cov = np.array([[4.0, 0.5], [0.5, 1.0]])
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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

    def test_full_cov_passed_through(self, product_space, outdir):
        full_cov = np.eye(NDIM) * 2.0
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
            sample_cov=full_cov,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        np.testing.assert_array_equal(cs.sample_cov, full_cov)

    def test_per_source_mean_expanded(self, product_space, outdir):
        per_source_mean = np.array([2.5, 1.5])
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
            sample_mean=per_source_mean,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert cs.sample_mean.shape == (NDIM,)
        for i in range(MAX_SOURCES):
            sl = slice(i * NUM_PARAMS, (i + 1) * NUM_PARAMS)
            np.testing.assert_array_equal(cs.sample_mean[sl], per_source_mean)
        assert cs.sample_mean[-1] == 0.0

    def test_short_run_with_per_source_cov(self, product_space, outdir):
        """Smoke test: sampler runs with per-source covariance."""
        per_source_cov = np.diag([1.0, 0.5])
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
            sample_cov=per_source_cov,
            save_freq=200,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=200)
        chain = sampler.load_chain()
        assert chain["samples"].shape == (2, 200, NDIM)


class TestPriorEnforcement:
    """Verify that ALL source parameters (active + inactive) stay within prior."""

    def test_all_params_in_bounds(self, product_space, outdir):
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
            save_freq=1000,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=2000)

        chain = sampler.load_chain()
        cold = chain["samples"][0]

        for i in range(MAX_SOURCES):
            block = cold[:, i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
            assert np.all(block >= LO - 1e-10), f"Source {i} below lower bound"
            assert np.all(block <= HI + 1e-10), f"Source {i} above upper bound"


class TestPerModelStats:
    """Verify per-model adaptive proposal statistics."""

    def test_per_model_state_initialized(self, product_space, outdir):
        """Per-model state has correct groups for each model."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
        )
        cs = sampler.multi_chain_stats.chain_stats[0]
        assert hasattr(cs, "_per_model")
        assert len(cs._per_model) == MAX_SOURCES
        # nmodel=0 -> 1 active source group
        assert len(cs._per_model[0].groups) == 1
        assert cs._per_model[0].groups[0] == list(range(NUM_PARAMS))
        # nmodel=1 -> 2 active source groups
        assert len(cs._per_model[1].groups) == 2
        # nmodel=2 -> 3 active source groups
        assert len(cs._per_model[2].groups) == 3

    def test_update_sample_swaps_groups(self, product_space, outdir):
        """update_sample swaps in model-specific groups."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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

    def test_update_sample_swaps_buffer(self, product_space, outdir):
        """update_sample swaps in model-specific buffer and sample_total."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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

    def test_recursive_update_routes_samples(self, product_space, outdir):
        """recursive_update partitions samples by nmodel."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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

    def test_proposal_L_diverges(self, product_space, outdir):
        """proposal_L diverges between models after model-specific samples."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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
        assert not np.allclose(
            L0_after, L1_after
        ), "proposal_L should diverge after model-specific updates"

    def test_short_run_with_per_model(self, product_space, outdir):
        """Smoke test: RJMCMC sampling runs correctly with per-model stats."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=3,
            seed=42,
            outdir=outdir,
            save_freq=500,
        )
        rng = np.random.default_rng(42)
        x0 = product_space.draw_initial_position(rng, nmodel=0)
        sampler.sample(x0, num_iterations=500)
        chain = sampler.load_chain()
        assert chain["samples"].shape == (3, 500, NDIM)

    def test_pickle_roundtrip(self, product_space, outdir):
        """Pickle round-trip preserves per-model state."""
        sampler = PTSampler.from_product_space(
            product_space,
            ntemps=2,
            seed=42,
            outdir=outdir,
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
            pm_before = cs_before._per_model[k]
            pm_after = cs_after._per_model[k]
            assert len(pm_after.groups) == len(pm_before.groups)
            assert pm_after.sample_total == pm_before.sample_total
            for i in range(len(pm_after.groups)):
                if pm_after.proposal_L[i] is not None:
                    np.testing.assert_array_equal(
                        pm_after.proposal_L[i],
                        pm_before.proposal_L[i],
                    )

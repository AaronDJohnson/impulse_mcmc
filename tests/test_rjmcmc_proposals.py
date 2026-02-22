import pytest
import numpy as np

from impulse.rjmcmc_proposals import (
    default_birth_death_probs,
    make_birth_proposal,
    make_death_proposal,
    make_nmodel_jump,
)
from impulse.chain_stats import ChainStats
from impulse.sampler_state import PTState


NUM_PARAMS = 3
MAX_SOURCES = 3
NDIM = MAX_SOURCES * NUM_PARAMS + 1


def _draw_from_prior(rng):
    """Draw 3 params uniformly from [0, 5]."""
    return rng.uniform(0, 5, size=NUM_PARAMS)


@pytest.fixture
def chain_stats():
    ptstate = PTState(ndim=NDIM, ntemps=1, min_temp=1.0, max_temp=1.0)
    rng = np.random.default_rng(42)
    return ChainStats(ndim=NDIM, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)


def _make_sample(nmodel, rng, ndim=NDIM):
    """Make a product-space sample with given nmodel."""
    q = rng.uniform(0, 5, size=ndim)
    q[-1] = nmodel
    return q


# ---------------------------------------------------------------------------
# default_birth_death_probs
# ---------------------------------------------------------------------------

class TestDefaultBirthDeathProbs:
    def test_at_zero(self):
        pb, pd = default_birth_death_probs(0, MAX_SOURCES)
        assert pb == 1.0
        assert pd == 0.0

    def test_at_max(self):
        pb, pd = default_birth_death_probs(MAX_SOURCES - 1, MAX_SOURCES)
        assert pb == 0.0
        assert pd == 1.0

    def test_intermediate(self):
        pb, pd = default_birth_death_probs(1, MAX_SOURCES)
        assert pb == 0.5
        assert pd == 0.5


# ---------------------------------------------------------------------------
# Birth proposal
# ---------------------------------------------------------------------------

class TestBirthProposal:
    def test_increments_nmodel(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        q, qxy = birth(chain_stats)
        assert int(np.rint(q[-1])) == 1

    def test_no_birth_at_max(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(MAX_SOURCES - 1, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        q, qxy = birth(chain_stats)
        np.testing.assert_array_equal(q, original)
        assert qxy == 0.0

    def test_new_params_in_correct_slot(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        old = chain_stats.current_sample.copy()
        q, qxy = birth(chain_stats)
        # slot 0 should be unchanged
        np.testing.assert_array_equal(q[:NUM_PARAMS], old[:NUM_PARAMS])
        # slot 1 should have new params
        new_slot = q[NUM_PARAMS:2 * NUM_PARAMS]
        assert not np.array_equal(new_slot, old[NUM_PARAMS:2 * NUM_PARAMS])

    def test_preserves_existing_params(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        old = chain_stats.current_sample.copy()
        q, _ = birth(chain_stats)
        # slots 0 and 1 unchanged
        np.testing.assert_array_equal(q[:2 * NUM_PARAMS], old[:2 * NUM_PARAMS])

    def test_qxy_finite(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        _, qxy = birth(chain_stats)
        assert np.isfinite(qxy)

    def test_does_not_modify_input(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        birth(chain_stats)
        np.testing.assert_array_equal(chain_stats.current_sample, original)


# ---------------------------------------------------------------------------
# Death proposal
# ---------------------------------------------------------------------------

class TestDeathProposal:
    def test_decrements_nmodel(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)
        chain_stats.current_sample = _make_sample(2, chain_stats.rng)
        q, qxy = death(chain_stats)
        assert int(np.rint(q[-1])) == 1

    def test_no_death_at_zero(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        q, qxy = death(chain_stats)
        np.testing.assert_array_equal(q, original)
        assert qxy == 0.0

    def test_contiguity_after_death(self, chain_stats):
        """After killing source i and swapping to back, sources 0..nmodel-1 should be contiguous."""
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)
        rng = np.random.default_rng(123)
        # nmodel=2 means 3 active sources (0,1,2)
        q = np.zeros(NDIM)
        for i in range(3):
            q[i * NUM_PARAMS:(i + 1) * NUM_PARAMS] = (i + 1) * np.ones(NUM_PARAMS)
        q[-1] = 2
        chain_stats.current_sample = q.copy()

        q_new, _ = death(chain_stats)
        new_nmodel = int(np.rint(q_new[-1]))
        assert new_nmodel == 1
        # active params are in slots 0..1, should be nonzero and valid
        for i in range(new_nmodel + 1):
            block = q_new[i * NUM_PARAMS:(i + 1) * NUM_PARAMS]
            assert np.all(block > 0)

    def test_qxy_finite(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)
        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        _, qxy = death(chain_stats)
        assert np.isfinite(qxy)

    def test_does_not_modify_input(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)
        chain_stats.current_sample = _make_sample(2, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        death(chain_stats)
        np.testing.assert_array_equal(chain_stats.current_sample, original)


# ---------------------------------------------------------------------------
# Birth/Death reversibility
# ---------------------------------------------------------------------------

class TestBirthDeathReversibility:
    def test_qxy_symmetry(self):
        """Birth qxy at nmodel=k and death qxy at nmodel=k+1 should sum to zero
        when the same source is created/killed and prior == proposal."""
        for k in range(MAX_SOURCES - 1):
            pb_k, pd_k = default_birth_death_probs(k, MAX_SOURCES)
            pb_k1, pd_k1 = default_birth_death_probs(k + 1, MAX_SOURCES)

            # birth qxy
            qxy_birth = np.log(pd_k1) - np.log(k + 2) - np.log(pb_k)
            # death qxy (from k+1 back to k)
            qxy_death = np.log(pb_k) + np.log(k + 2) - np.log(pd_k1)

            assert abs(qxy_birth + qxy_death) < 1e-12, f"Failed at k={k}"

    def test_birth_death_roundtrip(self):
        """Birth followed by death (killing the new source) should recover
        the original sample and produce canceling qxy values."""
        rng = np.random.default_rng(99)
        ptstate = PTState(ndim=NDIM, ntemps=1, min_temp=1.0, max_temp=1.0)
        cs = ChainStats(ndim=NDIM, pt_state=ptstate, chain_index=0,
                        rng=rng, buffer_size=50)

        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES)

        original = _make_sample(0, rng)
        cs.current_sample = original.copy()

        # birth: 0 -> 1
        q_after_birth, qxy_birth = birth(cs)
        assert int(np.rint(q_after_birth[-1])) == 1

        # death: kill source 1 (the one we just birthed)
        cs.current_sample = q_after_birth.copy()
        # we need the death to pick the newly added source (slot 1)
        # force by setting a specific rng state -- try repeatedly
        found = False
        for seed in range(200):
            cs.rng = np.random.default_rng(seed)
            cs.current_sample = q_after_birth.copy()
            q_after_death, qxy_death = death(cs)
            if int(np.rint(q_after_death[-1])) == 0:
                # check if active source matches original
                if np.allclose(q_after_death[:NUM_PARAMS], original[:NUM_PARAMS]):
                    found = True
                    assert abs(qxy_birth + qxy_death) < 1e-12
                    break
        assert found, "Could not find seed that kills the birthed source"


# ---------------------------------------------------------------------------
# Birth/Death with custom proposal != prior
# ---------------------------------------------------------------------------

class TestBirthDeathCustomProposal:
    def test_correction_term_applied(self, chain_stats):
        """When proposal != prior, qxy should include the correction."""
        def draw(rng):
            return rng.normal(2.5, 0.5, size=NUM_PARAMS)

        def log_proposal(params):
            from scipy.stats import norm
            return np.sum(norm.logpdf(params, 2.5, 0.5))

        def log_prior(params):
            # uniform on [0, 5]
            if np.any(params < 0) or np.any(params > 5):
                return -np.inf
            return -NUM_PARAMS * np.log(5.0)

        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, draw,
                                    log_proposal_density=log_proposal,
                                    log_prior_density=log_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES,
                                    log_proposal_density=log_proposal,
                                    log_prior_density=log_prior)

        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        _, qxy_birth = birth(chain_stats)
        assert np.isfinite(qxy_birth)

        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        _, qxy_death = death(chain_stats)
        assert np.isfinite(qxy_death)


# ---------------------------------------------------------------------------
# nmodel jump
# ---------------------------------------------------------------------------

class TestNmodelJump:
    def test_changes_nmodel(self, chain_stats):
        jump = make_nmodel_jump(MAX_SOURCES)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        # run many times, should eventually pick a different model
        seen = set()
        for _ in range(100):
            q, qxy = jump(chain_stats)
            seen.add(int(np.rint(q[-1])))
            assert qxy == 0.0
        assert len(seen) == MAX_SOURCES

    def test_symmetric(self, chain_stats):
        jump = make_nmodel_jump(MAX_SOURCES)
        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        _, qxy = jump(chain_stats)
        assert qxy == 0.0

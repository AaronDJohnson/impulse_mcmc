import pickle
import warnings

import numpy as np
import pytest

from impulse.chain_stats import ChainStats
from impulse.rjmcmc_proposals import (
    BirthDeathProposal,
    default_birth_death_probs,
    make_birth_death_proposal,
    make_birth_proposal,
    make_death_proposal,
    make_nmodel_jump,
)
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


class _StatsStub:
    """Minimal ChainStats stand-in: proposals only use rng + current_sample."""

    def __init__(self, rng, sample):
        self.rng = rng
        self.current_sample = sample


def _stats_stub(rng, sample):
    return _StatsStub(rng, sample)


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
        # slot 1 should have new params (birth draws a fresh source from the prior)
        new_slot = q[NUM_PARAMS : 2 * NUM_PARAMS]
        assert not np.array_equal(new_slot, old[NUM_PARAMS : 2 * NUM_PARAMS])

    def test_preserves_existing_params(self, chain_stats):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        old = chain_stats.current_sample.copy()
        q, _ = birth(chain_stats)
        # slots 0 and 1 unchanged
        np.testing.assert_array_equal(q[: 2 * NUM_PARAMS], old[: 2 * NUM_PARAMS])

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


class _ConstantDraw:
    """Deterministic draw stub: always returns the same vector."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __call__(self, rng):
        return self.values.copy()


class TestDeathProposal:
    def test_decrements_nmodel(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(2, chain_stats.rng)
        q, qxy = death(chain_stats)
        assert int(np.rint(q[-1])) == 1

    def test_no_death_at_zero(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        q, qxy = death(chain_stats)
        np.testing.assert_array_equal(q, original)
        assert qxy == 0.0

    def test_kills_last_active_source(self, chain_stats):
        """Death removes the LAST active slot; surviving slots are untouched.

        Kill-last is what makes death the exact pointwise reverse of the
        append-last birth (a uniformly chosen victim with the same qxy is
        NOT the reverse move and leaves a residual bias toward fewer
        sources; see the DeathProposal docstring).
        """
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(123)
        # nmodel=2 means 3 active sources (0,1,2)
        q = np.zeros(NDIM)
        for i in range(3):
            q[i * NUM_PARAMS : (i + 1) * NUM_PARAMS] = (i + 1) * np.ones(NUM_PARAMS)
        q[-1] = 2
        chain_stats.current_sample = q.copy()

        q_new, _ = death(chain_stats)
        new_nmodel = int(np.rint(q_new[-1]))
        assert new_nmodel == 1
        # surviving active slots 0..1 keep their exact values
        np.testing.assert_array_equal(q_new[: 2 * NUM_PARAMS], q[: 2 * NUM_PARAMS])
        # the vacated last slot no longer holds the killed source
        assert not np.array_equal(
            q_new[2 * NUM_PARAMS : 3 * NUM_PARAMS], q[2 * NUM_PARAMS : 3 * NUM_PARAMS]
        )

    def test_qxy_finite(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        _, qxy = death(chain_stats)
        assert np.isfinite(qxy)

    def test_does_not_modify_input(self, chain_stats):
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        chain_stats.current_sample = _make_sample(2, chain_stats.rng)
        original = chain_stats.current_sample.copy()
        death(chain_stats)
        np.testing.assert_array_equal(chain_stats.current_sample, original)

    def test_vacated_slot_refreshed(self):
        """The vacated slot must hold a FRESH draw, not the killed source.

        Regression for the inactive-slot memory bug: a value-preserving death
        left the killed source's posterior-distributed parameters in the
        inactive slot, which ``nmodel_jump`` later re-activated with
        ``qxy = 0`` (only valid for draw-distributed slots), biasing the
        model posterior toward more sources.
        """
        sentinel = np.full(NUM_PARAMS, 4.75)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _ConstantDraw(sentinel))
        for seed in range(20):
            rng = np.random.default_rng(seed)
            q = np.zeros(NDIM)
            for i in range(3):
                q[i * NUM_PARAMS : (i + 1) * NUM_PARAMS] = i + 1.0
            q[-1] = 2  # 3 active sources
            q_new, _ = death(_stats_stub(rng, q))
            # the vacated last slot holds the fresh draw
            np.testing.assert_array_equal(q_new[2 * NUM_PARAMS : 3 * NUM_PARAMS], sentinel)
            # kill-last: the LAST source (value 3.0) was removed and the
            # first two survive in place
            survivors = {q_new[i * NUM_PARAMS] for i in range(2)}
            assert survivors == {1.0, 2.0}

    def test_qxy_prior_refresh_term(self):
        """q = p: death qxy = schedule ratio + log p(killed) - log p(fresh)."""

        def log_prior(params):
            return float(np.sum(-0.1 * np.asarray(params)))

        fresh = np.full(NUM_PARAMS, 2.0)
        death = make_death_proposal(
            NUM_PARAMS, MAX_SOURCES, _ConstantDraw(fresh), log_prior_density=log_prior
        )
        rng = np.random.default_rng(3)
        q = np.zeros(NDIM)
        for i in range(2):
            q[i * NUM_PARAMS : (i + 1) * NUM_PARAMS] = i + 1.0
        q[-1] = 1  # 2 active sources
        # kill-last: the victim is deterministically the last active slot
        killed = q[1 * NUM_PARAMS : 2 * NUM_PARAMS].copy()

        _, qxy = death(_stats_stub(rng, q))
        # schedule: log p_birth(0) - log p_death(1) = log(1) - log(0.5)
        expected = -np.log(0.5) + log_prior(killed) - log_prior(fresh)
        assert qxy == pytest.approx(expected, abs=1e-12)

    def test_qxy_proposal_density_term(self):
        """q != p: death qxy uses the DRAW density q for the re-fill ratio.

        Regression for the dropped death-side proposal Hastings term: the
        reverse birth redraws the killed parameters with density q, and the
        forward move re-fills the vacated slot with density q, so
        ``qxy = schedule + log q(killed) - log q(fresh)``.
        """

        def log_proposal(params):
            return float(np.sum(-0.5 * np.asarray(params) ** 2))

        def log_prior(params):
            # deliberately different from the proposal density
            return float(np.sum(-0.01 * np.asarray(params)))

        fresh = np.full(NUM_PARAMS, 0.5)
        death = make_death_proposal(
            NUM_PARAMS,
            MAX_SOURCES,
            _ConstantDraw(fresh),
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
        )
        rng = np.random.default_rng(11)
        q = np.zeros(NDIM)
        for i in range(2):
            q[i * NUM_PARAMS : (i + 1) * NUM_PARAMS] = i + 1.0
        q[-1] = 1
        # kill-last: the victim is deterministically the last active slot
        killed = q[1 * NUM_PARAMS : 2 * NUM_PARAMS].copy()

        _, qxy = death(_stats_stub(rng, q))
        expected = -np.log(0.5) + log_proposal(killed) - log_proposal(fresh)
        assert qxy == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Birth/Death reversibility
# ---------------------------------------------------------------------------


class TestBirthDeathReversibility:
    def test_qxy_symmetry(self):
        """Birth qxy at nmodel=k and death qxy at nmodel=k+1 should sum to zero
        when the same source is created/killed and prior == proposal."""
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(7)

        for k in range(MAX_SOURCES - 1):
            q = _make_sample(k, rng)
            q_up, qxy_birth = birth(_stats_stub(rng, q))
            assert int(np.rint(q_up[-1])) == k + 1

            # death qxy depends only on the model index, not on which source
            # is killed, so any death from the post-birth state must exactly
            # reverse the birth's Hastings factor
            _, qxy_death = death(_stats_stub(rng, q_up))

            # cross-check against the schedule the implementation must use
            pb_k, _ = default_birth_death_probs(k, MAX_SOURCES)
            _, pd_k1 = default_birth_death_probs(k + 1, MAX_SOURCES)
            assert qxy_birth == pytest.approx(np.log(pd_k1) - np.log(pb_k))
            assert abs(qxy_birth + qxy_death) < 1e-12, f"Failed at k={k}"

    def test_birth_death_roundtrip(self):
        """Birth followed by death should recover the original active sources
        and produce canceling qxy values.

        Kill-last makes this deterministic: the death always removes the
        slot the birth just appended, so every birth+death pair is an exact
        reverse — no seed search over kill choices is needed.
        """
        rng = np.random.default_rng(99)
        ptstate = PTState(ndim=NDIM, ntemps=1, min_temp=1.0, max_temp=1.0)
        cs = ChainStats(ndim=NDIM, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)

        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)

        original = _make_sample(0, rng)
        cs.current_sample = original.copy()

        # birth: 0 -> 1
        q_after_birth, qxy_birth = birth(cs)
        assert int(np.rint(q_after_birth[-1])) == 1

        # death: deterministically kills the just-birthed last active source
        cs.current_sample = q_after_birth.copy()
        q_after_death, qxy_death = death(cs)
        assert int(np.rint(q_after_death[-1])) == 0
        np.testing.assert_allclose(q_after_death[:NUM_PARAMS], original[:NUM_PARAMS])
        assert abs(qxy_birth + qxy_death) < 1e-12


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

        birth = make_birth_proposal(
            NUM_PARAMS,
            MAX_SOURCES,
            draw,
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
        )
        death = make_death_proposal(
            NUM_PARAMS,
            MAX_SOURCES,
            draw,
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
        )

        chain_stats.current_sample = _make_sample(0, chain_stats.rng)
        _, qxy_birth = birth(chain_stats)
        assert np.isfinite(qxy_birth)

        chain_stats.current_sample = _make_sample(1, chain_stats.rng)
        _, qxy_death = death(chain_stats)
        assert np.isfinite(qxy_death)


# ---------------------------------------------------------------------------
# Combined birth-death kernel
# ---------------------------------------------------------------------------


def _zero_schedule(nmodel, max_sources):
    return 0.0, 0.0


def _uniform_schedule(nmodel, max_sources):
    """Interior-style probabilities everywhere (disagrees with the default)."""
    return 0.5, 0.5


def _nonconstant_sum_schedule(nmodel, max_sources):
    """p_birth + p_death varies with the model index (invalid)."""
    if nmodel == 1:
        return 0.5, 0.25
    return default_birth_death_probs(nmodel, max_sources)


class TestBirthDeathProposal:
    def test_factory_returns_kernel(self):
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        assert isinstance(kernel, BirthDeathProposal)
        assert kernel.__name__ == "birth_death"

    def test_always_birth_at_zero(self):
        """Default schedule: p_birth(0) = 1, so k=0 always proposes k=1."""
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(1)
        for _ in range(50):
            q, _ = kernel(_stats_stub(rng, _make_sample(0, rng)))
            assert int(np.rint(q[-1])) == 1

    def test_always_death_at_max(self):
        """Default schedule: p_death(K-1) = 1, so k=K-1 always proposes k=K-2."""
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(2)
        for _ in range(50):
            q, _ = kernel(_stats_stub(rng, _make_sample(MAX_SOURCES - 1, rng)))
            assert int(np.rint(q[-1])) == MAX_SOURCES - 2

    def test_interior_mixes_birth_and_death(self):
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(3)
        proposed = set()
        for _ in range(200):
            q, _ = kernel(_stats_stub(rng, _make_sample(1, rng)))
            proposed.add(int(np.rint(q[-1])))
        assert proposed == {0, 2}

    def test_qxy_is_schedule_ratio(self):
        """Birth from k=0: qxy = log(p_death(1)) - log(p_birth(0)) = log(0.5)."""
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        rng = np.random.default_rng(5)
        q, qxy = kernel(_stats_stub(rng, _make_sample(0, rng)))
        assert int(np.rint(q[-1])) == 1
        assert qxy == pytest.approx(np.log(0.5))

    def test_degenerate_schedule_rejects(self):
        """p_birth = p_death = 0 must reject, never a silently-accepted no-op."""
        kernel = make_birth_death_proposal(
            NUM_PARAMS,
            MAX_SOURCES,
            _draw_from_prior,
            prob_schedule=_zero_schedule,
        )
        rng = np.random.default_rng(4)
        sample = _make_sample(1, rng)
        q, qxy = kernel(_stats_stub(rng, sample))
        np.testing.assert_array_equal(q, sample)
        assert qxy == -np.inf

    def test_mismatched_max_sources_raises(self):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES + 1, _draw_from_prior)
        with pytest.raises(ValueError):
            BirthDeathProposal(birth, death)

    def test_mismatched_num_params_raises(self):
        """Birth writes num_params-sized slots that the death must undo,
        so a num_params disagreement corrupts the slot layout."""
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS + 1, MAX_SOURCES, _draw_from_prior)
        with pytest.raises(ValueError, match="num_params"):
            BirthDeathProposal(birth, death)

    def test_distinct_draw_callables_warn(self):
        """The death re-fills the vacated slot from the SAME draw
        distribution the birth samples; identity is the only check the
        constructor can make, so distinct callables warn (distinct-but-
        equal callables should go through make_birth_death_proposal,
        which wires one shared callable)."""
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _ConstantDraw(np.zeros(NUM_PARAMS)))
        with pytest.warns(UserWarning, match="draw_from_prior"):
            BirthDeathProposal(birth, death)

    def test_shared_draw_callable_does_not_warn(self):
        birth = make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            BirthDeathProposal(birth, death)

    def test_factory_shares_draw_callable_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kernel = make_birth_death_proposal(
                NUM_PARAMS,
                MAX_SOURCES,
                _draw_from_prior,
            )
        assert kernel.birth.draw_from_prior is kernel.death.draw_from_prior

    def test_max_sources_one_raises(self):
        """A single model admits no trans-dimensional move: the default
        schedule at max_sources=1 returns (1, 0) (the nmodel == 0 branch wins),
        so birth would always be selected and no-op as a silently accepted
        move.  Construction must fail instead."""
        with pytest.raises(ValueError, match="max_sources >= 2"):
            make_birth_death_proposal(NUM_PARAMS, 1, _draw_from_prior)

    def test_disagreeing_schedules_raise(self):
        """Selection, birth, and death schedules must agree pointwise."""
        birth = make_birth_proposal(
            NUM_PARAMS, MAX_SOURCES, _draw_from_prior, prob_schedule=_uniform_schedule
        )
        death = make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        with pytest.raises(ValueError, match="disagrees"):
            BirthDeathProposal(birth, death)

    def test_nonconstant_schedule_sum_raises(self):
        """p_birth + p_death must be constant across model indices: the
        sub-proposals use the raw schedule values in their Hastings ratios
        while selection normalizes, so a varying sum biases acceptance."""
        with pytest.raises(ValueError, match="constant across model indices"):
            make_birth_death_proposal(
                NUM_PARAMS,
                MAX_SOURCES,
                _draw_from_prior,
                prob_schedule=_nonconstant_sum_schedule,
            )

    def test_pickle_roundtrip(self):
        """Checkpointing pickles the sampler, so the kernel must round-trip."""
        kernel = make_birth_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior)
        restored = pickle.loads(pickle.dumps(kernel))
        assert restored.__name__ == "birth_death"
        assert restored.max_sources == MAX_SOURCES
        rng = np.random.default_rng(6)
        q, qxy = restored(_stats_stub(rng, _make_sample(0, rng)))
        assert int(np.rint(q[-1])) == 1
        assert np.isfinite(qxy)


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


# ---------------------------------------------------------------------------
# ParameterLayout plumbing
# ---------------------------------------------------------------------------


class TestProposalLayoutPlumbing:
    """The proposals consume one ParameterLayout instead of re-deriving it."""

    def _birth_death(self):
        return (
            make_birth_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior),
            make_death_proposal(NUM_PARAMS, MAX_SOURCES, _draw_from_prior),
        )

    def test_birth_death_layout_property(self):
        from impulse.product_space import ParameterLayout

        birth, death = self._birth_death()
        expected = ParameterLayout(num_params=NUM_PARAMS, num_models=MAX_SOURCES)
        assert birth.layout == expected
        assert death.layout == expected
        # Serialization format guard: layout is DERIVED from the pickled
        # num_params/max_sources scalars, never stored, so the checkpoint
        # attribute set is unchanged and legacy checkpoints work unchanged.
        assert "layout" not in vars(birth)
        assert "layout" not in vars(death)

    def test_nmodel_jump_layout_optional_and_identical(self, chain_stats):
        from impulse.product_space import ParameterLayout

        layout = ParameterLayout(num_params=NUM_PARAMS, num_models=MAX_SOURCES)
        sample = _make_sample(0, np.random.default_rng(3))
        with_layout = make_nmodel_jump(MAX_SOURCES, layout=layout)
        without_layout = make_nmodel_jump(MAX_SOURCES)
        q1, qxy1 = with_layout(_stats_stub(np.random.default_rng(7), sample.copy()))
        q2, qxy2 = without_layout(_stats_stub(np.random.default_rng(7), sample.copy()))
        np.testing.assert_array_equal(q1, q2)
        assert qxy1 == qxy2 == 0.0
        # only the model index moved
        np.testing.assert_array_equal(q1[: layout.nmodel_index], sample[: layout.nmodel_index])

    def test_nmodel_jump_legacy_unpickle_backfill(self):
        # Pre-layout checkpoints hold NmodelJump instances without the
        # ``layout`` attribute; __getattr__ back-fills None and the jump
        # still works (the write targets the trailing coordinate).
        jump = make_nmodel_jump(MAX_SOURCES)
        vars(jump).pop("layout", None)
        rng = np.random.default_rng(11)
        q, qxy = jump(_stats_stub(rng, _make_sample(0, np.random.default_rng(4))))
        assert jump.layout is None
        assert 0 <= int(np.rint(q[-1])) < MAX_SOURCES
        assert qxy == 0.0

    def test_source_swap_layout_injection(self):
        from impulse.product_space import ParameterLayout
        from impulse.proposals import make_source_swap_proposal

        layout = ParameterLayout(num_params=NUM_PARAMS, num_models=MAX_SOURCES)
        sample = _make_sample(MAX_SOURCES - 1, np.random.default_rng(5))
        swap_with = make_source_swap_proposal(NUM_PARAMS, layout=layout)
        swap_without = make_source_swap_proposal(NUM_PARAMS)
        q1, qxy1 = swap_with(_stats_stub(np.random.default_rng(9), sample.copy()))
        q2, qxy2 = swap_without(_stats_stub(np.random.default_rng(9), sample.copy()))
        np.testing.assert_array_equal(q1, q2)
        assert qxy1 == qxy2 == 0
        # a swap permutes source blocks; the sorted continuous params and
        # the model index are invariant
        np.testing.assert_array_equal(
            np.sort(q1[: layout.nmodel_index]), np.sort(sample[: layout.nmodel_index])
        )
        assert q1[-1] == sample[-1]

    def test_source_swap_legacy_unpickle_backfill(self):
        from impulse.proposals import make_source_swap_proposal

        swap = make_source_swap_proposal(NUM_PARAMS)
        vars(swap).pop("layout", None)
        sample = _make_sample(MAX_SOURCES - 1, np.random.default_rng(6))
        q, qxy = swap(_stats_stub(np.random.default_rng(13), sample.copy()))
        assert swap.layout is None
        assert qxy == 0
        np.testing.assert_array_equal(np.sort(q[:-1]), np.sort(sample[:-1]))

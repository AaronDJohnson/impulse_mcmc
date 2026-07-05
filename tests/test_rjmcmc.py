import numpy as np
import pytest

from impulse.rjmcmc import RJMCMCProductSpace

NUM_PARAMS = 3
MAX_SOURCES = 3
NDIM = MAX_SOURCES * NUM_PARAMS + 1

A_MIN, A_MAX = 0.0, 5.0
F_MIN, F_MAX = 0.0, 3.0
PHI_MIN, PHI_MAX = 0.0, np.pi


def _source_draw(rng):
    return np.array(
        [
            rng.uniform(A_MIN, A_MAX),
            rng.uniform(F_MIN, F_MAX),
            rng.uniform(PHI_MIN, PHI_MAX),
        ]
    )


def _loglike(params):
    return -0.5 * np.sum(params**2)


def _logprior(params):
    n = len(params)
    for i in range(n // NUM_PARAMS):
        a, f, phi = params[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
        if not (A_MIN <= a <= A_MAX and F_MIN <= f <= F_MAX and PHI_MIN <= phi <= PHI_MAX):
            return -np.inf
    return 0.0


@pytest.fixture
def space():
    return RJMCMCProductSpace(
        loglikelihood=_loglike,
        logprior=_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_source_draw,
    )


class TestRJMCMCProductSpaceInit:
    def test_ndim(self, space):
        assert space.ndim == NDIM

    def test_num_models(self, space):
        assert space.num_models == MAX_SOURCES

    def test_inherits_nested(self, space):
        # get_loglikelihood / get_logprior should work
        rng = np.random.default_rng(0)
        x = space.draw_initial_position(rng, nmodel=1)
        ll = space.get_loglikelihood(x)
        lp = space.get_logprior(x)
        assert np.isfinite(ll)
        assert np.isfinite(lp)


class TestGetDefaultGroups:
    def test_correct_number(self, space):
        groups = space.get_default_groups()
        assert len(groups) == MAX_SOURCES

    def test_correct_indices(self, space):
        groups = space.get_default_groups()
        assert groups[0] == [0, 1, 2]
        assert groups[1] == [3, 4, 5]
        assert groups[2] == [6, 7, 8]

    def test_model_index_excluded(self, space):
        groups = space.get_default_groups()
        all_indices = [idx for g in groups for idx in g]
        assert NDIM - 1 not in all_indices


class TestProposalFactories:
    def test_birth_returns_callable(self, space):
        birth = space.get_birth_proposal()
        assert callable(birth)

    def test_death_returns_callable(self, space):
        death = space.get_death_proposal()
        assert callable(death)

    def test_nmodel_jump_returns_callable(self, space):
        jump = space.get_nmodel_jump()
        assert callable(jump)

    def test_source_swap_returns_callable(self, space):
        swap = space.get_source_swap_proposal()
        assert callable(swap)


class TestModelPosteriorProbs:
    def test_sums_to_one(self, space):
        rng = np.random.default_rng(42)
        chain = np.zeros((1000, NDIM))
        chain[:, -1] = rng.integers(0, MAX_SOURCES, size=1000)
        probs = space.model_posterior_probs(chain, burn=0)
        assert abs(probs.sum() - 1.0) < 1e-12

    def test_known_distribution(self, space):
        chain = np.zeros((1000, NDIM))
        chain[:500, -1] = 0
        chain[500:, -1] = 1
        probs = space.model_posterior_probs(chain, burn=0)
        assert abs(probs[0] - 0.5) < 1e-12
        assert abs(probs[1] - 0.5) < 1e-12
        assert abs(probs[2] - 0.0) < 1e-12

    def test_burn(self, space):
        chain = np.zeros((1000, NDIM))
        chain[:200, -1] = 2  # burn-in dominated by model 2
        chain[200:, -1] = 0
        probs = space.model_posterior_probs(chain, burn=200)
        assert abs(probs[0] - 1.0) < 1e-12


class TestDrawInitialPosition:
    def test_shape(self, space):
        rng = np.random.default_rng(42)
        x0 = space.draw_initial_position(rng, nmodel=0)
        assert x0.shape == (NDIM,)

    def test_nmodel_set(self, space):
        rng = np.random.default_rng(42)
        for nm in range(MAX_SOURCES):
            x0 = space.draw_initial_position(rng, nmodel=nm)
            assert int(np.rint(x0[-1])) == nm

    def test_all_slots_filled(self, space):
        rng = np.random.default_rng(42)
        x0 = space.draw_initial_position(rng, nmodel=0)
        # all source params should be from the prior (nonzero with high prob)
        for i in range(MAX_SOURCES):
            block = x0[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
            assert np.all(np.isfinite(block))

    def test_within_prior_bounds(self, space):
        rng = np.random.default_rng(42)
        for _ in range(50):
            x0 = space.draw_initial_position(rng)
            for i in range(MAX_SOURCES):
                a = x0[i * NUM_PARAMS]
                f = x0[i * NUM_PARAMS + 1]
                phi = x0[i * NUM_PARAMS + 2]
                assert A_MIN <= a <= A_MAX
                assert F_MIN <= f <= F_MAX
                assert PHI_MIN <= phi <= PHI_MAX

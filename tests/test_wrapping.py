"""Tests for periodic-parameter wrapping.

impulse/wrapping.py had no test file at all despite being live in the sampler
path (impulse/_pt_base.py, impulse/sampler_step.py, impulse/hybrid_sampler.py)
and documented in docs/user_guide/parallel-tempering.md.

Its module docstring makes a correctness claim that nothing checked: that
period-shifted Gaussian proposals stay symmetric on the torus, so ``qxy``
remains 0 for AM/SCAM/DE/gaussian. In a package that already shipped one silent
detailed-balance bug, that claim needs a guard, so the statistical tests at the
bottom of this file assert the property the claim implies -- a flat circular
target must come back uniform, at every offset and with no drift to infinity.
"""

import numpy as np
import pytest

from impulse import PTSampler
from impulse.wrapping import WrapSpec

# ---------------------------------------------------------------------------
# WrapSpec.from_dict
# ---------------------------------------------------------------------------


class TestWrapSpecFromDict:
    def test_none_and_empty_give_no_spec(self):
        assert WrapSpec.from_dict(None) is None
        assert WrapSpec.from_dict({}) is None

    def test_scalar_period_implies_low_zero(self):
        spec = WrapSpec.from_dict({2: 2 * np.pi})
        assert spec is not None
        np.testing.assert_array_equal(spec.indices, [2])
        np.testing.assert_allclose(spec.lows, [0.0])
        np.testing.assert_allclose(spec.periods, [2 * np.pi])

    def test_low_high_tuple(self):
        spec = WrapSpec.from_dict({1: (-np.pi, np.pi)})
        np.testing.assert_allclose(spec.lows, [-np.pi])
        np.testing.assert_allclose(spec.periods, [2 * np.pi])

    def test_multiple_dims_are_sorted(self):
        spec = WrapSpec.from_dict({3: (-0.5, 0.5), 0: 1.0})
        np.testing.assert_array_equal(spec.indices, [0, 3])
        np.testing.assert_allclose(spec.lows, [0.0, -0.5])
        np.testing.assert_allclose(spec.periods, [1.0, 1.0])

    @pytest.mark.parametrize("bad", [(1.0,), (0.0, 1.0, 2.0)])
    def test_bad_tuple_length_raises(self, bad):
        with pytest.raises(ValueError, match="scalar period or a"):
            WrapSpec.from_dict({0: bad})

    @pytest.mark.parametrize("bad", [(1.0, 1.0), (2.0, -1.0), 0.0, -3.0])
    def test_non_positive_period_raises(self, bad):
        with pytest.raises(ValueError, match="must be >"):
            WrapSpec.from_dict({0: bad})


# ---------------------------------------------------------------------------
# WrapSpec.apply / apply_inplace
# ---------------------------------------------------------------------------


class TestWrapSpecApply:
    def test_wraps_into_half_open_interval_1d(self):
        spec = WrapSpec.from_dict({0: (-np.pi, np.pi)})
        x = np.array([3 * np.pi])  # equivalent to +pi -> must land on -pi
        out = spec.apply(x)
        assert -np.pi <= out[0] < np.pi
        np.testing.assert_allclose(out[0], -np.pi)

    def test_leaves_non_periodic_dims_untouched(self):
        spec = WrapSpec.from_dict({1: 1.0})
        x = np.array([100.0, 2.5, -7.0])
        out = spec.apply(x)
        np.testing.assert_allclose(out[[0, 2]], [100.0, -7.0])
        np.testing.assert_allclose(out[1], 0.5)

    def test_apply_is_a_copy(self):
        spec = WrapSpec.from_dict({0: 1.0})
        x = np.array([5.5, 1.0])
        out = spec.apply(x)
        np.testing.assert_allclose(x[0], 5.5)  # original untouched
        np.testing.assert_allclose(out[0], 0.5)

    def test_apply_inplace_mutates(self):
        spec = WrapSpec.from_dict({0: 1.0})
        x = np.array([5.5, 1.0])
        spec.apply_inplace(x)
        np.testing.assert_allclose(x[0], 0.5)

    def test_2d_batch(self):
        spec = WrapSpec.from_dict({1: (-np.pi, np.pi)})
        x = np.array([[0.0, 4 * np.pi], [1.0, -5 * np.pi]])
        out = spec.apply(x)
        assert np.all(out[:, 1] >= -np.pi) and np.all(out[:, 1] < np.pi)
        np.testing.assert_allclose(out[:, 0], [0.0, 1.0])  # non-periodic dim intact

    def test_idempotent(self):
        spec = WrapSpec.from_dict({0: (2.0, 7.0), 2: 3.0})
        rng = np.random.default_rng(0)
        x = rng.uniform(-50, 50, size=(20, 3))
        once = spec.apply(x)
        twice = spec.apply(once)
        np.testing.assert_allclose(once, twice)

    def test_preserves_value_modulo_period(self):
        """Wrapping may only shift by whole periods."""
        spec = WrapSpec.from_dict({0: (-np.pi, np.pi)})
        rng = np.random.default_rng(1)
        x = rng.uniform(-100, 100, size=(200, 1))
        out = spec.apply(x)
        shifts = (x - out) / (2 * np.pi)
        np.testing.assert_allclose(shifts, np.round(shifts), atol=1e-9)


# ---------------------------------------------------------------------------
# The correctness claim: wrapping must not distort the target.
# ---------------------------------------------------------------------------


def _flat_circular_lnlike(x):
    x = np.asarray(x)
    return 0.0 if x.ndim == 1 else np.zeros(x.shape[0])


def _make_circular_prior(low, high):
    """Flat prior on a circular dim 0 and a bounded ordinary dim 1."""

    def lnprior(x):
        x = np.asarray(x)
        if x.ndim == 1:
            ok = (low <= x[0] < high) and (abs(x[1]) <= 5)
            return 0.0 if ok else -np.inf
        out = np.zeros(x.shape[0])
        bad = ~((x[:, 0] >= low) & (x[:, 0] < high) & (np.abs(x[:, 1]) <= 5))
        out[bad] = -np.inf
        return out

    return lnprior


class TestPeriodicSamplingIsUnbiased:
    """A flat target on the circle must come back uniform, not merely finite."""

    @pytest.mark.parametrize("low,high", [(0.0, 2 * np.pi), (-np.pi, np.pi), (2.0, 5.0)])
    def test_flat_circular_marginal_is_uniform(self, low, high, temp_dir):
        """Guards the module's symmetry claim.

        If period-shifted Gaussian proposals were NOT symmetric on the torus,
        qxy = 0 would be wrong and this marginal would come back skewed.
        Tested at several offsets so a bug in the low/period bookkeeping cannot
        hide behind a symmetric special case.
        """
        period = high - low
        sampler = PTSampler(
            ndim=2,
            lnlike=_flat_circular_lnlike,
            lnprior=_make_circular_prior(low, high),
            ntemps=2,
            seed=12345,
            outdir=temp_dir,
            periodic={0: (low, high)},
            save_freq=20000,
        )
        sampler.sample([0.5 * (low + high), 0.0], num_iterations=20000)
        phase = sampler.load_chain()["samples"][0][2000:, 0]

        # every stored draw is inside the stated period
        assert np.all(phase >= low) and np.all(phase < high)

        # uniform on [low, high): compare decile occupancy
        counts, _ = np.histogram(phase, bins=10, range=(low, high))
        frac = counts / counts.sum()
        assert (
            np.max(np.abs(frac - 0.1)) < 0.025
        ), f"non-uniform circular marginal on [{low}, {high}): deciles {frac}"

        # circular mean resultant length ~ 0 for a uniform circular variable
        theta = 2 * np.pi * (phase - low) / period
        R = np.hypot(np.mean(np.cos(theta)), np.mean(np.sin(theta)))
        assert R < 0.05, f"circular marginal is concentrated (R={R:.3f})"

    def test_periodic_dim_does_not_random_walk_away(self, temp_dir):
        """The motivating failure: an unwrapped circular parameter escapes to +-inf."""
        low, high = 0.0, 2 * np.pi
        sampler = PTSampler(
            ndim=2,
            lnlike=_flat_circular_lnlike,
            lnprior=_make_circular_prior(low, high),
            ntemps=2,
            seed=7,
            outdir=temp_dir,
            periodic={0: (low, high)},
            save_freq=8000,
        )
        sampler.sample([1.0, 0.0], num_iterations=8000)
        phase = sampler.load_chain()["samples"][0][:, 0]

        assert np.all(np.isfinite(phase))
        assert phase.min() >= low and phase.max() < high

    def test_initial_position_outside_period_is_wrapped(self, temp_dir):
        """A user may legitimately pass an unwrapped starting phase."""
        low, high = -np.pi, np.pi
        sampler = PTSampler(
            ndim=2,
            lnlike=_flat_circular_lnlike,
            lnprior=_make_circular_prior(low, high),
            ntemps=2,
            seed=3,
            outdir=temp_dir,
            periodic={0: (low, high)},
            save_freq=500,
        )
        # 7.0 rad is outside [-pi, pi) and would be -inf under the prior if the
        # sampler did not wrap the initial position before the first evaluation.
        sampler.sample([7.0, 0.0], num_iterations=500)
        phase = sampler.load_chain()["samples"][0][:, 0]
        assert np.all(phase >= low) and np.all(phase < high)

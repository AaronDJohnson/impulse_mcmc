"""Masked moments and MaskedChainStats."""

import math

import numpy as np
import pytest
from rj_helpers import make_stats, padded_rows

from impulse import RJLayout
from impulse.reversible_jump import (
    MaskedChainStats,
    clip_to_psd,
    masked_moments,
    pairwise_covariance,
)
from impulse.samplers import PTState

LAYOUT = RJLayout(n_shared=1, n_component=2, max_components=3)  # D = 7


def _brute_force(params, fallback, min_rows):
    d = params.shape[1]
    out = np.empty((d, d))
    for a in range(d):
        for b in range(d):
            both = ~np.isnan(params[:, a]) & ~np.isnan(params[:, b])
            if both.sum() < min_rows:
                out[a, b] = fallback[a, b]
            else:
                out[a, b] = np.cov(params[both][:, [a, b]].T)[0, 1]
    return out


def test_pairwise_covariance_matches_brute_force():
    params = LAYOUT.params(padded_rows(LAYOUT, 300))
    fallback = np.diag(np.full(7, 5.0))
    got = pairwise_covariance(params, fallback, min_rows=10)
    np.testing.assert_allclose(got, _brute_force(params, fallback, 10), rtol=1e-10, atol=1e-12)


def test_pairwise_covariance_falls_back_below_min_rows():
    params = LAYOUT.params(padded_rows(LAYOUT, 300))
    params[:, 5:] = np.nan
    params[:8, 5:] = 1.0 + np.arange(8)[:, None]  # only 8 rows have the last slot
    fallback = np.full((7, 7), 0.25) + np.eye(7)
    got = pairwise_covariance(params, fallback, min_rows=10)
    assert np.array_equal(got[5:, :], fallback[5:, :])
    assert np.array_equal(got[:, 5:], fallback[:, 5:])


def test_clip_to_psd():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    fixed = clip_to_psd(bad, floor=1e-6)
    assert np.linalg.eigvalsh(fixed).min() >= 1e-6 * (1 - 1e-6)
    np.testing.assert_allclose(fixed, fixed.T)
    good = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert clip_to_psd(good, floor=1e-6) is good


def test_masked_moments_repairs_an_inconsistent_covariance():
    t = np.random.default_rng(2).normal(size=(3, 20))
    nan = np.full(20, np.nan)
    params = np.concatenate(
        [
            np.column_stack([t[0], t[0], nan]),  # a == b
            np.column_stack([nan, t[1], t[1]]),  # b == c
            np.column_stack([t[2], nan, -t[2]]),  # a == -c
        ]
    )
    assert np.linalg.eigvalsh(pairwise_covariance(params, np.eye(3), 10)).min() < -0.1
    mean, cov = masked_moments(params, np.zeros(3), np.eye(3), 10, 1e-8)
    assert np.linalg.eigvalsh(cov).min() >= 1e-8 * (1 - 1e-6)
    np.testing.assert_allclose(cov, cov.T)
    np.testing.assert_allclose(mean, np.nanmean(params, axis=0))


def test_requires_a_matching_layout():
    ptstate = PTState(ndim=8, ntemps=1)
    with pytest.raises(ValueError, match="layout"):
        MaskedChainStats(7, ptstate, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="ndim"):
        MaskedChainStats(8, ptstate, 0, np.random.default_rng(0), layout=LAYOUT)


def test_update_from_window_drops_k_and_uses_masked_moments():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    rows = padded_rows(LAYOUT, 300)
    cs.update_from_window(cs.sample_total, rows)
    params = LAYOUT.params(rows)
    assert cs._buffer.shape == (500, 7)
    np.testing.assert_array_equal(cs._buffer[-300:], params)
    expected = clip_to_psd(pairwise_covariance(params, np.eye(7), 10), cs._cov_ridge)
    np.testing.assert_allclose(cs.sample_cov, expected, rtol=1e-12)
    np.testing.assert_allclose(cs.sample_mean, np.nanmean(params, axis=0))


def test_update_skips_a_window_with_a_non_finite_active_value():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    rows = padded_rows(LAYOUT, 50)
    rows[3, 0] = np.inf
    cs.update_from_window(0, rows)
    assert cs.sample_total == 0 and cs._seen_raw == 0
    cs.update_from_window(0, padded_rows(LAYOUT, 50, seed=3))
    assert cs.sample_total == 50 and cs._seen_raw == 50
    assert not np.array_equal(cs.sample_cov, np.eye(7))


def test_active_block_factor_matches_the_block_of_the_covariance():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    rows = padded_rows(LAYOUT, 300)
    cs.update_from_window(0, rows)
    idx, factor, neff, sqrt_2neff = cs.active_block(0, 2)
    assert idx.tolist() == [0, 1, 2, 3, 4] and neff == 5
    assert sqrt_2neff == math.sqrt(10)
    own = np.cov(LAYOUT.params(rows[rows[:, -1] == 2])[:, :5], rowvar=False)
    block = own + cs._cov_ridge * np.eye(5)
    np.testing.assert_allclose(factor @ factor.T, block, rtol=1e-10, atol=1e-12)
    assert cs.active_block(0, 2) is cs.active_block(0, 2)
    cs.update_from_window(cs.sample_total, padded_rows(LAYOUT, 20, seed=1))
    assert cs.active_block(0, 2)[1] is not factor


def test_groups_restrict_blocks_to_active_coordinates():
    groups = [np.array([0]), np.array([1, 2, 5, 6]), np.array([3, 4])]
    cs = make_stats(LAYOUT, groups=groups)
    assert cs.eligible_groups(0) == [0]
    assert cs.eligible_groups(1) == [0, 1]
    assert cs.eligible_groups(3) == [0, 1, 2]
    assert cs.active_block(1, 1)[0].tolist() == [1, 2]
    assert cs.active_block(1, 0) is None


def test_checkpoint_round_trip_rebuilds_identical_factors():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    cs.update_from_window(0, padded_rows(LAYOUT, 300))
    cs.active_block(0, 3)
    arrays, meta = cs.get_checkpoint_state()
    fresh = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    fresh.active_block(0, 3)
    fresh.set_checkpoint_state(arrays, meta)
    assert fresh._factors == {} and fresh._eligible == {}
    np.testing.assert_array_equal(fresh._buffer, cs._buffer)
    np.testing.assert_array_equal(fresh.active_block(0, 3)[1], cs.active_block(0, 3)[1])


def test_pooled_fallback_keeps_a_non_identity_initial_covariance():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1, sample_cov=4 * np.eye(7))
    rows = padded_rows(LAYOUT, 300)
    rows = rows[rows[:, -1] < 3]  # the last slot is never active
    cs.update_from_window(0, rows)
    assert cs.sample_cov[5, 5] == 4.0 and cs.sample_cov[6, 6] == 4.0
    assert cs.sample_cov[0, 0] != 4.0


def test_inactive_slots_are_forced_to_nan_before_storage():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    clean = padded_rows(LAYOUT, 300)
    cs.update_from_window(0, clean)
    ref = cs.sample_cov.copy()
    cs2 = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    dirty = clean.copy()
    i = int(np.flatnonzero(dirty[:, -1] == 0)[0])
    j = int(np.flatnonzero(dirty[:, -1] == 1)[0])
    dirty[i, 5] = 1e6
    dirty[j, 6] = np.inf
    cs2.update_from_window(0, dirty)
    assert cs2.sample_total == 300
    np.testing.assert_array_equal(cs2.sample_cov, ref)
    assert np.isnan(cs2._buffer[-300:][i, 5]) and np.isnan(cs2._buffer[-300:][j, 6])


@pytest.mark.parametrize("bad_k", [np.nan, np.inf, 1.5, -1.0, 4.0])
def test_a_window_with_an_invalid_k_is_skipped(bad_k):
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    rows = padded_rows(LAYOUT, 50)
    rows[7, -1] = bad_k
    cs.update_from_window(0, rows)
    assert cs.sample_total == 0 and cs._seen_raw == 0


def test_fallback_state_round_trips_through_a_checkpoint():
    cs = make_stats(LAYOUT, sample_cov=4 * np.eye(7), sample_mean=np.arange(7.0))
    arrays, meta = cs.get_checkpoint_state()
    fresh = make_stats(LAYOUT)
    fresh.set_checkpoint_state(arrays, meta)
    np.testing.assert_array_equal(fresh._fallback_cov, 4 * np.eye(7))
    np.testing.assert_array_equal(fresh._fallback_mean, np.arange(7.0))


def test_pairwise_covariance_rejects_tiny_min_rows():
    with pytest.raises(ValueError, match="min_rows"):
        pairwise_covariance(np.zeros((5, 2)), np.eye(2), 1)


def _mixture_rows(layout, n0=1800, n1=200, seed=0):
    rng = np.random.default_rng(seed)
    a0 = rng.normal(0.0, 0.05, n0)
    z = rng.normal(size=n1)
    a1, b1 = z, z + 0.05 * rng.normal(size=n1)
    r0 = np.stack([layout.make_row([a, 0.0], 0) for a in a0])
    r1 = np.stack([layout.make_row([a, b], 1) for a, b in zip(a1, b1)])
    rows = np.concatenate([r0, r1])
    return rows[rng.permutation(len(rows))], a0, np.column_stack([a1, b1])


def test_blocks_come_from_each_models_own_rows():
    lay = RJLayout(n_shared=1, n_component=1, max_components=1)
    rows, a0, ab1 = _mixture_rows(lay)
    cs = make_stats(lay, buffer_size=2000, buffer_thin=1)
    cs.update_from_window(0, rows)
    assert np.linalg.eigvalsh(cs.sample_cov).min() >= 0
    for k, truth in ((0, np.atleast_2d(a0.var(ddof=1))), (1, np.cov(ab1.T))):
        idx, factor, neff, _ = cs.active_block(0, k)
        assert neff == truth.shape[0]
        got = factor @ factor.T
        assert np.linalg.eigvalsh(got).min() > 0
        np.testing.assert_allclose(np.diag(got), np.diag(truth), rtol=0.1)
    # the pooled matrix would give a k = 0 variance far from the k = 0 rows'
    assert cs.sample_cov[0, 0] > 10 * a0.var(ddof=1)


def test_a_model_with_too_few_rows_uses_the_pooled_block():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    rows = padded_rows(LAYOUT, 300)
    rows = rows[(rows[:, -1] != 2) | (np.cumsum(rows[:, -1] == 2) <= 5)]  # 5 rows at k = 2
    cs.update_from_window(0, rows)
    assert cs._model_covariance(2) is None
    idx, factor, _, _ = cs.active_block(0, 2)
    block = cs.sample_cov[np.ix_(idx, idx)] + cs._cov_ridge * np.eye(len(idx))
    np.testing.assert_allclose(factor @ factor.T, block, rtol=1e-10, atol=1e-12)
    assert cs._model_covariance(1) is not None


def test_draw_active_block_only_returns_eligible_groups():
    groups = [np.array([0]), np.array([1, 2, 5, 6]), np.array([3, 4])]
    cs = make_stats(LAYOUT, groups=groups)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        idx = cs.draw_active_block(rng, 1)[0]
        seen.add(tuple(idx.tolist()))
    assert seen == {(0,), (1, 2)}
    assert (
        make_stats(RJLayout(0, 1, 2), groups=[np.array([0, 1])]).draw_active_block(rng, 0) is None
    )

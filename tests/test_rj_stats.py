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
    assert cs.sample_total == 0


def test_active_block_factor_matches_the_block_of_the_covariance():
    cs = make_stats(LAYOUT, buffer_size=500, buffer_thin=1)
    cs.update_from_window(0, padded_rows(LAYOUT, 300))
    idx, factor, neff, sqrt_2neff = cs.active_block(0, 2)
    assert idx.tolist() == [0, 1, 2, 3, 4] and neff == 5
    assert sqrt_2neff == math.sqrt(10)
    block = cs.sample_cov[np.ix_(idx, idx)] + cs._cov_ridge * np.eye(5)
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

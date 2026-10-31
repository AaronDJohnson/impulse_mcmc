"""Reversible-jump moves."""

import math

import numpy as np
import pytest
from rj_helpers import make_stats, padded_rows

from impulse import RJLayout
from impulse.reversible_jump import RJDEProposal, masked_difference, rj_am, rj_de, rj_scam

LAYOUT = RJLayout(n_shared=1, n_component=2, max_components=3)  # D = 7, rows of 8


def _at(k, seed=0, fill=0, current=None, **kwargs):
    cs = make_stats(LAYOUT, seed=seed, buffer_size=200, buffer_thin=1, **kwargs)
    if fill:
        cs.update_from_window(0, padded_rows(LAYOUT, fill, seed))
    x = np.arange(7.0) if current is None else np.asarray(current, dtype=float)
    cs.update_sample(LAYOUT.make_row(x, k))
    return cs


@pytest.mark.parametrize("move", [rj_am, rj_scam, rj_de])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_within_model_moves_touch_only_active_coordinates(move, k):
    cs = _at(k, fill=150)
    x = cs.current_sample.copy()
    n_act = LAYOUT.n_active(k)
    moved = False
    for _ in range(50):
        y, qxy = move(cs)
        assert qxy == 0.0
        assert y.shape == (8,) and y[-1] == k
        assert np.all(np.isnan(y[n_act:7])) and np.all(np.isfinite(y[:n_act]))
        assert not np.shares_memory(y, cs.current_sample)
        moved |= bool(np.any(y[:n_act] != x[:n_act]))
    assert moved
    np.testing.assert_array_equal(cs.current_sample, x)


def test_moves_return_the_current_state_when_nothing_is_active():
    layout = RJLayout(n_shared=0, n_component=2, max_components=2)
    cs = make_stats(layout, buffer_size=200, buffer_thin=1)
    cs.update_from_window(0, padded_rows(layout, 150))
    cs.update_sample(layout.make_row(np.zeros(4), 0))
    for move in (rj_am, rj_scam, rj_de):
        y, qxy = move(cs)
        assert qxy == 0.0 and np.array_equal(y, cs.current_sample, equal_nan=True)


def test_rj_am_step_uses_the_active_count():
    cs = _at(1)  # identity covariance, one group, 3 active coordinates
    twin = np.random.default_rng(0)
    y, _ = rj_am(cs)
    # Replay the draws: one eligible group (no group draw), the jump scale, 3 normals.
    u = twin.random()
    scale = 10.0 if u > 0.97 else (0.2 if u > 0.9 else 1.0)
    z = twin.standard_normal(3)
    _, factor, neff, _ = cs.active_block(0, 1)
    assert neff == 3
    expected = np.arange(3.0) + 2.4 / math.sqrt(6) * scale * (factor @ z)
    np.testing.assert_allclose(y[:3], expected, rtol=1e-12)


def test_rj_de_waits_for_min_fill():
    cs = _at(2, fill=5)
    y, qxy = RJDEProposal(min_fill=10)(cs)
    assert qxy == 0.0 and np.array_equal(y, cs.current_sample, equal_nan=True)


def test_rj_de_uses_zero_where_a_stored_row_lacks_the_slot():
    cs = _at(3, seed=4, current=np.full(7, 5.0))
    rows = np.stack(
        [LAYOUT.make_row(np.full(7, 1.0 if i % 2 else 0.0), 1 if i % 2 else 3) for i in range(40)]
    )
    cs.update_from_window(0, rows)
    proposal = RJDEProposal(min_fill=10)
    steps = np.array([proposal(cs)[0][:7] - cs.current_sample[:7] for _ in range(200)])
    assert np.all(steps[:, 3:] == 0.0)  # slots 1-2 are missing from every k = 1 row
    assert np.all(steps[:, 0] == steps[:, 1]) and np.all(steps[:, 1] == steps[:, 2])
    assert np.any(steps[:, 0] != 0.0)


def test_masked_difference_is_antisymmetric_with_the_same_mask():
    a = np.array([1.0, np.nan, 3.0, np.nan])
    b = np.array([0.5, 2.0, np.nan, np.nan])
    assert np.array_equal(masked_difference(a, b), [0.5, 0.0, 0.0, 0.0])
    assert np.array_equal(masked_difference(b, a), -masked_difference(a, b))


def test_groups_without_active_coordinates_are_skipped():
    groups = [np.array([0]), np.array([1, 2]), np.array([3, 4, 5, 6])]
    cs = _at(1, groups=groups, fill=150)
    for _ in range(100):
        y, _ = rj_am(cs)
        assert np.all(np.isnan(y[3:7]))
    assert cs.eligible_groups(1) == [0, 1]


def test_rj_de_min_fill_validation():
    with pytest.raises(ValueError, match="min_fill"):
        RJDEProposal(min_fill=1)

"""Reversible-jump moves."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from rj_helpers import make_stats, padded_rows

from impulse import IndependentComponentProposal, RJLayout
from impulse.reversible_jump import (
    BirthDeath,
    ComponentRedraw,
    RJDEProposal,
    _CheckedWithinModelJump,
    _ModelJump,
    _RJJumpProposals,
    masked_difference,
    rj_am,
    rj_de,
    rj_scam,
)

LAYOUT = RJLayout(n_shared=1, component_dim=2, max_components=3)  # D = 7, rows of 8


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
    layout = RJLayout(n_shared=0, component_dim=2, max_components=2)
    cs = make_stats(layout, buffer_size=200, buffer_thin=1)
    cs.update_from_window(0, padded_rows(layout, 150))
    cs.update_sample(layout.make_row(np.zeros(4), 0))
    for move in (rj_am, rj_scam, rj_de):
        y, qxy = move(cs)
        assert qxy == 0.0 and np.array_equal(y, cs.current_sample, equal_nan=True)


def _first_scale_seed():
    for seed in range(100):
        if np.random.default_rng(seed).random() > 0.9:
            return seed
    raise AssertionError("no seed found")


def test_rj_am_step_uses_the_active_count():
    seed = _first_scale_seed()
    cs = _at(1, seed=seed)  # identity covariance, one group, 3 active coordinates
    twin = np.random.default_rng(seed)
    y, _ = rj_am(cs)
    # Replay the draws: one eligible group (no group draw), the jump scale, 3 normals.
    u = twin.random()
    scale = 10.0 if u > 0.97 else (0.2 if u > 0.9 else 1.0)
    assert scale != 1.0
    z = twin.standard_normal(3)
    _, factor, neff, _ = cs.active_block(0, 1)
    assert neff == 3
    expected = np.arange(3.0) + 2.4 / math.sqrt(6) * scale * (factor @ z)
    np.testing.assert_allclose(y[:3], expected, rtol=1e-12)


def test_rj_scam_step_uses_the_fixed_axis_scale():
    seed = _first_scale_seed()
    cs = _at(1, seed=seed)
    twin = np.random.default_rng(seed)
    y, _ = rj_scam(cs)
    u = twin.random()
    scale = 10.0 if u > 0.97 else (0.2 if u > 0.9 else 1.0)
    assert scale != 1.0
    _, factor, neff, _ = cs.active_block(0, 1)
    ind = twin.integers(0, neff)
    z = twin.standard_normal()
    expected = np.arange(3.0) + z * (2.4 / math.sqrt(2)) * scale * factor[:, ind]
    np.testing.assert_allclose(y[:3], expected, rtol=1e-12)


def test_rj_de_gamma_uses_the_active_count():
    seen = set()
    for seed in range(30):
        cs = _at(2, seed=seed, fill=150)  # 5 active coordinates
        twin = np.random.default_rng(seed)
        y, _ = RJDEProposal(min_fill=10)(cs)
        buf = cs._buffer
        n = min(cs.sample_total, len(buf))
        off = len(buf) - n
        mm, nn = twin.integers(0, n), twin.integers(0, n)
        while mm == nn:
            nn = twin.integers(0, n)
        diff = masked_difference(buf[off + mm, :5], buf[off + nn, :5])
        assert np.any(diff)
        if twin.random() > 0.5:
            gamma = 1.0
            seen.add("one")
        else:
            gamma = twin.random() * 2.4 / math.sqrt(10)
            seen.add("uniform")
        np.testing.assert_allclose(y[:5], np.arange(5.0) + gamma * diff, rtol=1e-12)
    assert seen == {"one", "uniform"}


def test_rj_de_step_is_symmetric_on_a_trending_buffer():
    # Rows in time order with a strong trend: any order-dependent pair choice shows up as drift.
    cs = make_stats(LAYOUT, seed=1, buffer_size=200, buffer_thin=1)
    ks = np.tile([1, 2, 3], 67)[:200]
    rows = np.stack([LAYOUT.make_row(np.full(7, float(i)), ks[i]) for i in range(200)])
    cs.update_from_window(0, rows)
    cs.update_sample(LAYOUT.make_row(np.zeros(7), 3))
    de = RJDEProposal(min_fill=10)
    steps = np.array([de(cs)[0][:7] for _ in range(20000)])
    se = steps.std(axis=0) / np.sqrt(len(steps))
    assert np.all(np.abs(steps.mean(axis=0)) < 5 * se)


def test_within_model_steps_do_not_depend_on_the_current_point():
    for move in (rj_am, rj_scam, RJDEProposal(min_fill=10)):
        out = []
        for x in (np.arange(7.0), 50.0 - 3.0 * np.arange(7.0)):
            cs = make_stats(LAYOUT, seed=2, buffer_size=200, buffer_thin=1)
            rows = np.stack(
                [
                    LAYOUT.make_row(np.random.default_rng(i).normal(size=7), i % 4)
                    for i in range(150)
                ]
            )
            cs.update_from_window(0, rows)
            cs.update_sample(LAYOUT.make_row(x, 2))
            out.append(np.array([move(cs)[0][:5] - x[:5] for _ in range(200)]))
        np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-12)


def test_rj_de_redraws_pairs_that_give_a_zero_difference():
    # Half of the stored rows lack every slot, so a single pair is usable only 1 time in 4.
    layout = RJLayout(n_shared=0, component_dim=2, max_components=3)
    rows = np.stack([layout.make_row(np.full(6, float(i)), 3 if i % 2 else 0) for i in range(80)])
    moved = 0
    for seed in range(200):
        cs = make_stats(layout, seed=seed, buffer_size=200, buffer_thin=1)
        cs.update_from_window(0, rows)
        cs.update_sample(layout.make_row(np.full(6, 5.0), 3))
        y, qxy = RJDEProposal(min_fill=10)(cs)
        assert qxy == 0.0 and np.all(np.isfinite(y[:6]))
        moved += bool(np.any(y[:6] != 5.0))
    assert moved > 190


@pytest.mark.filterwarnings("ignore:groups do not cover")
def test_rj_de_returns_the_current_state_when_no_buffered_row_has_the_slot():
    # Every stored row has k = 0, so none has a component slot: the one group (the slots) always
    # gives a zero difference and the move returns the current state.
    layout = LAYOUT
    groups = [np.array([1, 2, 3, 4, 5, 6])]
    rows = np.stack([layout.make_row(np.full(7, float(i)), 0) for i in range(40)])
    cs = make_stats(layout, groups=groups, buffer_size=200, buffer_thin=1)
    cs.update_from_window(0, rows)
    cs.update_sample(layout.make_row(np.full(7, 5.0), 3))
    for _ in range(20):
        y, qxy = RJDEProposal(min_fill=10)(cs)
        assert qxy == 0.0 and np.array_equal(y, cs.current_sample, equal_nan=True)


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


P = 2
BD_LAYOUT = RJLayout(n_shared=0, component_dim=P, max_components=3)  # D = 6, rows of 7


def _uniform_draw(rng):
    return rng.random(P)


def _uniform_logpdf(u):
    u = np.asarray(u)
    return 0.0 if np.all((u >= 0) & (u <= 1)) else -np.inf


def _ramp_draw(rng):
    return np.sqrt(rng.random(P))  # density 2u on [0, 1] in each coordinate


def _ramp_logpdf(u):
    u = np.asarray(u)
    return float(np.sum(np.log(2 * u))) if np.all((u > 0) & (u <= 1)) else -np.inf


UNIFORM = IndependentComponentProposal(_uniform_draw, _uniform_logpdf)
RAMP = IndependentComponentProposal(_ramp_draw, _ramp_logpdf)


def _bd_at(k, seed=0, comps=None):
    cs = make_stats(BD_LAYOUT, seed=seed)
    x = np.full(BD_LAYOUT.ndim, 0.5) if comps is None else np.asarray(comps, dtype=float)
    cs.update_sample(BD_LAYOUT.make_row(x, k))
    return cs


def test_independent_component_proposal_ignores_the_state():
    u = UNIFORM.draw(np.random.default_rng(0), np.zeros(6), 2)
    assert u.shape == (2,) and UNIFORM.logpdf(u, np.zeros(6), 2) == 0.0
    with pytest.raises(TypeError, match="callable"):
        IndependentComponentProposal(_uniform_draw, None)


def test_birth_at_min_components():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=1)
    y, qxy = bd(_bd_at(0))
    assert y[-1] == 1
    assert np.all((y[:2] >= 0) & (y[:2] <= 1)) and np.all(np.isnan(y[2:6]))
    assert qxy == pytest.approx(math.log(0.5))  # log d_1 - log b_0 - log g = log(1/2) - 0 - 0


def test_death_at_max_components():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=1)
    y, qxy = bd(_bd_at(3))
    assert y[-1] == 2 and np.all(np.isnan(y[4:6]))
    assert qxy == pytest.approx(math.log(0.5))  # log b_2 - log d_3 + log g = log(1/2) - 0 + 0


def test_interior_selection_ratios():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=1)
    cs = _bd_at(1)
    seen = {}
    for seed in range(40):
        cs.rng = np.random.default_rng(seed)
        y, qxy = bd(cs)
        seen[int(y[-1])] = qxy
    assert seen[2] == pytest.approx(0.0)  # birth 1 -> 2: log d_2 - log b_1 = 0
    assert seen[0] == pytest.approx(math.log(2.0))  # death 1 -> 0: log b_0 - log d_1


def test_birth_and_death_use_the_component_density():
    bd = BirthDeath(BD_LAYOUT, RAMP, ntemps=1)
    y, qxy = bd(_bd_at(0))
    assert qxy == pytest.approx(math.log(0.5) - _ramp_logpdf(y[:2]))
    comps = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    y, qxy = bd(_bd_at(3, comps=comps))
    kept = BD_LAYOUT.components(y)[:2]
    removed = [c for c in comps.reshape(3, 2) if not any(np.array_equal(c, r) for r in kept)]
    assert len(removed) == 1
    assert qxy == pytest.approx(math.log(0.5) + _ramp_logpdf(removed[0]))


def test_death_with_zero_density_is_rejected_and_counted():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=1)
    cs = _bd_at(3, comps=[2.0] * 6)  # outside the proposal's support
    y, qxy = bd(cs)
    assert qxy == -math.inf
    np.testing.assert_array_equal(y, cs.current_sample)
    assert bd.rejected_deaths == 1


def test_branch_counts_follow_reports_and_checkpoint():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=2)
    bd(_bd_at(0))
    bd.report_branch_accept(0, True)
    bd(_bd_at(3))
    bd.report_branch_accept(0, False)
    assert bd.branch_counts(0) == {"birth": (1, 1), "death": (1, 0)}
    assert bd.branch_counts(1) == {"birth": (0, 0), "death": (0, 0)}
    state = json.loads(json.dumps(bd.get_checkpoint_state()))
    fresh = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=2)
    fresh.set_checkpoint_state(state)
    assert fresh.branch_counts(0) == bd.branch_counts(0)
    assert fresh.rejected_deaths == bd.rejected_deaths


def test_component_redraw():
    redraw = ComponentRedraw(BD_LAYOUT, UNIFORM)
    y, qxy = redraw(_bd_at(0))
    assert y[-1] == 0 and qxy == 0.0
    cs = _bd_at(2, comps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y, qxy = redraw(cs)
    assert y[-1] == 2 and qxy == 0.0 and np.all(np.isnan(y[4:6]))
    before, after = BD_LAYOUT.components(cs.current_sample)[:2], BD_LAYOUT.components(y)[:2]
    assert sum(not np.array_equal(b, a) for b, a in zip(before, after)) == 1


def test_component_redraw_qxy_is_old_minus_new_density():
    redraw = ComponentRedraw(BD_LAYOUT, RAMP)
    cs = _bd_at(2, comps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y, qxy = redraw(cs)
    before, after = BD_LAYOUT.components(cs.current_sample)[:2], BD_LAYOUT.components(y)[:2]
    j = next(i for i in range(2) if not np.array_equal(before[i], after[i]))
    assert qxy == pytest.approx(_ramp_logpdf(before[j]) - _ramp_logpdf(after[j]))


def _grow(cs):
    return BD_LAYOUT.insert(cs.current_sample, 0, [0.5, 0.5]), 0.0


def _scribble(cs):
    y = cs.current_sample.copy()
    y[5] = 1.0
    return y, 0.0


def _short(cs):
    return np.zeros(3), 0.0


def test_within_model_wrapper_checks_k_and_inactive_slots():
    with pytest.raises(ValueError, match="'_grow' changed k"):
        _CheckedWithinModelJump(_grow, BD_LAYOUT)(_bd_at(1))
    with pytest.raises(ValueError, match="'_scribble' wrote to an inactive slot"):
        _CheckedWithinModelJump(_scribble, BD_LAYOUT)(_bd_at(1))
    with pytest.raises(ValueError, match="'_short' returned a row of shape"):
        _ModelJump(_short, BD_LAYOUT, 0.0)(_bd_at(1))


class _Stateful:
    __name__ = "stateful_move"

    def __init__(self):
        self.count = 0
        self.frozen = False

    def __call__(self, cs):
        self.count += 1
        return cs.current_sample.copy(), 0.0

    def get_checkpoint_state(self):
        return {"count": self.count}

    def set_checkpoint_state(self, state):
        self.count = state["count"]

    def freeze_adaptation(self):
        self.frozen = True


def test_wrappers_keep_the_name_and_hooks():
    move = _Stateful()
    for wrapped in (_CheckedWithinModelJump(move, BD_LAYOUT), _ModelJump(move, BD_LAYOUT, 0.0)):
        assert wrapped.__name__ == "stateful_move"
        wrapped(_bd_at(1))
        assert wrapped.get_checkpoint_state() == {"count": move.count}
        wrapped.set_checkpoint_state({"count": 7})
        assert move.count == 7
        wrapped.freeze_adaptation()
        assert move.frozen
    plain = _CheckedWithinModelJump(_grow, BD_LAYOUT)
    assert not hasattr(plain, "get_checkpoint_state")
    assert not hasattr(plain, "freeze_adaptation")


def _bad_k(cs):
    return np.append(np.full(6, np.nan), 4.0), 0.0


def _nan_active(cs):
    return np.append(np.full(6, np.nan), 1.0), 0.0


def _values_past_k(cs):
    return np.array([0.5, 0.5, 0.5, np.nan, np.nan, np.nan, 1.0]), 0.0


@pytest.mark.parametrize(
    "move, message",
    [(_bad_k, "invalid k"), (_nan_active, "non-finite value"), (_values_past_k, "past k=1")],
)
def test_model_jump_checks_name_the_jump(move, message):
    with pytest.raises(ValueError, match=f"'{move.__name__}'.*{message}"):
        _ModelJump(move, BD_LAYOUT, 0.0)(_bd_at(1))


def test_model_jump_adds_the_log_jacobian_by_direction():
    def up(cs):
        return BD_LAYOUT.insert(cs.current_sample, 0, [0.5, 0.5]), 0.25

    def down(cs):
        return BD_LAYOUT.remove(cs.current_sample, 0)[0], 0.25

    def stay(cs):
        return cs.current_sample.copy(), 0.25

    assert _ModelJump(up, BD_LAYOUT, 1.5)(_bd_at(1))[1] == pytest.approx(1.75)
    assert _ModelJump(down, BD_LAYOUT, 1.5)(_bd_at(1))[1] == pytest.approx(-1.25)
    assert _ModelJump(stay, BD_LAYOUT, 1.5)(_bd_at(1))[1] == pytest.approx(0.25)


def test_jump_proposals_report_birth_and_death_separately():
    cs = _bd_at(0)
    jp = _RJJumpProposals(cs)
    jp.add_jump(BirthDeath(BD_LAYOUT, UNIFORM, ntemps=1), 1.0)
    jp(SimpleNamespace(positions=cs.current_sample[None, :].copy()))
    jp.report_accept(True)
    rates = jp.acceptance_rates()
    assert set(rates) == {"birth", "death"}
    assert rates["birth"] == {"calls": 1, "accepts": 1, "rate": 1.0}
    assert rates["death"] == {"calls": 0, "accepts": 0, "rate": 0.0}


def _inplace_k(cs):
    x = cs.current_sample
    x[-1] = 2.0
    return x, 0.0


def _returns_current(cs):
    return cs.current_sample, 0.0


def test_in_place_moves_are_rejected_not_misread():
    # current_sample is a read-only view, so an in-place write fails before k is read back.
    with pytest.raises(ValueError, match="read-only"):
        _CheckedWithinModelJump(_inplace_k, BD_LAYOUT)(_bd_at(1))
    with pytest.raises(ValueError, match="read-only"):
        _ModelJump(_inplace_k, BD_LAYOUT, 1.5)(_bd_at(1))
    with pytest.raises(ValueError, match="'_returns_current'.*sharing memory"):
        _CheckedWithinModelJump(_returns_current, BD_LAYOUT)(_bd_at(1))
    with pytest.raises(ValueError, match="'_returns_current'.*sharing memory"):
        _ModelJump(_returns_current, BD_LAYOUT, 1.5)(_bd_at(1))


def test_within_model_wrapper_rejects_non_finite_active_values():
    def nan_active(cs):
        y = cs.current_sample.copy()
        y[0] = np.nan
        return y, 0.0

    with pytest.raises(ValueError, match="'nan_active'.*non-finite"):
        _CheckedWithinModelJump(nan_active, BD_LAYOUT)(_bd_at(1))


class _Writer:
    def draw(self, rng, x, k):
        x[np.isnan(x)] = 0.0
        return rng.random(P)

    def logpdf(self, u, x, k):
        return 0.0


@pytest.mark.parametrize("move_cls, k", [(BirthDeath, 1), (ComponentRedraw, 1)])
def test_component_proposal_cannot_write_into_the_state(move_cls, k):
    cs = _bd_at(k)
    before = cs.current_sample.copy()
    args = (BD_LAYOUT, _Writer()) + ((1,) if move_cls is BirthDeath else ())
    move = move_cls(*args)
    with pytest.raises(ValueError, match="read-only"):
        for seed in range(20):  # BirthDeath at k=1 may take the death branch first
            cs.rng = np.random.default_rng(seed)
            move(cs)
    np.testing.assert_array_equal(cs.current_sample, before)


def test_birth_death_needs_room_to_move():
    layout = RJLayout(n_shared=0, component_dim=P, max_components=2, min_components=2)
    with pytest.raises(ValueError, match="min_components < max_components"):
        BirthDeath(layout, UNIFORM, ntemps=1)


class _Column:
    def draw(self, rng, x, k):
        return rng.random((P, 1))

    def logpdf(self, u, x, k):
        return 0.0


@pytest.mark.parametrize(
    "move, k",
    [(BirthDeath(BD_LAYOUT, _Column(), 1), 0), (ComponentRedraw(BD_LAYOUT, _Column()), 1)],
)
def test_component_draw_shape_is_checked(move, k):
    with pytest.raises(ValueError, match=r"_Column\.draw returned shape \(2, 1\); expected \(2,\)"):
        move(_bd_at(k))


def test_birth_with_non_finite_density_is_rejected():
    bad = IndependentComponentProposal(_uniform_draw, lambda u: -np.inf)
    cs = _bd_at(0)
    y, qxy = BirthDeath(BD_LAYOUT, bad, ntemps=1)(cs)
    assert qxy == -math.inf
    np.testing.assert_array_equal(y, cs.current_sample)


def test_selection_ratios_with_a_positive_minimum():
    layout = RJLayout(n_shared=0, component_dim=P, max_components=3, min_components=1)
    bd = BirthDeath(layout, UNIFORM, ntemps=1)
    cs = make_stats(layout)
    cs.update_sample(layout.make_row(np.full(layout.ndim, 0.5), 1))
    assert bd._birth_probability(1) == 1.0
    y, qxy = bd(cs)  # at k = min only a birth is possible; d_2 / b_1 = 1/2
    assert y[-1] == 2 and qxy == pytest.approx(math.log(0.5))
    assert bd._selection_log_ratio(1) == pytest.approx(math.log(0.5))
    cs.update_sample(layout.make_row(np.full(layout.ndim, 0.5), 2))
    for seed in range(40):
        cs.rng = np.random.default_rng(seed)
        y, qxy = bd(cs)
        if y[-1] == 1:
            assert qxy == pytest.approx(math.log(2.0))  # log b_1 - log d_2 = -log(1/2)
            break
    else:
        pytest.fail("no death drawn")


def test_two_chains_sharing_one_birth_death_keep_separate_counts():
    bd = BirthDeath(BD_LAYOUT, UNIFORM, ntemps=2)
    jps = []
    for chain in range(2):
        cs = make_stats(BD_LAYOUT, seed=chain)
        cs.chain_index = chain
        cs.update_sample(BD_LAYOUT.make_row(np.full(BD_LAYOUT.ndim, 0.5), 1))
        jp = _RJJumpProposals(cs)
        jp.add_jump(bd, 1.0)
        jps.append(jp)
    rng = np.random.default_rng(5)
    for _ in range(60):
        for chain, jp in enumerate(jps):
            row = BD_LAYOUT.make_row(np.full(BD_LAYOUT.ndim, 0.5), 1)
            jp(SimpleNamespace(positions=np.stack([row, row])))
            jp.report_accept(bool(rng.random() < (0.3 if chain == 0 else 0.8)))
    for chain, jp in enumerate(jps):
        counts = bd.branch_counts(chain)
        assert sum(c for c, _ in counts.values()) == jp._proposal_calls[0] == 60
        assert sum(a for _, a in counts.values()) == jp._proposal_accepts[0]
        rates = jp.acceptance_rates()
        assert rates["birth"]["calls"] + rates["death"]["calls"] == 60
        assert rates["birth"]["accepts"] + rates["death"]["accepts"] == jp._proposal_accepts[0]

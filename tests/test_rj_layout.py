"""RJLayout: the padded-row convention for reversible-jump states."""

import numpy as np
import pytest

from impulse import RJLayout

L = RJLayout(n_shared=2, component_dim=3, max_components=4, min_components=0)


def _row(k, seed=0):
    return L.make_row(np.random.default_rng(seed).normal(size=L.ndim), k)


def test_sizes():
    assert L.ndim == 14 and L.width == 15
    assert L.n_active(0) == 2 and L.n_active(4) == 14
    assert L.slot(1) == slice(5, 8)


def test_make_row_pads_with_nan_and_stores_k():
    row = _row(2)
    assert row.shape == (15,) and row[-1] == 2
    assert np.all(np.isfinite(row[:8])) and np.all(np.isnan(row[8:14]))


def test_make_row_rejects_bad_input():
    with pytest.raises(ValueError, match="k must be an integer"):
        L.make_row(np.zeros(14), 5)
    with pytest.raises(ValueError, match="k must be an integer"):
        L.make_row(np.zeros(14), 1.5)
    with pytest.raises(ValueError, match="shape"):
        L.make_row(np.zeros(13), 1)
    x = np.zeros(14)
    x[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        L.make_row(x, 1)


def test_active_mask():
    assert L.active_mask(0).tolist() == [True, True] + [False] * 12
    assert L.active_mask(2).sum() == 8
    assert L.active_mask(2) is L.active_mask(2)
    with pytest.raises(ValueError):
        L.active_mask(2)[0] = False
    rows = L.active_mask_rows(np.array([0.0, 2.0, 4.0]))
    assert rows.shape == (3, 14)
    assert np.array_equal(rows[1], L.active_mask(2))


LAYOUTS = [(2, 3, 4, 0), (0, 1, 1, 0), (0, 1, 5, 0), (3, 1, 6, 2)]


def _any_row(lay, k, seed=0):
    return lay.make_row(np.random.default_rng(seed).normal(size=lay.ndim), k)


@pytest.mark.parametrize("args", LAYOUTS)
def test_insert_then_remove_is_identity(args):
    lay = RJLayout(*args)
    u = np.arange(7.0, 7.0 + lay.component_dim)
    for k in range(lay.min_components, lay.max_components):
        row = _any_row(lay, k, seed=k)
        for j in range(k + 1):
            grown = lay.insert(row, j, u)
            assert grown[-1] == k + 1
            assert np.all(np.isnan(lay.params(grown)[lay.n_active(k + 1) :]))
            assert np.array_equal(lay.components(grown)[j], u)
            back, removed = lay.remove(grown, j)
            assert np.array_equal(back, row, equal_nan=True)
            assert np.array_equal(removed, u)


def test_insert_shifts_later_components():
    row = _row(2)
    before = L.components(row)[:2].copy()
    grown = L.insert(row, 1, [7, 8, 9])
    after = L.components(grown)
    assert np.array_equal(after[0], before[0]) and np.array_equal(after[2], before[1])
    assert np.array_equal(L.shared(grown), L.shared(row))


@pytest.mark.parametrize("args", LAYOUTS)
def test_remove_then_insert_is_identity(args):
    lay = RJLayout(*args)
    for k in range(max(lay.min_components, 1), lay.max_components + 1):
        row = _any_row(lay, k, seed=5 + k)
        for j in range(k):
            smaller, u = lay.remove(row, j)
            assert smaller[-1] == k - 1
            assert np.all(np.isnan(lay.params(smaller)[lay.n_active(k - 1) :]))
            assert np.array_equal(lay.insert(smaller, j, u), row, equal_nan=True)


def test_below_min_components_rejected():
    lay = RJLayout(2, 3, 4, 2)
    with pytest.raises(ValueError, match="k must be an integer"):
        lay.check_k(1)
    with pytest.raises(ValueError, match="k must be an integer"):
        lay.make_row(np.zeros(lay.ndim), 1)


def test_insert_and_remove_check_row_width():
    params_only = L.params(_row(2))
    with pytest.raises(ValueError, match="shape"):
        L.insert(params_only, 0, [1, 2, 3])
    with pytest.raises(ValueError, match="shape"):
        L.remove(params_only, 0)


def test_insert_rejects_non_finite_component():
    with pytest.raises(ValueError, match="finite"):
        L.insert(_row(1), 0, [1, np.nan, 3])


def test_active_mask_cannot_be_made_writable():
    with pytest.raises(ValueError):
        L.active_mask(2).setflags(write=True)


def test_insert_and_remove_do_not_mutate_their_input():
    row = _row(2)
    before = row.copy()
    L.insert(row, 0, [1, 2, 3])
    L.remove(row, 1)
    assert np.array_equal(row, before, equal_nan=True)


def test_insert_and_remove_bounds():
    with pytest.raises(ValueError, match="max_components"):
        L.insert(_row(4), 0, [1, 2, 3])
    with pytest.raises(ValueError, match="insert position"):
        L.insert(_row(1), 2, [1, 2, 3])
    with pytest.raises(ValueError, match="remove position"):
        L.remove(_row(1), 1)
    with pytest.raises(ValueError, match="3 parameters"):
        L.insert(_row(1), 0, [1, 2])


def test_views_on_batches():
    rows = np.stack([_row(k) for k in range(5)])
    params = L.params(rows)
    assert params.shape == (5, 14)
    assert L.shared(params).shape == (5, 2)
    assert L.components(params).shape == (5, 4, 3)
    assert np.shares_memory(L.components(params), rows)


@pytest.mark.parametrize(
    "args, match",
    [
        ((0, 0, 3), "component_dim"),
        ((0, 1, 3, 4), "min_components"),
        ((-1, 1, 3), "n_shared"),
        ((0, 1, 0), "max_components"),
        ((0, 1.5, 3), "component_dim must be an integer"),
    ],
)
def test_construction_checks(args, match):
    with pytest.raises(ValueError, match=match):
        RJLayout(*args)

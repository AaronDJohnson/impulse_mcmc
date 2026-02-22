"""Tests for Stan-style warmup."""

import numpy as np
import pytest

from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.warmup import DualAveraging, WarmupSchedule, find_reasonable_step_size
from impulse.nuts.core import NUTSState


def gaussian_logp_and_grad(x):
    logp = -0.5 * np.sum(x ** 2)
    grad = -x
    return logp, grad


class TestDualAveraging:
    def test_converges_to_target(self):
        """Dual averaging should converge step size to hit target accept."""
        da = DualAveraging(target_accept=0.8, initial_step_size=1.0)
        # Simulate constant acceptance of 0.9 -> should increase step size
        for _ in range(200):
            step = da.update(0.9)
        final = da.finalize()
        assert final > 1.0, "Step size should increase when accept > target"

    def test_decreases_step_for_low_accept(self):
        da = DualAveraging(target_accept=0.8, initial_step_size=1.0)
        for _ in range(200):
            step = da.update(0.3)
        final = da.finalize()
        assert final < 1.0, "Step size should decrease when accept < target"

    def test_reset(self):
        da = DualAveraging(initial_step_size=1.0)
        for _ in range(50):
            da.update(0.7)
        da.reset(0.5)
        assert da.count == 0
        assert np.isclose(np.exp(da.log_step), 0.5)


class TestWarmupSchedule:
    def test_window_schedule(self):
        """Windows should cover the middle phase and double in size."""
        ws = WarmupSchedule(num_warmup=1000, ndim=2)
        windows = ws._windows
        assert len(windows) > 0
        # First window starts at init_buffer
        assert windows[0][0] == 75
        # Last window ends at num_warmup - term_buffer
        assert windows[-1][1] == 950

    def test_short_warmup(self):
        """Gracefully handle very short warmup."""
        ws = WarmupSchedule(num_warmup=50, ndim=2, init_buffer=75, term_buffer=50)
        # With only 50 warmup iters and init_buffer=50, no windows
        # Just verify it doesn't crash
        assert isinstance(ws._windows, list)

    def test_mass_matrix_update(self):
        """Mass matrix should be updated at window boundaries."""
        ws = WarmupSchedule(num_warmup=200, ndim=2,
                           mass_matrix_type=MassMatrixType.DIAGONAL,
                           init_buffer=25, term_buffer=25)

        mm = MassMatrix(2, MassMatrixType.UNIT)
        mass_matrix_updated = False

        for i in range(200):
            pos = np.random.randn(2) * np.array([2.0, 0.5])
            state = NUTSState(
                position=pos, logp=-0.5 * np.sum(pos ** 2),
                grad=-pos, step_size=0.1, mass_matrix=mm,
            )
            _, new_mm = ws.update(i, state, 0.7)
            if new_mm is not None:
                mass_matrix_updated = True
                mm = new_mm

        assert mass_matrix_updated, "Mass matrix should be updated during warmup"

    def test_finalize(self):
        ws = WarmupSchedule(num_warmup=100, ndim=2, initial_step_size=0.5)
        mm = MassMatrix(2, MassMatrixType.UNIT)
        for i in range(100):
            state = NUTSState(
                position=np.zeros(2), logp=0.0, grad=np.zeros(2),
                step_size=0.5, mass_matrix=mm,
            )
            ws.update(i, state, 0.8)
        final = ws.finalize()
        assert np.isfinite(final)
        assert final > 0


class TestFindReasonableStepSize:
    def test_finds_step_size(self):
        """Should return a positive, finite step size."""
        x = np.array([0.0, 0.0])
        logp, grad = gaussian_logp_and_grad(x)
        mm = MassMatrix(2, MassMatrixType.UNIT)
        rng = np.random.default_rng(42)

        step = find_reasonable_step_size(x, logp, grad, gaussian_logp_and_grad, mm, rng)
        assert np.isfinite(step)
        assert step > 0

    def test_reasonable_range(self):
        """Step size should be in a reasonable range for standard Gaussian."""
        x = np.array([1.0, -0.5])
        logp, grad = gaussian_logp_and_grad(x)
        mm = MassMatrix(2, MassMatrixType.UNIT)
        rng = np.random.default_rng(42)

        step = find_reasonable_step_size(x, logp, grad, gaussian_logp_and_grad, mm, rng)
        # For a 2D standard Gaussian, step size should be O(1)
        assert 0.01 < step < 100

"""Tests for the core NUTS algorithm."""

import numpy as np
import pytest

from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType
from impulse.nuts.core import leapfrog, _build_tree, nuts_step, NUTSState


def gaussian_logp_and_grad(x):
    """Standard multivariate Gaussian: logp = -0.5 * x^T x."""
    logp = -0.5 * np.sum(x ** 2)
    grad = -x
    return logp, grad


def make_correlated_gaussian(ndim=2, rho=0.9):
    """Create a correlated Gaussian with given correlation."""
    cov = np.eye(ndim)
    for i in range(ndim):
        for j in range(ndim):
            if i != j:
                cov[i, j] = rho ** abs(i - j)
    prec = np.linalg.inv(cov)

    def logp_and_grad(x):
        logp = -0.5 * x @ prec @ x
        grad = -prec @ x
        return logp, grad

    return logp_and_grad, cov


class TestLeapfrog:
    def test_energy_conservation(self):
        """Leapfrog should approximately conserve energy with small step size."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([1.0, 0.5])
        logp, grad = gaussian_logp_and_grad(x)
        p = np.array([0.5, -0.3])

        H0 = -logp + mm.kinetic_energy(p)

        # Many small steps
        step_size = 0.01
        for _ in range(100):
            x, p, logp, grad = leapfrog(x, p, grad, step_size, mm, gaussian_logp_and_grad)

        H1 = -logp + mm.kinetic_energy(p)
        assert abs(H1 - H0) < 0.01, f"Energy drift: {abs(H1 - H0)}"

    def test_reversibility(self):
        """Leapfrog should be time-reversible."""
        mm = MassMatrix(3, MassMatrixType.UNIT)
        x0 = np.array([1.0, 0.5, -0.3])
        _, grad0 = gaussian_logp_and_grad(x0)
        p0 = np.array([0.5, -0.3, 0.1])

        step_size = 0.1
        x, p, logp, grad = leapfrog(x0, p0, grad0, step_size, mm, gaussian_logp_and_grad)

        # Reverse: negate momentum and step back
        x_rev, p_rev, _, _ = leapfrog(x, -p, grad, step_size, mm, gaussian_logp_and_grad)

        np.testing.assert_allclose(x_rev, x0, atol=1e-10)
        np.testing.assert_allclose(-p_rev, p0, atol=1e-10)

    def test_with_diagonal_mass_matrix(self):
        """Leapfrog with diagonal mass matrix should conserve energy."""
        diag = np.array([2.0, 0.5])
        mm = MassMatrix(2, MassMatrixType.DIAGONAL, diagonal=diag)
        x = np.array([1.0, 0.5])
        logp, grad = gaussian_logp_and_grad(x)
        p = np.array([0.5, -0.3])

        H0 = -logp + mm.kinetic_energy(p)

        step_size = 0.01
        for _ in range(100):
            x, p, logp, grad = leapfrog(x, p, grad, step_size, mm, gaussian_logp_and_grad)

        H1 = -logp + mm.kinetic_energy(p)
        assert abs(H1 - H0) < 0.01


class TestBuildTree:
    def test_single_step(self):
        """Depth-0 tree should do one leapfrog step."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([0.5, 0.5])
        logp, grad = gaussian_logp_and_grad(x)
        p = np.array([1.0, -1.0])
        H0 = -logp + mm.kinetic_energy(p)
        rng = np.random.default_rng(42)

        tree = _build_tree(x, p, grad, logp, 0, 0.1, 1, mm,
                          gaussian_logp_and_grad, H0, 1000.0, rng)

        assert tree["n_leapfrog"] == 1
        assert not tree["divergent"]
        assert not tree["turning"]

    def test_deeper_tree(self):
        """Depth-2 tree should do 4 leapfrog steps."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([0.5, 0.5])
        logp, grad = gaussian_logp_and_grad(x)
        p = np.array([1.0, -1.0])
        H0 = -logp + mm.kinetic_energy(p)
        rng = np.random.default_rng(42)

        tree = _build_tree(x, p, grad, logp, 2, 0.1, 1, mm,
                          gaussian_logp_and_grad, H0, 1000.0, rng)

        assert tree["n_leapfrog"] == 4

    def test_divergence_detection(self):
        """Very large step size should trigger divergence."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([0.5, 0.5])
        logp, grad = gaussian_logp_and_grad(x)
        p = np.array([1.0, -1.0])
        H0 = -logp + mm.kinetic_energy(p)
        rng = np.random.default_rng(42)

        tree = _build_tree(x, p, grad, logp, 3, 100.0, 1, mm,
                          gaussian_logp_and_grad, H0, 10.0, rng)

        assert tree["divergent"]


class TestNutsStep:
    def test_basic_transition(self):
        """NUTS step should produce a valid state."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([0.5, 0.5])
        logp, grad = gaussian_logp_and_grad(x)

        state = NUTSState(
            position=x, logp=logp, grad=grad,
            step_size=0.1, mass_matrix=mm,
        )
        rng = np.random.default_rng(42)
        new_state = nuts_step(state, gaussian_logp_and_grad, rng)

        assert new_state.iteration == 1
        assert new_state.position.shape == (2,)
        assert np.isfinite(new_state.logp)
        assert new_state.tree_depth >= 0
        assert 0 <= new_state.mean_accept_prob <= 1

    def test_samples_gaussian(self):
        """NUTS should produce samples with correct mean for a Gaussian."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([3.0, -2.0])
        logp, grad = gaussian_logp_and_grad(x)

        state = NUTSState(
            position=x, logp=logp, grad=grad,
            step_size=0.5, mass_matrix=mm,
        )
        rng = np.random.default_rng(42)

        samples = []
        for _ in range(2000):
            state = nuts_step(state, gaussian_logp_and_grad, rng)
            samples.append(state.position.copy())

        samples = np.array(samples)
        mean = np.mean(samples[500:], axis=0)
        # Mean of standard Gaussian should be near 0
        np.testing.assert_allclose(mean, [0, 0], atol=0.15)

    def test_u_turn_limits_depth(self):
        """Tree should stop before max_tree_depth for reasonable problems."""
        mm = MassMatrix(2, MassMatrixType.UNIT)
        x = np.array([0.0, 0.0])
        logp, grad = gaussian_logp_and_grad(x)

        state = NUTSState(
            position=x, logp=logp, grad=grad,
            step_size=0.5, mass_matrix=mm,
        )
        rng = np.random.default_rng(42)

        depths = []
        for _ in range(50):
            state = nuts_step(state, gaussian_logp_and_grad, rng, max_tree_depth=10)
            depths.append(state.tree_depth)

        # Should rarely hit max depth for a well-conditioned Gaussian
        assert np.mean(depths) < 8

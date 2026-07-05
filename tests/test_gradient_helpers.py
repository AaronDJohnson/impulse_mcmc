"""Tests for gradient helper utilities."""

import numpy as np
import pytest

from impulse.nuts.gradient_helpers import (
    compose_logp_and_grad,
    make_logp_and_grad_numerical,
    numerical_gradient,
)


class TestNumericalGradient:
    def test_quadratic(self):
        """Gradient of -0.5 * x^T x is -x."""

        def logp(x):
            return -0.5 * np.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        grad = numerical_gradient(logp, x)
        np.testing.assert_allclose(grad, -x, atol=1e-5)

    def test_linear(self):
        """Gradient of a^T x is a."""
        a = np.array([3.0, -1.0])

        def logp(x):
            return np.dot(a, x)

        x = np.array([0.0, 0.0])
        grad = numerical_gradient(logp, x)
        np.testing.assert_allclose(grad, a, atol=1e-5)

    def test_single_dim(self):
        def logp(x):
            return -x[0] ** 3

        x = np.array([2.0])
        grad = numerical_gradient(logp, x)
        # d/dx(-x^3) = -3x^2 = -12
        np.testing.assert_allclose(grad, [-12.0], atol=1e-4)


class TestMakeLogpAndGradNumerical:
    def test_interface(self):
        def logp(x):
            return -0.5 * np.sum(x**2)

        fn = make_logp_and_grad_numerical(logp)
        x = np.array([1.0, 2.0])
        val, grad = fn(x)
        assert np.isclose(val, -2.5)
        np.testing.assert_allclose(grad, [-1.0, -2.0], atol=1e-5)


class TestComposeLogpAndGrad:
    def test_with_analytical_grads(self):
        def lnlike(x):
            return -0.5 * np.sum(x**2)

        def lnprior(x):
            return 0.0  # flat prior

        def lnlike_grad(x):
            return -x

        def lnprior_grad(x):
            return np.zeros_like(x)

        fn = compose_logp_and_grad(lnlike, lnprior, lnlike_grad, lnprior_grad)
        x = np.array([1.0, 2.0])
        val, grad = fn(x)
        assert np.isclose(val, -2.5)
        np.testing.assert_allclose(grad, [-1.0, -2.0])

    def test_with_numerical_fallback(self):
        """No gradient functions provided -> uses numerical."""

        def lnlike(x):
            return -0.5 * np.sum(x**2)

        def lnprior(x):
            return 0.0

        fn = compose_logp_and_grad(lnlike, lnprior)
        x = np.array([1.0, 2.0])
        val, grad = fn(x)
        assert np.isclose(val, -2.5)
        np.testing.assert_allclose(grad, [-1.0, -2.0], atol=1e-5)

    def test_mixed_gradients(self):
        """One analytical, one numerical."""

        def lnlike(x):
            return -0.5 * np.sum(x**2)

        def lnprior(x):
            return -0.1 * np.sum(x**2)

        def lnlike_grad(x):
            return -x

        fn = compose_logp_and_grad(lnlike, lnprior, lnlike_grad=lnlike_grad)
        x = np.array([1.0, 2.0])
        val, grad = fn(x)
        expected_val = -0.5 * 5 - 0.1 * 5
        assert np.isclose(val, expected_val)
        expected_grad = -x + (-0.2 * x)
        np.testing.assert_allclose(grad, expected_grad, atol=1e-5)

    def test_nonfinite_logp_returns_zero_grad(self):
        def lnlike(x):
            return -0.5 * np.sum(x**2)

        def lnprior(x):
            return -np.inf  # always outside prior

        fn = compose_logp_and_grad(lnlike, lnprior)
        x = np.array([1.0, 2.0])
        val, grad = fn(x)
        assert val == -np.inf
        np.testing.assert_array_equal(grad, [0.0, 0.0])

    def test_matches_analytical(self):
        """Numerical gradient should match analytical to ~1e-5."""

        def lnlike(x):
            return -0.5 * (x[0] ** 2 + 0.5 * x[1] ** 2 + 0.3 * x[0] * x[1])

        def lnprior(x):
            return -0.01 * np.sum(x**2)

        def analytical_grad(x):
            g0 = -(x[0] + 0.15 * x[1]) - 0.02 * x[0]
            g1 = -(0.5 * x[1] + 0.15 * x[0]) - 0.02 * x[1]
            return np.array([g0, g1])

        fn = compose_logp_and_grad(lnlike, lnprior)
        x = np.array([1.5, -0.7])
        val, grad = fn(x)
        expected = analytical_grad(x)
        np.testing.assert_allclose(grad, expected, atol=1e-5)

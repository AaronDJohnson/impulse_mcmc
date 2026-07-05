"""Gradient utilities for NUTS sampler."""

from typing import Callable, Optional

import numpy as np


def numerical_gradient(logp_fn: Callable, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Central-difference gradient approximation.

    Parameters
    ----------
    logp_fn : callable
        Scalar function f(x) -> float.
    x : np.ndarray
        Point at which to evaluate gradient.
    epsilon : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Gradient estimate, same shape as x.
    """
    grad = np.empty_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (logp_fn(x_plus) - logp_fn(x_minus)) / (2 * epsilon)
    return grad


def make_logp_and_grad_numerical(logp_fn: Callable, epsilon: float = 1e-6) -> Callable:
    """Wrap a scalar logp function into (x) -> (logp, grad) interface.

    Parameters
    ----------
    logp_fn : callable
        Scalar function f(x) -> float.
    epsilon : float
        Step size for numerical gradient.

    Returns
    -------
    callable
        Function (x) -> (logp, grad).
    """

    def logp_and_grad(x):
        val = logp_fn(x)
        grad = numerical_gradient(logp_fn, x, epsilon)
        return val, grad

    return logp_and_grad


def compose_logp_and_grad(
    lnlike: Callable,
    lnprior: Callable,
    lnlike_grad: Optional[Callable] = None,
    lnprior_grad: Optional[Callable] = None,
    epsilon: float = 1e-6,
) -> Callable:
    """Combine separate likelihood/prior into single logp_and_grad callable.

    Falls back to numerical differentiation for any missing gradient function.

    Parameters
    ----------
    lnlike : callable
        Log-likelihood function, x -> float.
    lnprior : callable
        Log-prior function, x -> float.
    lnlike_grad : callable, optional
        Gradient of log-likelihood, x -> np.ndarray. If None, uses numerical.
    lnprior_grad : callable, optional
        Gradient of log-prior, x -> np.ndarray. If None, uses numerical.
    epsilon : float
        Step size for numerical gradients.

    Returns
    -------
    callable
        Function (x) -> (logp, grad) where logp = lnlike(x) + lnprior(x).
    """

    def logp_and_grad(x):
        ll = lnlike(x)
        lp = lnprior(x)
        logp = ll + lp

        if not np.isfinite(logp):
            return logp, np.zeros_like(x)

        if lnlike_grad is not None:
            g_like = lnlike_grad(x)
        else:
            g_like = numerical_gradient(lnlike, x, epsilon)

        if lnprior_grad is not None:
            g_prior = lnprior_grad(x)
        else:
            g_prior = numerical_gradient(lnprior, x, epsilon)

        return logp, g_like + g_prior

    return logp_and_grad

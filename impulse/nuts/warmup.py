"""Stan-style three-phase warmup for NUTS.

Implements Nesterov dual averaging for step size adaptation and
a windowed scheme for mass matrix estimation.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from impulse.nuts.core import NUTSState, leapfrog
from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType


@dataclass
class DualAveraging:
    """Nesterov dual averaging for step size adaptation.

    Parameters
    ----------
    target_accept : float
        Target mean acceptance probability.
    gamma : float
        Regularization scale.
    t0 : int
        Stabilization offset.
    kappa : float
        Power for step size schedule.
    initial_step_size : float
        Starting step size.
    """

    target_accept: float = 0.8
    gamma: float = 0.05
    t0: int = 10
    kappa: float = 0.75
    initial_step_size: float = 1.0

    def __post_init__(self) -> None:
        self.log_step = np.log(self.initial_step_size)
        self.log_step_bar = np.log(self.initial_step_size)
        self.mu = np.log(10.0 * self.initial_step_size)
        self.h_bar = 0.0
        self.count = 0

    def update(self, accept_prob: float) -> float:
        """Update step size given observed acceptance probability.

        Parameters
        ----------
        accept_prob : float
            Acceptance probability from last NUTS transition.

        Returns
        -------
        float
            Updated step size.
        """
        self.count += 1
        m = self.count

        # Update running mean of acceptance statistic
        w = 1.0 / (m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target_accept - accept_prob)

        # Primal update
        self.log_step = self.mu - np.sqrt(m) / self.gamma * self.h_bar

        # Dual averaging update
        m_kappa = m ** (-self.kappa)
        self.log_step_bar = m_kappa * self.log_step + (1 - m_kappa) * self.log_step_bar

        return np.exp(self.log_step)

    def finalize(self) -> float:
        """Return the smoothed step size.

        Returns
        -------
        float
            Final adapted step size.
        """
        return np.exp(self.log_step_bar)

    def reset(self, step_size: float) -> None:
        """Reset for a new adaptation window.

        Parameters
        ----------
        step_size : float
            New initial step size.
        """
        self.initial_step_size = step_size
        self.log_step = np.log(step_size)
        self.log_step_bar = np.log(step_size)
        self.mu = np.log(10.0 * step_size)
        self.h_bar = 0.0
        self.count = 0


def regularized_mass_matrix(
    samples: ArrayLike, ndim: int, mass_matrix_type: MassMatrixType
) -> MassMatrix:
    """Estimate a mass matrix from samples with Stan-style regularization.

    Shared by :meth:`WarmupSchedule._adapt_mass_matrix` and the online
    per-model adaptation in :class:`impulse.nuts.adapter.PerModelNUTSAdapter`
    (internal helper — not exported publicly).

    Parameters
    ----------
    samples : list of np.ndarray
        Position samples, each of shape ``(ndim,)``.
    ndim : int
        Dimensionality.
    mass_matrix_type : MassMatrixType
        Desired mass matrix type.

    Returns
    -------
    MassMatrix
        New mass matrix M = (regularized covariance)^{-1}.
    """
    samples = np.array(samples)
    n = len(samples)
    if n < 2:
        return MassMatrix(ndim, MassMatrixType.UNIT)

    sample_cov = np.cov(samples, rowvar=False)
    if sample_cov.ndim == 0:
        sample_cov = sample_cov.reshape(1, 1)

    # Regularization: shrink toward diagonal (Stan's approach)
    shrinkage = 5.0 / (n + 5.0)
    reg_cov = (1 - shrinkage) * sample_cov + shrinkage * np.diag(np.diag(sample_cov) + 1e-3)

    # from_covariance inverts: mass matrix M = reg_cov^{-1} (Stan's
    # inverse metric equals the posterior covariance)
    if mass_matrix_type == MassMatrixType.DIAGONAL:
        return MassMatrix.from_covariance(reg_cov, MassMatrixType.DIAGONAL)
    elif mass_matrix_type == MassMatrixType.DENSE:
        # Add small diagonal for numerical stability
        reg_cov += 1e-8 * np.eye(ndim)
        try:
            return MassMatrix.from_covariance(reg_cov, MassMatrixType.DENSE)
        except np.linalg.LinAlgError:
            # Fall back to diagonal
            return MassMatrix.from_covariance(reg_cov, MassMatrixType.DIAGONAL)
    else:
        return MassMatrix(ndim, MassMatrixType.UNIT)


class WarmupSchedule:
    """Stan-style three-phase warmup with mass matrix adaptation.

    Phase I (init_buffer): step size adaptation only.
    Phase II (doubling windows): mass matrix + step size adaptation.
    Phase III (term_buffer): step size adaptation with fixed mass matrix.

    Parameters
    ----------
    num_warmup : int
        Total warmup iterations.
    ndim : int
        Dimensionality of parameter space.
    mass_matrix_type : MassMatrixType
        Type of mass matrix to adapt.
    target_accept : float
        Target acceptance probability.
    init_buffer : int
        Size of initial step-size-only phase.
    term_buffer : int
        Size of terminal step-size-only phase.
    initial_step_size : float
        Starting step size.
    """

    def __init__(
        self,
        num_warmup: int,
        ndim: int,
        mass_matrix_type: MassMatrixType = MassMatrixType.DIAGONAL,
        target_accept: float = 0.8,
        init_buffer: int = 75,
        term_buffer: int = 50,
        initial_step_size: float = 1.0,
    ) -> None:
        self.num_warmup = num_warmup
        self.ndim = ndim
        self.mass_matrix_type = mass_matrix_type
        self.target_accept = target_accept
        self.init_buffer = min(init_buffer, num_warmup)
        self.term_buffer = min(term_buffer, num_warmup - self.init_buffer)
        self.initial_step_size = initial_step_size

        self.dual_averaging = DualAveraging(
            target_accept=target_accept,
            initial_step_size=initial_step_size,
        )

        self._window_samples: List[np.ndarray] = []
        self._windows = self._compute_windows()
        self._current_window_idx = 0

    def _compute_windows(self) -> List[Tuple[int, int]]:
        """Compute Stan's doubling window schedule.

        Returns list of (start, end) iteration pairs for adaptation windows.
        """
        middle_start = self.init_buffer
        middle_end = self.num_warmup - self.term_buffer

        if middle_end <= middle_start:
            return []

        windows: List[Tuple[int, int]] = []
        window_size = 25
        start = middle_start
        while start < middle_end:
            end = start + window_size
            # If the next window wouldn't fit, expand this one to fill
            if end + window_size > middle_end:
                end = middle_end
            windows.append((start, end))
            start = end
            window_size = (
                min(window_size * 2, middle_end - start) if start < middle_end else window_size
            )
        return windows

    def _adapt_mass_matrix(self, samples: ArrayLike) -> MassMatrix:
        """Estimate mass matrix from window samples with regularization.

        Parameters
        ----------
        samples : list of np.ndarray
            Samples collected during adaptation window.

        Returns
        -------
        MassMatrix
            Updated mass matrix.
        """
        return regularized_mass_matrix(samples, self.ndim, self.mass_matrix_type)

    def _in_window(self, iteration: int) -> Optional[int]:
        """Check if iteration falls within any adaptation window."""
        for i, (start, end) in enumerate(self._windows):
            if start <= iteration < end:
                return i
            if iteration == end:
                return -(i + 1)  # signal: at window boundary
        return None

    def update(
        self, iteration: int, state: NUTSState, accept_prob: float
    ) -> Tuple[float, Optional[MassMatrix]]:
        """Update adaptation state for the given iteration.

        Parameters
        ----------
        iteration : int
            Current warmup iteration (0-indexed).
        state : NUTSState
            Current sampler state.
        accept_prob : float
            Acceptance probability from last transition.

        Returns
        -------
        step_size : float
            Updated step size.
        mass_matrix : MassMatrix or None
            Updated mass matrix, or None if unchanged.
        """
        new_step_size = self.dual_averaging.update(accept_prob)
        new_mass_matrix = None

        # Collect samples during adaptation windows
        window_idx = self._in_window(iteration)
        if window_idx is not None:
            if window_idx >= 0:
                # Inside a window: collect samples
                self._window_samples.append(state.position.copy())
            else:
                # At window boundary: update mass matrix
                self._window_samples.append(state.position.copy())
                new_mass_matrix = self._adapt_mass_matrix(self._window_samples)
                self._window_samples = []
                # Reset dual averaging with current step size
                self.dual_averaging.reset(np.exp(self.dual_averaging.log_step))

        return new_step_size, new_mass_matrix

    def finalize(self) -> float:
        """Finalize warmup: return smoothed step size.

        Returns
        -------
        float
            Final adapted step size.
        """
        return self.dual_averaging.finalize()


def find_reasonable_step_size(
    position: np.ndarray,
    logp: float,
    grad: np.ndarray,
    logp_and_grad: Callable,
    mass_matrix: MassMatrix,
    rng: np.random.Generator,
) -> float:
    """Stan's heuristic to find a reasonable initial step size.

    Doubles or halves the step size until the acceptance probability
    crosses 0.5.

    Parameters
    ----------
    position : np.ndarray
        Current position.
    logp : float
        Log-probability at position.
    grad : np.ndarray
        Gradient at position.
    logp_and_grad : callable
        Function (x) -> (logp, grad).
    mass_matrix : MassMatrix
        Current mass matrix.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    float
        Reasonable initial step size.
    """
    step_size = 1.0

    # Sample momentum
    momentum = mass_matrix.sample_momentum(rng)

    # Initial Hamiltonian
    H0 = -logp + mass_matrix.kinetic_energy(momentum)

    # Trial leapfrog
    new_pos, new_mom, new_logp, new_grad = leapfrog(
        position, momentum, grad, step_size, mass_matrix, logp_and_grad
    )
    H_new = -new_logp + mass_matrix.kinetic_energy(new_mom)

    delta_H = H_new - H0
    if not np.isfinite(delta_H):
        delta_H = np.inf

    # Decide direction: if accept prob > 0.5, make step bigger; else smaller
    direction = 1 if -delta_H > np.log(0.5) else -1

    for _ in range(100):  # safety limit
        if direction == 1:
            step_size *= 2.0
        else:
            step_size /= 2.0

        new_pos, new_mom, new_logp, new_grad = leapfrog(
            position, momentum, grad, step_size, mass_matrix, logp_and_grad
        )
        H_new = -new_logp + mass_matrix.kinetic_energy(new_mom)
        delta_H = H_new - H0
        if not np.isfinite(delta_H):
            delta_H = np.inf

        # Stop when acceptance crosses 0.5
        if direction == 1 and -delta_H <= np.log(0.5):
            break
        if direction == -1 and -delta_H >= np.log(0.5):
            break

        if step_size > 1e7 or step_size < 1e-15:
            break

    return step_size

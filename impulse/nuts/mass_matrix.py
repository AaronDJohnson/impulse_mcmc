"""Mass matrix representations for Hamiltonian Monte Carlo."""

from enum import Enum
import numpy as np


class MassMatrixType(Enum):
    UNIT = "unit"
    DIAGONAL = "diagonal"
    DENSE = "dense"


class MassMatrix:
    """Mass matrix M for HMC momentum: p ~ N(0, M).

    Stores precomputed factorizations for efficient sampling and
    kinetic energy evaluation. Fully picklable (numpy arrays + enum only).

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    matrix_type : MassMatrixType
        Type of mass matrix representation.
    diagonal : np.ndarray, optional
        Diagonal entries of M for DIAGONAL type.
    dense : np.ndarray, optional
        Full matrix M for DENSE type.
    """

    def __init__(self, ndim, matrix_type=MassMatrixType.UNIT,
                 diagonal=None, dense=None):
        self.ndim = ndim
        self.matrix_type = matrix_type

        if matrix_type == MassMatrixType.UNIT:
            self._inv_diag = np.ones(ndim)
            self._sqrt_diag = np.ones(ndim)
        elif matrix_type == MassMatrixType.DIAGONAL:
            if diagonal is None:
                raise ValueError("diagonal required for DIAGONAL type")
            self._inv_diag = 1.0 / diagonal
            self._sqrt_diag = np.sqrt(diagonal)
        elif matrix_type == MassMatrixType.DENSE:
            if dense is None:
                raise ValueError("dense required for DENSE type")
            self._dense = dense
            self._cholesky = np.linalg.cholesky(dense)
            self._inv = np.linalg.inv(dense)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

    def sample_momentum(self, rng):
        """Draw p ~ N(0, M).

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Momentum vector of shape (ndim,).
        """
        z = rng.standard_normal(self.ndim)
        if self.matrix_type == MassMatrixType.DENSE:
            return self._cholesky @ z
        return self._sqrt_diag * z

    def kinetic_energy(self, p):
        """Compute 0.5 * p^T M^{-1} p.

        Parameters
        ----------
        p : np.ndarray
            Momentum vector.

        Returns
        -------
        float
            Kinetic energy.
        """
        if self.matrix_type == MassMatrixType.DENSE:
            return 0.5 * p @ self._inv @ p
        return 0.5 * np.sum(self._inv_diag * p ** 2)

    def inverse_multiply(self, p):
        """Compute M^{-1} p (velocity).

        Parameters
        ----------
        p : np.ndarray
            Momentum vector.

        Returns
        -------
        np.ndarray
            Velocity vector M^{-1} p.
        """
        if self.matrix_type == MassMatrixType.DENSE:
            return self._inv @ p
        return self._inv_diag * p

    @classmethod
    def from_covariance(cls, cov, matrix_type=MassMatrixType.DIAGONAL):
        """Build mass matrix from sample covariance.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix estimate, shape (ndim, ndim).
        matrix_type : MassMatrixType
            Desired representation type.

        Returns
        -------
        MassMatrix
            Mass matrix set to the covariance estimate.
        """
        ndim = cov.shape[0]
        if matrix_type == MassMatrixType.UNIT:
            return cls(ndim, MassMatrixType.UNIT)
        elif matrix_type == MassMatrixType.DIAGONAL:
            return cls(ndim, MassMatrixType.DIAGONAL, diagonal=np.diag(cov))
        elif matrix_type == MassMatrixType.DENSE:
            return cls(ndim, MassMatrixType.DENSE, dense=cov)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

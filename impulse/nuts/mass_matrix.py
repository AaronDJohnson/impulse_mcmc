"""Mass matrix representations for Hamiltonian Monte Carlo."""

from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


class MassMatrixType(Enum):
    UNIT = "unit"
    DIAGONAL = "diagonal"
    DENSE = "dense"


class MassMatrix:
    """Mass matrix M for HMC momentum: p ~ N(0, M).

    Convention: momenta are drawn p ~ N(0, M) and the leapfrog velocity is
    M^{-1} p, so M should be set to the inverse of the target covariance
    (use :meth:`from_covariance` to build M from a covariance estimate,
    or :meth:`from_precision` to inject a Fisher/precision matrix as M).

    Stores precomputed factorizations for efficient sampling and
    kinetic energy evaluation. Fully picklable (numpy arrays + enum only).
    All array arguments (``diagonal``, ``dense``, ``inverse``) are copied
    defensively on construction, so callers may freely mutate their
    buffers afterwards without corrupting the mass matrix.

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
    inverse : np.ndarray, optional
        Keyword-only. Known inverse M^{-1} of ``dense``, supplied when the
        caller already holds it (e.g. :meth:`from_covariance` inverts the
        covariance to get M, so the covariance itself is M^{-1}). Skips a
        second O(ndim^3) inversion whose error grows like cond(M)^2.
        Ignored unless ``matrix_type`` is DENSE.
    """

    def __init__(
        self,
        ndim: int,
        matrix_type: MassMatrixType = MassMatrixType.UNIT,
        diagonal: Optional[ArrayLike] = None,
        dense: Optional[ArrayLike] = None,
        *,
        inverse: Optional[ArrayLike] = None,
    ) -> None:
        self.ndim = ndim
        self.matrix_type = matrix_type

        if matrix_type == MassMatrixType.UNIT:
            self._inv_diag = np.ones(ndim)
            self._sqrt_diag = np.ones(ndim)
        elif matrix_type == MassMatrixType.DIAGONAL:
            if diagonal is None:
                raise ValueError("diagonal required for DIAGONAL type")
            # Copy: never retain (or derive state from) a live view of the
            # caller's buffer — a caller mutating its array in place (e.g.
            # a rolling adaptation estimate) must not corrupt the matrix.
            diagonal = np.array(diagonal, dtype=float, copy=True)
            self._inv_diag = 1.0 / diagonal
            self._sqrt_diag = np.sqrt(diagonal)
        elif matrix_type == MassMatrixType.DENSE:
            if dense is None:
                raise ValueError("dense required for DENSE type")
            # Copy: self._dense / self._inv retain these arrays, and storing
            # the caller's buffer by reference would let an in-place caller
            # mutation silently corrupt the mass matrix (see DIAGONAL).
            dense = np.array(dense, dtype=float, copy=True)
            self._dense = dense
            self._cholesky = np.linalg.cholesky(dense)
            self._inv = (
                np.linalg.inv(dense)
                if inverse is None
                else np.array(inverse, dtype=float, copy=True)
            )
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

    def sample_momentum(self, rng: np.random.Generator) -> np.ndarray:
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

    def kinetic_energy(self, p: np.ndarray) -> float:
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
        return 0.5 * np.sum(self._inv_diag * p**2)

    def inverse_multiply(self, p: np.ndarray) -> np.ndarray:
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

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        """Serialize to ``(arrays, meta)`` for the no-code-execution checkpoint.

        Stores the exact internal factorization (never a recomputed
        approximation) so a restored matrix reproduces momentum sampling and
        kinetic-energy evaluation bit-for-bit.  ``arrays`` maps local keys to
        ``np.ndarray``; ``meta`` holds JSON-serializable scalars.
        """
        meta = {"ndim": int(self.ndim), "matrix_type": self.matrix_type.value}
        arrays: dict = {}
        if self.matrix_type == MassMatrixType.DENSE:
            arrays["dense"] = np.asarray(self._dense)
            arrays["cholesky"] = np.asarray(self._cholesky)
            arrays["inv"] = np.asarray(self._inv)
        else:
            # UNIT and DIAGONAL both store the two diagonal factors.
            arrays["inv_diag"] = np.asarray(self._inv_diag)
            arrays["sqrt_diag"] = np.asarray(self._sqrt_diag)
        return arrays, meta

    @classmethod
    def from_checkpoint_state(cls, arrays: dict, meta: dict) -> "MassMatrix":
        """Rebuild a :class:`MassMatrix` from :meth:`get_checkpoint_state` output.

        Bypasses ``__init__`` (which would recompute the factorization and
        could differ in the last bit) and installs the stored factors
        directly.
        """
        obj = cls.__new__(cls)
        obj.ndim = int(meta["ndim"])
        obj.matrix_type = MassMatrixType(meta["matrix_type"])
        if obj.matrix_type == MassMatrixType.DENSE:
            obj._dense = np.array(arrays["dense"], dtype=float)
            obj._cholesky = np.array(arrays["cholesky"], dtype=float)
            obj._inv = np.array(arrays["inv"], dtype=float)
        else:
            obj._inv_diag = np.array(arrays["inv_diag"], dtype=float)
            obj._sqrt_diag = np.array(arrays["sqrt_diag"], dtype=float)
        return obj

    @classmethod
    def from_covariance(
        cls, cov: ArrayLike, matrix_type: MassMatrixType = MassMatrixType.DIAGONAL
    ) -> "MassMatrix":
        """Build mass matrix from a posterior covariance estimate.

        Follows Stan's convention: the inverse metric equals the posterior
        covariance, i.e. the stored mass matrix is M = Sigma^{-1}. Momenta
        are drawn p ~ N(0, M) and velocities are M^{-1} p, so position
        updates scale with the target covariance.

        .. versionchanged:: BREAKING
            ``from_covariance`` now INVERTS its argument (M = Sigma^{-1});
            it previously stored the covariance as M directly. Callers who
            passed a Fisher information / precision matrix here to inject
            it as the mass matrix now silently get the inverse of what
            they want — use :meth:`from_precision` (or the raw
            constructor) for Fisher/precision injection instead.

        Parameters
        ----------
        cov : np.ndarray
            Posterior covariance matrix estimate Sigma, shape (ndim, ndim).
            Should already be regularized/positive definite.
        matrix_type : MassMatrixType
            Desired representation type.

        Returns
        -------
        MassMatrix
            Mass matrix M = Sigma^{-1}.
        """
        # __init__ copies its array arguments defensively (including the
        # inverse=cov pass-through below); asarray here only normalizes
        # sequence input for the shape/diag/inv computations.
        cov = np.asarray(cov, dtype=float)
        ndim = cov.shape[0]
        if matrix_type == MassMatrixType.UNIT:
            return cls(ndim, MassMatrixType.UNIT)
        elif matrix_type == MassMatrixType.DIAGONAL:
            diagonal = 1.0 / np.maximum(np.diag(cov), 1e-10)
            return cls(ndim, MassMatrixType.DIAGONAL, diagonal=diagonal)
        elif matrix_type == MassMatrixType.DENSE:
            inv_cov = np.linalg.inv(cov)
            # Symmetrize so the Cholesky factorization in __init__ is stable
            inv_cov = 0.5 * (inv_cov + inv_cov.T)
            # cov IS the inverse metric; pass it through instead of letting
            # __init__ invert inv_cov back (double inversion amplifies error
            # ~cond(cov)^2 and costs a second O(ndim^3) factorization)
            return cls(ndim, MassMatrixType.DENSE, dense=inv_cov, inverse=cov)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

    @classmethod
    def from_precision(
        cls, precision: ArrayLike, matrix_type: MassMatrixType = MassMatrixType.DIAGONAL
    ) -> "MassMatrix":
        """Build mass matrix directly from a precision (inverse-covariance) matrix.

        Stores M = precision without any inversion. This is the correct
        entry point for injecting a Fisher information matrix: the Fisher
        matrix approximates the posterior *precision*, which under the
        Stan convention (inverse metric = posterior covariance) is exactly
        the mass matrix.

        Parameters
        ----------
        precision : np.ndarray
            Precision matrix estimate Sigma^{-1} (e.g. a Fisher matrix),
            shape (ndim, ndim). Should be positive definite.
        matrix_type : MassMatrixType
            Desired representation type.

        Returns
        -------
        MassMatrix
            Mass matrix M = precision.
        """
        # __init__ copies its array arguments defensively (including the
        # dense=precision pass-through below); asarray here only normalizes
        # sequence input for the shape/diag/inv computations.
        precision = np.asarray(precision, dtype=float)
        ndim = precision.shape[0]
        if matrix_type == MassMatrixType.UNIT:
            return cls(ndim, MassMatrixType.UNIT)
        elif matrix_type == MassMatrixType.DIAGONAL:
            diagonal = np.maximum(np.diag(precision), 1e-10)
            return cls(ndim, MassMatrixType.DIAGONAL, diagonal=diagonal)
        elif matrix_type == MassMatrixType.DENSE:
            # The inverse metric is inv(precision); computing it here is the
            # same single inversion __init__ would do, just relocated, with
            # a symmetrization for numerical hygiene.
            inv_prec = np.linalg.inv(precision)
            inv_prec = 0.5 * (inv_prec + inv_prec.T)
            return cls(ndim, MassMatrixType.DENSE, dense=precision, inverse=inv_prec)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

"""Tests for mass matrix representations."""

import pickle
import numpy as np
import pytest

from impulse.nuts.mass_matrix import MassMatrix, MassMatrixType


class TestMassMatrixUnit:
    def test_sample_momentum_shape(self):
        mm = MassMatrix(5, MassMatrixType.UNIT)
        rng = np.random.default_rng(42)
        p = mm.sample_momentum(rng)
        assert p.shape == (5,)

    def test_kinetic_energy(self):
        mm = MassMatrix(3, MassMatrixType.UNIT)
        p = np.array([1.0, 2.0, 3.0])
        ke = mm.kinetic_energy(p)
        assert np.isclose(ke, 0.5 * (1 + 4 + 9))

    def test_inverse_multiply_identity(self):
        mm = MassMatrix(3, MassMatrixType.UNIT)
        p = np.array([1.0, 2.0, 3.0])
        v = mm.inverse_multiply(p)
        np.testing.assert_array_equal(v, p)

    def test_pickle(self):
        mm = MassMatrix(4, MassMatrixType.UNIT)
        data = pickle.dumps(mm)
        mm2 = pickle.loads(data)
        assert mm2.ndim == 4
        assert mm2.matrix_type == MassMatrixType.UNIT


class TestMassMatrixDiagonal:
    def test_kinetic_energy(self):
        diag = np.array([2.0, 4.0])
        mm = MassMatrix(2, MassMatrixType.DIAGONAL, diagonal=diag)
        p = np.array([1.0, 1.0])
        # KE = 0.5 * (1/2 * 1 + 1/4 * 1) = 0.5 * 0.75 = 0.375
        ke = mm.kinetic_energy(p)
        assert np.isclose(ke, 0.375)

    def test_inverse_multiply(self):
        diag = np.array([2.0, 4.0])
        mm = MassMatrix(2, MassMatrixType.DIAGONAL, diagonal=diag)
        p = np.array([2.0, 4.0])
        v = mm.inverse_multiply(p)
        np.testing.assert_array_almost_equal(v, [1.0, 1.0])

    def test_sample_momentum_variance(self):
        """Momentum samples should have variance = diagonal entries."""
        diag = np.array([4.0, 9.0])
        mm = MassMatrix(2, MassMatrixType.DIAGONAL, diagonal=diag)
        rng = np.random.default_rng(42)
        samples = np.array([mm.sample_momentum(rng) for _ in range(10000)])
        np.testing.assert_allclose(np.var(samples, axis=0), diag, rtol=0.1)

    def test_pickle(self):
        diag = np.array([1.0, 2.0, 3.0])
        mm = MassMatrix(3, MassMatrixType.DIAGONAL, diagonal=diag)
        data = pickle.dumps(mm)
        mm2 = pickle.loads(data)
        assert mm2.matrix_type == MassMatrixType.DIAGONAL
        np.testing.assert_array_equal(mm2._inv_diag, 1.0 / diag)


class TestMassMatrixDense:
    def test_kinetic_energy(self):
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=M)
        p = np.array([1.0, 0.0])
        # KE = 0.5 * p^T M^{-1} p
        M_inv = np.linalg.inv(M)
        expected = 0.5 * p @ M_inv @ p
        assert np.isclose(mm.kinetic_energy(p), expected)

    def test_inverse_multiply(self):
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=M)
        p = np.array([1.0, 1.0])
        M_inv = np.linalg.inv(M)
        expected = M_inv @ p
        np.testing.assert_array_almost_equal(mm.inverse_multiply(p), expected)

    def test_sample_momentum_covariance(self):
        """Momentum samples should have covariance = M."""
        M = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=M)
        rng = np.random.default_rng(42)
        samples = np.array([mm.sample_momentum(rng) for _ in range(20000)])
        sample_cov = np.cov(samples, rowvar=False)
        np.testing.assert_allclose(sample_cov, M, atol=0.15)

    def test_pickle(self):
        M = np.eye(3)
        mm = MassMatrix(3, MassMatrixType.DENSE, dense=M)
        data = pickle.dumps(mm)
        mm2 = pickle.loads(data)
        assert mm2.matrix_type == MassMatrixType.DENSE
        np.testing.assert_array_equal(mm2._dense, M)


class TestFromCovariance:
    def test_diagonal(self):
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        assert mm.matrix_type == MassMatrixType.DIAGONAL
        np.testing.assert_array_almost_equal(mm._sqrt_diag, np.sqrt([4.0, 2.0]))

    def test_dense(self):
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        assert mm.matrix_type == MassMatrixType.DENSE
        np.testing.assert_array_almost_equal(mm._dense, cov)

    def test_unit(self):
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.UNIT)
        assert mm.matrix_type == MassMatrixType.UNIT

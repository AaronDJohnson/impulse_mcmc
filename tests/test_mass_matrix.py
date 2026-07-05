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
        """M = Sigma^{-1}: diagonal entries are the reciprocal variances."""
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        assert mm.matrix_type == MassMatrixType.DIAGONAL
        np.testing.assert_array_almost_equal(mm._sqrt_diag, np.sqrt([0.25, 0.5]))
        np.testing.assert_array_almost_equal(mm._inv_diag, [4.0, 2.0])

    def test_dense(self):
        """M = Sigma^{-1}: stored dense matrix is the inverse covariance."""
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        assert mm.matrix_type == MassMatrixType.DENSE
        np.testing.assert_array_almost_equal(mm._dense, np.linalg.inv(cov))

    def test_unit(self):
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.UNIT)
        assert mm.matrix_type == MassMatrixType.UNIT


class TestFromCovarianceConvention:
    """Regression tests: from_covariance must invert (M = Sigma^{-1}).

    Before the fix the sample covariance was stored as M directly, making
    adaptation actively harmful on anisotropic targets.
    """

    def test_anisotropic_diagonal_inverted(self):
        cov = np.diag([100.0, 1.0])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        # M = diag([0.01, 1]); _inv_diag holds M^{-1} = diag(cov)
        np.testing.assert_allclose(1.0 / mm._inv_diag, [0.01, 1.0])

    def test_momentum_covariance_matches_M(self):
        cov = np.diag([100.0, 1.0])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        rng = np.random.default_rng(42)
        samples = np.array([mm.sample_momentum(rng) for _ in range(20000)])
        np.testing.assert_allclose(np.var(samples, axis=0), [0.01, 1.0], rtol=0.1)

    def test_velocity_scales_with_target_covariance(self):
        cov = np.diag([100.0, 1.0])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        p = np.array([1.0, 1.0])
        # velocity = M^{-1} p = Sigma p: wide dimensions move faster
        np.testing.assert_allclose(mm.inverse_multiply(p), [100.0, 1.0])

    def test_dense_velocity_scales_with_target_covariance(self):
        cov = np.array([[100.0, 5.0], [5.0, 1.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        p = np.array([1.0, 1.0])
        np.testing.assert_allclose(mm.inverse_multiply(p), cov @ p, rtol=1e-8)


class TestFromPrecision:
    """from_precision stores M = precision directly (Fisher injection)."""

    def test_diagonal(self):
        prec = np.diag([4.0, 0.25])
        mm = MassMatrix.from_precision(prec, MassMatrixType.DIAGONAL)
        assert mm.matrix_type == MassMatrixType.DIAGONAL
        # M = diag(precision): momentum scale sqrt(diag), velocity 1/diag
        np.testing.assert_array_almost_equal(mm._sqrt_diag, np.sqrt([4.0, 0.25]))
        np.testing.assert_array_almost_equal(mm._inv_diag, [0.25, 4.0])

    def test_dense_stores_precision_as_M(self):
        prec = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_precision(prec, MassMatrixType.DENSE)
        assert mm.matrix_type == MassMatrixType.DENSE
        np.testing.assert_array_equal(mm._dense, prec)
        np.testing.assert_allclose(mm._inv, np.linalg.inv(prec), rtol=1e-12)

    def test_unit(self):
        prec = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_precision(prec, MassMatrixType.UNIT)
        assert mm.matrix_type == MassMatrixType.UNIT

    def test_fisher_roundtrip_with_from_covariance(self):
        """from_precision(F) equals from_covariance(inv(F)) up to roundoff.

        A caller who used from_covariance(fisher) under the OLD identity
        semantics must migrate to from_precision(fisher) to keep M = F.
        """
        fisher = np.array([[10.0, 2.0], [2.0, 1.0]])
        mm_prec = MassMatrix.from_precision(fisher, MassMatrixType.DENSE)
        mm_cov = MassMatrix.from_covariance(np.linalg.inv(fisher), MassMatrixType.DENSE)
        np.testing.assert_allclose(mm_prec._dense, mm_cov._dense, rtol=1e-10)
        # and from_covariance(fisher) is now the WRONG call for injection:
        mm_wrong = MassMatrix.from_covariance(fisher, MassMatrixType.DENSE)
        assert not np.allclose(mm_wrong._dense, fisher)


class TestDenseInversePath:
    """Regression: the dense paths must not invert twice (Issue: error
    amplification ~cond^2 and a redundant O(n^3) factorization)."""

    def test_from_covariance_inverse_is_exactly_cov(self):
        """The known inverse metric (the covariance) is passed through
        verbatim rather than recovered via inv(inv(cov))."""
        cov = np.array([[100.0, 5.0], [5.0, 1.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        np.testing.assert_array_equal(mm._inv, cov)

    def test_ill_conditioned_covariance_roundtrip(self):
        """With cond(cov) ~ 1e12, inv(inv(cov)) loses most digits; the
        pass-through inverse keeps the metric exact."""
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        cov = q @ np.diag([1e6, 1.0, 1e-3, 1e-6]) @ q.T
        cov = 0.5 * (cov + cov.T)
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        np.testing.assert_array_equal(mm._inv, cov)

    def test_explicit_inverse_keyword(self):
        """The supplied inverse is used verbatim (a defensive copy of it,
        not a recomputed inv(dense)): a deliberately scaled marker inverse
        must show up unchanged in the stored metric."""
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        M_inv_marker = 2.0 * np.linalg.inv(M)  # NOT inv(M): pass-through marker
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=M, inverse=M_inv_marker)
        assert mm._inv is not M_inv_marker  # __init__ copies defensively
        np.testing.assert_array_equal(mm._inv, M_inv_marker)
        np.testing.assert_allclose(mm.inverse_multiply(np.ones(2)), M_inv_marker @ np.ones(2))

    def test_inverse_keyword_is_keyword_only(self):
        M = np.eye(2)
        with pytest.raises(TypeError):
            MassMatrix(2, MassMatrixType.DENSE, None, M, np.eye(2))

    def test_pickle_roundtrip_new_and_legacy(self):
        """New-style objects pickle fine, and objects lacking any new state
        (i.e. old checkpointed MassMatrix payloads, which pickle via plain
        __dict__ with no __setstate__) still unpickle and work."""
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        mm2 = pickle.loads(pickle.dumps(mm))
        np.testing.assert_array_equal(mm2._inv, cov)
        np.testing.assert_array_equal(mm2._dense, mm._dense)

        # Simulate a legacy checkpointed object: default pickling restores
        # __dict__ directly without calling __init__.
        legacy = MassMatrix.__new__(MassMatrix)
        legacy.__dict__ = {
            "ndim": 2,
            "matrix_type": MassMatrixType.DENSE,
            "_dense": np.linalg.inv(cov),
            "_cholesky": np.linalg.cholesky(np.linalg.inv(cov)),
            "_inv": cov,
        }
        legacy2 = pickle.loads(pickle.dumps(legacy))
        np.testing.assert_allclose(legacy2.inverse_multiply(np.ones(2)), cov @ np.ones(2))
        assert np.isfinite(legacy2.kinetic_energy(np.ones(2)))


class TestNoAliasingOfCallerArrays:
    """No construction path may retain the caller's array by reference:
    mutating the input afterwards must not change the mass matrix
    (regression for the DENSE branches, which previously stored the
    np.asarray view of the caller's buffer — the defensive copies now
    live in __init__ itself, so the raw constructor is covered too)."""

    def test_from_covariance_dense_copies_input(self):
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DENSE)
        ke_before = mm.kinetic_energy(np.ones(2))
        vel_before = mm.inverse_multiply(np.ones(2))

        cov *= 100.0  # caller mutates its covariance in place

        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)
        np.testing.assert_allclose(mm.inverse_multiply(np.ones(2)), vel_before)

    def test_from_precision_dense_copies_input(self):
        prec = np.array([[2.0, 0.5], [0.5, 1.0]])
        mm = MassMatrix.from_precision(prec, MassMatrixType.DENSE)
        ke_before = mm.kinetic_energy(np.ones(2))
        vel_before = mm.inverse_multiply(np.ones(2))
        rng = np.random.default_rng(0)
        p_before = mm.sample_momentum(rng)

        prec *= 100.0  # caller mutates its precision in place

        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)
        np.testing.assert_allclose(mm.inverse_multiply(np.ones(2)), vel_before)
        rng = np.random.default_rng(0)
        np.testing.assert_allclose(mm.sample_momentum(rng), p_before)

    def test_from_covariance_diagonal_copies_input(self):
        cov = np.diag([4.0, 9.0])
        mm = MassMatrix.from_covariance(cov, MassMatrixType.DIAGONAL)
        ke_before = mm.kinetic_energy(np.ones(2))
        cov *= 100.0
        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)

    def test_raw_constructor_dense_copies_inputs(self):
        """The raw constructor stores dense and inverse; both must be
        copies, not views of the caller's buffers."""
        dense = np.array([[2.0, 0.5], [0.5, 1.0]])
        inverse = np.linalg.inv(dense)
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=dense, inverse=inverse)
        ke_before = mm.kinetic_energy(np.ones(2))
        vel_before = mm.inverse_multiply(np.ones(2))
        rng = np.random.default_rng(0)
        p_before = mm.sample_momentum(rng)

        dense *= 100.0  # caller mutates its buffers in place
        inverse *= 100.0

        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)
        np.testing.assert_allclose(mm.inverse_multiply(np.ones(2)), vel_before)
        rng = np.random.default_rng(0)
        np.testing.assert_allclose(mm.sample_momentum(rng), p_before)

    def test_raw_constructor_dense_computed_inverse_unaffected(self):
        """Without an explicit inverse, the stored M must be a copy:
        mutating the caller's dense array must change neither the stored
        matrix nor the derived quantities."""
        dense = np.array([[4.0, 1.0], [1.0, 2.0]])
        dense_before = dense.copy()
        mm = MassMatrix(2, MassMatrixType.DENSE, dense=dense)
        ke_before = mm.kinetic_energy(np.ones(2))
        dense *= 100.0
        np.testing.assert_array_equal(mm._dense, dense_before)
        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)

    def test_raw_constructor_diagonal_copies_input(self):
        diagonal = np.array([4.0, 9.0])
        mm = MassMatrix(2, MassMatrixType.DIAGONAL, diagonal=diagonal)
        ke_before = mm.kinetic_energy(np.ones(2))
        rng = np.random.default_rng(0)
        p_before = mm.sample_momentum(rng)

        diagonal *= 100.0

        assert mm.kinetic_energy(np.ones(2)) == pytest.approx(ke_before)
        rng = np.random.default_rng(0)
        np.testing.assert_allclose(mm.sample_momentum(rng), p_before)

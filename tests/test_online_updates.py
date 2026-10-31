import numpy as np
import pytest

from impulse.chain_stats import svd_groups


class TestSvdGroupsRidge:
    """The ridge that stops the adaptive proposal becoming an absorbing state.

    am/scam take their whole step scale from proposal_L. A singular covariance
    therefore gives proposal_L == 0 in some direction, after which no default
    proposal can move that coordinate ever again -- the chain reports a point
    mass with zero variance while its acceptance rate looks perfect.
    """

    def test_singular_covariance_gives_zero_proposal_without_ridge(self):
        """Precondition: this is the failure mode the ridge exists to prevent."""
        cov = np.zeros((2, 2))  # what a never-moving chain's history produces
        _, _, L = svd_groups([None], [None], [np.array([0, 1])], cov)
        assert np.all(L[0] == 0.0)

    def test_ridge_keeps_proposal_non_degenerate(self):
        cov = np.zeros((2, 2))
        _, _, L = svd_groups([None], [None], [np.array([0, 1])], cov, ridge=1e-10)
        assert not np.all(L[0] == 0.0)
        # the step scale is sqrt(ridge) in every direction
        np.testing.assert_allclose(np.abs(L[0]).max(), np.sqrt(1e-10), rtol=1e-6)

    def test_ridge_is_negligible_for_a_healthy_covariance(self):
        """A well-conditioned covariance must be essentially unchanged."""
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        _, _, plain = svd_groups([None], [None], [np.array([0, 1])], cov)
        _, _, ridged = svd_groups([None], [None], [np.array([0, 1])], cov, ridge=1e-10)
        np.testing.assert_allclose(ridged[0], plain[0], atol=1e-8)

    def test_ridge_rescues_a_partially_collapsed_covariance(self):
        """One dead direction must not zero out that column of L."""
        cov = np.diag([1.0, 0.0])  # dim 1 collapsed, dim 0 healthy
        _, _, L = svd_groups([None], [None], [np.array([0, 1])], cov, ridge=1e-10)
        # every direction retains some scale
        assert np.linalg.matrix_rank(L[0], tol=1e-12) == 2

    def test_negative_ridge_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            svd_groups([None], [None], [np.array([0, 1])], np.eye(2), ridge=-1.0)


class TestSvdGroups:
    """Test suite for svd_groups function"""

    def test_svd_groups_single_group(self):
        """Test eigh decomposition with single parameter group"""
        svd_U = [None]
        svd_S = [None]
        groups = [np.array([0, 1])]
        sample_cov = np.array([[2.0, 0.5], [0.5, 1.0]])

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        assert len(updated_U) == 1
        assert len(updated_S) == 1
        assert len(updated_L) == 1
        assert updated_U[0].shape == (2, 2)
        assert updated_S[0].shape == (2,)
        assert updated_L[0].shape == (2, 2)

        # Eigenvalues should match SVD singular values for symmetric PSD matrix
        U, s, Vt = np.linalg.svd(sample_cov)
        np.testing.assert_array_almost_equal(np.sort(updated_S[0])[::-1], np.sort(s)[::-1])

        # L @ L.T should reconstruct the covariance
        np.testing.assert_array_almost_equal(updated_L[0] @ updated_L[0].T, sample_cov)

    def test_svd_groups_multiple_groups(self):
        """Test eigh decomposition with multiple parameter groups"""
        svd_U = [None, None]
        svd_S = [None, None]
        groups = [np.array([0, 1]), np.array([2, 3])]
        sample_cov = np.eye(4)

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        assert len(updated_U) == 2
        assert len(updated_S) == 2
        assert len(updated_L) == 2

        for i in range(2):
            assert updated_U[i].shape == (2, 2)
            assert updated_S[i].shape == (2,)
            assert updated_L[i].shape == (2, 2)
            # For identity submatrices, eigenvalues should be 1
            np.testing.assert_array_almost_equal(updated_S[i], np.ones(2))
            # L should equal U for identity covariance (sqrt(1)=1)
            np.testing.assert_array_almost_equal(np.abs(updated_L[i]), np.abs(updated_U[i]))

    def test_svd_groups_single_parameter_groups(self):
        """Test eigh with single-parameter groups"""
        svd_U = [None, None]
        svd_S = [None, None]
        groups = [np.array([0]), np.array([1])]
        sample_cov = np.array([[4.0, 1.0], [1.0, 2.0]])

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        assert len(updated_U) == 2
        assert len(updated_S) == 2
        assert len(updated_L) == 2

        # Single parameter groups should have 1x1 U matrices and 1-element S vectors
        assert updated_U[0].shape == (1, 1)
        assert updated_U[1].shape == (1, 1)
        assert updated_S[0].shape == (1,)
        assert updated_S[1].shape == (1,)

        # Eigenvalues should match diagonal elements
        np.testing.assert_array_almost_equal(updated_S[0], [4.0])
        np.testing.assert_array_almost_equal(updated_S[1], [2.0])

        # L[:, j] should equal U[:, j] * sqrt(S[j])
        np.testing.assert_array_almost_equal(updated_L[0], [[2.0]])  # sqrt(4)
        np.testing.assert_array_almost_equal(updated_L[1], [[np.sqrt(2.0)]])

    def test_svd_groups_overlapping_indices(self):
        """Test eigh with realistic covariance matrix"""
        svd_U = [None]
        svd_S = [None]
        groups = [np.array([0, 1, 2])]

        # Create a realistic covariance matrix
        np.random.seed(42)
        A = np.random.randn(3, 3)
        sample_cov = A @ A.T  # Ensure positive definite

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        # Verify decomposition properties
        U = updated_U[0]
        s = updated_S[0]
        L = updated_L[0]

        # U should be orthogonal
        np.testing.assert_array_almost_equal(U @ U.T, np.eye(3), decimal=10)

        # Eigenvalues should be positive
        assert np.all(s > 0)

        # Reconstruction check: U @ diag(S) @ U.T == cov
        reconstructed = U @ np.diag(s) @ U.T
        np.testing.assert_array_almost_equal(reconstructed, sample_cov, decimal=10)

        # L @ L.T should also reconstruct the covariance
        np.testing.assert_array_almost_equal(L @ L.T, sample_cov, decimal=10)

        # L[:, j] == U[:, j] * sqrt(S[j])
        for j in range(3):
            np.testing.assert_array_almost_equal(L[:, j], U[:, j] * np.sqrt(s[j]))

    def test_svd_groups_preserves_list_length(self):
        """Test that decomposition preserves input list lengths"""
        original_U = [None, None, None]
        original_S = [None, None, None]
        original_L = [None, None, None]
        groups = [np.array([0]), np.array([1]), np.array([2])]
        sample_cov = np.eye(3)

        updated_U, updated_S, updated_L = svd_groups(
            original_U, original_S, groups, sample_cov, original_L
        )

        assert len(updated_U) == len(original_U)
        assert len(updated_S) == len(original_S)
        assert len(updated_L) == len(original_L)
        assert updated_U is original_U  # Should modify in place
        assert updated_S is original_S  # Should modify in place
        assert updated_L is original_L  # Should modify in place

    def test_svd_groups_empty_groups(self):
        """Test decomposition with empty groups list"""
        svd_U = []
        svd_S = []
        groups = []
        sample_cov = np.eye(2)

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        assert len(updated_U) == 0
        assert len(updated_S) == 0
        assert len(updated_L) == 0

    def test_svd_groups_proposal_L_none_creates_list(self):
        """Test that proposal_L=None creates a fresh list"""
        svd_U = [None]
        svd_S = [None]
        groups = [np.array([0, 1])]
        sample_cov = np.eye(2)

        updated_U, updated_S, updated_L = svd_groups(svd_U, svd_S, groups, sample_cov)

        assert len(updated_L) == 1
        assert updated_L[0] is not None
        assert updated_L[0].shape == (2, 2)

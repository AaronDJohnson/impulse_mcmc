import pytest
import numpy as np
from impulse.online_updates import update_mean, update_covariance, svd_groups


class TestUpdateMean:
    """Test suite for update_mean function"""

    def test_update_mean_basic(self):
        """Test basic mean update"""
        old_length = 100
        old_avg = np.array([1.0, 2.0])
        new_arr = np.array([[1.5, 2.5], [0.5, 1.5]])
        
        result = update_mean(old_length, old_avg, new_arr)
        
        # Expected: (100*[1,2] + sum(new_samples)) / 102 
        # Sum of new samples = [1.5+0.5, 2.5+1.5] = [2.0, 4.0]
        # Total = [100+2, 200+4] / 102 = [102, 204] / 102 = [1.0, 2.0]
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_update_mean_single_new_sample(self):
        """Test update with single new sample"""
        old_length = 10
        old_avg = np.array([0.0, 0.0])
        new_arr = np.array([[1.0, 2.0]])
        
        result = update_mean(old_length, old_avg, new_arr)
        expected = np.array([1.0, 2.0]) / 11.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_update_mean_zero_old_length(self):
        """Test update when old length is zero"""
        old_length = 0
        old_avg = np.array([0.0, 0.0])
        new_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = update_mean(old_length, old_avg, new_arr)
        expected = np.array([2.0, 3.0])  # Mean of new samples
        np.testing.assert_array_almost_equal(result, expected)

    def test_update_mean_single_dimension(self):
        """Test update with single dimension"""
        old_length = 5
        old_avg = np.array([1.0])
        new_arr = np.array([[2.0], [3.0]])
        
        result = update_mean(old_length, old_avg, new_arr)
        # (5*1 + 2 + 3) / 7 = 10/7
        expected = np.array([10.0/7.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_update_mean_large_numbers(self):
        """Test update with large numbers"""
        old_length = 1000
        old_avg = np.array([100.0, 200.0])
        new_arr = np.array([[101.0, 201.0]])
        
        result = update_mean(old_length, old_avg, new_arr)
        # Should be very close to original mean
        expected = (1000 * np.array([100.0, 200.0]) + np.array([101.0, 201.0])) / 1001
        np.testing.assert_array_almost_equal(result, expected)


class TestUpdateCovariance:
    """Test suite for update_covariance function"""

    def test_update_covariance_basic(self):
        """Test basic covariance update"""
        old_length = 2
        old_cov = np.eye(2)
        old_avg = np.array([0.0, 0.0])
        new_arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        new_avg, new_cov = update_covariance(old_length, old_cov, old_avg, new_arr)
        
        # Check that dimensions are preserved
        assert new_cov.shape == (2, 2)
        assert new_avg.shape == (2,)
        
        # Should be positive definite
        eigenvals = np.linalg.eigvals(new_cov)
        assert np.all(eigenvals >= 0)

    def test_update_covariance_identity_preservation(self):
        """Test that updating with similar data preserves structure"""
        old_length = 100
        old_cov = np.eye(3)
        old_avg = np.array([0.0, 0.0, 0.0])
        # Add samples from same distribution
        np.random.seed(42)
        new_arr = np.random.randn(10, 3)
        
        new_avg, new_cov = update_covariance(old_length, old_cov, old_avg, new_arr)
        
        assert new_cov.shape == (3, 3)
        assert new_avg.shape == (3,)
        # Covariance should still be reasonable
        assert np.allclose(new_cov, new_cov.T)  # Symmetric

    def test_update_covariance_single_sample(self):
        """Test update with single new sample"""
        old_length = 1
        old_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        old_avg = np.array([0.0, 0.0])
        new_arr = np.array([[1.0, 1.0]])
        
        new_avg, new_cov = update_covariance(old_length, old_cov, old_avg, new_arr)
        
        assert new_cov.shape == (2, 2)
        assert new_avg.shape == (2,)

    def test_update_covariance_zero_old_length(self):
        """Test update when old length is zero"""
        old_length = 0
        old_cov = np.zeros((2, 2))
        old_avg = np.array([0.0, 0.0])
        new_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        new_avg, new_cov = update_covariance(old_length, old_cov, old_avg, new_arr)
        
        # Should compute fresh statistics
        expected_avg = np.mean(new_arr, axis=0)
        np.testing.assert_array_almost_equal(new_avg, expected_avg)

    def test_update_covariance_symmetry(self):
        """Test that resulting covariance is symmetric"""
        old_length = 10
        old_cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        old_avg = np.array([1.0, -1.0])
        new_arr = np.array([[0.5, -0.5], [1.5, -1.5]])
        
        new_avg, new_cov = update_covariance(old_length, old_cov, old_avg, new_arr)
        
        np.testing.assert_array_almost_equal(new_cov, new_cov.T)


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
            np.testing.assert_array_almost_equal(
                np.abs(updated_L[i]), np.abs(updated_U[i])
            )

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
        np.testing.assert_array_almost_equal(updated_L[0], [[2.0]])   # sqrt(4)
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
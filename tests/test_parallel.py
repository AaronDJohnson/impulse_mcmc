"""
Tests for parallel likelihood evaluation module.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from impulse.parallel import ParallelLikelihood

# Multiprocessing tests can deadlock; cap each test at 120s (requires pytest-timeout).
pytestmark = pytest.mark.timeout(120)


# Test likelihood functions defined at module level for pickle compatibility
def simple_likelihood(x):
    """Simple likelihood for testing"""
    return -0.5 * np.sum(x**2, axis=1)


def quadratic_likelihood(x):
    """Simple quadratic likelihood for testing"""
    return -0.5 * np.sum(x**2, axis=1)


def sum_likelihood(x):
    """Simple sum likelihood for testing"""
    return np.sum(x, axis=1)


def complex_likelihood(x):
    """More complex likelihood with multiple operations"""
    # Mixture of operations
    term1 = -0.5 * np.sum(x**2, axis=1)
    term2 = np.sum(np.sin(x), axis=1) * 0.1
    term3 = -np.sum(np.abs(x), axis=1) * 0.01
    return term1 + term2 + term3


def medium_likelihood(x):
    """Likelihood with moderate computational cost"""
    # Add some computation to make parallelization worthwhile
    result = np.zeros(x.shape[0])
    for i in range(10):  # Some loops to increase computation time
        result += -0.1 * np.sum(x**2, axis=1)
        result += 0.01 * np.sum(np.sin(x * (i + 1)), axis=1)
    return result


def cpu_intensive_likelihood(x):
    """CPU-intensive likelihood for testing parallelization"""
    result = np.zeros(x.shape[0])
    # Add computation to make parallelization beneficial
    for _ in range(50):
        result += -0.001 * np.sum(x**2, axis=1)
        result += 0.0001 * np.sum(np.exp(-(x**2)), axis=1)
    return result


def base_likelihood(x):
    """Base likelihood for integration testing"""
    return -0.5 * np.sum(x**2, axis=1)


class TestParallelLikelihood:
    """Test cases for ParallelLikelihood class"""

    def test_init_basic(self):
        """Test basic initialization"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(dummy_likelihood)
        assert parallel_like.likelihood_fn is dummy_likelihood
        assert parallel_like.n_workers > 0
        assert parallel_like.batch_size == 1000
        assert parallel_like.max_batch_size == 10000

    def test_init_custom_params(self):
        """Test initialization with custom parameters"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(
            dummy_likelihood, n_workers=4, batch_size=500, max_batch_size=5000
        )
        assert parallel_like.n_workers == 4
        assert parallel_like.batch_size == 500
        assert parallel_like.max_batch_size == 5000

    def test_call_empty_input(self):
        """Test handling of empty input"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(dummy_likelihood)
        result = parallel_like(np.empty((0, 5)))
        assert result.shape == (0,)

    def test_call_1d_input_error(self):
        """Test that 1D input raises error"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(dummy_likelihood)

        with pytest.raises(ValueError, match="params must be 2-D array"):
            parallel_like(np.array([1, 2, 3]))

    def test_small_batch_direct_evaluation(self):
        """Test that small batches use direct evaluation"""
        parallel_like = ParallelLikelihood(simple_likelihood, batch_size=100)

        # Small batch should be evaluated directly
        params = np.random.randn(50, 5)
        expected = simple_likelihood(params)
        result = parallel_like(params)

        np.testing.assert_array_almost_equal(result, expected)

    def test_correctness_vs_direct_evaluation(self):
        """Test that parallel evaluation gives same results as direct"""
        # Test with different batch sizes
        np.random.seed(42)
        params = np.random.randn(2000, 10)

        # Direct evaluation
        expected = quadratic_likelihood(params)

        # Parallel evaluation
        parallel_like = ParallelLikelihood(quadratic_likelihood, n_workers=2, batch_size=100)
        result = parallel_like(params)

        # Results should be identical
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

        # Clean up
        parallel_like.cleanup()

    def test_different_parameter_dimensions(self):
        """Test handling of different parameter dimensions"""
        parallel_like = ParallelLikelihood(simple_likelihood, batch_size=50)

        # Test different dimensions
        for dim in [1, 5, 20, 100]:
            params = np.random.randn(200, dim)
            expected = simple_likelihood(params)
            result = parallel_like(params)
            np.testing.assert_array_almost_equal(result, expected)

        parallel_like.cleanup()

    def test_large_batch_processing(self):
        """Test processing of large batches that exceed max_batch_size"""
        parallel_like = ParallelLikelihood(
            sum_likelihood, batch_size=100, max_batch_size=500, n_workers=2
        )

        # Large batch that will be chunked
        params = np.random.randn(1200, 5)
        expected = sum_likelihood(params)
        result = parallel_like(params)

        np.testing.assert_array_almost_equal(result, expected)
        parallel_like.cleanup()

    def test_single_parameter_sample(self):
        """Test handling of single parameter sample"""
        parallel_like = ParallelLikelihood(simple_likelihood)

        # Single sample
        params = np.random.randn(1, 10)
        expected = simple_likelihood(params)
        result = parallel_like(params)

        np.testing.assert_array_almost_equal(result, expected)
        parallel_like.cleanup()

    def test_complex_likelihood_function(self):
        """Test with a more complex likelihood function"""
        parallel_like = ParallelLikelihood(complex_likelihood, batch_size=100)

        params = np.random.randn(500, 8)
        expected = complex_likelihood(params)
        result = parallel_like(params)

        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        parallel_like.cleanup()

    def test_cleanup_method(self):
        """Test that cleanup method works without errors"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(dummy_likelihood)

        # Force initialization by calling with large batch
        params = np.random.randn(2000, 5)
        _ = parallel_like(params)

        # Cleanup should work without errors
        parallel_like.cleanup()

        # Should be able to call cleanup multiple times
        parallel_like.cleanup()

    def test_repr(self):
        """Test string representation"""

        def dummy_likelihood(x):
            return np.sum(x**2, axis=1)

        parallel_like = ParallelLikelihood(
            dummy_likelihood, n_workers=4, batch_size=500, max_batch_size=2000
        )

        repr_str = repr(parallel_like)
        assert "ParallelLikelihood" in repr_str
        assert "n_workers=4" in repr_str
        assert "batch_size=500" in repr_str
        assert "max_batch_size=2000" in repr_str

    def test_performance_comparison(self):
        """Test that parallel evaluation can be faster for appropriate workloads"""
        params = np.random.randn(1000, 20)

        # Direct evaluation
        start_time = time.time()
        expected = medium_likelihood(params)
        direct_time = time.time() - start_time

        # Parallel evaluation
        parallel_like = ParallelLikelihood(medium_likelihood, n_workers=2, batch_size=100)

        start_time = time.time()
        result = parallel_like(params)
        parallel_time = time.time() - start_time

        # Results should be correct
        np.testing.assert_array_almost_equal(result, expected, decimal=8)

        # Note: We don't assert parallel is faster since it depends on system
        # and likelihood complexity, but we verify it produces correct results
        print(f"Direct time: {direct_time:.3f}s, Parallel time: {parallel_time:.3f}s")

        parallel_like.cleanup()

    def test_fallback_on_worker_failure(self):
        """Test fallback to direct evaluation when workers fail"""
        parallel_like = ParallelLikelihood(simple_likelihood, batch_size=100)

        params = np.random.randn(500, 5)

        # Mock pool.map to raise exception
        with patch.object(parallel_like, "_pool") as mock_pool:
            mock_pool.map.side_effect = Exception("Worker failed")

            # Should fallback to direct evaluation without raising
            with pytest.warns(UserWarning, match="Parallel evaluation failed"):
                result = parallel_like(params)

            # Result should still be correct
            expected = simple_likelihood(params)
            np.testing.assert_array_almost_equal(result, expected)

        parallel_like.cleanup()

    def test_memory_efficiency(self):
        """Test that memory usage doesn't grow with sample count"""
        parallel_like = ParallelLikelihood(
            sum_likelihood, batch_size=100, max_batch_size=200, n_workers=2
        )

        # Process multiple batches of different sizes
        for n_samples in [500, 1000, 2000]:
            params = np.random.randn(n_samples, 10)
            expected = sum_likelihood(params)
            result = parallel_like(params)
            np.testing.assert_array_almost_equal(result, expected)

        # Memory should be reused, not growing
        # (This is more of a design verification than a direct test)
        parallel_like.cleanup()

    def test_with_function_wrapper_integration(self):
        """Test integration with _function_wrapper"""
        from impulse.input_function_wrapper import _function_wrapper

        # Wrap with function wrapper (vectorized)
        wrapped_like = _function_wrapper(base_likelihood, vectorized=True)

        # Wrap with parallel likelihood
        parallel_like = ParallelLikelihood(wrapped_like, batch_size=100)

        params = np.random.randn(500, 8)
        expected = wrapped_like(params)
        result = parallel_like(params)

        np.testing.assert_array_almost_equal(result, expected)
        parallel_like.cleanup()


# Integration test with actual multiprocessing
@pytest.mark.slow
class TestParallelLikelihoodIntegration:
    """Integration tests that actually use multiprocessing"""

    def test_multiprocessing_execution(self):
        """Test actual multiprocessing execution"""
        params = np.random.randn(400, 10)

        # Test with actual multiprocessing
        parallel_like = ParallelLikelihood(
            cpu_intensive_likelihood, n_workers=2, batch_size=50, max_batch_size=100
        )

        # This will actually use multiprocessing
        result = parallel_like(params)
        expected = cpu_intensive_likelihood(params)

        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        parallel_like.cleanup()

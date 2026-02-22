import pytest
import numpy as np
import os
import pickle
import tempfile
from unittest.mock import patch

from impulse.resume import checkpoint_sampler, load_checkpoint, check_for_checkpoint


class _FakeSampler:
    """Simple pickleable stand-in for PTSampler in checkpoint tests."""
    pass


class TestCheckpointSampler:
    """Test suite for checkpoint_sampler function"""

    def test_checkpoint_sampler_basic(self, temp_dir):
        """Test basic checkpoint functionality"""
        sampler = _FakeSampler()
        sampler.outdir = temp_dir
        sampler.lnlike = lambda x: -0.5 * np.sum(x**2)
        sampler.lnprior = lambda x: 0.0
        sampler.some_data = [1, 2, 3]
        sampler.ndim = 2

        # Checkpoint the sampler
        checkpoint_path = checkpoint_sampler(sampler)

        expected_path = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        assert checkpoint_path == expected_path
        assert os.path.exists(checkpoint_path)

        # Functions should be None in the checkpoint
        with open(checkpoint_path, 'rb') as f:
            loaded = pickle.load(f)
            assert loaded.lnlike is None
            assert loaded.lnprior is None
            assert loaded.some_data == [1, 2, 3]  # Other attributes preserved

        # Original sampler should have functions restored
        assert sampler.lnlike is not None
        assert sampler.lnprior is not None

    def test_checkpoint_sampler_custom_path(self, temp_dir):
        """Test checkpoint with custom path"""
        sampler = _FakeSampler()
        sampler.lnlike = lambda x: x[0]
        sampler.lnprior = lambda x: 0.0

        custom_path = os.path.join(temp_dir, "custom_checkpoint.pkl")
        result_path = checkpoint_sampler(sampler, path=custom_path)

        assert result_path == custom_path
        assert os.path.exists(custom_path)

    def test_checkpoint_sampler_custom_omit(self, temp_dir):
        """Test checkpoint with custom omit list"""
        sampler = _FakeSampler()
        sampler.func1 = lambda x: x
        sampler.func2 = lambda x: x**2
        sampler.keep_me = "important_data"

        checkpoint_path = checkpoint_sampler(sampler, omit=('func1', 'func2'))

        # Check that specified functions are omitted
        with open(checkpoint_path, 'rb') as f:
            loaded = pickle.load(f)
            assert loaded.func1 is None
            assert loaded.func2 is None
            assert loaded.keep_me == "important_data"

        # Original should be restored
        assert sampler.func1 is not None
        assert sampler.func2 is not None

    def test_checkpoint_sampler_nonexistent_attribute(self, temp_dir):
        """Test checkpointing when omit list contains non-existent attributes"""
        sampler = _FakeSampler()
        sampler.existing_func = lambda x: x

        # Should not raise error even if 'nonexistent' doesn't exist
        checkpoint_path = checkpoint_sampler(sampler, omit=('existing_func', 'nonexistent'))

        assert os.path.exists(checkpoint_path)

        # Only existing function should be restored
        assert sampler.existing_func is not None

    def test_checkpoint_sampler_creates_directories(self):
        """Test that checkpoint creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            sampler = _FakeSampler()
            sampler.lnlike = lambda x: x[0]

            nested_path = os.path.join(temp_dir, "deep", "nested", "checkpoint.pkl")
            result_path = checkpoint_sampler(sampler, path=nested_path)

            assert result_path == nested_path
            assert os.path.exists(nested_path)
            assert os.path.exists(os.path.dirname(nested_path))

    def test_checkpoint_sampler_atomic_write(self, temp_dir):
        """Test that checkpoint uses atomic write operations"""
        sampler = _FakeSampler()
        sampler.lnlike = lambda x: x[0]

        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pkl")

        # Mock tempfile.mkstemp to verify atomic write behavior
        with patch('impulse.resume.tempfile.mkstemp') as mock_mkstemp:
            mock_fd = 123
            mock_tmp_path = os.path.join(temp_dir, ".ckpt.test.tmp")
            mock_mkstemp.return_value = (mock_fd, mock_tmp_path)

            with patch('impulse.resume.os.close') as mock_close, \
                 patch('impulse.resume.os.replace') as mock_replace, \
                 patch('builtins.open', create=True) as mock_open, \
                 patch('pickle.dump'):

                result_path = checkpoint_sampler(sampler, path=checkpoint_path)

                # Verify atomic operations were called
                mock_mkstemp.assert_called_once()
                mock_close.assert_called_once_with(mock_fd)
                mock_replace.assert_called_once_with(mock_tmp_path, checkpoint_path)

    def test_checkpoint_sampler_exception_handling(self, temp_dir):
        """Test that exceptions during checkpointing restore original state"""
        sampler = _FakeSampler()
        sampler.lnlike = lambda x: x[0]
        original_func = sampler.lnlike

        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pkl")

        # Mock pickle.dump to raise exception
        with patch('pickle.dump', side_effect=Exception("Pickle failed")):
            with pytest.raises(Exception, match="Pickle failed"):
                checkpoint_sampler(sampler, path=checkpoint_path)

        # Original function should be restored even after exception
        assert sampler.lnlike is original_func


class TestLoadCheckpoint:
    """Test suite for load_checkpoint function"""

    def test_load_checkpoint_basic(self, temp_dir):
        """Test basic checkpoint loading"""
        original_sampler = _FakeSampler()
        original_sampler.ndim = 3
        original_sampler.ntemps = 5
        original_sampler.data = [1, 2, 3]
        original_sampler.lnlike = None  # Simulate checkpointed state
        original_sampler.lnprior = None

        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(original_sampler, f)

        # Define functions to restore
        def log_likelihood(x):
            return -0.5 * np.sum(x**2)

        def log_prior(x):
            return 0.0 if np.all(np.abs(x) <= 5) else -np.inf

        # Load checkpoint
        loaded_sampler = load_checkpoint(checkpoint_path, log_likelihood, log_prior)

        # Check that data is preserved
        assert loaded_sampler.ndim == 3
        assert loaded_sampler.ntemps == 5
        assert loaded_sampler.data == [1, 2, 3]

        # Check that functions are restored
        assert loaded_sampler.lnlike is log_likelihood
        assert loaded_sampler.lnprior is log_prior

    def test_load_checkpoint_functions_work(self, temp_dir):
        """Test that loaded functions work correctly"""
        sampler = _FakeSampler()
        sampler.lnlike = None
        sampler.lnprior = None

        checkpoint_path = os.path.join(temp_dir, "functional_test.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(sampler, f)

        # Define working functions
        def log_likelihood(x):
            return -np.sum(x**2)

        def log_prior(x):
            return 0.0

        loaded_sampler = load_checkpoint(checkpoint_path, log_likelihood, log_prior)

        # Test that functions work
        test_input = np.array([1.0, 2.0])
        assert loaded_sampler.lnlike(test_input) == -5.0  # -(1^2 + 2^2)
        assert loaded_sampler.lnprior(test_input) == 0.0

    def test_load_checkpoint_file_not_found(self, temp_dir):
        """Test error handling when checkpoint file doesn't exist"""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.pkl")
        
        def dummy_func(x):
            return 0.0
        
        with pytest.raises(FileNotFoundError):
            load_checkpoint(nonexistent_path, dummy_func, dummy_func)

    def test_load_checkpoint_invalid_pickle(self, temp_dir):
        """Test error handling with corrupted checkpoint file"""
        corrupt_path = os.path.join(temp_dir, "corrupt.pkl")
        
        # Create corrupted file
        with open(corrupt_path, 'w') as f:
            f.write("This is not a pickle file")
        
        def dummy_func(x):
            return 0.0
        
        with pytest.raises((pickle.UnpicklingError, UnicodeDecodeError)):
            load_checkpoint(corrupt_path, dummy_func, dummy_func)


class TestCheckForCheckpoint:
    """Test suite for check_for_checkpoint function"""

    def test_check_for_checkpoint_exists(self, temp_dir):
        """Test finding existing checkpoint"""
        checkpoint_path = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        
        # Create checkpoint file
        with open(checkpoint_path, 'w') as f:
            f.write("dummy checkpoint")
        
        result = check_for_checkpoint(temp_dir)
        assert result == checkpoint_path

    def test_check_for_checkpoint_not_exists(self, temp_dir):
        """Test when no checkpoint exists"""
        result = check_for_checkpoint(temp_dir)
        assert result is None

    def test_check_for_checkpoint_nonexistent_dir(self):
        """Test with non-existent directory"""
        result = check_for_checkpoint("/nonexistent/directory")
        assert result is None

    def test_check_for_checkpoint_standard_name(self, temp_dir):
        """Test that it looks for standard checkpoint name"""
        # Create file with different name
        wrong_name = os.path.join(temp_dir, "different_checkpoint.pkl")
        with open(wrong_name, 'w') as f:
            f.write("dummy")
        
        result = check_for_checkpoint(temp_dir)
        assert result is None  # Should not find wrong name
        
        # Create file with correct name
        correct_name = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        with open(correct_name, 'w') as f:
            f.write("dummy")
        
        result = check_for_checkpoint(temp_dir)
        assert result == correct_name


class TestIntegrationCheckpointResumeWorkflow:
    """Integration tests for checkpoint-resume workflow"""

    def test_full_checkpoint_resume_cycle(self, temp_dir):
        """Test complete checkpoint and resume cycle"""
        original_sampler = _FakeSampler()
        original_sampler.outdir = temp_dir
        original_sampler.ndim = 2
        original_sampler.ntemps = 3
        original_sampler.iteration = 1000
        original_sampler.chain_data = np.random.randn(3, 100, 2)
        
        # Define functions
        def log_likelihood(x):
            return -0.5 * np.sum(x**2, axis=-1) if x.ndim > 1 else -0.5 * np.sum(x**2)
        
        def log_prior(x):
            return 0.0
        
        original_sampler.lnlike = log_likelihood
        original_sampler.lnprior = log_prior
        
        # Step 1: Checkpoint
        checkpoint_path = checkpoint_sampler(original_sampler)
        assert os.path.exists(checkpoint_path)
        
        # Step 2: Check for checkpoint
        found_path = check_for_checkpoint(temp_dir)
        assert found_path == checkpoint_path
        
        # Step 3: Load checkpoint
        loaded_sampler = load_checkpoint(checkpoint_path, log_likelihood, log_prior)
        
        # Step 4: Verify everything is restored correctly
        assert loaded_sampler.ndim == original_sampler.ndim
        assert loaded_sampler.ntemps == original_sampler.ntemps
        assert loaded_sampler.iteration == original_sampler.iteration
        np.testing.assert_array_equal(loaded_sampler.chain_data, original_sampler.chain_data)
        
        # Test that functions work
        test_input = np.array([[1.0, 2.0]])
        assert loaded_sampler.lnlike(test_input)[0] == -2.5
        assert loaded_sampler.lnprior(test_input) == 0.0
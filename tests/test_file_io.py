import os
import shutil
import tempfile
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from impulse.file_io import ShortChain
from impulse.sampler_state import SamplerState


class TestShortChain:
    """Test suite for ShortChain class"""

    def test_init_basic(self, temp_dir):
        """Test basic ShortChain initialization"""
        chain = ShortChain(ndim=2, ntemps=3, short_iters=100, outdir=temp_dir)

        assert chain.ndim == 2
        assert chain.ntemps == 3
        assert chain.short_iters == 100
        assert chain.iteration == 0
        assert chain.outdir == temp_dir
        assert chain.resume == False
        assert chain.thin == 1

        # Check array shapes
        assert chain.samples.shape == (3, 100, 2)
        assert chain.lnprob.shape == (3, 100)
        assert chain.lnlike.shape == (3, 100)
        assert chain.accept.shape == (3, 100)
        assert chain.var_temp.shape == (3, 100)

    def test_init_with_custom_params(self, temp_dir):
        """Test initialization with custom parameters"""
        chain = ShortChain(
            ndim=4, ntemps=5, short_iters=50, iteration=10, outdir=temp_dir, resume=True, thin=2
        )

        assert chain.ndim == 4
        assert chain.ntemps == 5
        assert chain.short_iters == 50
        assert chain.iteration == 10
        assert chain.resume == True
        assert chain.thin == 2

    def test_init_creates_filenames(self, temp_dir):
        """Test that initialization creates correct filenames"""
        chain = ShortChain(ndim=2, ntemps=3, short_iters=10, outdir=temp_dir)

        expected_filenames = ["chain_0.txt", "chain_1.txt", "chain_2.txt"]
        assert chain.filenames == expected_filenames

        expected_paths = [os.path.join(temp_dir, f) for f in expected_filenames]
        assert chain.filepaths == expected_paths

    def test_init_thin_validation(self, temp_dir):
        """Test validation of thin parameter"""
        # Should raise error if thin > short_iters
        with pytest.raises(ValueError, match="There are not enough samples to thin"):
            ShortChain(ndim=2, ntemps=3, short_iters=10, thin=15, outdir=temp_dir)

    def test_init_creates_files(self, temp_dir):
        """Test that initialization creates output files"""
        chain = ShortChain(ndim=2, ntemps=3, short_iters=10, outdir=temp_dir)

        # All files should be created
        for filepath in chain.filepaths:
            assert os.path.exists(filepath)
            assert os.path.isfile(filepath)

    def test_add_state_basic(self, temp_dir):
        """Test adding a sampler state to the chain"""
        chain = ShortChain(ndim=2, ntemps=3, short_iters=10, outdir=temp_dir)

        # Create a sample state
        positions = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
        lnlikes = np.array([-1.0, -1.1, -1.2])
        lnpriors = np.array([0.0, 0.0, 0.0])
        lnprobs = np.array([-1.0, -1.1, -1.2])
        accepted = np.array([1, 0, 1])
        temps = np.array([1.0, 2.0, 4.0])

        state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)

        initial_iteration = chain.iteration
        chain.add_state(state)

        # Check that iteration was incremented
        assert chain.iteration == initial_iteration + 1

        # Check that data was stored correctly
        save_iter = initial_iteration % chain.short_iters
        np.testing.assert_array_equal(chain.samples[:, save_iter], positions)
        np.testing.assert_array_equal(chain.lnprob[:, save_iter], lnprobs)
        np.testing.assert_array_equal(chain.lnlike[:, save_iter], lnlikes)
        np.testing.assert_array_equal(chain.accept[:, save_iter], accepted)
        np.testing.assert_array_equal(chain.var_temp[:, save_iter], temps)

    def test_add_state_circular_buffer(self, temp_dir):
        """Test that add_state works as circular buffer"""
        chain = ShortChain(ndim=1, ntemps=2, short_iters=3, outdir=temp_dir)

        # Add more states than buffer size
        for i in range(5):
            positions = np.array([[i], [i + 10]])
            lnlikes = np.array([-(i + 1), -(i + 11)])
            lnpriors = np.array([0.0, 0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1, 1])
            temps = np.array([1.0, 2.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        assert chain.iteration == 5

        # Check circular buffer behavior - should have overwritten early entries
        # After 5 iterations: positions 0,1,2,0,1 filled with values 0,1,2,3,4
        # So position 2 contains iteration 2 (value 2)
        np.testing.assert_array_equal(chain.samples[:, 2], np.array([[2], [12]]))

    def test_exists_method(self, temp_dir):
        """Test the exists method"""
        chain = ShortChain(ndim=1, ntemps=1, short_iters=10, outdir=temp_dir)

        # Test with existing file
        existing_file = "test_existing.txt"
        with open(os.path.join(temp_dir, existing_file), "w") as f:
            f.write("test")

        assert chain.exists(temp_dir, existing_file) == True

        # Test with non-existing file
        assert chain.exists(temp_dir, "nonexistent.txt") == False

    def test_save_chain_basic(self, temp_dir):
        """Test basic chain saving functionality"""
        chain = ShortChain(ndim=2, ntemps=2, short_iters=3, outdir=temp_dir)

        # Add some states
        for i in range(3):
            positions = np.array([[i, i + 0.5], [i + 1, i + 1.5]])
            lnlikes = np.array([-(i + 1), -(i + 2)])
            lnpriors = np.array([0.0, 0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1, 0])
            temps = np.array([1.0, 2.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        chain.save_chain()

        # Check that files have content
        for filepath in chain.filepaths:
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

            # Check file format (should have 6 columns: 2 params + lnlike + lnprob + accept + temp)
            data = np.loadtxt(filepath)
            assert data.shape == (3, 6)  # 3 iterations, 6 columns

    def test_save_chain_with_thinning(self, temp_dir):
        """Test chain saving with thinning"""
        chain = ShortChain(ndim=1, ntemps=1, short_iters=6, thin=2, outdir=temp_dir)

        # Add 6 states
        for i in range(6):
            positions = np.array([[i]])
            lnlikes = np.array([-(i + 1)])
            lnpriors = np.array([0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1])
            temps = np.array([1.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        chain.save_chain()

        # With thin=2, should only save every 2nd sample
        data = np.loadtxt(chain.filepaths[0])
        assert data.shape == (
            3,
            5,
        )  # 3 thinned samples, 5 columns (param + lnlike + lnprob + accept + temp)

        # Check that we got the right thinned values (0, 2, 4)
        np.testing.assert_array_equal(data[:, 0], [0.0, 2.0, 4.0])

    def test_save_chain_empty_buffer(self, temp_dir):
        """Test saving with empty buffer writes nothing"""
        chain = ShortChain(ndim=1, ntemps=1, short_iters=3, outdir=temp_dir)

        # Don't add any states, just save
        chain.save_chain()

        # File should exist but be empty (no unsaved samples)
        import os

        assert os.path.exists(chain.filepaths[0])
        assert os.path.getsize(chain.filepaths[0]) == 0

    def test_save_chain_append_mode(self, temp_dir):
        """Test that save_chain appends to files"""
        chain = ShortChain(ndim=1, ntemps=1, short_iters=2, outdir=temp_dir)

        # Add first batch
        for i in range(2):
            positions = np.array([[i]])
            lnlikes = np.array([-(i + 1)])
            lnpriors = np.array([0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1])
            temps = np.array([1.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        chain.save_chain()

        # Add second batch
        for i in range(2, 4):
            positions = np.array([[i]])
            lnlikes = np.array([-(i + 1)])
            lnpriors = np.array([0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1])
            temps = np.array([1.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        chain.save_chain()

        # File should have 4 rows total
        data = np.loadtxt(chain.filepaths[0])
        assert data.shape == (4, 5)  # 1D: 1 param + lnlike + lnprob + accept + temp

    def test_save_chain_multiple_temperatures(self, temp_dir):
        """Test saving with multiple temperature chains"""
        chain = ShortChain(ndim=1, ntemps=3, short_iters=2, outdir=temp_dir)

        for i in range(2):
            positions = np.array([[i], [i + 10], [i + 20]])
            lnlikes = np.array([-(i + 1), -(i + 11), -(i + 21)])
            lnpriors = np.array([0.0, 0.0, 0.0])
            lnprobs = lnlikes + lnpriors
            accepted = np.array([1, 0, 1])
            temps = np.array([1.0, 2.0, 4.0])

            state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
            chain.add_state(state)

        chain.save_chain()

        # Check that all 3 files were created with correct data
        assert len(chain.filepaths) == 3

        for temp_idx, filepath in enumerate(chain.filepaths):
            assert os.path.exists(filepath)
            data = np.loadtxt(filepath)
            assert data.shape == (2, 5)  # 1D: 1 param + lnlike + lnprob + accept + temp

            # Check first column contains the right parameter values
            expected_params = [temp_idx * 10, temp_idx * 10 + 1]
            np.testing.assert_array_equal(data[:, 0], expected_params)

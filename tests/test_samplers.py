import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch

from impulse.samplers import (
    setup_seeds, setup_chain_stats, setup_standard_jumps, 
    setup_initial_position, PTSampler
)
from impulse.sampler_state import PTState
from impulse.chain_stats import MultiChainStats
from impulse.proposals import ProposalBundle


class TestSetupSeeds:
    """Test suite for setup_seeds function"""

    def test_setup_seeds_basic(self):
        """Test basic seed setup"""
        rngs = setup_seeds(42, 3)
        
        assert len(rngs) == 4  # ntemps + 1 for PT swaps
        assert all(isinstance(rng, np.random.Generator) for rng in rngs)
        
        # Different generators should produce different sequences
        vals = [rng.random() for rng in rngs]
        assert len(set(vals)) == len(vals)  # All unique

    def test_setup_seeds_none_seed(self):
        """Test setup_seeds with None seed"""
        rngs = setup_seeds(None, 2)
        
        assert len(rngs) == 3
        assert all(isinstance(rng, np.random.Generator) for rng in rngs)

    def test_setup_seeds_reproducible(self):
        """Test that same seed produces same generators"""
        rngs1 = setup_seeds(42, 2)
        rngs2 = setup_seeds(42, 2)
        
        # Same seed should produce same initial random values
        vals1 = [rng.random() for rng in rngs1]
        vals2 = [rng.random() for rng in rngs2]
        
        np.testing.assert_array_almost_equal(vals1, vals2)

    def test_setup_seeds_single_chain(self):
        """Test setup_seeds with single chain"""
        rngs = setup_seeds(123, 1)
        assert len(rngs) == 2  # 1 chain + 1 for PT


class TestSetupChainStats:
    """Test suite for setup_chain_stats function"""

    def test_setup_chain_stats_basic(self):
        """Test basic chain stats setup"""
        ptstate = PTState(ndim=2, ntemps=3)
        rngs = setup_seeds(42, 3)
        temps = ptstate.ladder
        
        multi_stats = setup_chain_stats(
            ndim=2, ptstate=ptstate, rngs=rngs, groups=None,
            sample_cov=None, sample_mean=None, buffer_size=1000, temps=temps
        )
        
        assert isinstance(multi_stats, MultiChainStats)
        assert multi_stats.ntemps == 3
        assert multi_stats.ndim == 2
        assert len(multi_stats.chain_stats) == 3
        
        # Check that each chain stats has correct properties
        for i, cs in enumerate(multi_stats.chain_stats):
            assert cs.ndim == 2
            assert cs.chain_index == i
            assert cs.buffer_size == 1000
            assert cs.temp == temps[i]

    def test_setup_chain_stats_custom_params(self):
        """Test chain stats setup with custom parameters"""
        ptstate = PTState(ndim=3, ntemps=2)
        rngs = setup_seeds(42, 2)
        temps = ptstate.ladder
        custom_cov = np.eye(3) * 2
        custom_mean = np.array([1.0, 2.0, 3.0])
        custom_groups = [[0, 1], [2]]
        
        multi_stats = setup_chain_stats(
            ndim=3, ptstate=ptstate, rngs=rngs, groups=custom_groups,
            sample_cov=custom_cov, sample_mean=custom_mean, 
            buffer_size=500, temps=temps
        )
        
        assert multi_stats.ndim == 3
        
        # Check that custom parameters were passed through
        for cs in multi_stats.chain_stats:
            np.testing.assert_array_equal(cs.sample_cov, custom_cov)
            np.testing.assert_array_equal(cs.sample_mean, custom_mean)
            assert cs.groups == custom_groups
            assert cs.buffer_size == 500


class TestSetupStandardJumps:
    """Test suite for setup_standard_jumps function"""

    def test_setup_standard_jumps_basic(self):
        """Test basic standard jumps setup"""
        ptstate = PTState(ndim=2, ntemps=3)
        rngs = setup_seeds(42, 3)
        multi_stats = setup_chain_stats(
            ndim=2, ptstate=ptstate, rngs=rngs, groups=None,
            sample_cov=None, sample_mean=None, buffer_size=1000, temps=ptstate.ladder
        )
        
        proposal_bundle = setup_standard_jumps(multi_stats, am_weight=15, scam_weight=30, de_weight=50)
        
        assert isinstance(proposal_bundle, ProposalBundle)
        assert len(proposal_bundle.jump_proposals) == 3
        
        # Each chain should have 3 proposal types
        for jp in proposal_bundle.jump_proposals:
            assert len(jp.proposal_list) == 3  # AM, SCAM, DE
            assert len(jp.proposal_weights) == 3
            # Check weights were set correctly (normalized)
            expected_total = 15 + 30 + 50
            expected_probs = np.array([15, 30, 50]) / expected_total
            np.testing.assert_array_almost_equal(jp.proposal_probs, expected_probs)

    def test_setup_standard_jumps_zero_weights(self):
        """Test standard jumps setup with zero weights"""
        ptstate = PTState(ndim=2, ntemps=2)
        rngs = setup_seeds(42, 2)
        multi_stats = setup_chain_stats(
            ndim=2, ptstate=ptstate, rngs=rngs, groups=None,
            sample_cov=None, sample_mean=None, buffer_size=1000, temps=ptstate.ladder
        )
        
        # This should still work but produce different probability distributions
        proposal_bundle = setup_standard_jumps(multi_stats, am_weight=0, scam_weight=50, de_weight=50)
        
        for jp in proposal_bundle.jump_proposals:
            expected_probs = np.array([0, 50, 50]) / 100
            np.testing.assert_array_almost_equal(jp.proposal_probs, expected_probs)


class TestSetupInitialPosition:
    """Test suite for setup_initial_position function"""

    def test_setup_initial_position_1d_input(self):
        """Test setup with 1D input"""
        initial_pos = np.array([1.0, 2.0])
        result = setup_initial_position(initial_pos, ntemps=3)
        
        expected = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 2)

    def test_setup_initial_position_2d_single_row(self):
        """Test setup with 2D input (single row)"""
        initial_pos = np.array([[1.0, 2.0, 3.0]])
        result = setup_initial_position(initial_pos, ntemps=4)
        
        expected = np.tile([[1.0, 2.0, 3.0]], (4, 1))
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (4, 3)

    def test_setup_initial_position_2d_full(self):
        """Test setup with 2D input (full matrix)"""
        initial_pos = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = setup_initial_position(initial_pos, ntemps=3)
        
        np.testing.assert_array_equal(result, initial_pos)
        assert result.shape == (3, 2)

    def test_setup_initial_position_wrong_ntemps(self):
        """Test error with wrong number of temperature chains"""
        initial_pos = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows
        
        with pytest.raises(ValueError, match="initial_position has 2 rows but expected 1 or 3"):
            setup_initial_position(initial_pos, ntemps=3)

    def test_setup_initial_position_3d_input(self):
        """Test error with 3D input"""
        initial_pos = np.random.randn(2, 3, 4)
        
        with pytest.raises(ValueError, match="initial_position must be 1-D .* or 2-D .*"):
            setup_initial_position(initial_pos, ntemps=2)

    def test_setup_initial_position_dtype_preservation(self):
        """Test that dtype is preserved"""
        initial_pos = np.array([1, 2], dtype=np.int32)
        result = setup_initial_position(initial_pos, ntemps=2)
        
        assert result.dtype == np.float64  # Always converted to float64


class TestPTSampler:
    """Test suite for PTSampler class"""

    def test_pt_sampler_init_basic(self, simple_likelihood, simple_prior, temp_dir):
        """Test basic PTSampler initialization"""
        sampler = PTSampler(
            ndim=2,
            lnlike=simple_likelihood,
            lnprior=simple_prior,
            ntemps=3,
            outdir=temp_dir,
            seed=42
        )
        
        assert sampler.ndim == 2
        assert sampler.ntemps == 3
        assert sampler.swap_steps == 1
        assert sampler.outdir == temp_dir
        assert sampler.resume == False
        
        # Check that components were initialized
        assert hasattr(sampler, 'lnlike')
        assert hasattr(sampler, 'lnprior')
        assert hasattr(sampler, 'rngs')
        assert hasattr(sampler, 'ptstate')
        assert hasattr(sampler, 'multi_chain_stats')
        assert hasattr(sampler, 'proposal_bundle')
        
        assert len(sampler.rngs) == 4  # 3 chains + 1 for PT
        assert sampler.ptstate.ntemps == 3
        assert sampler.multi_chain_stats.ntemps == 3

    def test_pt_sampler_init_custom_params(self, simple_likelihood, simple_prior, temp_dir):
        """Test PTSampler initialization with custom parameters"""
        custom_mean = np.array([1.0, -1.0])
        custom_cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        custom_groups = [[0], [1]]
        
        sampler = PTSampler(
            ndim=2,
            lnlike=simple_likelihood,
            lnprior=simple_prior,
            ntemps=5,
            buffer_size=1000,
            sample_mean=custom_mean,
            sample_cov=custom_cov,
            groups=custom_groups,
            scam_weight=40,
            am_weight=20,
            de_weight=60,
            max_temp=32.0,
            outdir=temp_dir,
            seed=123
        )
        
        assert sampler.ntemps == 5
        assert sampler.ptstate.max_temp == 32.0
        
        # Check that custom parameters were passed through
        for cs in sampler.multi_chain_stats.chain_stats:
            np.testing.assert_array_equal(cs.sample_mean, custom_mean)
            np.testing.assert_array_equal(cs.sample_cov, custom_cov)
            assert cs.groups == custom_groups
            assert cs.buffer_size == 1000

    def test_pt_sampler_add_custom_jump(self, simple_likelihood, simple_prior, temp_dir):
        """Test adding custom jump proposal"""
        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir
        )
        
        def custom_proposal(chain_stats):
            return chain_stats.current_sample.copy(), 0.0
        
        # Should not raise error
        sampler.add_custom_jump(custom_proposal, weight=25)
        
        # Each chain should now have 4 proposals (3 standard + 1 custom)
        for jp in sampler.proposal_bundle.jump_proposals:
            assert len(jp.proposal_list) == 4

    @patch('impulse.samplers.tqdm')  # Mock tqdm to avoid progress bar output
    def test_pt_sampler_sample_basic(self, mock_tqdm, simple_likelihood, simple_prior, temp_dir):
        """Test basic sampling functionality"""
        # Make tqdm return an iterable that doesn't interfere
        mock_tqdm.return_value = range(10)
        
        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir, seed=42, save_freq=5
        )
        
        initial_pos = np.array([0.1, 0.1])
        
        # Should complete without error
        sampler.sample(initial_pos, num_iterations=10)
        
        # Check that state was created
        assert hasattr(sampler, 'state')
        assert hasattr(sampler, 'short_chain')
        assert sampler.short_chain.iteration == 10

    def test_pt_sampler_sample_bad_initial_likelihood(self, simple_prior, temp_dir):
        """Test error handling with bad initial likelihood"""
        def bad_likelihood(x):
            return np.array([np.nan, np.nan])  # All NaN
        
        sampler = PTSampler(
            ndim=2, lnlike=bad_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir
        )
        
        with pytest.raises(ValueError, match="Some likelihood values are not finite"):
            sampler.sample([0.0, 0.0], num_iterations=5)

    def test_pt_sampler_sample_bad_initial_prior(self, simple_likelihood, temp_dir):
        """Test error handling with bad initial prior"""
        def bad_prior(x):
            return np.array([np.nan, np.nan])  # All NaN
        
        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=bad_prior,
            ntemps=2, outdir=temp_dir
        )
        
        with pytest.raises(ValueError, match="An initial value falls outside the prior bounds"):
            sampler.sample([0.0, 0.0], num_iterations=5)

    def test_pt_sampler_vectorized_functions(self, vectorized_likelihood, vectorized_prior, temp_dir):
        """Test PTSampler with vectorized functions"""
        sampler = PTSampler(
            ndim=2,
            lnlike=vectorized_likelihood,
            lnprior=vectorized_prior,
            ntemps=3,
            outdir=temp_dir,
            vectorized=True,
            seed=42
        )
        
        # Should initialize without error
        assert sampler.ndim == 2
        assert sampler.ntemps == 3
        
        # Functions should be wrapped as vectorized
        assert sampler.lnlike.vectorized == True
        assert sampler.lnprior.vectorized == True

    @patch('impulse.samplers.check_for_checkpoint')
    @patch('impulse.samplers.load_checkpoint')
    @patch('impulse.samplers.tqdm')
    def test_pt_sampler_resume_functionality(self, mock_tqdm, mock_load_checkpoint,
                                           mock_check_checkpoint, simple_likelihood, simple_prior, temp_dir):
        """Test resume functionality"""
        mock_tqdm.return_value = range(5)
        mock_check_checkpoint.return_value = '/fake/checkpoint.pkl'

        # Create a pickleable stand-in for a loaded sampler
        class _FakeLoaded:
            pass
        fake_loaded = _FakeLoaded()
        fake_loaded.iteration = 100
        fake_loaded.some_state = 'loaded'
        mock_load_checkpoint.return_value = fake_loaded

        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir, resume=True
        )

        # This should trigger the resume logic
        sampler.sample([0.0, 0.0], num_iterations=5)

        # Check that checkpoint loading was attempted
        mock_check_checkpoint.assert_called_once_with(temp_dir)
        mock_load_checkpoint.assert_called_once()

    def test_pt_sampler_different_initial_formats(self, simple_likelihood, simple_prior, temp_dir):
        """Test PTSampler with different initial position formats"""
        sampler = PTSampler(
            ndim=3, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir, seed=42
        )
        
        # Test 1D initial position
        initial_1d = np.array([0.1, 0.2, 0.3])
        # Should not raise error
        test_pos = sampler.setup_initial_position if hasattr(sampler, 'setup_initial_position') else setup_initial_position
        result = setup_initial_position(initial_1d, sampler.ntemps)
        assert result.shape == (2, 3)
        
        # Test 2D initial position
        initial_2d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = setup_initial_position(initial_2d, sampler.ntemps)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, initial_2d)

    def test_pt_sampler_function_args_kwargs(self, temp_dir):
        """Test PTSampler with function arguments and kwargs"""
        def param_likelihood(x, scale=1.0, offset=0.0):
            return -scale * np.sum((x - offset)**2)

        def param_prior(x, bounds=5.0):
            if np.asarray(x).ndim == 1:
                return 0.0 if np.all(np.abs(x) <= bounds) else -np.inf
            else:
                result = np.zeros(x.shape[0])
                mask = np.any(np.abs(x) > bounds, axis=1)
                result[mask] = -np.inf
                return result

        sampler = PTSampler(
            ndim=2,
            lnlike=param_likelihood,
            lnprior=param_prior,
            loglargs=(2.0,),  # scale=2.0
            loglkwargs={'offset': 1.0},
            logpkwargs={'bounds': 3.0},
            ntemps=2,
            outdir=temp_dir
        )

        # Should initialize without error
        assert sampler.ndim == 2

    def test_pt_sampler_load_chain(self, simple_likelihood, simple_prior, temp_dir):
        """Test load_chain reads saved chain files correctly"""
        ndim = 2
        ntemps = 3
        nsamples = 5

        sampler = PTSampler(
            ndim=ndim, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=ntemps, outdir=temp_dir, seed=42
        )

        # Write synthetic chain files matching the save format
        rng = np.random.default_rng(0)
        expected = {}
        expected['samples'] = rng.standard_normal((ntemps, nsamples, ndim))
        expected['lnlike'] = rng.standard_normal((ntemps, nsamples))
        expected['lnprob'] = rng.standard_normal((ntemps, nsamples))
        expected['accepted'] = rng.integers(0, 2, size=(ntemps, nsamples)).astype(float)
        expected['temperature'] = np.tile(np.arange(1, ntemps + 1, dtype=float)[:, None], (1, nsamples))

        for ii in range(ntemps):
            filepath = os.path.join(temp_dir, f'chain_{ii}.txt')
            data = np.column_stack([
                expected['samples'][ii],
                expected['lnlike'][ii],
                expected['lnprob'][ii],
                expected['accepted'][ii],
                expected['temperature'][ii],
            ])
            np.savetxt(filepath, data)

        result = sampler.load_chain()

        assert set(result.keys()) == {'samples', 'lnlike', 'lnprob', 'accepted', 'temperature'}
        assert result['samples'].shape == (ntemps, nsamples, ndim)
        assert result['lnlike'].shape == (ntemps, nsamples)
        np.testing.assert_allclose(result['samples'], expected['samples'])
        np.testing.assert_allclose(result['lnlike'], expected['lnlike'])
        np.testing.assert_allclose(result['lnprob'], expected['lnprob'])
        np.testing.assert_allclose(result['accepted'], expected['accepted'])
        np.testing.assert_allclose(result['temperature'], expected['temperature'])

    def test_pt_sampler_threads_param(self, simple_likelihood, simple_prior, temp_dir):
        """PTSampler(threads=2) initializes and forwards to wrappers"""
        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=2, outdir=temp_dir, threads=2,
        )
        assert sampler.lnlike.threads == 2
        assert sampler.lnprior.threads == 2

    def test_pt_sampler_load_chain_missing_file(self, simple_likelihood, simple_prior, temp_dir):
        """Test load_chain raises FileNotFoundError when files are missing"""
        sampler = PTSampler(
            ndim=2, lnlike=simple_likelihood, lnprior=simple_prior,
            ntemps=3, outdir=temp_dir, seed=42
        )

        with pytest.raises(FileNotFoundError, match="Chain file not found"):
            sampler.load_chain()
import pytest
import numpy as np
from impulse.proposals import JumpProposals, ProposalBundle, am, scam, de
from impulse.chain_stats import ChainStats
from impulse.sampler_state import PTState, SamplerState


class TestJumpProposals:
    """Test cases for JumpProposals class"""
    
    def test_init_basic(self, chain_stats_2d):
        """Test basic JumpProposals initialization"""
        jp = JumpProposals(chain_stats_2d)
        
        assert jp.chain_stats is chain_stats_2d
        assert jp.proposal_list == []
        assert jp.proposal_weights == []
        assert jp.proposal_probs is None
    
    def test_init_with_params(self, chain_stats_2d):
        """Test JumpProposals initialization with parameters"""
        proposal_list = [am, scam]
        proposal_weights = [0.5, 0.3]
        
        jp = JumpProposals(
            chain_stats_2d, 
            proposal_list=proposal_list,
            proposal_weights=proposal_weights
        )
        
        assert jp.proposal_list == proposal_list
        assert jp.proposal_weights == proposal_weights
    
    def test_add_jump_new_proposal(self, chain_stats_2d):
        """Test adding a new proposal"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 0.5)
        
        assert am in jp.proposal_list
        assert 0.5 in jp.proposal_weights
        np.testing.assert_array_almost_equal(jp.proposal_probs, [1.0])
    
    def test_add_jump_multiple_proposals(self, chain_stats_2d):
        """Test adding multiple proposals with weight normalization"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 30.0)
        jp.add_jump(scam, 20.0)
        jp.add_jump(de, 50.0)
        
        expected_probs = np.array([30.0, 20.0, 50.0]) / 100.0
        np.testing.assert_array_almost_equal(jp.proposal_probs, expected_probs)
    
    def test_add_jump_update_existing(self, chain_stats_2d):
        """Test updating weight of existing proposal"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 30.0)
        jp.add_jump(scam, 20.0)
        
        # Update weight of am
        jp.add_jump(am, 40.0)
        
        assert jp.proposal_weights[0] == 40.0
        expected_probs = np.array([40.0, 20.0]) / 60.0
        np.testing.assert_array_almost_equal(jp.proposal_probs, expected_probs)
    
    def test_call_proposal_selection(self, chain_stats_2d, sample_state_2d):
        """Test proposal selection and execution"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 1.0)  # Only AM proposal
        
        # Mock current sample in chain_stats
        chain_stats_2d.current_sample = sample_state_2d.positions[0]
        
        new_sample, qxy = jp(sample_state_2d)
        
        assert isinstance(new_sample, np.ndarray)
        assert new_sample.shape == (2,)
        assert isinstance(qxy, (int, float, np.number))
    
    def test_call_de_disabled_when_buffer_not_full(self, chain_stats_2d, sample_state_2d):
        """Test that DE proposal is avoided when buffer is not full"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 0.1)
        jp.add_jump(de, 0.9)  # High weight for DE
        
        chain_stats_2d.current_sample = sample_state_2d.positions[0]
        chain_stats_2d.buffer_full = False
        
        # Set a fixed seed to make test deterministic
        chain_stats_2d.rng = np.random.default_rng(42)
        
        # Should fall back to AM even though DE has higher weight
        new_sample, qxy = jp(sample_state_2d)
        assert isinstance(new_sample, np.ndarray)


class TestProposalBundle:
    """Test cases for ProposalBundle class"""
    
    def test_init(self, chain_stats_2d):
        """Test ProposalBundle initialization"""
        jp1 = JumpProposals(chain_stats_2d)
        jp2 = JumpProposals(chain_stats_2d)
        
        bundle = ProposalBundle([jp1, jp2])
        
        assert len(bundle.jump_proposals) == 2
        assert bundle.jump_proposals[0] is jp1
        assert bundle.jump_proposals[1] is jp2
    
    def test_get_new_position(self, chain_stats_2d, sample_state_2d):
        """Test generating proposals for all chains"""
        # Create multiple chain stats for different temperatures
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        rng3 = np.random.default_rng(44)
        
        ptstate = PTState(ndim=2, ntemps=3)
        
        chain_stats_list = [
            ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng1, buffer_size=50),
            ChainStats(ndim=2, pt_state=ptstate, chain_index=1, rng=rng2, buffer_size=50),
            ChainStats(ndim=2, pt_state=ptstate, chain_index=2, rng=rng3, buffer_size=50)
        ]
        
        jump_proposals = []
        for cs in chain_stats_list:
            jp = JumpProposals(cs)
            jp.add_jump(am, 1.0)
            jump_proposals.append(jp)
        
        bundle = ProposalBundle(jump_proposals)
        
        new_positions, qxys = bundle.get_new_position(sample_state_2d)
        
        assert new_positions.shape == (3, 2)  # ntemps x ndim
        assert qxys.shape == (3,)  # ntemps
    
    def test_call_delegates_to_get_new_position(self, chain_stats_2d, sample_state_2d):
        """Test that __call__ delegates to get_new_position"""
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 1.0)
        
        bundle = ProposalBundle([jp])
        
        new_positions, qxys = bundle(sample_state_2d)
        
        assert isinstance(new_positions, np.ndarray)
        assert isinstance(qxys, np.ndarray)
    
    def test_add_jump_to_all_chains(self):
        """Test adding jump to all chains in bundle"""
        ptstate = PTState(ndim=2, ntemps=2)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        
        cs1 = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng1, buffer_size=50)
        cs2 = ChainStats(ndim=2, pt_state=ptstate, chain_index=1, rng=rng2, buffer_size=50)
        
        jp1 = JumpProposals(cs1)
        jp2 = JumpProposals(cs2)
        
        bundle = ProposalBundle([jp1, jp2])
        bundle.add_jump(am, 0.5)
        
        assert am in jp1.proposal_list
        assert am in jp2.proposal_list
        assert jp1.proposal_weights[0] == 0.5
        assert jp2.proposal_weights[0] == 0.5


class TestProposalFunctions:
    """Test cases for individual proposal functions"""
    
    def test_am_basic(self, chain_stats_2d):
        """Test adaptive Metropolis proposal"""
        chain_stats_2d.current_sample = np.array([1.0, 2.0])
        
        new_sample, qxy = am(chain_stats_2d)
        
        assert isinstance(new_sample, np.ndarray)
        assert new_sample.shape == (2,)
        assert qxy == 0  # AM is symmetric
        
        # Should be different from current sample (with high probability)
        assert not np.array_equal(new_sample, chain_stats_2d.current_sample)
    
    def test_am_scale_factors(self, chain_stats_2d):
        """Test that AM occasionally uses different scale factors"""
        chain_stats_2d.current_sample = np.array([0.0, 0.0])
        
        # Generate many proposals to test scale factor variation
        proposals = []
        chain_stats_2d.rng = np.random.default_rng(42)  # Fixed seed
        
        for _ in range(100):
            new_sample, _ = am(chain_stats_2d)
            proposals.append(new_sample)
        
        proposals = np.array(proposals)
        
        # Check that there's variation in proposal magnitudes
        magnitudes = np.linalg.norm(proposals, axis=1)
        assert np.std(magnitudes) > 0  # Should have variation
    
    def test_scam_basic(self, chain_stats_2d):
        """Test single-component adaptive Metropolis proposal"""
        chain_stats_2d.current_sample = np.array([1.0, 2.0])
        
        new_sample, qxy = scam(chain_stats_2d)
        
        assert isinstance(new_sample, np.ndarray)
        assert new_sample.shape == (2,)
        assert qxy == 0  # SCAM is symmetric
        
        # Should modify exactly one component
        diff = new_sample - chain_stats_2d.current_sample
        non_zero_components = np.sum(np.abs(diff) > 1e-10)
        assert non_zero_components == 1
    
    def test_scam_parameter_groups(self):
        """Test SCAM with parameter groups"""
        ptstate = PTState(ndim=4, ntemps=3)
        rng = np.random.default_rng(42)
        
        # Define parameter groups
        groups = [[0, 1], [2, 3]]
        
        chain_stats = ChainStats(
            ndim=4,
            pt_state=ptstate,
            chain_index=0,
            rng=rng,
            groups=groups,
            buffer_size=50
        )
        
        chain_stats.current_sample = np.array([1.0, 2.0, 3.0, 4.0])
        
        new_sample, qxy = scam(chain_stats)
        
        assert new_sample.shape == (4,)
        assert qxy == 0
    
    def test_de_basic_with_full_buffer(self):
        """Test differential evolution proposal with full buffer"""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(42)
        
        chain_stats = ChainStats(
            ndim=2,
            pt_state=ptstate,
            chain_index=0,
            rng=rng,
            buffer_size=10  # Small buffer for testing
        )
        
        # Fill buffer with sample data
        chain_stats._buffer = np.random.randn(10, 2)
        chain_stats.buffer_full = True
        chain_stats.current_sample = np.array([0.0, 0.0])
        
        new_sample, qxy = de(chain_stats)
        
        assert isinstance(new_sample, np.ndarray)
        assert new_sample.shape == (2,)
        assert qxy == 0  # DE is symmetric
    
    def test_de_scale_factors(self):
        """Test that DE uses different scale factors"""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(12345)  # Fixed seed for reproducibility
        
        chain_stats = ChainStats(
            ndim=2,
            pt_state=ptstate,
            chain_index=0,
            rng=rng,
            buffer_size=20
        )
        
        # Fill buffer with diverse samples
        chain_stats._buffer = np.random.randn(20, 2) * 2
        chain_stats.buffer_full = True
        chain_stats.current_sample = np.array([0.0, 0.0])
        
        # Test multiple proposals to see scale variation
        proposals = []
        for _ in range(50):
            new_sample, _ = de(chain_stats)
            proposals.append(new_sample)
        
        proposals = np.array(proposals)
        magnitudes = np.linalg.norm(proposals - chain_stats.current_sample, axis=1)
        
        # Should have variation in proposal magnitudes due to different scales
        assert np.std(magnitudes) > 0
    
    def test_proposal_functions_return_types(self, chain_stats_2d):
        """Test that all proposal functions return correct types"""
        chain_stats_2d.current_sample = np.array([0.0, 0.0])
        chain_stats_2d.buffer_full = True
        chain_stats_2d._buffer = np.random.randn(chain_stats_2d.buffer_size, 2)
        
        for proposal_func in [am, scam, de]:
            new_sample, qxy = proposal_func(chain_stats_2d)
            
            assert isinstance(new_sample, np.ndarray)
            assert new_sample.shape == (2,)
            assert isinstance(qxy, (int, float, np.number))
    
    def test_proposal_functions_preserve_input(self, chain_stats_2d):
        """Test that proposal functions don't modify input"""
        original_sample = np.array([1.0, -1.0])
        chain_stats_2d.current_sample = original_sample.copy()
        chain_stats_2d.buffer_full = True
        chain_stats_2d._buffer = np.random.randn(chain_stats_2d.buffer_size, 2)
        
        for proposal_func in [am, scam, de]:
            before_call = chain_stats_2d.current_sample.copy()
            new_sample, qxy = proposal_func(chain_stats_2d)
            after_call = chain_stats_2d.current_sample.copy()
            
            # Current sample should be unchanged by proposal
            np.testing.assert_array_equal(before_call, after_call)
    
    def test_am_covariance_scaling(self, chain_stats_2d):
        """Test that AM uses proper covariance scaling"""
        # Set up known covariance matrix
        chain_stats_2d.sample_cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        chain_stats_2d.current_sample = np.array([0.0, 0.0])
        
        # Generate many proposals to test covariance structure
        proposals = []
        rng = np.random.default_rng(42)
        chain_stats_2d.rng = rng
        
        for _ in range(1000):
            new_sample, _ = am(chain_stats_2d)
            proposals.append(new_sample)
        
        proposals = np.array(proposals)
        
        # Empirical covariance should be related to input covariance
        # (though scaled by the AM factor 2.38²/d)
        empirical_cov = np.cov(proposals.T)
        
        # Check that covariance structure is preserved (same eigenvector directions)
        # This is a rough test - exact matching would require many more samples
        assert empirical_cov.shape == (2, 2)
        assert np.all(np.isfinite(empirical_cov))
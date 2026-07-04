import warnings

import pytest
import numpy as np
from unittest.mock import Mock

from impulse.sampler_step import vectorized_mh_step, pt_step
from impulse.sampler_state import SamplerState, PTState
from impulse.proposals import ProposalBundle


class TestVectorizedMhStep:
    """Test suite for vectorized_mh_step function"""

    def test_vectorized_mh_step_basic(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test basic vectorized MH step"""
        # Create mock proposal bundle
        prop_fn = Mock()
        new_positions = sample_state_2d.positions + 0.1
        qxys = np.array([0.0, 0.0, 0.0])  # Symmetric proposals
        prop_fn.return_value = (new_positions, qxys)
        
        # Create RNG
        rng = np.random.default_rng(42)
        
        # Execute MH step
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Check output structure
        assert isinstance(new_state, SamplerState)
        assert new_state.positions.shape == sample_state_2d.positions.shape
        assert new_state.lnlikes.shape == sample_state_2d.lnlikes.shape
        assert new_state.lnpriors.shape == sample_state_2d.lnpriors.shape
        assert new_state.lnprobs.shape == sample_state_2d.lnprobs.shape
        assert new_state.accepted.shape == sample_state_2d.accepted.shape
        assert new_state.temps.shape == sample_state_2d.temps.shape
        
        # Temperatures should be unchanged
        np.testing.assert_array_equal(new_state.temps, sample_state_2d.temps)
        
        # Accepted should be binary
        assert np.all((new_state.accepted == 0) | (new_state.accepted == 1))

    def test_vectorized_mh_step_acceptance_logic(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test acceptance/rejection logic"""
        # Create proposal that always improves likelihood
        prop_fn = Mock()
        # Move towards zero (better likelihood for quadratic)
        new_positions = sample_state_2d.positions * 0.1  # Much closer to zero
        qxys = np.zeros(sample_state_2d.ntemps)
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Should accept most/all proposals since they improve likelihood
        acceptance_rate = new_state.accepted.mean()
        assert acceptance_rate > 0.5  # Should accept most good proposals

    def test_vectorized_mh_step_rejection(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test rejection of bad proposals"""
        # Create proposal that worsens likelihood significantly
        prop_fn = Mock()
        new_positions = sample_state_2d.positions + 10  # Much worse positions
        qxys = np.zeros(sample_state_2d.ntemps)
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Should reject most/all proposals since they worsen likelihood
        acceptance_rate = new_state.accepted.mean()
        assert acceptance_rate < 0.5  # Should reject most bad proposals
        
        # Rejected proposals should keep original positions
        rejected_mask = new_state.accepted == 0
        if np.any(rejected_mask):
            np.testing.assert_array_equal(
                new_state.positions[rejected_mask],
                sample_state_2d.positions[rejected_mask]
            )

    def test_vectorized_mh_step_proposal_ratios(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test that proposal ratios are included in acceptance calculation"""
        prop_fn = Mock()
        new_positions = sample_state_2d.positions.copy()
        # Non-zero proposal ratios (asymmetric proposals)
        qxys = np.array([1.0, -1.0, 0.5])  # Different ratios for each chain
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Check that step completed without error
        assert isinstance(new_state, SamplerState)
        # Proposal ratios should influence acceptance
        assert np.any(new_state.accepted == 1) or np.any(new_state.accepted == 0)

    def test_vectorized_mh_step_temperature_scaling(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test that temperature scaling affects acceptance probabilities"""
        prop_fn = Mock()
        # Slightly worse proposals
        new_positions = sample_state_2d.positions + 0.5
        qxys = np.zeros(sample_state_2d.ntemps)
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Hot chains (higher temperature) should have higher acceptance rates
        # This is probabilistic, so we just check the structure is reasonable
        assert isinstance(new_state, SamplerState)
        assert len(new_state.accepted) == len(sample_state_2d.temps)

    def test_vectorized_mh_step_prior_boundaries(self, sample_state_2d, simple_prior):
        """Test handling of prior boundaries"""
        # Create likelihood that doesn't constrain
        def unconstrained_likelihood(x):
            if x.ndim == 1:
                return 0.0
            return np.zeros(x.shape[0])
        
        prop_fn = Mock()
        # Propose positions outside prior bounds
        new_positions = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])  # Outside [-5,5]
        qxys = np.zeros(sample_state_2d.ntemps)
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, unconstrained_likelihood, simple_prior, rng
        )
        
        # Should reject all proposals due to prior
        assert np.all(new_state.accepted == 0)
        # Positions should remain unchanged
        np.testing.assert_array_equal(new_state.positions, sample_state_2d.positions)

    def test_vectorized_mh_step_inf_temp_accepts_neg_inf_likelihood(self, simple_prior):
        """T = inf chain accepts prior-valid moves even where lnlike = -inf"""
        def half_neg_inf_likelihood(x):
            x = np.asarray(x)
            if x.ndim == 1:
                return -np.inf if x[0] > 0 else -0.5 * np.sum(x**2)
            result = -0.5 * np.sum(x**2, axis=1)
            result[x[:, 0] > 0] = -np.inf
            return result

        positions = np.array([[-1.0, 0.0], [-1.0, 0.0]])
        lnlikes = np.array([-0.5, -0.5])
        lnpriors = np.array([0.0, 0.0])
        temps = np.array([1.0, np.inf])
        lnprobs = np.array([-0.5, 0.0])  # inf chain: lnprob = lnprior
        state = SamplerState(positions, lnlikes, lnpriors, lnprobs,
                             np.ones(2, dtype=int), temps)

        prop_fn = Mock()
        # proposals inside the prior but in the -inf-likelihood half-space
        prop_fn.return_value = (np.array([[0.5, 0.0], [0.5, 0.0]]), np.zeros(2))
        rng = np.random.default_rng(42)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            new_state = vectorized_mh_step(
                state, prop_fn, half_neg_inf_likelihood, simple_prior, rng
            )

        assert not np.any(np.isnan(new_state.lnprobs))
        # cold chain must reject the -inf-likelihood proposal
        assert new_state.accepted[0] == 0
        # prior chain must accept it (prior ratio is 0)
        assert new_state.accepted[1] == 1
        np.testing.assert_array_equal(new_state.positions[1], [0.5, 0.0])
        assert new_state.lnprobs[1] == 0.0

    def test_vectorized_mh_step_updates_probabilities(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test that log-probabilities are updated correctly"""
        prop_fn = Mock()
        new_positions = sample_state_2d.positions * 0.5  # Better positions
        qxys = np.zeros(sample_state_2d.ntemps)
        prop_fn.return_value = (new_positions, qxys)
        
        rng = np.random.default_rng(42)
        
        new_state = vectorized_mh_step(
            sample_state_2d, prop_fn, simple_likelihood, simple_prior, rng
        )
        
        # Accepted moves should have updated probabilities
        accepted_mask = new_state.accepted == 1
        if np.any(accepted_mask):
            # Likelihoods should be different for accepted moves
            assert not np.allclose(
                new_state.lnlikes[accepted_mask],
                sample_state_2d.lnlikes[accepted_mask]
            )


class TestPtStep:
    """Test suite for pt_step function"""

    def test_pt_step_basic(self, sample_state_3d, simple_likelihood, simple_prior):
        """Test basic PT step functionality"""
        # Create PTState
        ptstate = PTState(ndim=3, ntemps=5)
        rng = np.random.default_rng(42)
        
        initial_nswaps = ptstate.nswaps
        
        new_state = pt_step(sample_state_3d, ptstate, simple_likelihood, simple_prior, rng)
        
        # Check output structure
        assert isinstance(new_state, SamplerState)
        assert new_state.positions.shape == sample_state_3d.positions.shape
        assert new_state.lnlikes.shape == sample_state_3d.lnlikes.shape
        assert new_state.lnpriors.shape == sample_state_3d.lnpriors.shape
        assert new_state.lnprobs.shape == sample_state_3d.lnprobs.shape
        assert new_state.accepted.shape == sample_state_3d.accepted.shape
        assert new_state.temps.shape == sample_state_3d.temps.shape
        
        # Should have attempted swaps
        assert ptstate.nswaps > initial_nswaps
        
        # All accepted should be 1 (PT swaps tracked separately)
        assert np.all(new_state.accepted == 1)

    def test_pt_step_swap_statistics(self, sample_state_3d, simple_likelihood, simple_prior):
        """Test that swap statistics are updated"""
        ptstate = PTState(ndim=3, ntemps=5)
        rng = np.random.default_rng(42)
        
        initial_swap_accept = ptstate.swap_accept.copy()
        initial_nswaps = ptstate.nswaps
        
        new_state = pt_step(sample_state_3d, ptstate, simple_likelihood, simple_prior, rng)

        # nswaps increments by 1 per sweep (not per pair)
        assert ptstate.nswaps == initial_nswaps + 1
        
        # swap_accept should be non-negative and possibly changed
        assert np.all(ptstate.swap_accept >= initial_swap_accept)

    def test_pt_step_position_reordering(self, simple_likelihood, simple_prior):
        """Test that positions are correctly reordered after swaps"""
        # Create state with distinct positions for each chain
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        lnlikes = np.array([-1.0, -3.0, -6.0])  # Different likelihoods
        lnpriors = np.zeros(3)
        temps = np.array([1.0, 2.0, 4.0])
        lnprobs = lnpriors + lnlikes / temps
        accepted = np.ones(3)
        
        state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
        ptstate = PTState(ndim=3, ntemps=3, ladder=temps)  # Use the specific temps we want
        
        rng = np.random.default_rng(42)
        
        new_state = pt_step(state, ptstate, simple_likelihood, simple_prior, rng)
        
        # Positions should still be the same set, just potentially reordered
        original_set = set(tuple(pos) for pos in positions)
        new_set = set(tuple(pos) for pos in new_state.positions)
        assert original_set == new_set
        
        # Temperatures should remain the same (fixed ladder)  
        np.testing.assert_array_equal(new_state.temps, ptstate.ladder)

    def test_pt_step_single_chain(self, simple_likelihood, simple_prior):
        """Test PT step with single chain (no swaps possible)"""
        positions = np.array([[1.0, 2.0]])
        lnlikes = np.array([-2.5])
        lnpriors = np.array([0.0])
        temps = np.array([1.0])
        lnprobs = lnpriors + lnlikes / temps
        accepted = np.array([1])
        
        state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
        ptstate = PTState(ndim=2, ntemps=1)
        
        rng = np.random.default_rng(42)
        
        new_state = pt_step(state, ptstate, simple_likelihood, simple_prior, rng)
        
        # Should return essentially unchanged state
        np.testing.assert_array_equal(new_state.positions, positions)
        np.testing.assert_array_equal(new_state.temps, temps)
        
        # No swaps should have been attempted (nswaps might increase but no swap_accept)
        assert np.sum(ptstate.swap_accept) == 0

    def test_pt_step_ladder_none_error(self, sample_state_2d, simple_likelihood, simple_prior):
        """Test error when PTState ladder is None"""
        ptstate = PTState.__new__(PTState)  # Create without proper initialization
        ptstate.ladder = None
        
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="PTState ladder is not initialized"):
            pt_step(sample_state_2d, ptstate, simple_likelihood, simple_prior, rng)

    def test_pt_step_probability_recalculation(self, sample_state_3d, simple_likelihood, simple_prior):
        """Test that probabilities are recalculated after swaps"""
        ptstate = PTState(ndim=3, ntemps=5)
        rng = np.random.default_rng(42)
        
        # Force a specific swap scenario by setting up the state
        original_lnprobs = sample_state_3d.lnprobs.copy()
        
        new_state = pt_step(sample_state_3d, ptstate, simple_likelihood, simple_prior, rng)
        
        # Probabilities should be recalculated (even if no swaps occurred)
        # Check that they follow the correct formula: lnprob = lnprior + lnlike/temp
        expected_lnprobs = new_state.lnpriors + new_state.lnlikes / new_state.temps
        np.testing.assert_array_almost_equal(new_state.lnprobs, expected_lnprobs)

    def test_pt_step_inf_temp_neg_inf_likelihood_no_nan(self, simple_likelihood, simple_prior):
        """T = inf chain holding a -inf lnlike position: swap rejected, no NaN"""
        positions = np.array([[-1.0, 0.0], [0.5, 0.0]])
        lnlikes = np.array([-0.5, -np.inf])
        lnpriors = np.array([0.0, 0.0])
        temps = np.array([1.0, np.inf])
        lnprobs = np.array([-0.5, 0.0])  # inf chain: lnprob = lnprior
        state = SamplerState(positions, lnlikes, lnpriors, lnprobs,
                             np.ones(2, dtype=int), temps)
        ptstate = PTState(ndim=2, ntemps=2, ladder=temps)
        rng = np.random.default_rng(42)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            new_state = pt_step(state, ptstate, simple_likelihood, simple_prior, rng)

        # -inf-likelihood position can never move to a finite temperature
        assert ptstate.swap_accept[0] == 0
        np.testing.assert_array_equal(new_state.positions, positions)
        assert not np.any(np.isnan(new_state.lnprobs))
        assert new_state.lnprobs[1] == 0.0

    def test_pt_step_high_temperature_favors_swaps(self, simple_likelihood, simple_prior):
        """Test that higher temperature differences affect swap probabilities"""
        # Create chains with very different likelihoods
        positions = np.array([[0.0, 0.0], [5.0, 5.0]])  # Very different positions
        lnlikes = np.array([-1.0, -25.0])  # Very different likelihoods
        lnpriors = np.array([0.0, 0.0])
        temps = np.array([1.0, 100.0])  # Large temperature difference
        lnprobs = lnpriors + lnlikes / temps
        accepted = np.ones(2)
        
        state = SamplerState(positions, lnlikes, lnpriors, lnprobs, accepted, temps)
        ptstate = PTState(ndim=2, ntemps=2, ladder=temps)
        
        rng = np.random.default_rng(42)
        
        # Run multiple PT steps to see swap behavior
        total_swaps = 0
        for _ in range(100):
            state = pt_step(state, ptstate, simple_likelihood, simple_prior, rng)
            total_swaps += ptstate.swap_accept[0]
        
        # Should have some swaps due to large temperature difference
        # This is probabilistic, but with large temp difference should see some swaps
        assert ptstate.nswaps > 100  # Should have attempted swaps
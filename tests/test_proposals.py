import pickle
import pytest
import numpy as np
from impulse.proposals import (
    EarlyDE,
    JumpProposals,
    ProposalBundle,
    am,
    de,
    gaussian,
    make_early_de,
    scam,
)
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


class TestProposalAcceptanceTracking:
    """Test per-proposal acceptance counter logic in JumpProposals."""

    def test_counters_initialized_to_zero(self, chain_stats_2d):
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 15)
        jp.add_jump(scam, 30)
        assert jp._last_proposal_idx == -1
        np.testing.assert_array_equal(jp._proposal_calls, [0, 0])
        np.testing.assert_array_equal(jp._proposal_accepts, [0, 0])

    def test_calls_attributed_to_correct_proposal(self, sample_state_2d):
        """With multiple proposals, each call is attributed to the selected one."""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(42)
        cs = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)
        jp = JumpProposals(cs)
        jp.add_jump(am, 50)
        jp.add_jump(scam, 50)

        n_iters = 200
        for _ in range(n_iters):
            jp(sample_state_2d)

        # Total calls across proposals must equal n_iters
        assert jp._proposal_calls.sum() == n_iters
        # With 50/50 weights and 200 draws both should get selected at least once
        assert jp._proposal_calls[0] > 0, "am never selected"
        assert jp._proposal_calls[1] > 0, "scam never selected"
        # _last_proposal_idx should be a valid index
        assert jp._last_proposal_idx in (0, 1)

    def test_report_accept_credits_correct_proposal(self, sample_state_2d):
        """report_accept credits the proposal that was last called, not always idx 0."""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(42)
        cs = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)
        jp = JumpProposals(cs)
        jp.add_jump(am, 50)
        jp.add_jump(scam, 50)

        am_accepts = 0
        scam_accepts = 0
        for _ in range(200):
            jp(sample_state_2d)
            # Always accept — each accept should credit the last-selected proposal
            jp.report_accept(True)
            if jp._last_proposal_idx == 0:
                am_accepts += 1
            else:
                scam_accepts += 1

        assert jp._proposal_accepts[0] == am_accepts
        assert jp._proposal_accepts[1] == scam_accepts
        # Sanity: both were called (same check as above, but for accepts)
        assert am_accepts > 0
        assert scam_accepts > 0

    def test_report_accept_false_does_not_increment(self, chain_stats_2d, sample_state_2d):
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 1.0)

        jp(sample_state_2d)
        jp.report_accept(False)
        assert jp._proposal_accepts[0] == 0

        jp(sample_state_2d)
        jp.report_accept(True)
        assert jp._proposal_accepts[0] == 1

        jp(sample_state_2d)
        jp.report_accept(False)
        # Still 1 — the False should not have changed it
        assert jp._proposal_accepts[0] == 1

    def test_acceptance_rates_multiple_proposals(self, sample_state_2d):
        """acceptance_rates returns correct per-proposal rates with mixed accept/reject."""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(99)
        cs = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)
        jp = JumpProposals(cs)
        jp.add_jump(am, 50)
        jp.add_jump(scam, 50)

        # Run enough iterations and accept every other call
        for i in range(100):
            jp(sample_state_2d)
            jp.report_accept(i % 2 == 0)

        rates = jp.acceptance_rates()
        assert set(rates.keys()) == {'am', 'scam'}
        total_calls = rates['am']['calls'] + rates['scam']['calls']
        total_accepts = rates['am']['accepts'] + rates['scam']['accepts']
        assert total_calls == 100
        assert total_accepts == 50  # every other call accepted
        for name in ('am', 'scam'):
            assert rates[name]['rate'] == pytest.approx(
                rates[name]['accepts'] / rates[name]['calls'], abs=1e-12
            )

    def test_de_fallback_attributed_to_de_slot(self, sample_state_2d):
        """When DE falls back to gaussian, the call is still counted under the DE slot."""
        ptstate = PTState(ndim=2, ntemps=3)
        rng = np.random.default_rng(7)
        cs = ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=50)
        cs.buffer_full = False  # force DE to always fall back

        jp = JumpProposals(cs)
        jp.add_jump(de, 1.0)  # only DE, but it will always fall back to gaussian

        for _ in range(20):
            jp(sample_state_2d)

        assert jp._proposal_calls[0] == 20  # all counted under de's slot
        assert jp._last_proposal_idx == 0
        # gaussian is not in proposal_list, so it should have no slot
        rates = jp.acceptance_rates()
        assert 'de' in rates
        assert 'gaussian' not in rates

    def test_counters_survive_pickle(self, chain_stats_2d, sample_state_2d):
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 50)
        jp.add_jump(scam, 50)

        for _ in range(10):
            jp(sample_state_2d)
            jp.report_accept(True)

        jp2 = pickle.loads(pickle.dumps(jp))
        np.testing.assert_array_equal(jp2._proposal_calls, jp._proposal_calls)
        np.testing.assert_array_equal(jp2._proposal_accepts, jp._proposal_accepts)
        assert jp2._last_proposal_idx == jp._last_proposal_idx

    def test_setstate_handles_old_checkpoint(self, chain_stats_2d):
        jp = JumpProposals(chain_stats_2d)
        jp.add_jump(am, 15)
        jp.add_jump(scam, 30)

        # Simulate an old checkpoint missing counter attrs
        state = jp.__dict__.copy()
        del state['_last_proposal_idx']
        del state['_proposal_calls']
        del state['_proposal_accepts']

        jp2 = JumpProposals.__new__(JumpProposals)
        jp2.__setstate__(state)

        assert jp2._last_proposal_idx == -1
        np.testing.assert_array_equal(jp2._proposal_calls, [0, 0])
        np.testing.assert_array_equal(jp2._proposal_accepts, [0, 0])
        # Should be usable immediately after restore
        rates = jp2.acceptance_rates()
        assert rates['am']['calls'] == 0
        assert rates['scam']['calls'] == 0


class TestProposalBundleAcceptanceReport:
    """Test ProposalBundle.report_accepts and acceptance_report."""

    def _make_bundle(self, ntemps=2):
        """Helper: create a bundle with am + scam on each chain."""
        ptstate = PTState(ndim=2, ntemps=ntemps)
        jps = []
        for i in range(ntemps):
            rng = np.random.default_rng(42 + i)
            cs = ChainStats(ndim=2, pt_state=ptstate, chain_index=i, rng=rng, buffer_size=50)
            jp = JumpProposals(cs)
            jp.add_jump(am, 50)
            jp.add_jump(scam, 50)
            jps.append(jp)
        state = SamplerState(
            np.zeros((ntemps, 2)),
            -np.ones(ntemps),
            np.zeros(ntemps),
            -np.ones(ntemps),
            np.ones(ntemps),
            ptstate.ladder,
        )
        return ProposalBundle(jps), state, jps

    def test_report_accepts_per_chain(self):
        """report_accepts dispatches True/False independently to each chain."""
        bundle, state, jps = self._make_bundle(ntemps=3)

        bundle(state)
        # chain 0 accepted, chain 1 rejected, chain 2 accepted
        bundle.report_accepts(np.array([1, 0, 1]))

        # chain 0: last proposal should have 1 accept
        idx0 = jps[0]._last_proposal_idx
        assert jps[0]._proposal_accepts[idx0] == 1
        # chain 1: no accepts
        assert jps[1]._proposal_accepts.sum() == 0
        # chain 2: last proposal should have 1 accept
        idx2 = jps[2]._last_proposal_idx
        assert jps[2]._proposal_accepts[idx2] == 1

    def test_acceptance_report_aggregates_multiple_proposals(self):
        """acceptance_report sums calls/accepts across chains for each proposal name."""
        bundle, state, jps = self._make_bundle(ntemps=2)

        n_iters = 100
        for i in range(n_iters):
            bundle(state)
            # chain 0 always accepts, chain 1 always rejects
            bundle.report_accepts(np.array([1, 0]))

        report = bundle.acceptance_report()
        assert set(report.keys()) == {'am', 'scam'}

        # Total calls across all proposals and chains must equal n_iters * ntemps
        total_calls = sum(report[name]['calls'] for name in report)
        assert total_calls == n_iters * 2

        # Total accepts = chain 0's accepts (chain 1 has 0)
        total_accepts = sum(report[name]['accepts'] for name in report)
        assert total_accepts == n_iters  # chain 0 accepted all n_iters

        # Per-chain breakdown present
        for name in ('am', 'scam'):
            assert len(report[name]['per_chain']) == 2
            # rate is consistent with calls/accepts
            if report[name]['calls'] > 0:
                assert report[name]['rate'] == pytest.approx(
                    report[name]['accepts'] / report[name]['calls'], abs=1e-12
                )

    def test_acceptance_report_zero_calls(self):
        """acceptance_report handles proposals with zero calls (rate=0)."""
        bundle, state, jps = self._make_bundle(ntemps=1)

        # No calls made — everything should be zero
        report = bundle.acceptance_report()
        for name in ('am', 'scam'):
            assert report[name]['calls'] == 0
            assert report[name]['accepts'] == 0
            assert report[name]['rate'] == 0.0

    def test_save_chain_acceptance_rates_writes_json(self, tmp_path):
        """End-to-end: a PTSampler run writes a usable JSON acceptance file."""
        import json
        import os
        from impulse.samplers import PTSampler

        def lnlike(x):
            return float(-0.5 * (x[0] ** 2 + x[1] ** 2))

        def lnprior(x):
            return -np.inf if np.any(np.abs(x) > 5) else 0.0

        outdir = str(tmp_path / "chains")
        s = PTSampler(
            ndim=2, lnlike=lnlike, lnprior=lnprior,
            am_weight=15, scam_weight=30, de_weight=50,
            ntemps=3, min_temp=1.0, max_temp=4.0,
            seed=0, outdir=outdir, buffer_size=500, save_freq=2000,
        )
        s.sample(np.zeros((3, 2)), num_iterations=1500)

        path = os.path.join(outdir, "chain_acceptance.json")
        assert os.path.exists(path)
        with open(path) as f:
            rep = json.load(f)
        assert set(rep.keys()) >= {"temperatures", "mh", "pt_swap", "per_proposal"}
        assert len(rep["temperatures"]) == 3
        assert len(rep["mh"]) == 3
        for ch in rep["mh"]:
            assert {"calls", "accepts", "rate", "per_proposal"} <= set(ch.keys())
            assert ch["calls"] > 0
            assert 0.0 <= ch["rate"] <= 1.0
            assert {"am", "scam", "de"} <= set(ch["per_proposal"].keys())
        assert len(rep["pt_swap"]) == 2  # ntemps - 1 neighbour pairs

    def test_chain_acceptance_rates_per_chain(self):
        """chain_acceptance_rates summarises by chain, including per-proposal."""
        bundle, state, jps = self._make_bundle(ntemps=3)

        n_iters = 50
        for _ in range(n_iters):
            bundle(state)
            # chain 0 always accepts, chain 1 always rejects, chain 2 accepts half
            accepts = np.array([1, 0, 1 if _ % 2 == 0 else 0])
            bundle.report_accepts(accepts)

        rates = bundle.chain_acceptance_rates()
        assert len(rates) == 3
        # Chain-level totals match
        assert rates[0]['calls'] == n_iters
        assert rates[0]['accepts'] == n_iters
        assert rates[0]['rate'] == pytest.approx(1.0)
        assert rates[1]['rate'] == pytest.approx(0.0)
        assert rates[2]['rate'] == pytest.approx(0.5, abs=0.05)
        # per_proposal field present and keyed by proposal name
        for ch in rates:
            assert set(ch['per_proposal'].keys()) == {'am', 'scam'}

class TestEarlyDE:
    """Test cases for the min-fill-gated differential evolution proposal."""

    def _stats(self, buffer_size=100, ndim=2, seed=42):
        ptstate = PTState(ndim=ndim, ntemps=3)
        rng = np.random.default_rng(seed)
        return ChainStats(
            ndim=ndim, pt_state=ptstate, chain_index=0, rng=rng,
            buffer_size=buffer_size,
        )

    def test_name_is_not_de(self):
        """__name__ must differ from 'de': JumpProposals substitutes gaussian
        for proposals named 'de' whenever the buffer is not full, which would
        defeat the min-fill gating."""
        prop = make_early_de()
        assert prop.__name__ == 'early_de'

    def test_identity_below_min_fill(self):
        """Below min_fill the proposal is the identity kernel with qxy=0."""
        cs = self._stats()
        cs.current_sample = np.array([1.0, 2.0])
        cs.sample_total = 5  # < min_fill
        prop = make_early_de(min_fill=10)

        new_sample, qxy = prop(cs)

        np.testing.assert_array_equal(new_sample, cs.current_sample)
        assert new_sample is not cs.current_sample  # a copy, not an alias
        assert qxy == 0.0

    def test_difference_move_from_partial_buffer_tail(self):
        """Above min_fill the move is a difference of two TAIL buffer rows.

        The circular buffer fills from the tail, so the head of a partially
        filled buffer is zero padding; drawing from the head (as the stock
        de indexes it) would produce degenerate near-zero jumps.
        """
        cs = self._stats(buffer_size=100)
        rng_fill = np.random.default_rng(7)
        n_filled = 20
        # head of buffer left as zero padding; tail holds the history
        cs._buffer[-n_filled:] = 5.0 + rng_fill.random((n_filled, 2))
        cs.sample_total = n_filled
        cs.current_sample = np.array([1.0, 2.0])
        prop = make_early_de(min_fill=10)

        moved = 0
        for _ in range(50):
            new_sample, qxy = prop(cs)
            assert qxy == 0.0
            delta = new_sample - cs.current_sample
            if np.any(delta != 0.0):
                moved += 1
                # every delta must be a scaled difference of tail rows:
                # tail values are in [5, 6], so |row_i - row_j| < 1 per
                # coordinate and scale <= 1  =>  |delta| < 1.  A head
                # (zero-padding) row would give |delta| >= 4.
                assert np.max(np.abs(delta)) < 1.0
        assert moved > 0

    def test_min_fill_validation(self):
        """min_fill < 2 cannot produce two distinct rows and must raise."""
        with pytest.raises(ValueError, match="min_fill"):
            make_early_de(min_fill=1)

    def test_respects_groups(self):
        """Only the chosen group's parameters move; others are untouched."""
        ptstate = PTState(ndim=4, ntemps=3)
        rng = np.random.default_rng(3)
        groups = [[0, 1], [2, 3]]
        cs = ChainStats(ndim=4, pt_state=ptstate, chain_index=0, rng=rng,
                        groups=groups, buffer_size=50)
        cs._buffer[-20:] = np.random.default_rng(4).random((20, 4))
        cs.sample_total = 20
        cs.current_sample = np.zeros(4)
        prop = make_early_de(min_fill=10)

        for _ in range(50):
            new_sample, _ = prop(cs)
            delta = new_sample != 0.0
            # a single group per call: never parameters from both groups
            assert not (np.any(delta[:2]) and np.any(delta[2:]))

    def test_pickle_roundtrip(self):
        """Checkpointing pickles every registered proposal."""
        prop = make_early_de(min_fill=37)
        restored = pickle.loads(pickle.dumps(prop))
        assert isinstance(restored, EarlyDE)
        assert restored.min_fill == 37
        assert restored.__name__ == 'early_de'

    def test_symmetry_forward_reverse_rates(self):
        """Empirical check of q(y|x) = q(x|y) (the qxy = 0 claim).

        With a frozen two-row buffer and scale fixed to the mode-jump
        branch (prob > 0.5 -> scale = 1), the only proposable moves from x
        are x + (b0 - b1) and x + (b1 - b0), each with probability 1/2 (of
        the mode-jump branch): the forward and reverse displacement are
        proposed at identical rates from any point.
        """
        cs = self._stats(buffer_size=10, seed=11)
        cs._buffer[-2:] = np.array([[0.3, 0.1], [0.1, 0.4]])
        cs.sample_total = 2
        cs.current_sample = np.array([0.0, 0.0])
        prop = make_early_de(min_fill=2)

        diff = cs._buffer[-2] - cs._buffer[-1]
        n_fwd = n_rev = 0
        for _ in range(4000):
            new_sample, qxy = prop(cs)
            assert qxy == 0.0
            delta = new_sample - cs.current_sample
            if np.allclose(delta, diff):
                n_fwd += 1
            elif np.allclose(delta, -diff):
                n_rev += 1
        # mode-jump branch fires ~half the time, split evenly across signs
        assert n_fwd + n_rev > 1000
        assert abs(n_fwd - n_rev) / (n_fwd + n_rev) < 0.1

from unittest.mock import Mock

import numpy as np
import pytest

from impulse.chain_stats import ChainStats, MultiChainStats, _HistoryBuffer
from impulse.sampler_state import PTState, SamplerState


class TestChainStats:
    """Test suite for ChainStats class"""

    def test_chain_stats_init_basic(self, ptstate_2d):
        """Test basic ChainStats initialization"""
        rng = np.random.default_rng(42)

        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        assert stats.ndim == 2
        assert stats.pt_state is ptstate_2d
        assert stats.chain_index == 0
        assert stats.rng is rng
        assert stats.sample_total == 0
        assert stats.buffer_size == 50_000  # default
        assert stats.buffer_full == False

        # Check default initialization
        assert stats.sample_cov.shape == (2, 2)
        np.testing.assert_array_equal(stats.sample_cov, np.eye(2))
        np.testing.assert_array_equal(stats.sample_mean, np.zeros(2))
        # Check default groups - should be [array([0, 1])]
        assert len(stats.groups) == 1
        np.testing.assert_array_equal(stats.groups[0], np.arange(2))
        assert len(stats.svd_U) == 1
        assert len(stats.svd_S) == 1
        assert len(stats.proposal_L) == 1
        assert stats.proposal_L[0] is not None
        assert stats.proposal_L[0].shape == (2, 2)

    def test_chain_stats_init_custom_params(self, ptstate_2d):
        """Test ChainStats initialization with custom parameters"""
        rng = np.random.default_rng(42)
        custom_cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        custom_mean = np.array([1.0, -1.0])
        custom_groups = [[0], [1]]  # Two single-parameter groups

        stats = ChainStats(
            ndim=2,
            pt_state=ptstate_2d,
            chain_index=1,
            rng=rng,
            groups=custom_groups,
            sample_cov=custom_cov,
            sample_mean=custom_mean,
            buffer_size=1000,
        )

        assert stats.chain_index == 1
        assert stats.buffer_size == 1000
        np.testing.assert_array_equal(stats.sample_cov, custom_cov)
        np.testing.assert_array_equal(stats.sample_mean, custom_mean)
        assert len(stats.groups) == 2
        assert len(stats.svd_U) == 2  # One per group
        assert len(stats.svd_S) == 2
        assert len(stats.proposal_L) == 2

    def test_chain_stats_init_ladder_error(self):
        """Test error when PTState ladder is None"""
        ptstate = PTState.__new__(PTState)
        ptstate.ladder = None
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="pt_state.ladder must be initialized"):
            ChainStats(ndim=2, pt_state=ptstate, chain_index=0, rng=rng)

    def test_chain_stats_temperature_assignment(self, ptstate_2d):
        """Test that temperature is correctly assigned from ladder"""
        rng = np.random.default_rng(42)

        for chain_idx in range(ptstate_2d.ntemps):
            stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=chain_idx, rng=rng)
            assert stats.temp == ptstate_2d.ladder[chain_idx]

    def test_chain_stats_buffer_initialization(self, ptstate_2d):
        """Test that buffer is properly initialized"""
        rng = np.random.default_rng(42)
        buffer_size = 100

        stats = ChainStats(
            ndim=3, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=buffer_size
        )

        assert stats._buffer.shape == (buffer_size, 3)
        np.testing.assert_array_equal(stats._buffer, np.zeros((buffer_size, 3)))

    def test_update_buffer_basic(self, ptstate_2d):
        """Test basic buffer update functionality"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=5)

        # Add some samples
        new_samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats.update_buffer(new_samples)

        # Check that samples are in the buffer (at the end)
        np.testing.assert_array_equal(stats._buffer[-2:], new_samples)
        # Earlier entries should still be zeros
        np.testing.assert_array_equal(stats._buffer[:-2], np.zeros((3, 2)))

    def test_update_buffer_circular(self, ptstate_2d):
        """Test buffer circular behavior"""
        rng = np.random.default_rng(42)
        buffer_size = 3
        stats = ChainStats(
            ndim=1, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=buffer_size
        )

        # Fill buffer beyond capacity
        for i in range(5):
            new_sample = np.array([[i]])
            stats.update_buffer(new_sample)

        # Buffer should contain the last 3 samples: [2, 3, 4]
        expected = np.array([[2.0], [3.0], [4.0]])
        np.testing.assert_array_equal(stats._buffer, expected)

    def test_update_buffer_sets_buffer_full(self, ptstate_2d):
        """Test that buffer_full flag is set correctly"""
        rng = np.random.default_rng(42)
        buffer_size = 3
        stats = ChainStats(
            ndim=1, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=buffer_size
        )

        assert stats.buffer_full == False

        # Add samples but not enough to exceed buffer size
        stats.sample_total = 2
        stats.update_buffer(np.array([[1.0]]))
        assert stats.buffer_full == False

        # Add enough to exceed buffer size
        stats.sample_total = 5  # > buffer_size
        stats.update_buffer(np.array([[2.0]]))
        assert stats.buffer_full == True

    def test_recursive_update_basic(self, ptstate_2d):
        """Test basic recursive update functionality"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng, buffer_size=10)

        # Set up initial state
        initial_sample_total = 5
        new_samples = np.array([[1.0, 2.0], [3.0, 4.0]])

        stats.recursive_update(initial_sample_total, new_samples)

        # Sample total should be updated (starts at 0, adds len(new_samples))
        assert stats.sample_total == len(new_samples)

        # Buffer should contain new samples
        assert np.any(np.all(stats._buffer == new_samples[0], axis=1))
        assert np.any(np.all(stats._buffer == new_samples[1], axis=1))

        # Statistics should be updated (non-default values)
        assert not np.allclose(stats.sample_mean, np.zeros(2))

    def test_recursive_update_1d_keeps_cov_2d(self):
        """ndim=1: np.cov returns a 0-d scalar, which must be promoted to (1, 1).

        Regression test: without the promotion, svd_groups raises
        "IndexError: too many indices for array: array is 0-dimensional"
        on the first covariance update of any 1-parameter run.
        """
        ptstate = PTState(ndim=1, ntemps=1, min_temp=1.0, max_temp=1.0)
        rng = np.random.default_rng(0)
        stats = ChainStats(ndim=1, pt_state=ptstate, chain_index=0, rng=rng, buffer_size=10)

        stats.recursive_update(0, rng.standard_normal((6, 1)))

        assert stats.sample_cov.shape == (1, 1)
        assert stats.sample_mean.shape == (1,)
        # the group SVD must have run and produced a usable 1x1 factor
        assert stats.proposal_L is not None
        assert stats.proposal_L[0].shape == (1, 1)

    def test_user_supplied_1d_sample_cov_is_promoted(self):
        """A caller-supplied sample_cov for ndim=1 must be accepted.

        Regression test: the obvious way to seed a 1-parameter run,
        np.cov(pilot, rowvar=False), returns a 0-d scalar. Passing it raised
        "IndexError: too many indices for array: array is 0-dimensional" from
        svd_groups during construction -- the same failure as the recursive
        update path, reachable through a documented public argument.
        """
        ptstate = PTState(ndim=1, ntemps=1, min_temp=1.0, max_temp=1.0)
        rng = np.random.default_rng(0)
        pilot = rng.standard_normal((50, 1))
        scalar_cov = np.cov(pilot, rowvar=False)
        assert np.asarray(scalar_cov).ndim == 0  # precondition

        stats = ChainStats(
            ndim=1,
            pt_state=ptstate,
            chain_index=0,
            rng=rng,
            buffer_size=10,
            sample_cov=scalar_cov,
            sample_mean=np.mean(pilot, axis=0),
        )

        assert stats.sample_cov.shape == (1, 1)
        assert stats.proposal_L[0].shape == (1, 1)

    def test_recursive_update_none_checks(self, ptstate_2d):
        """Test recursive_update error handling for None values"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        # Manually set required attributes to None to test error handling
        stats.sample_cov = None

        with pytest.raises(ValueError, match="sample_cov and sample_mean must be initialized"):
            stats.recursive_update(10, np.array([[1.0, 2.0]]))

    def test_get_group_u_basic(self, ptstate_2d):
        """Test getting U matrix for parameter group"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        U = stats.get_group_U(0)

        assert isinstance(U, np.ndarray)
        assert U.shape == (2, 2)  # 2D problem, single group

        # Should be orthogonal matrix
        np.testing.assert_array_almost_equal(U @ U.T, np.eye(2), decimal=10)

    def test_get_group_u_error_handling(self, ptstate_2d):
        """Test error handling in get_group_U"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        # Test with uninitialized svd_U
        stats.svd_U = None
        with pytest.raises(ValueError, match="svd_U is not initialized"):
            stats.get_group_U(0)

        # Test with None entry
        stats.svd_U = [None]
        with pytest.raises(ValueError, match="U for group 0 is not initialized"):
            stats.get_group_U(0)

    def test_get_group_s_basic(self, ptstate_2d):
        """Test getting singular values for parameter group"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        S = stats.get_group_S(0)

        assert isinstance(S, np.ndarray)
        assert S.shape == (2,)  # 2D problem, single group
        assert np.all(S > 0)  # Singular values should be positive

    def test_get_group_s_error_handling(self, ptstate_2d):
        """Test error handling in get_group_S"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        # Test with uninitialized svd_S
        stats.svd_S = None
        with pytest.raises(ValueError, match="svd_S is not initialized"):
            stats.get_group_S(0)

        # Test with None entry
        stats.svd_S = [None]
        with pytest.raises(ValueError, match="Singular values for group 0 are not initialized"):
            stats.get_group_S(0)

    def test_update_sample(self, ptstate_2d):
        """Test updating current sample"""
        rng = np.random.default_rng(42)
        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng)

        new_position = np.array([1.5, -0.8])
        stats.update_sample(new_position)

        np.testing.assert_array_equal(stats.current_sample, new_position)

    def test_multiple_parameter_groups(self, ptstate_2d):
        """Test ChainStats with multiple parameter groups"""
        rng = np.random.default_rng(42)
        groups = [[0], [1]]  # Two single-parameter groups

        stats = ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng, groups=groups)

        assert len(stats.groups) == 2
        assert len(stats.svd_U) == 2
        assert len(stats.svd_S) == 2

        # Each group should have appropriate dimensions
        for i in range(2):
            U = stats.get_group_U(i)
            S = stats.get_group_S(i)
            assert U.shape == (1, 1)  # Single parameter groups
            assert S.shape == (1,)


class TestMultiChainStats:
    """Test suite for MultiChainStats class"""

    def test_multi_chain_stats_init(self, ptstate_3d):
        """Test MultiChainStats initialization"""
        rng_list = [np.random.default_rng(i) for i in range(5)]
        chain_stats_list = [
            ChainStats(ndim=3, pt_state=ptstate_3d, chain_index=i, rng=rng_list[i])
            for i in range(5)
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        assert multi_stats.ntemps == 5
        assert multi_stats.ndim == 3
        assert multi_stats.sample_total == 0  # All chains start at 0
        assert len(multi_stats.chain_stats) == 5

    def test_multi_chain_stats_properties(self, ptstate_3d):
        """Test MultiChainStats property methods"""
        rng_list = [np.random.default_rng(i) for i in range(3)]
        chain_stats_list = [
            ChainStats(ndim=2, pt_state=ptstate_3d, chain_index=i, rng=rng_list[i])
            for i in range(3)
        ]

        # Modify sample_total for one chain
        chain_stats_list[0].sample_total = 100

        multi_stats = MultiChainStats(chain_stats_list)

        assert multi_stats.ntemps == 3
        assert multi_stats.ndim == 2
        assert multi_stats.sample_total == 100  # Takes from first chain

    def test_multi_chain_stats_recursive_update(self, ptstate_3d):
        """Test MultiChainStats recursive update"""
        rng_list = [np.random.default_rng(i) for i in range(3)]
        chain_stats_list = [
            ChainStats(ndim=2, pt_state=ptstate_3d, chain_index=i, rng=rng_list[i], buffer_size=10)
            for i in range(3)
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        # Create new samples for all chains
        new_samples = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])

        multi_stats.recursive_update(new_samples)

        # Check that all chains were updated
        for i, cs in enumerate(multi_stats.chain_stats):
            assert cs.sample_total == 1
            # Check that the right sample was added to each chain
            expected_sample = new_samples[i, 0]
            assert np.any(np.all(cs._buffer == expected_sample, axis=1))

    def test_multi_chain_stats_get_group_methods(self, ptstate_3d):
        """Test MultiChainStats group access methods"""
        rng_list = [np.random.default_rng(i) for i in range(2)]
        chain_stats_list = [
            ChainStats(ndim=3, pt_state=ptstate_3d, chain_index=i, rng=rng_list[i])
            for i in range(2)
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        # Test get_group_U
        U = multi_stats.get_group_U(chain_idx=0, group_idx=0)
        assert U.shape == (3, 3)

        # Test get_group_S
        S = multi_stats.get_group_S(chain_idx=1, group_idx=0)
        assert S.shape == (3,)
        assert np.all(S > 0)

    def test_multi_chain_stats_update_sample(self, ptstate_2d, sample_state_2d):
        """Test MultiChainStats update_sample method"""
        rng_list = [np.random.default_rng(i) for i in range(3)]
        chain_stats_list = [
            ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=i, rng=rng_list[i])
            for i in range(3)
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        # Update with sampler state
        multi_stats.update_sample(sample_state_2d)

        # Check that each chain was updated with its corresponding position
        for i, cs in enumerate(multi_stats.chain_stats):
            expected_position = sample_state_2d.positions[i]
            np.testing.assert_array_equal(cs.current_sample, expected_position)

    def test_multi_chain_stats_empty_list(self):
        """Test MultiChainStats with empty chain list"""
        multi_stats = MultiChainStats([])

        # Properties should handle empty list gracefully
        assert multi_stats.ntemps == 0
        # Note: ndim and sample_total will raise IndexError for empty list
        # This might be a design choice - accessing properties of empty MultiChainStats

    def test_multi_chain_stats_different_buffer_sizes(self, ptstate_2d):
        """Test MultiChainStats with chains having different buffer sizes"""
        rng_list = [np.random.default_rng(i) for i in range(2)]
        chain_stats_list = [
            ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=0, rng=rng_list[0], buffer_size=10),
            ChainStats(ndim=2, pt_state=ptstate_2d, chain_index=1, rng=rng_list[1], buffer_size=20),
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        # Should work fine - each chain manages its own buffer
        assert multi_stats.chain_stats[0].buffer_size == 10
        assert multi_stats.chain_stats[1].buffer_size == 20

        # Updates should work
        new_samples = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        multi_stats.recursive_update(new_samples)

        for cs in multi_stats.chain_stats:
            assert cs.sample_total == 1

    def test_multi_chain_stats_integration(self, ptstate_3d):
        """Test MultiChainStats integration with realistic usage"""
        # Create realistic setup
        rng_list = [np.random.default_rng(42 + i) for i in range(3)]
        groups = [[0, 1], [2]]  # Two groups: (0,1) and (2)

        chain_stats_list = [
            ChainStats(
                ndim=3,
                pt_state=ptstate_3d,
                chain_index=i,
                rng=rng_list[i],
                groups=groups,
                buffer_size=50,
            )
            for i in range(3)
        ]

        multi_stats = MultiChainStats(chain_stats_list)

        # Simulate several updates
        for iteration in range(5):
            new_samples = np.random.randn(3, 1, 3) * 0.1  # Small random samples
            multi_stats.recursive_update(new_samples)

        # Check that all chains have been updated
        assert multi_stats.sample_total == 5

        # Check that statistics have evolved
        for cs in multi_stats.chain_stats:
            assert cs.sample_total == 5
            assert not np.allclose(cs.sample_mean, np.zeros(3))  # Should have changed

            # Check group-wise access
            for group_idx in range(len(groups)):
                U = multi_stats.get_group_U(cs.chain_index, group_idx)
                S = multi_stats.get_group_S(cs.chain_index, group_idx)
                expected_size = len(groups[group_idx])
                assert U.shape == (expected_size, expected_size)
                assert S.shape == (expected_size,)


class TestEnablePerModelLayout:
    """enable_per_model derives its indices from ParameterLayout."""

    @staticmethod
    def _make_stats(ndim, groups):
        ptstate = PTState(ndim=ndim, ntemps=1, min_temp=1.0, max_temp=1.0)
        rng = np.random.default_rng(0)
        return ChainStats(
            ndim=ndim,
            pt_state=ptstate,
            chain_index=0,
            rng=rng,
            groups=groups,
            buffer_size=10,
        )

    def test_scalar_and_layout_paths_agree(self):
        from impulse.product_space import ParameterLayout

        num_models, num_params = 2, 3
        layout = ParameterLayout(num_params=num_params, num_models=num_models)
        groups = [list(range(k * num_params, (k + 1) * num_params)) for k in range(num_models)]

        scalar_stats = self._make_stats(layout.total_dim, [list(g) for g in groups])
        scalar_stats.enable_per_model(num_models, num_params)

        layout_stats = self._make_stats(layout.total_dim, [list(g) for g in groups])
        layout_stats.enable_per_model(num_models, num_params, layout=layout)

        for stats in (scalar_stats, layout_stats):
            assert stats._num_models == num_models
            assert stats._num_params == num_params
            assert stats._nmodel_idx == layout.nmodel_index == num_models * num_params
            assert set(stats._per_model) == set(range(num_models))

    def test_multi_chain_forwards_layout(self):
        from impulse.product_space import ParameterLayout

        num_models, num_params = 2, 2
        layout = ParameterLayout(num_params=num_params, num_models=num_models)
        groups = [list(range(k * num_params, (k + 1) * num_params)) for k in range(num_models)]
        chains = [self._make_stats(layout.total_dim, [list(g) for g in groups]) for _ in range(2)]
        multi = MultiChainStats(chains)
        multi.enable_per_model(num_models, num_params, layout=layout)
        for cs in multi.chain_stats:
            assert cs._nmodel_idx == layout.nmodel_index
            assert cs._num_models == num_models


class TestHistoryBufferThinning:
    """Storing every buffer_thin-th state instead of every consecutive one.

    The DE history buffer holds ~89% redundant rows at realistic autocorrelation
    (measured ESS 4362 of 39901 at ACT~9). Retaining every k-th state keeps the
    same iteration HORIZON with k times fewer rows, which is what makes it safe:
    naively shrinking buffer_size instead shortens the horizon and measurably
    biases the adaptive covariance (a 500-row buffer at cov_update=10 samples a
    posterior 2.8% too narrow; the same buffer with thin=25 is 0.35% off, versus
    0.07% for the 100x larger default buffer).
    """

    def test_phase_is_global_not_per_batch(self):
        """Retained rows are the global iterations divisible by thin."""
        rows = np.arange(100, dtype=float).reshape(-1, 1)
        kept, seen = [], 0
        for start in range(0, 100, 7):  # ragged batches, deliberately not a multiple
            batch = rows[start : start + 7]
            sel = _HistoryBuffer.select_for_storage(batch, seen, 5)
            kept.extend(sel[:, 0].tolist())
            seen += len(batch)
        assert kept == list(range(0, 100, 5))

    def test_thin_one_is_identity(self):
        rows = np.arange(10, dtype=float).reshape(-1, 1)
        out = _HistoryBuffer.select_for_storage(rows, 3, 1)
        np.testing.assert_array_equal(out, rows)

    def test_buffer_stores_every_kth_state(self):
        ptstate = PTState(ndim=1, ntemps=1, min_temp=1.0, max_temp=1.0)
        stats = ChainStats(
            ndim=1,
            pt_state=ptstate,
            chain_index=0,
            rng=np.random.default_rng(0),
            buffer_size=50,
            buffer_thin=4,
        )
        samples = np.arange(80, dtype=float).reshape(-1, 1)
        for start in range(0, 80, 10):
            stats.recursive_update(start, samples[start : start + 10])

        # sample_total counts STORED rows -- `de` derives its window from it and
        # indexes buffer[-n_filled:], so counting raw iterations here would make
        # DE read zero padding.
        assert stats.sample_total == 20
        assert stats._seen_raw == 80
        stored = stats._buffer[-stats.sample_total :][:, 0]
        np.testing.assert_array_equal(stored, np.arange(0, 80, 4))

    def test_rejects_bad_thin(self):
        ptstate = PTState(ndim=1, ntemps=1, min_temp=1.0, max_temp=1.0)
        with pytest.raises(ValueError, match="buffer_thin must be >= 1"):
            ChainStats(
                ndim=1,
                pt_state=ptstate,
                chain_index=0,
                rng=np.random.default_rng(0),
                buffer_thin=0,
            )


class TestGroupsCoverage:
    """Groups must cover every parameter or the chain is reducible.

    am/scam/de each pick ONE group per call, so a coordinate that appears in no
    group is never proposed: it stays pinned at its initial value and the chain
    samples a CONDITIONAL of the posterior instead of the marginal. Measured on
    a correlated 2-D Gaussian, that produced a reported mean 44% off and a width
    40% off, silently, with a healthy acceptance rate.
    """

    def _stats(self, ndim, groups):
        ptstate = PTState(ndim=ndim, ntemps=1, min_temp=1.0, max_temp=1.0)
        return ChainStats(
            ndim=ndim,
            pt_state=ptstate,
            chain_index=0,
            rng=np.random.default_rng(0),
            groups=groups,
        )

    def test_uncovered_parameter_warns(self):
        """A warning, not an error: the product-space layout legitimately leaves
        the model index out of groups because birth/death moves it instead."""
        with pytest.warns(UserWarning, match="do not cover parameter"):
            self._stats(3, [[0, 1]])

    def test_warning_names_the_missing_indices(self):
        with pytest.warns(UserWarning, match=r"\[1, 3\]"):
            self._stats(4, [[0], [2]])

    def test_out_of_range_index_rejected(self):
        with pytest.raises(ValueError, match="out-of-range"):
            self._stats(2, [[0, 1, 5]])

    @pytest.mark.parametrize(
        "ndim,groups",
        [
            (3, [[0, 1], [2]]),
            (3, [[0], [1], [2]]),
            (3, [[0, 1, 2]]),
            (3, [[0, 1], [1, 2]]),  # overlapping is fine, all covered
            (1, [[0]]),
        ],
    )
    def test_valid_groupings_accepted(self, ndim, groups):
        stats = self._stats(ndim, groups)
        assert len(stats.groups) == len(groups)

    def test_default_groups_cover_everything(self):
        ptstate = PTState(ndim=4, ntemps=1, min_temp=1.0, max_temp=1.0)
        stats = ChainStats(ndim=4, pt_state=ptstate, chain_index=0, rng=np.random.default_rng(0))
        covered = {int(i) for g in stats.groups for i in np.atleast_1d(g)}
        assert covered == set(range(4))

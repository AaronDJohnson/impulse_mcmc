import logging
import os
import pickle
import tempfile
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from impulse.resume import check_for_checkpoint, checkpoint_sampler, load_checkpoint


class _FakeSampler:
    """Simple pickleable stand-in for PTSampler in checkpoint tests."""

    pass


def _gauss_lnlike(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return -0.5 * np.sum(x**2)
    return -0.5 * np.sum(x**2, axis=1)


def _flat_lnprior(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return 0.0 if np.all(np.abs(x) <= 5) else -np.inf
    result = np.zeros(x.shape[0])
    result[np.any(np.abs(x) > 5, axis=1)] = -np.inf
    return result


class _LegacyNamedProposal:
    """Picklable stand-in for a pre-fix separate birth/death proposal.

    Carries only ``__name__`` — none of the legacy ``BirthProposal``
    attributes — so it is detected as legacy wiring but can never be
    migrated (the reconstruction fallback path).
    """

    def __init__(self, name):
        self.__name__ = name

    def __call__(self, chain_stats):
        return chain_stats.current_sample.copy(), 0.0


class _UnitIntervalDraw:
    """Picklable ``draw_from_prior``: one parameter uniform in [-0.4, 0.4]."""

    def __call__(self, rng):
        return rng.uniform(-0.4, 0.4, size=1)


class _FlatSourceLogPrior:
    """Picklable flat per-source log prior density."""

    def __call__(self, params):
        return 0.0


def _tight_nmodel_lnprior(x):
    """Flat prior with ``|x[1]| <= 0.4`` so ``rint(x[-1])`` (the model
    index the birth/death moves read) is pinned to 0: births propose the
    out-of-bounds model index 1 and are always rejected, deaths no-op, and
    the legacy death fixture (which has no ``draw_from_prior``) is never
    asked to draw."""
    x = np.asarray(x)
    if x.ndim == 1:
        ok = np.abs(x[0]) <= 5 and np.abs(x[1]) <= 0.4
        return 0.0 if ok else -np.inf
    result = np.zeros(x.shape[0])
    bad = (np.abs(x[:, 0]) > 5) | (np.abs(x[:, 1]) > 0.4)
    result[bad] = -np.inf
    return result


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
        with open(checkpoint_path, "rb") as f:
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

        checkpoint_path = checkpoint_sampler(sampler, omit=("func1", "func2"))

        # Check that specified functions are omitted
        with open(checkpoint_path, "rb") as f:
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
        checkpoint_path = checkpoint_sampler(sampler, omit=("existing_func", "nonexistent"))

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
        with patch("impulse.resume.tempfile.mkstemp") as mock_mkstemp:
            mock_fd = 123
            mock_tmp_path = os.path.join(temp_dir, ".ckpt.test.tmp")
            mock_mkstemp.return_value = (mock_fd, mock_tmp_path)

            with (
                patch("impulse.resume.os.close") as mock_close,
                patch("impulse.resume.os.replace") as mock_replace,
                patch("builtins.open", create=True) as mock_open,
                patch("pickle.dump"),
            ):

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
        with patch("pickle.dump", side_effect=Exception("Pickle failed")):
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
        with open(checkpoint_path, "wb") as f:
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
        with open(checkpoint_path, "wb") as f:
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
        with open(corrupt_path, "w") as f:
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
        with open(checkpoint_path, "w") as f:
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
        with open(wrong_name, "w") as f:
            f.write("dummy")

        result = check_for_checkpoint(temp_dir)
        assert result is None  # Should not find wrong name

        # Create file with correct name
        correct_name = os.path.join(temp_dir, "sampler_checkpoint.pkl")
        with open(correct_name, "w") as f:
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


_OMIT = object()  # test-local marker: do not pass num_adapt at all


class TestResumeNumAdaptOverride:
    """Resume semantics of the ``num_adapt`` constructor argument.

    Not passing ``num_adapt`` (the default) keeps the checkpointed value on
    resume: silently un-freezing a checkpointed freeze would produce a
    half-frozen kernel (proposals whose frozen state is pickled — e.g. a
    frozen normalizing flow — stay frozen while everything else adapts
    again).  An EXPLICITLY passed value, including an explicit ``None``,
    overrides the checkpointed value with a warning when they differ.
    """

    @staticmethod
    def _run_pt(outdir, num_adapt=_OMIT, resume=False, num_iterations=25):
        from impulse.samplers import PTSampler

        kwargs = {} if num_adapt is _OMIT else {"num_adapt": num_adapt}
        sampler = PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
            **kwargs,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=num_iterations)
        return sampler

    @staticmethod
    def _run_rjpt(outdir, num_adapt=_OMIT, resume=False, num_iterations=25):
        from impulse.rjpt_sampler import RJPTSampler

        kwargs = {} if num_adapt is _OMIT else {"num_adapt": num_adapt}
        sampler = RJPTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
            **kwargs,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=num_iterations)
        return sampler

    def test_pt_explicit_num_adapt_wins_on_resume(self, temp_dir, caplog):
        """An explicitly passed int overrides the checkpointed value."""
        self._run_pt(temp_dir, num_adapt=None)
        assert check_for_checkpoint(temp_dir) is not None

        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, num_adapt=123, resume=True, num_iterations=30)
        assert resumed.num_adapt == 123
        assert any("overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records)

    def test_pt_no_warning_when_num_adapt_matches(self, temp_dir, caplog):
        self._run_pt(temp_dir, num_adapt=1000)
        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, num_adapt=1000, resume=True, num_iterations=30)
        assert resumed.num_adapt == 1000
        assert not any(
            "overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records
        )

    def test_pt_default_resume_keeps_checkpointed_freeze(self, temp_dir, caplog):
        """Resuming WITHOUT passing num_adapt keeps the checkpointed freeze
        (no warning): the default must not silently un-freeze the kernel."""
        self._run_pt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, resume=True, num_iterations=30)
        assert resumed.num_adapt == 15
        assert not any(
            "overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records
        )

    def test_pt_explicit_none_unfreezes_with_warning(self, temp_dir, caplog):
        """An EXPLICIT None un-freezes a checkpointed freeze, with a warning
        (deliberate override, unlike the omitted default)."""
        self._run_pt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, num_adapt=None, resume=True, num_iterations=30)
        assert resumed.num_adapt is None
        assert any("overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records)

    def test_rjpt_explicit_num_adapt_wins_on_resume(self, temp_dir, caplog):
        self._run_rjpt(temp_dir, num_adapt=None)
        assert check_for_checkpoint(temp_dir) is not None

        with caplog.at_level(logging.WARNING, logger="impulse.rjpt_sampler"):
            resumed = self._run_rjpt(temp_dir, num_adapt=77, resume=True, num_iterations=30)
        assert resumed.num_adapt == 77
        assert any("overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records)

    def test_rjpt_default_resume_keeps_checkpointed_freeze(self, temp_dir, caplog):
        self._run_rjpt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.rjpt_sampler"):
            resumed = self._run_rjpt(temp_dir, resume=True, num_iterations=30)
        assert resumed.num_adapt == 15
        assert not any(
            "overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records
        )

    def test_rjpt_explicit_none_unfreezes_with_warning(self, temp_dir, caplog):
        self._run_rjpt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.rjpt_sampler"):
            resumed = self._run_rjpt(temp_dir, num_adapt=None, resume=True, num_iterations=30)
        assert resumed.num_adapt is None
        assert any("overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records)

    def test_pt_pre_num_adapt_sampler_resumes_without_attribute_error(self, temp_dir):
        """The public load_checkpoint(...)->sample() path must survive
        samplers unpickled from checkpoints written before num_adapt
        existed (unpickling bypasses __init__, so neither ``num_adapt``
        nor ``_num_adapt_explicit`` is present); the capture site must use
        getattr, not raw attribute access."""
        from impulse.samplers import PTSampler

        self._run_pt(temp_dir, num_adapt=15)
        sampler = PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=10,
            resume=True,
        )
        # emulate the unpickled pre-num_adapt sampler
        del sampler.num_adapt
        del sampler._num_adapt_explicit
        sampler.sample(np.array([0.1, 0.1]), num_iterations=30)
        # attributes missing == not explicitly passed -> checkpoint kept
        assert sampler.num_adapt == 15

    def test_rjpt_pre_num_adapt_sampler_resumes_without_attribute_error(self, temp_dir):
        from impulse.rjpt_sampler import RJPTSampler

        self._run_rjpt(temp_dir, num_adapt=15)
        sampler = RJPTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=10,
            resume=True,
        )
        del sampler.num_adapt
        del sampler._num_adapt_explicit
        sampler.sample(np.array([0.1, 0.1]), num_iterations=30)
        assert sampler.num_adapt == 15


class TestResumeLegacyBirthDeathWarning:
    """Resuming a checkpoint that registers separate birth/death jumps
    (pre-detailed-balance-fix wiring) must emit a loud UserWarning when
    the pair cannot be migrated (these fixtures carry none of the legacy
    ``BirthProposal`` attributes, so reconstruction always fails and the
    warn-only fallback fires).  The migratable case is covered by
    ``TestResumeLegacyBirthDeathMigration``."""

    LEGACY_MATCH = "predates the detailed-balance fix"

    @staticmethod
    def _make_pt(outdir, resume=False):
        from impulse.samplers import PTSampler

        return PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
        )

    @staticmethod
    def _make_rjpt(outdir, resume=False):
        from impulse.rjpt_sampler import RJPTSampler

        return RJPTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
        )

    def test_pt_resume_warns_on_legacy_birth_death(self, temp_dir):
        legacy = self._make_pt(temp_dir)
        legacy.add_custom_jump(_LegacyNamedProposal("birth_proposal"), weight=5)
        legacy.add_custom_jump(_LegacyNamedProposal("death_proposal"), weight=5)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)

        resuming = self._make_pt(temp_dir, resume=True)
        with pytest.warns(UserWarning, match=self.LEGACY_MATCH):
            resuming.sample(np.array([0.1, 0.1]), num_iterations=30)

    def test_pt_resume_no_warning_without_legacy_proposals(self, temp_dir):
        clean = self._make_pt(temp_dir)
        clean.sample(np.array([0.1, 0.1]), num_iterations=25)

        resuming = self._make_pt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resuming.sample(np.array([0.1, 0.1]), num_iterations=30)
        assert not any(self.LEGACY_MATCH in str(w.message) for w in caught)

    def test_rjpt_resume_warns_on_legacy_birth_death(self, temp_dir):
        legacy = self._make_rjpt(temp_dir)
        legacy.add_custom_jump(_LegacyNamedProposal("birth_proposal"), weight=5)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)

        resuming = self._make_rjpt(temp_dir, resume=True)
        with pytest.warns(UserWarning, match=self.LEGACY_MATCH):
            resuming.sample(np.array([0.1, 0.1]), num_iterations=30)

    def test_rjpt_resume_no_warning_without_legacy_proposals(self, temp_dir):
        clean = self._make_rjpt(temp_dir)
        clean.sample(np.array([0.1, 0.1]), num_iterations=25)

        resuming = self._make_rjpt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resuming.sample(np.array([0.1, 0.1]), num_iterations=30)
        assert not any(self.LEGACY_MATCH in str(w.message) for w in caught)


class TestResumeLegacyBirthDeathMigration:
    """Automatic migration of resumed pre-fix separate birth/death wiring.

    The legacy fixtures are FAITHFUL to the attribute layout at git HEAD
    (``git show HEAD:impulse/rjmcmc_proposals.py``): the legacy
    ``BirthProposal`` stored ``num_params``, ``max_sources``,
    ``draw_from_prior``, ``log_proposal_density``, ``log_prior_density``
    and ``prob_schedule`` — identical names to today's class — while the
    legacy ``DeathProposal`` stored the same set WITHOUT
    ``draw_from_prior``.  The fixtures instantiate the current classes and
    strip attributes down to that layout, then are pickled into a real
    checkpoint by ``sample()``.

    ``_tight_nmodel_lnprior`` pins ``rint(x[-1])`` to 0, so the legacy
    pair (and the migrated combined kernel) only ever takes the
    guaranteed-safe nmodel=0 code paths during the fixture and resumed
    runs.
    """

    MIGRATED_MATCH = "migrated automatically"
    FALLBACK_MATCH = "predates the detailed-balance fix"
    STANDALONE_MATCH = "current-code standalone registrations"

    # Attribute layouts at git HEAD (verified against
    # ``git show HEAD:impulse/rjmcmc_proposals.py``).
    HEAD_BIRTH_ATTRS = {
        "num_params",
        "max_sources",
        "draw_from_prior",
        "log_proposal_density",
        "log_prior_density",
        "prob_schedule",
    }
    HEAD_DEATH_ATTRS = HEAD_BIRTH_ATTRS - {"draw_from_prior"}

    BIRTH_WEIGHT = 5.0
    DEATH_WEIGHT = 7.0

    @staticmethod
    def _make_pt(outdir, resume=False):
        from impulse.samplers import PTSampler

        return PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_tight_nmodel_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
        )

    @staticmethod
    def _make_rjpt(outdir, resume=False):
        from impulse.rjpt_sampler import RJPTSampler

        return RJPTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_tight_nmodel_lnprior,
            ntemps=2,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
        )

    def _legacy_pair(self):
        """Build a HEAD-layout birth/death pair from the current classes."""
        from impulse.rjmcmc_proposals import BirthProposal, DeathProposal

        draw = _UnitIntervalDraw()
        birth = BirthProposal(
            num_params=1,
            max_sources=2,
            draw_from_prior=draw,
            log_prior_density=_FlatSourceLogPrior(),
        )
        death = DeathProposal(
            num_params=1,
            max_sources=2,
            draw_from_prior=draw,
            log_prior_density=_FlatSourceLogPrior(),
        )
        # The HEAD DeathProposal never stored draw_from_prior; strip it so
        # the pickled fixture is attribute-faithful to a real legacy
        # checkpoint.
        del death.draw_from_prior
        assert set(vars(birth)) == self.HEAD_BIRTH_ATTRS
        assert set(vars(death)) == self.HEAD_DEATH_ATTRS
        return birth, death

    def _legacy_pair_no_densities(self):
        """HEAD-layout pair whose densities are both ``None``.

        Both ``log_proposal_density`` and ``log_prior_density`` defaulted
        to ``None`` at HEAD unless the user supplied one, so this is the
        layout of a real legacy checkpoint from a user who never passed a
        density.  Such a pair must NOT be migrated: the combined kernel's
        correctness requires the true per-source draw density, and
        silently assuming a flat one gives wrong acceptance ratios for
        non-flat priors.
        """
        from impulse.rjmcmc_proposals import BirthProposal, DeathProposal

        draw = _UnitIntervalDraw()
        birth = BirthProposal(
            num_params=1,
            max_sources=2,
            draw_from_prior=draw,
        )
        death = DeathProposal(
            num_params=1,
            max_sources=2,
            draw_from_prior=draw,
        )
        del death.draw_from_prior
        assert set(vars(birth)) == self.HEAD_BIRTH_ATTRS
        assert set(vars(death)) == self.HEAD_DEATH_ATTRS
        assert birth.log_proposal_density is None
        assert birth.log_prior_density is None
        return birth, death

    def _current_standalone_pair(self):
        """CURRENT-code standalone birth/death registrations.

        Same ``__name__``\\ s as the legacy pair, but built by today's
        factories: in particular the current ``DeathProposal`` DOES store
        ``draw_from_prior`` (it re-fills the vacated slot), which is what
        distinguishes it from a pre-fix legacy checkpoint.
        """
        from impulse.rjmcmc_proposals import (
            make_birth_proposal,
            make_death_proposal,
        )

        draw = _UnitIntervalDraw()
        birth = make_birth_proposal(
            1,
            2,
            draw,
            log_prior_density=_FlatSourceLogPrior(),
        )
        death = make_death_proposal(
            1,
            2,
            draw,
            log_prior_density=_FlatSourceLogPrior(),
        )
        assert callable(death.draw_from_prior)
        return birth, death

    def _write_legacy_checkpoint(self, make_sampler, outdir):
        """Sample a sampler carrying the HEAD-layout pair to a checkpoint."""
        legacy = make_sampler(outdir)
        birth, death = self._legacy_pair()
        legacy.add_custom_jump(birth, weight=self.BIRTH_WEIGHT)
        legacy.add_custom_jump(death, weight=self.DEATH_WEIGHT)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)
        assert check_for_checkpoint(outdir) is not None
        # Weights of the untouched proposals, for comparison after resume.
        pre_weights = [list(jp.proposal_weights) for jp in legacy.proposal_bundle.jump_proposals]
        return pre_weights

    def _assert_migrated(self, resumed, pre_weights):
        from impulse.rjmcmc_proposals import BirthDeathProposal

        combined = None
        for jp, old_weights in zip(resumed.proposal_bundle.jump_proposals, pre_weights):
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert names.count("birth_death") == 1
            assert "birth_proposal" not in names
            assert "death_proposal" not in names
            idx = names.index("birth_death")
            kernel = jp.proposal_list[idx]
            if combined is None:
                combined = kernel
            # One shared instance across chains, like add_custom_jump.
            assert kernel is combined
            assert isinstance(kernel, BirthDeathProposal)
            # Reconstructed from the legacy birth attributes, sharing one
            # draw_from_prior between birth and death.
            assert kernel.birth.num_params == 1
            assert kernel.max_sources == 2
            assert kernel.birth.draw_from_prior is kernel.death.draw_from_prior
            assert isinstance(kernel.birth.draw_from_prior, _UnitIntervalDraw)
            assert isinstance(kernel.birth.log_prior_density, _FlatSourceLogPrior)
            # Weight = sum of the two legacy weights; other proposals and
            # weights untouched; normalization consistent.
            assert jp.proposal_weights[idx] == pytest.approx(self.BIRTH_WEIGHT + self.DEATH_WEIGHT)
            expected_weights = old_weights[:-2] + [self.BIRTH_WEIGHT + self.DEATH_WEIGHT]
            assert jp.proposal_weights == pytest.approx(expected_weights)
            assert len(jp.proposal_list) == len(jp.proposal_weights)
            np.testing.assert_allclose(
                jp.proposal_probs, np.asarray(expected_weights) / sum(expected_weights)
            )
            # Acceptance counters stay index-aligned with the list.
            assert len(jp._proposal_calls) == len(jp.proposal_list)
            assert len(jp._proposal_accepts) == len(jp.proposal_list)

    def test_pt_resume_migrates_legacy_pair(self, temp_dir):
        pre_weights = self._write_legacy_checkpoint(self._make_pt, temp_dir)

        resumed = self._make_pt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert any(self.MIGRATED_MATCH in m for m in messages)
        assert sum(self.MIGRATED_MATCH in m for m in messages) == 1
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        self._assert_migrated(resumed, pre_weights)

    def test_rjpt_resume_migrates_legacy_pair(self, temp_dir):
        pre_weights = self._write_legacy_checkpoint(self._make_rjpt, temp_dir)

        resumed = self._make_rjpt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert any(self.MIGRATED_MATCH in m for m in messages)
        assert sum(self.MIGRATED_MATCH in m for m in messages) == 1
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        self._assert_migrated(resumed, pre_weights)

    def test_pt_corrupted_legacy_pair_falls_back_to_warning(self, temp_dir):
        """A legacy pair missing the reconstruction attributes must not be
        migrated (or half-migrated): the warn-only fallback fires and the
        checkpointed wiring is left exactly as loaded."""
        legacy = self._make_pt(temp_dir)
        legacy.add_custom_jump(_LegacyNamedProposal("birth_proposal"), weight=self.BIRTH_WEIGHT)
        legacy.add_custom_jump(_LegacyNamedProposal("death_proposal"), weight=self.DEATH_WEIGHT)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)
        pre_weights = [list(jp.proposal_weights) for jp in legacy.proposal_bundle.jump_proposals]

        resumed = self._make_pt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert any(self.FALLBACK_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        for jp, old_weights in zip(resumed.proposal_bundle.jump_proposals, pre_weights):
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert "birth_death" not in names
            assert names.count("birth_proposal") == 1
            assert names.count("death_proposal") == 1
            assert jp.proposal_weights == pytest.approx(old_weights)

    def test_rjpt_corrupted_legacy_pair_falls_back_to_warning(self, temp_dir):
        legacy = self._make_rjpt(temp_dir)
        legacy.add_custom_jump(_LegacyNamedProposal("birth_proposal"), weight=self.BIRTH_WEIGHT)
        legacy.add_custom_jump(_LegacyNamedProposal("death_proposal"), weight=self.DEATH_WEIGHT)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)

        resumed = self._make_rjpt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert any(self.FALLBACK_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        for jp in resumed.proposal_bundle.jump_proposals:
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert "birth_death" not in names
            assert names.count("birth_proposal") == 1
            assert names.count("death_proposal") == 1

    def _resume_and_collect(self, make_sampler, outdir):
        resumed = make_sampler(outdir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        return resumed, [str(w.message) for w in caught]

    def _assert_pair_untouched(self, resumed, pre_weights):
        for jp, old_weights in zip(resumed.proposal_bundle.jump_proposals, pre_weights):
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert "birth_death" not in names
            assert names.count("birth_proposal") == 1
            assert names.count("death_proposal") == 1
            assert jp.proposal_weights == pytest.approx(old_weights)

    def _write_pair_checkpoint(self, make_sampler, outdir, birth, death):
        legacy = make_sampler(outdir)
        legacy.add_custom_jump(birth, weight=self.BIRTH_WEIGHT)
        legacy.add_custom_jump(death, weight=self.DEATH_WEIGHT)
        legacy.sample(np.array([0.1, 0.1]), num_iterations=25)
        assert check_for_checkpoint(outdir) is not None
        return [list(jp.proposal_weights) for jp in legacy.proposal_bundle.jump_proposals]

    def test_pt_legacy_pair_without_densities_falls_back_to_warning(self, temp_dir):
        """A true legacy pair whose birth carries NEITHER density (the
        HEAD default when the user supplied none) must NOT be migrated —
        rebuilding the combined kernel would silently assume a flat draw
        density, which is wrong for non-flat priors.  The warn-only
        fallback fires and the wiring is left exactly as loaded."""
        birth, death = self._legacy_pair_no_densities()
        pre_weights = self._write_pair_checkpoint(self._make_pt, temp_dir, birth, death)

        resumed, messages = self._resume_and_collect(self._make_pt, temp_dir)
        assert any(self.FALLBACK_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.STANDALONE_MATCH in m for m in messages)
        self._assert_pair_untouched(resumed, pre_weights)

    def test_rjpt_legacy_pair_without_densities_falls_back_to_warning(self, temp_dir):
        birth, death = self._legacy_pair_no_densities()
        pre_weights = self._write_pair_checkpoint(self._make_rjpt, temp_dir, birth, death)

        resumed, messages = self._resume_and_collect(self._make_rjpt, temp_dir)
        assert any(self.FALLBACK_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.STANDALONE_MATCH in m for m in messages)
        self._assert_pair_untouched(resumed, pre_weights)

    def test_pt_current_standalone_pair_not_migrated_accurate_warning(self, temp_dir):
        """CURRENT-code standalone birth/death registrations carry the
        same ``__name__``\\ s as the legacy pair but their attribute layout
        (the death proposal stores ``draw_from_prior``) shows they are not
        legacy.  They must NOT be migrated, and the warning must say
        standalone registration violates detailed balance — not falsely
        claim the checkpoint predates the fix."""
        birth, death = self._current_standalone_pair()
        pre_weights = self._write_pair_checkpoint(self._make_pt, temp_dir, birth, death)

        resumed, messages = self._resume_and_collect(self._make_pt, temp_dir)
        assert any(self.STANDALONE_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        self._assert_pair_untouched(resumed, pre_weights)

    def test_rjpt_current_standalone_pair_not_migrated_accurate_warning(self, temp_dir):
        birth, death = self._current_standalone_pair()
        pre_weights = self._write_pair_checkpoint(self._make_rjpt, temp_dir, birth, death)

        resumed, messages = self._resume_and_collect(self._make_rjpt, temp_dir)
        assert any(self.STANDALONE_MATCH in m for m in messages)
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        self._assert_pair_untouched(resumed, pre_weights)

    def test_pt_post_fix_checkpoint_round_trip_untouched(self, temp_dir):
        """A checkpoint already carrying the combined kernel must resume
        with no migration and no legacy warning."""
        from impulse.rjmcmc_proposals import make_birth_death_proposal

        clean = self._make_pt(temp_dir)
        kernel = make_birth_death_proposal(
            1,
            2,
            _UnitIntervalDraw(),
            log_prior_density=_FlatSourceLogPrior(),
        )
        clean.add_custom_jump(kernel, weight=12.0)
        clean.sample(np.array([0.1, 0.1]), num_iterations=25)

        resumed = self._make_pt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        for jp in resumed.proposal_bundle.jump_proposals:
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert names.count("birth_death") == 1
            assert "birth_proposal" not in names
            assert "death_proposal" not in names
            assert jp.proposal_weights[names.index("birth_death")] == (pytest.approx(12.0))

    def test_rjpt_post_fix_checkpoint_round_trip_untouched(self, temp_dir):
        from impulse.rjmcmc_proposals import make_birth_death_proposal

        clean = self._make_rjpt(temp_dir)
        kernel = make_birth_death_proposal(
            1,
            2,
            _UnitIntervalDraw(),
            log_prior_density=_FlatSourceLogPrior(),
        )
        clean.add_custom_jump(kernel, weight=12.0)
        clean.sample(np.array([0.1, 0.1]), num_iterations=25)

        resumed = self._make_rjpt(temp_dir, resume=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resumed.sample(np.array([0.1, 0.1]), num_iterations=30)
        messages = [str(w.message) for w in caught]
        assert not any(self.MIGRATED_MATCH in m for m in messages)
        assert not any(self.FALLBACK_MATCH in m for m in messages)
        for jp in resumed.proposal_bundle.jump_proposals:
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            assert names.count("birth_death") == 1
            assert "birth_proposal" not in names
            assert "death_proposal" not in names

import logging
import os

import numpy as np
import pytest
from conftest import count_chain_rows

from impulse import PTSampler
from impulse.resume import check_for_checkpoint, checkpoint_sampler

# ndim=2 plus the four bookkeeping columns (lnlike, lnprob, accepted, temperature)
_GUARD_NCOLS = 2 + 4


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


class TestCheckForCheckpoint:
    """Test suite for check_for_checkpoint function"""

    def test_check_for_checkpoint_not_exists(self, temp_dir):
        result = check_for_checkpoint(temp_dir)
        assert result is None

    def test_check_for_checkpoint_nonexistent_dir(self):
        result = check_for_checkpoint("/nonexistent/directory")
        assert result is None

    def test_checkpoint_sampler_creates_json_and_npz(self, temp_dir):
        sampler = PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=1,
            outdir=temp_dir,
            save_freq=50,
        )
        sampler.sample(np.array([0.1, 0.1]), num_iterations=10)
        json_path = checkpoint_sampler(sampler)
        assert json_path == os.path.join(temp_dir, "sampler_checkpoint.json")
        assert os.path.exists(json_path)
        assert os.path.exists(os.path.join(temp_dir, "sampler_checkpoint.npz"))
        assert check_for_checkpoint(temp_dir) == json_path


_OMIT = object()


class TestResumeNumAdaptOverride:
    """Resume semantics of the ``num_adapt`` constructor argument."""

    @staticmethod
    def _run_pt(outdir, num_adapt=_OMIT, resume=False, num_iterations=25):
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

    def test_pt_explicit_num_adapt_wins_on_resume(self, temp_dir, caplog):
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
        self._run_pt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, resume=True, num_iterations=30)
        assert resumed.num_adapt == 15
        assert not any(
            "overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records
        )

    def test_pt_explicit_none_unfreezes_with_warning(self, temp_dir, caplog):
        self._run_pt(temp_dir, num_adapt=15)
        with caplog.at_level(logging.WARNING, logger="impulse.samplers"):
            resumed = self._run_pt(temp_dir, num_adapt=None, resume=True, num_iterations=30)
        assert resumed.num_adapt is None
        assert any("overriding checkpointed num_adapt" in r.getMessage() for r in caplog.records)


class TestResumeWithoutUsableCheckpoint:
    """resume=True must not append a cold-start run onto an existing chain."""

    @staticmethod
    def _lnlike(x):
        x = np.asarray(x, dtype=float)
        return float(-0.5 * np.sum(x**2)) if x.ndim == 1 else -0.5 * np.sum(x**2, axis=1)

    @staticmethod
    def _lnprior(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return 0.0 if np.all(np.abs(x) <= 10) else -np.inf
        out = np.zeros(x.shape[0])
        out[np.any(np.abs(x) > 10, axis=1)] = -np.inf
        return out

    def _sampler(self, outdir, **kw):
        return PTSampler(
            ndim=2,
            lnlike=self._lnlike,
            lnprior=self._lnprior,
            ntemps=1,
            seed=1,
            outdir=outdir,
            save_freq=500,
            **kw,
        )

    def test_raises_when_chain_exists_but_checkpoint_does_not(self, tmp_path):
        outdir = str(tmp_path)
        self._sampler(outdir).sample(np.zeros(2), num_iterations=1000)
        rows_before = count_chain_rows(os.path.join(outdir, "chain_0.bin"), _GUARD_NCOLS)
        os.remove(os.path.join(outdir, "sampler_checkpoint.json"))

        with pytest.raises(RuntimeError, match="no usable checkpoint"):
            self._sampler(outdir, resume=True).sample(np.zeros(2), num_iterations=1000)

        rows_after = count_chain_rows(os.path.join(outdir, "chain_0.bin"), _GUARD_NCOLS)
        assert rows_after == rows_before

    def test_message_warns_that_resume_false_deletes_chains(self, tmp_path):
        outdir = str(tmp_path)
        self._sampler(outdir).sample(np.zeros(2), num_iterations=1000)
        os.remove(os.path.join(outdir, "sampler_checkpoint.json"))
        with pytest.raises(RuntimeError, match="resume=False DELETES"):
            self._sampler(outdir, resume=True).sample(np.zeros(2), num_iterations=1000)

    def test_message_names_an_unsupported_legacy_pickle(self, tmp_path):
        outdir = str(tmp_path)
        self._sampler(outdir).sample(np.zeros(2), num_iterations=1000)
        os.remove(os.path.join(outdir, "sampler_checkpoint.json"))
        with open(os.path.join(outdir, "sampler_checkpoint.pkl"), "wb") as fp:
            fp.write(b"not read")
        with pytest.raises(RuntimeError, match=r"sampler_checkpoint\.pkl"):
            self._sampler(outdir, resume=True).sample(np.zeros(2), num_iterations=1000)

    def test_resume_true_on_empty_outdir_still_works(self, tmp_path):
        outdir = str(tmp_path / "fresh")
        self._sampler(outdir, resume=True).sample(np.zeros(2), num_iterations=500)
        assert count_chain_rows(os.path.join(outdir, "chain_0.bin"), _GUARD_NCOLS) == 500

    def test_normal_resume_still_works(self, tmp_path):
        outdir = str(tmp_path)
        self._sampler(outdir).sample(np.zeros(2), num_iterations=1000)
        self._sampler(outdir, resume=True).sample(np.zeros(2), num_iterations=2000)
        assert count_chain_rows(os.path.join(outdir, "chain_0.bin"), _GUARD_NCOLS) == 2000


class TestCorruptCheckpointIsSkipped:
    """A damaged .npz is a torn checkpoint: check_for_checkpoint returns None, never raises."""

    @staticmethod
    def _write(outdir):
        sampler = PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=2,
            seed=0,
            outdir=outdir,
            save_freq=50,
            verbose=False,
        )
        sampler.sample(np.zeros(2), num_iterations=60)
        return os.path.join(outdir, "sampler_checkpoint.npz")

    def test_truncated_npz(self, tmp_path):
        npz = self._write(str(tmp_path))
        data = open(npz, "rb").read()
        with open(npz, "wb") as fp:
            fp.write(data[: len(data) // 2])
        assert check_for_checkpoint(str(tmp_path)) is None

    def test_empty_npz(self, tmp_path):
        npz = self._write(str(tmp_path))
        open(npz, "wb").close()
        assert check_for_checkpoint(str(tmp_path)) is None


def test_pkl_suffix_is_treated_as_a_base_name(tmp_path):
    """checkpoint_sampler(s, 'run.pkl') used to write run.pkl.npz / run.pkl.json."""
    from impulse.resume import load_state_checkpoint

    sampler = PTSampler(
        ndim=2,
        lnlike=_gauss_lnlike,
        lnprior=_flat_lnprior,
        ntemps=2,
        seed=0,
        outdir=str(tmp_path / "chains"),
        save_freq=50,
        verbose=False,
    )
    sampler.sample(np.zeros(2), num_iterations=60)
    written = checkpoint_sampler(sampler, path=str(tmp_path / "run.pkl"))
    assert written == str(tmp_path / "run.json")
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["run.json", "run.npz"]
    arrays, meta = load_state_checkpoint(str(tmp_path / "run.pkl"))
    assert meta["ndim"] == 2

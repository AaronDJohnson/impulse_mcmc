"""The ``verbose`` flag silences progress output without touching the chain."""

import io
import os
from contextlib import redirect_stderr

import numpy as np

from impulse import PTSampler, chain_io


def _lnlike(x):
    return -0.5 * np.sum(np.asarray(x) ** 2)


def _lnprior(x):
    return 0.0 if np.all(np.abs(np.asarray(x)) <= 10) else -np.inf


_DEFAULT = object()


def _run(outdir, verbose=_DEFAULT, resume=False, iters=150):
    stderr = io.StringIO()
    kwargs = {} if verbose is _DEFAULT else {"verbose": verbose}
    with redirect_stderr(stderr):
        PTSampler(
            ndim=2,
            lnlike=_lnlike,
            lnprior=_lnprior,
            ntemps=3,
            seed=1,
            outdir=outdir,
            save_freq=50,
            resume=resume,
            **kwargs,
        ).sample(np.zeros(2), num_iterations=iters)
    return stderr.getvalue()


def _chain_bytes(outdir):
    base = os.path.join(outdir, "chain_0")
    fmt = chain_io.detect_format(base)
    assert fmt is not None, f"no chain file under {base}"
    with open(base + chain_io.chain_suffix(fmt), "rb") as fp:
        return fp.read()


class TestVerboseFlag:
    def test_verbose_false_is_silent(self, tmp_path):
        out = _run(str(tmp_path / "pt_quiet"), verbose=False)
        assert out == "", f"verbose=False still wrote to stderr: {out[:200]!r}"

    def test_verbose_true_shows_progress(self, tmp_path):
        out = _run(str(tmp_path / "pt_loud"), verbose=True)
        assert out != "", "verbose=True produced no progress output"

    def test_verbose_does_not_change_the_chain(self, tmp_path):
        loud_dir = str(tmp_path / "pt_a")
        quiet_dir = str(tmp_path / "pt_b")
        _run(loud_dir, verbose=True)
        _run(quiet_dir, verbose=False)
        assert _chain_bytes(loud_dir) == _chain_bytes(
            quiet_dir
        ), "verbose changed the sampled chain; it must be presentation-only"

    def test_default_is_verbose(self, tmp_path):
        out = _run(str(tmp_path / "pt_default"))
        assert out != ""

    def test_verbose_survives_resume(self, tmp_path):
        outdir = str(tmp_path / "resume")
        _run(outdir, verbose=True, resume=False, iters=100)
        out = _run(outdir, verbose=False, resume=True, iters=150)
        assert out == "", f"resuming with verbose=False produced progress output: {out[:200]!r}"

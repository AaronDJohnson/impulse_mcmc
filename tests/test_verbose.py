"""The ``verbose`` flag silences progress output without touching the chain.

``verbose`` is presentation-only. Two properties matter and are easy to break
independently:

1. ``verbose=False`` writes nothing to stderr (tqdm's stream).
2. It does not perturb sampling -- the chain must be byte-identical to a
   ``verbose=True`` run with the same seed. A progress bar that consumed a
   random number, or a flag accidentally wired into a run-shaping argument,
   would show up here.
"""

import io
import os
from contextlib import redirect_stderr

import numpy as np
import pytest

from impulse import NUTSSampler, PTSampler
from impulse.experimental import HybridPTSampler


def _lnlike(x):
    return -0.5 * np.sum(np.asarray(x) ** 2)


def _lnprior(x):
    return 0.0 if np.all(np.abs(np.asarray(x)) <= 10) else -np.inf


def _logp_and_grad(x):
    x = np.asarray(x, dtype=float)
    return -0.5 * float(np.sum(x**2)), -x


def _run(kind, outdir, verbose):
    """Run a short chain of the given sampler and return its stderr output."""
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        if kind == "pt":
            PTSampler(
                ndim=2,
                lnlike=_lnlike,
                lnprior=_lnprior,
                ntemps=3,
                seed=1,
                outdir=outdir,
                verbose=verbose,
            ).sample(np.zeros(2), num_iterations=150)
        elif kind == "hybrid":
            HybridPTSampler(
                ndim=2,
                lnlike=_lnlike,
                lnprior=_lnprior,
                ntemps=3,
                seed=1,
                outdir=outdir,
                verbose=verbose,
            ).sample(np.zeros(2), num_iterations=150)
        else:
            NUTSSampler(
                ndim=2,
                logp_and_grad=_logp_and_grad,
                num_warmup=40,
                seed=1,
                outdir=outdir,
                verbose=verbose,
            ).sample(np.zeros(2), num_iterations=150)
    return stderr.getvalue()


def _chain_bytes(kind, outdir):
    name = "chain_nuts.txt" if kind == "nuts" else "chain_1.0.txt"
    path = os.path.join(outdir, name)
    if not os.path.exists(path):  # PT chain filenames follow the ladder
        cands = sorted(f for f in os.listdir(outdir) if f.endswith(".txt"))
        path = os.path.join(outdir, cands[0])
    with open(path) as fp:
        return fp.read()


@pytest.mark.parametrize("kind", ["pt", "hybrid", "nuts"])
class TestVerboseFlag:
    def test_verbose_false_is_silent(self, tmp_path, kind):
        out = _run(kind, str(tmp_path / f"{kind}_quiet"), verbose=False)
        assert out == "", f"{kind}: verbose=False still wrote to stderr: {out[:200]!r}"

    def test_verbose_true_shows_progress(self, tmp_path, kind):
        """Guards against the flag being inverted or the bar removed entirely."""
        out = _run(kind, str(tmp_path / f"{kind}_loud"), verbose=True)
        assert out != "", f"{kind}: verbose=True produced no progress output"

    def test_verbose_does_not_change_the_chain(self, tmp_path, kind):
        loud_dir = str(tmp_path / f"{kind}_a")
        quiet_dir = str(tmp_path / f"{kind}_b")
        _run(kind, loud_dir, verbose=True)
        _run(kind, quiet_dir, verbose=False)
        assert _chain_bytes(kind, loud_dir) == _chain_bytes(kind, quiet_dir), (
            f"{kind}: verbose changed the sampled chain; it must be " "presentation-only"
        )

    def test_default_is_verbose(self, tmp_path, kind):
        """Silencing by default would be a surprising API change."""
        out = _run(kind, str(tmp_path / f"{kind}_default"), verbose=True)
        assert out != ""


def test_verbose_survives_a_legacy_pickle_resume(tmp_path):
    """A run resumed with verbose=False stays quiet.

    The legacy pickle restore path replaces state via ``__dict__.update``, so a
    checkpoint written by a verbose run would otherwise re-enable the progress
    bar in a process that explicitly asked for silence.
    """
    from impulse.resume import checkpoint_sampler

    outdir = str(tmp_path / "resume")

    def make(verbose, resume):
        return PTSampler(
            ndim=2,
            lnlike=_lnlike,
            lnprior=_lnprior,
            ntemps=3,
            seed=1,
            outdir=outdir,
            save_freq=50,
            verbose=verbose,
            resume=resume,
        )

    loud = make(verbose=True, resume=False)
    loud.sample(np.zeros(2), num_iterations=100)
    # Force the legacy pickle so the __dict__.update path is the one exercised.
    for ext in (".json", ".npz"):
        path = os.path.join(outdir, "sampler_checkpoint" + ext)
        if os.path.exists(path):
            os.remove(path)
    checkpoint_sampler(loud, path=os.path.join(outdir, "sampler_checkpoint.pkl"))

    stderr = io.StringIO()
    with redirect_stderr(stderr):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make(verbose=False, resume=True).sample(np.zeros(2), num_iterations=150)
    assert stderr.getvalue() == "", (
        "resuming with verbose=False produced progress output: the checkpointed "
        f"verbose flag overwrote the constructor's: {stderr.getvalue()[:200]!r}"
    )

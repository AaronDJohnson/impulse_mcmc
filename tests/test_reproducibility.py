"""Reproducibility guarantees: seed determinism and checkpoint-resume equivalence.

Two properties are pinned down here, both BIT-IDENTICAL (``np.array_equal``),
not statistical:

1. **Seed determinism** — two fresh runs with the same seed and configuration
   produce identical chain files (samples, lnlike, lnprob for every
   temperature), for both :class:`~impulse.samplers.PTSampler` and
   :class:`~impulse.rjpt_sampler.RJPTSampler` (including trans-dimensional
   birth/death activity).

2. **Resume equivalence** — an uninterrupted run of ``N + M`` iterations and a
   run of ``N`` iterations followed by a fresh constructor with
   ``resume=True`` continuing to ``N + M`` produce identical chain files over
   the FULL ``N + M`` history.

Resume semantics being tested (see the individual test docstrings):

- ``num_iterations`` is a GLOBAL iteration target: on resume the loop runs
  ``range(short_chain.iteration, num_iterations)``, so the resumed call must
  pass the total ``N + M``, not the increment ``M``.
- Checkpoints are written at the END of every iteration ``jj`` with ``jj > 0``
  and ``jj % save_freq == 0``, i.e. after the MH step, the buffer append, the
  PT swap, ladder adaptation, and the covariance update of iteration ``jj``
  have all completed.  The pickle therefore captures every RNG stream, the
  :class:`~impulse.file_io.ShortChain` ring buffer (with its unsaved-sample
  count and flushed-row count), the adaptive-proposal statistics, and the
  PT ladder exactly at an iteration boundary, and a resumed run continues at
  ``jj + 1`` with bit-identical draws.
- Iterations executed after the last checkpoint (e.g. the tail of a run whose
  ``num_iterations`` is not a multiple of ``save_freq``) are RE-GENERATED
  deterministically by the resumed run; on resume the chain files on disk are
  truncated back to the checkpointed flushed-row count first, so the
  regenerated tail neither duplicates nor skips rows.
"""

import os
import pickle

import numpy as np
import pytest

from impulse.resume import check_for_checkpoint
from impulse.rjmcmc import RJMCMCProductSpace
from impulse.rjpt_sampler import RJPTSampler
from impulse.samplers import PTSampler

SEED = 1234

# ---------------------------------------------------------------------------
# Picklable model functions (module-level: they end up inside checkpoints)
# ---------------------------------------------------------------------------


def _gauss_lnlike(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return -0.5 * np.sum(x**2)
    return -0.5 * np.sum(x**2, axis=1)


def _flat_lnprior(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return 0.0 if np.all(np.abs(x) <= 10) else -np.inf
    result = np.zeros(x.shape[0])
    result[np.any(np.abs(x) > 10, axis=1)] = -np.inf
    return result


def _gauss_lnlike_grad(x):
    """``(active_params) -> (loglike, grad)`` for a standard Gaussian."""
    x = np.asarray(x, dtype=np.float64)
    return -0.5 * np.sum(x**2), -x


# Small RJ problem: sum-of-sinusoids fit, 2 params (amplitude, frequency)
# per source, up to 3 sources.  One true source, so birth/death moves are
# genuinely exercised in both directions.
NUM_PARAMS = 2
MAX_SOURCES = 3
LO = np.array([0.0, 0.0])
HI = np.array([5.0, 3.0])

_RNG_DATA = np.random.default_rng(0)
N_PTS = 50
T_GRID = np.linspace(0, 2 * np.pi, N_PTS)
SIGMA = 1.0
TRUE_A, TRUE_F = 2.0, 1.0
DATA = TRUE_A * np.sin(2 * np.pi * TRUE_F * T_GRID) + SIGMA * _RNG_DATA.standard_normal(N_PTS)


def _rj_source_draw(rng):
    return rng.uniform(LO, HI)


def _rj_logprior(params):
    n = len(params)
    for i in range(n // NUM_PARAMS):
        p = params[i * NUM_PARAMS : (i + 1) * NUM_PARAMS]
        if np.any(p < LO) or np.any(p > HI):
            return -np.inf
    return 0.0


def _rj_loglike(params):
    n_sources = len(params) // NUM_PARAMS
    model = np.zeros(N_PTS)
    for i in range(n_sources):
        a = params[i * NUM_PARAMS]
        f = params[i * NUM_PARAMS + 1]
        model += a * np.sin(2 * np.pi * f * T_GRID)
    return -0.5 * np.sum(((DATA - model) / SIGMA) ** 2)


def _make_rj_space():
    return RJMCMCProductSpace(
        loglikelihood=_rj_loglike,
        logprior=_rj_logprior,
        num_sources=MAX_SOURCES,
        num_params=NUM_PARAMS,
        source_prior_draw=_rj_source_draw,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_chains_bit_identical(chain_a, chain_b, keys=("samples", "lnlike", "lnprob")):
    """Assert two load_chain() results are bit-identical for every temperature.

    On mismatch, report the first differing iteration per temperature so RNG
    stream divergences are easy to localize.
    """
    for key in keys:
        a, b = chain_a[key], chain_b[key]
        assert a.shape == b.shape, f"{key}: shape mismatch {a.shape} vs {b.shape}"
        if np.array_equal(a, b):
            continue
        # Build a precise failure message: first differing row per temp.
        details = []
        for temp in range(a.shape[0]):
            diff = a[temp] != b[temp]
            if diff.ndim > 1:
                diff = diff.any(axis=-1)
            bad = np.nonzero(diff)[0]
            if bad.size:
                details.append(
                    f"temp {temp}: first mismatch at iteration {bad[0]} "
                    f"({bad.size} rows differ)"
                )
        raise AssertionError(f"{key} not bit-identical: " + "; ".join(details))


def _pt_sampler(outdir, resume=False):
    return PTSampler(
        ndim=2,
        lnlike=_gauss_lnlike,
        lnprior=_flat_lnprior,
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=200,
        cov_update=100,
        buffer_size=200,
    )


def _pt_resume_sampler(outdir, resume=False):
    # cov_update deliberately NOT a divisor of save_freq: the covariance
    # refresh cadence must survive the checkpoint even when the checkpoint
    # iteration is not a covariance-update boundary.
    return PTSampler(
        ndim=2,
        lnlike=_gauss_lnlike,
        lnprior=_flat_lnprior,
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=40,
        cov_update=25,
        buffer_size=120,
    )


def _rjpt_nuts_sampler(outdir, resume=False):
    return RJPTSampler(
        ndim=2,
        lnlike=_gauss_lnlike,
        lnprior=_flat_lnprior,
        lnlike_grad=_gauss_lnlike_grad,
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=40,
        cov_update=25,
        buffer_size=120,
        max_tree_depth=4,
        hot_chain_max_depth=3,
        mass_matrix_adapt_interval=60,
        mass_matrix_min_samples=20,
    )


def _rjpt_rj_sampler(outdir, resume=False, save_freq=200, cov_update=100):
    return RJPTSampler.from_rjmcmc(
        _make_rj_space(),
        ntemps=3,
        seed=SEED,
        outdir=outdir,
        resume=resume,
        save_freq=save_freq,
        cov_update=cov_update,
        buffer_size=200,
    )


def _rj_x0():
    return _make_rj_space().draw_initial_position(
        np.random.default_rng(SEED),
        nmodel=0,
    )


# ---------------------------------------------------------------------------
# (1) Seed determinism
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """Two fresh runs, identical seed/config -> bit-identical chains."""

    def test_ptsampler_seed_determinism(self, tmp_path):
        """PTSampler: 600 iterations spanning several covariance updates
        (cov_update=100), per-iteration temperature-ladder adaptation, and
        multiple chain-file flushes (save_freq=200) must be bit-identical
        between two independent runs with the same seed."""
        x0 = np.array([0.5, -0.3])
        chains = []
        for sub in ("run_a", "run_b"):
            outdir = str(tmp_path / sub)
            sampler = _pt_sampler(outdir)
            sampler.sample(x0, num_iterations=600)
            chains.append(sampler.load_chain())

        # sanity: full history present, and the ladder actually adapted
        assert chains[0]["samples"].shape == (3, 600, 2)
        assert not np.all(
            chains[0]["temperature"][1] == chains[0]["temperature"][1][0]
        ), "ladder adaptation never moved the interior temperature"

        _assert_chains_bit_identical(
            chains[0],
            chains[1],
            keys=("samples", "lnlike", "lnprob", "accepted", "temperature"),
        )

    def test_rjpt_rj_seed_determinism(self, tmp_path):
        """RJPTSampler on a small RJ problem: 600 iterations with the
        combined birth/death kernel, model-index jumps, and source swaps
        registered.  Requires genuine trans-dimensional activity (accepted
        birth/death moves, more than one model index visited on the cold
        chain) and bit-identical chains between two same-seed runs."""
        x0 = _rj_x0()
        chains = []
        for sub in ("run_a", "run_b"):
            outdir = str(tmp_path / sub)
            sampler = _rjpt_rj_sampler(outdir)
            sampler.sample(x0, num_iterations=600)
            chains.append(sampler.load_chain())
            report = sampler.proposal_acceptance_rates()

        # sanity: birth/death moves fired AND were accepted
        assert report["birth_death"]["calls"] > 0
        assert report["birth_death"]["accepts"] > 0, (
            "no accepted birth/death move: trans-dimensional activity is "
            "not being exercised by this configuration/seed"
        )
        nmodel_cold = np.rint(chains[0]["samples"][0, :, -1]).astype(int)
        assert len(np.unique(nmodel_cold)) > 1, "cold chain never changed model index"

        _assert_chains_bit_identical(
            chains[0],
            chains[1],
            keys=("samples", "lnlike", "lnprob", "accepted", "temperature"),
        )


# ---------------------------------------------------------------------------
# (2) Resume equivalence
# ---------------------------------------------------------------------------

# N and M are chosen deliberately AGAINST the checkpoint boundaries
# (save_freq=40): the last checkpoint of the N=100 run is written at the end
# of iteration 80, so the resumed run must (a) truncate the 20 tail rows
# (iterations 80-99) that the graceful shutdown of run N flushed to disk
# after the checkpoint, and (b) re-generate iterations 81..99 bit-identically
# from the pickled RNG streams before sampling new ground up to 160.
# cov_update=25 does not divide save_freq=40, so the covariance-update
# cadence (_last_cov_iter) must itself be checkpointed state.
N_FIRST = 100
M_EXTRA = 60


class TestResumeEquivalence:
    """run(N) -> checkpoint -> fresh constructor + resume=True -> run to N+M
    must equal one uninterrupted run of N+M, bit-identically, over the FULL
    N+M history.

    Guarantee under test (see module docstring for the full statement): the
    checkpoint is written at the end of iteration ``jj`` for every ``jj > 0``
    with ``jj % save_freq == 0`` and captures a fully completed iteration;
    ``num_iterations`` is a global target, so the resumed call passes
    ``N + M``; the resumed run truncates the chain files to the checkpointed
    flushed-row count and re-generates everything after the checkpoint
    deterministically.
    """

    def _run_pair(self, tmp_path, make_sampler, x0):
        # Uninterrupted N+M run
        full_dir = str(tmp_path / "full")
        full = make_sampler(full_dir)
        full.sample(x0, num_iterations=N_FIRST + M_EXTRA)
        full_chain = full.load_chain()

        # Interrupted: N iterations, then a FRESH constructor with
        # resume=True continuing to the same global target N+M.
        split_dir = str(tmp_path / "split")
        first = make_sampler(split_dir)
        first.sample(x0, num_iterations=N_FIRST)
        # A checkpoint (the no-code-execution .npz/.json pair by default, or a
        # legacy .pkl) must exist; check_for_checkpoint resolves whichever.
        assert (
            check_for_checkpoint(split_dir) is not None
        ), "no checkpoint written during the first run"

        resumed = make_sampler(split_dir, resume=True)
        resumed.sample(x0, num_iterations=N_FIRST + M_EXTRA)
        resumed_chain = resumed.load_chain()

        # Full history, no duplicated / dropped rows (this catches the
        # append-after-graceful-shutdown failure mode where the tail rows
        # between the last checkpoint and N appear twice).
        assert full_chain["samples"].shape[1] == N_FIRST + M_EXTRA
        assert resumed_chain["samples"].shape[1] == N_FIRST + M_EXTRA, (
            f"resumed run wrote {resumed_chain['samples'].shape[1]} rows, "
            f"expected {N_FIRST + M_EXTRA}: chain files were duplicated or "
            "truncated across the resume boundary"
        )

        _assert_chains_bit_identical(
            full_chain,
            resumed_chain,
            keys=("samples", "lnlike", "lnprob", "accepted", "temperature"),
        )
        return full_chain, resumed_chain

    def test_ptsampler_resume_equivalence(self, tmp_path):
        """PTSampler resume equivalence with adaptive covariance, DE-buffer
        growth, and per-iteration ladder adaptation across the boundary."""
        self._run_pair(tmp_path, _pt_resume_sampler, np.array([0.5, -0.3]))

    def test_rjpt_nuts_resume_equivalence(self, tmp_path):
        """RJPTSampler with NUTS enabled: dual-averaging step-size state,
        mass-matrix adaptation buffers/counters, and the per-chain NUTS RNG
        streams must all be restored so the resumed trajectory is
        bit-identical (mass_matrix_adapt_interval=60 forces mass-matrix
        commits on both sides of the resume boundary)."""
        self._run_pair(tmp_path, _rjpt_nuts_sampler, np.array([0.5, -0.3]))

    def test_rjpt_rj_resume_equivalence(self, tmp_path):
        """RJPTSampler on the RJ problem (MH-only): birth/death kernel state
        and per-model adaptive statistics must survive the checkpoint;
        trans-dimensional moves must be active across the boundary."""

        def make(outdir, resume=False):
            return _rjpt_rj_sampler(outdir, resume=resume, save_freq=40, cov_update=25)

        full_chain, _ = self._run_pair(tmp_path, make, _rj_x0())
        nmodel_cold = np.rint(full_chain["samples"][0, :, -1]).astype(int)
        assert len(np.unique(nmodel_cold)) > 1, "cold chain never changed model index"


# ---------------------------------------------------------------------------
# (3) Legacy (pre-row-tracking) checkpoints must never lose chain rows
# ---------------------------------------------------------------------------


class TestLegacyCheckpointRowTracking:
    """Resume lineages passing through a pre-row-tracking checkpoint must
    never destroy chain-file history.

    Regression for a data-loss bug: a ``ShortChain`` unpickled from a
    checkpoint written before ``_rows_written`` existed lacked the
    attribute, so ``save_chain`` restarted the counter at 0 while the files
    already held many rows.  The undercount was pickled into the next
    checkpoint, and on the SECOND resume ``truncate_files_to_saved``
    trusted it and rewrote the chain files as a tiny prefix, permanently
    destroying the first run's history.  The fix re-seeds a missing counter
    from the CURRENT on-disk line count (never 0).
    """

    NTEMPS = 2

    @staticmethod
    def _make(outdir, resume=False):
        return PTSampler(
            ndim=2,
            lnlike=_gauss_lnlike,
            lnprior=_flat_lnprior,
            ntemps=TestLegacyCheckpointRowTracking.NTEMPS,
            seed=1,
            outdir=outdir,
            save_freq=10,
            resume=resume,
        )

    def test_double_resume_through_pre_row_tracking_checkpoint(self, tmp_path):
        from impulse.resume import checkpoint_sampler

        outdir = str(tmp_path)
        x0 = np.array([0.1, 0.1])

        # Run 1: fresh run to 25 iterations. This regression is specific to
        # the LEGACY pickle format — the new .npz/.json format always stores
        # the row counter, so a resume can never restart it at 0. Write the
        # run's checkpoint as a legacy pickle (and drop the new-format pair)
        # so the pre-row-tracking downgrade below is meaningful.
        first = self._make(outdir)
        first.sample(x0, num_iterations=25)
        ckpt = os.path.join(outdir, "sampler_checkpoint.pkl")
        checkpoint_sampler(first, path=ckpt, format="pickle")
        for _ext in (".json", ".npz"):
            _p = os.path.join(outdir, "sampler_checkpoint" + _ext)
            if os.path.exists(_p):
                os.remove(_p)
        original = []
        for ii in range(self.NTEMPS):
            with open(os.path.join(outdir, f"chain_{ii}.txt")) as fp:
                lines = fp.readlines()
            assert len(lines) == 25
            original.append(lines)

        # Simulate a legacy checkpoint written before row tracking existed:
        # strip the attribute from the pickled ShortChain.
        with open(ckpt, "rb") as fp:
            state = pickle.load(fp)
        assert hasattr(state.short_chain, "_rows_written")
        del state.short_chain._rows_written
        with open(ckpt, "wb") as fp:
            pickle.dump(state, fp)

        # Resume 1: the missing counter must be re-seeded from the on-disk
        # row count (25), not restart at 0.
        self._make(outdir, resume=True).sample(x0, num_iterations=40)
        with open(ckpt, "rb") as fp:
            state = pickle.load(fp)
        assert getattr(state.short_chain, "_rows_written", 0) >= 25, (
            "row counter restarted below the pre-existing on-disk row "
            "count: the next resume's truncation would destroy history"
        )

        # Resume 2: truncate_files_to_saved trusts the pickled counter; an
        # undercount here rewrote the files as lines[:rows], destroying the
        # first run's rows.
        self._make(outdir, resume=True).sample(x0, num_iterations=50)

        for ii in range(self.NTEMPS):
            with open(os.path.join(outdir, f"chain_{ii}.txt")) as fp:
                lines = fp.readlines()
            # All 50 iterations present at least once (legacy resumes may
            # append a few duplicate rows — the historical append-only
            # behavior — but must never LOSE rows).
            assert len(lines) >= 50, (
                f"chain_{ii}.txt holds {len(lines)} rows after the double "
                "resume; history was truncated"
            )
            # Run 1's history is preserved verbatim at the head of the file.
            assert lines[:25] == original[ii], (
                f"chain_{ii}.txt no longer starts with the first run's "
                "rows: resume destroyed pre-checkpoint history"
            )

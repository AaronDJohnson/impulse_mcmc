"""Reproducibility guarantees: seed determinism and checkpoint-resume equivalence."""

import numpy as np

from impulse.resume import check_for_checkpoint
from impulse.samplers import PTSampler

SEED = 1234


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


def _assert_chains_bit_identical(chain_a, chain_b, keys=("samples", "lnlike", "lnprob")):
    for key in keys:
        a, b = chain_a[key], chain_b[key]
        assert a.shape == b.shape, f"{key}: shape mismatch {a.shape} vs {b.shape}"
        if np.array_equal(a, b):
            continue
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


class TestSeedDeterminism:
    """Two fresh runs, identical seed/config -> bit-identical chains."""

    def test_ptsampler_seed_determinism(self, tmp_path):
        x0 = np.array([0.5, -0.3])
        chains = []
        for sub in ("run_a", "run_b"):
            outdir = str(tmp_path / sub)
            sampler = _pt_sampler(outdir)
            sampler.sample(x0, num_iterations=600)
            chains.append(sampler.load_chain())

        assert chains[0]["samples"].shape == (3, 600, 2)
        assert not np.all(
            chains[0]["temperature"][1] == chains[0]["temperature"][1][0]
        ), "ladder adaptation never moved the interior temperature"

        _assert_chains_bit_identical(
            chains[0],
            chains[1],
            keys=("samples", "lnlike", "lnprob", "accepted", "temperature"),
        )


N_FIRST = 100
M_EXTRA = 60


class TestResumeEquivalence:
    """run(N) -> checkpoint -> fresh constructor + resume=True -> run to N+M
    must equal one uninterrupted run of N+M, bit-identically, over the FULL
    N+M history."""

    def _run_pair(self, tmp_path, make_sampler, x0):
        full_dir = str(tmp_path / "full")
        full = make_sampler(full_dir)
        full.sample(x0, num_iterations=N_FIRST + M_EXTRA)
        full_chain = full.load_chain()

        split_dir = str(tmp_path / "split")
        first = make_sampler(split_dir)
        first.sample(x0, num_iterations=N_FIRST)
        assert (
            check_for_checkpoint(split_dir) is not None
        ), "no checkpoint written during the first run"

        resumed = make_sampler(split_dir, resume=True)
        resumed.sample(x0, num_iterations=N_FIRST + M_EXTRA)
        resumed_chain = resumed.load_chain()

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
        self._run_pair(tmp_path, _pt_resume_sampler, np.array([0.5, -0.3]))

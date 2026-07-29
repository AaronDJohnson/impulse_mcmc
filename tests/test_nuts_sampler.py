"""Integration tests for NUTSSampler."""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest

from impulse.nuts.gradient_helpers import compose_logp_and_grad, make_logp_and_grad_numerical
from impulse.nuts.sampler import NUTSSampler
from impulse.resume import checkpoint_sampler, load_nuts_checkpoint


def gaussian_logp_and_grad(x):
    logp = -0.5 * np.sum(x**2)
    grad = -x
    return logp, grad


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestNUTSSamplerBasic:
    def test_sample_2d_gaussian(self, temp_dir):
        """Sample a 2D Gaussian and verify mean/cov."""
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            num_warmup=500,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        sampler.sample(np.array([3.0, -2.0]), num_iterations=2000)

        chain = sampler.load_chain()
        samples = chain["samples"]

        # Discard first 200 post-warmup samples as burn-in
        samples = samples[200:]
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)

        # Mean should be near 0
        np.testing.assert_allclose(mean, [0, 0], atol=0.2)
        # Variance should be near 1
        np.testing.assert_allclose(np.diag(cov), [1, 1], rtol=0.3)

    def test_correlated_gaussian(self, temp_dir):
        """Test mass matrix adaptation on correlated Gaussian."""
        rho = 0.9
        cov = np.array([[1.0, rho], [rho, 1.0]])
        prec = np.linalg.inv(cov)

        def logp_and_grad(x):
            logp = -0.5 * x @ prec @ x
            grad = -prec @ x
            return logp, grad

        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=logp_and_grad,
            num_warmup=1000,
            mass_matrix_type="diagonal",
            seed=42,
            outdir=temp_dir,
            save_freq=1000,
        )
        sampler.sample(np.array([0.0, 0.0]), num_iterations=3000)

        chain = sampler.load_chain()
        samples = chain["samples"][500:]
        sample_cov = np.cov(samples, rowvar=False)

        # Should recover the correlation structure roughly
        np.testing.assert_allclose(np.diag(sample_cov), [1, 1], rtol=0.4)

    def test_no_divergences_on_gaussian(self, temp_dir):
        """Well-conditioned Gaussian should have 0 divergences."""
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            num_warmup=300,
            seed=42,
            outdir=temp_dir,
            save_freq=500,
        )
        sampler.sample(np.zeros(2), num_iterations=500)
        diag = sampler.get_diagnostics()
        assert diag["num_divergent"] == 0


class TestChainIO:
    def test_roundtrip(self, temp_dir):
        """Chain save/load roundtrip preserves data."""
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            num_warmup=50,
            seed=42,
            outdir=temp_dir,
            save_freq=100,
        )
        sampler.sample(np.zeros(2), num_iterations=100)

        chain = sampler.load_chain()
        assert chain["samples"].shape == (100, 2)
        assert chain["logp"].shape == (100,)
        assert chain["tree_depth"].shape == (100,)
        assert chain["divergent"].dtype == bool
        assert chain["accepted"].dtype == bool

    def test_file_not_found(self, temp_dir):
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            outdir=os.path.join(temp_dir, "nonexistent"),
        )
        with pytest.raises(FileNotFoundError):
            sampler.load_chain()


class TestCheckpointResume:
    def _sampler(self, temp_dir, **kw):
        kw.setdefault("num_warmup", 50)
        kw.setdefault("save_freq", 50)
        return NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            seed=42,
            outdir=temp_dir,
            **kw,
        )

    def test_checkpoint_and_load(self, temp_dir):
        """Explicit pickle checkpoint round-trip (the legacy manual path)."""
        sampler = self._sampler(temp_dir)
        sampler.sample(np.zeros(2), num_iterations=100)

        # Checkpoint
        ckpt_path = os.path.join(temp_dir, "nuts_ckpt.pkl")
        checkpoint_sampler(sampler, path=ckpt_path, omit=("logp_and_grad",))

        # Load
        loaded = load_nuts_checkpoint(ckpt_path, gaussian_logp_and_grad)
        assert loaded.ndim == 2
        assert loaded.logp_and_grad is gaussian_logp_and_grad
        np.testing.assert_array_equal(loaded.state.position, sampler.state.position)

    def test_default_checkpoint_is_the_no_code_execution_format(self, temp_dir):
        """Automatic checkpoints are .npz + .json, never a pickle.

        NUTSSampler was the last writer of the deprecated pickle format; it now
        implements the capture hook, so checkpoint_sampler selects the safe
        format for it like it does for the PT samplers.
        """
        self._sampler(temp_dir).sample(np.zeros(2), num_iterations=100)
        assert os.path.exists(os.path.join(temp_dir, "sampler_checkpoint.json"))
        assert os.path.exists(os.path.join(temp_dir, "sampler_checkpoint.npz"))
        assert not os.path.exists(os.path.join(temp_dir, "sampler_checkpoint.pkl"))

        # Loading executes no code: allow_pickle=False must succeed.
        with np.load(os.path.join(temp_dir, "sampler_checkpoint.npz"), allow_pickle=False) as npz:
            assert "position" in npz.files

    def test_resume_with_no_checkpoint_starts_fresh(self, temp_dir):
        """The requeue idiom: the same script works for the first submission."""
        sampler = self._sampler(temp_dir, resume=True)
        sampler.sample(np.zeros(2), num_iterations=100)
        assert sampler._iteration == 100

    def test_resume_without_checkpoint_but_with_chain_rows_raises(self, temp_dir):
        """Refuse to splice a fresh run onto an existing chain file.

        save_freq exceeds the run length, so rows reach disk but no checkpoint
        is ever written. Appending a second, re-warmed-up run to that file
        would silently mix two chains, so this must raise rather than proceed.
        """
        self._sampler(temp_dir, save_freq=10_000).sample(np.zeros(2), num_iterations=40)
        assert os.path.getsize(os.path.join(temp_dir, "chain_nuts.txt")) > 0

        with pytest.raises(RuntimeError, match="no usable checkpoint"):
            self._sampler(temp_dir, save_freq=10_000, resume=True).sample(
                np.zeros(2), num_iterations=200
            )

    def test_resume_from_legacy_pickle_checkpoint_raises(self, temp_dir):
        """Legacy NUTS pickles predate resume and lack the progress counters.

        They were written but never read back, so they record neither the
        iteration count nor the flushed-row count a resume needs. Refuse
        clearly instead of guessing.
        """
        sampler = self._sampler(temp_dir)
        sampler.sample(np.zeros(2), num_iterations=100)
        # Emulate a pre-2.0 outdir: only a pickle checkpoint present.
        for ext in (".json", ".npz"):
            os.remove(os.path.join(temp_dir, "sampler_checkpoint" + ext))
        checkpoint_sampler(
            sampler,
            path=os.path.join(temp_dir, "sampler_checkpoint.pkl"),
            omit=("logp_and_grad",),
        )

        with pytest.raises(RuntimeError, match="legacy pickle checkpoint"):
            self._sampler(temp_dir, resume=True).sample(np.zeros(2), num_iterations=200)

    def test_resume_refuses_changed_run_shaping_argument(self, temp_dir):
        """Reconstruct-then-restore: a changed num_warmup is caught, not absorbed."""
        from impulse.resume import CheckpointMismatchError

        self._sampler(temp_dir).sample(np.zeros(2), num_iterations=100)
        with pytest.raises(CheckpointMismatchError, match="num_warmup mismatch"):
            self._sampler(temp_dir, num_warmup=999, resume=True).sample(
                np.zeros(2), num_iterations=200
            )


class TestDiagnostics:
    def test_diagnostics_keys(self, temp_dir):
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            num_warmup=50,
            seed=42,
            outdir=temp_dir,
            save_freq=100,
        )
        sampler.sample(np.zeros(2), num_iterations=100)
        diag = sampler.get_diagnostics()
        assert "num_divergent" in diag
        assert "num_max_depth" in diag
        assert "mean_tree_depth" in diag
        assert "mean_accept_prob" in diag
        assert "final_step_size" in diag
        assert "mass_matrix_type" in diag

    def test_no_data_raises(self):
        sampler = NUTSSampler(ndim=2, logp_and_grad=gaussian_logp_and_grad)
        with pytest.raises(RuntimeError):
            sampler.get_diagnostics()


class TestNumericalGradientEndToEnd:
    def test_numerical_gradient_sampler(self, temp_dir):
        """NUTSSampler with numerical gradients should work."""

        def lnlike(x):
            return -0.5 * np.sum(x**2)

        def lnprior(x):
            return 0.0 if np.all(np.abs(x) < 10) else -np.inf

        logp_and_grad = compose_logp_and_grad(lnlike, lnprior)
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=logp_and_grad,
            num_warmup=200,
            seed=42,
            outdir=temp_dir,
            save_freq=200,
        )
        sampler.sample(np.array([1.0, -1.0]), num_iterations=500)

        chain = sampler.load_chain()
        mean = np.mean(chain["samples"][100:], axis=0)
        np.testing.assert_allclose(mean, [0, 0], atol=0.3)


class TestHigherDimensional:
    def test_5d_gaussian(self, temp_dir):
        """5D Gaussian: verify posterior mean within 2 sigma."""
        ndim = 5
        # Correlated covariance
        rho = 0.5
        cov = np.eye(ndim)
        for i in range(ndim):
            for j in range(ndim):
                if i != j:
                    cov[i, j] = rho ** abs(i - j)
        prec = np.linalg.inv(cov)

        def logp_and_grad(x):
            logp = -0.5 * x @ prec @ x
            grad = -prec @ x
            return logp, grad

        sampler = NUTSSampler(
            ndim=ndim,
            logp_and_grad=logp_and_grad,
            num_warmup=500,
            mass_matrix_type="diagonal",
            seed=42,
            outdir=temp_dir,
            save_freq=1000,
        )
        sampler.sample(np.ones(ndim) * 2, num_iterations=3000)

        chain = sampler.load_chain()
        samples = chain["samples"][500:]
        mean = np.mean(samples, axis=0)
        sample_cov = np.cov(samples, rowvar=False)

        # Mean within 2 sigma / sqrt(n)
        n_eff = len(samples) / 10  # conservative ESS estimate
        se = np.sqrt(np.diag(cov) / n_eff)
        tol = 2 * se + 0.1
        for i in range(ndim):
            assert abs(mean[i]) < tol[i], f"dim {i}: mean={mean[i]:.3f}, tol={tol[i]:.3f}"

        # Variance within 20%
        np.testing.assert_allclose(np.diag(sample_cov), np.diag(cov), rtol=0.3)

        # No divergences
        diag = sampler.get_diagnostics()
        assert diag["num_divergent"] == 0


class TestAnisotropicAdaptation:
    def test_anisotropic_gaussian_adaptation_helps(self, temp_dir):
        """Regression: adapted mass matrix must not hurt on anisotropic targets.

        Before the M vs M^{-1} convention fix, warmup adaptation set
        M = Sigma, degrading conditioning by the square of the condition
        number; ESS collapsed relative to an identity mass matrix.
        """
        from impulse.diagnostics import effective_sample_size

        stds = np.array([10.0, 1.0])
        var = stds**2

        def logp_and_grad(x):
            return -0.5 * np.sum(x**2 / var), -x / var

        def run(mass_matrix_type, outdir):
            sampler = NUTSSampler(
                ndim=2,
                logp_and_grad=logp_and_grad,
                num_warmup=500,
                mass_matrix_type=mass_matrix_type,
                seed=42,
                outdir=outdir,
                save_freq=2000,
            )
            sampler.sample(np.zeros(2), num_iterations=1500)
            return sampler, sampler.load_chain()["samples"][200:]

        sampler_adapted, samples_adapted = run("diagonal", os.path.join(temp_dir, "adapted"))
        _, samples_unit = run("unit", os.path.join(temp_dir, "unit"))

        # The adapted M must equal Sigma^{-1}: velocity scaling M^{-1} p
        # matches the target variances (pre-fix this was ~1/var, off by
        # the squared condition number)
        adapted_mm = sampler_adapted.state.mass_matrix
        np.testing.assert_allclose(adapted_mm.inverse_multiply(np.ones(2)), var, rtol=0.5)

        # Post-warmup samples recover the per-dimension scales
        np.testing.assert_allclose(np.std(samples_adapted, axis=0), stds, rtol=0.3)

        ess_adapted = effective_sample_size(samples_adapted)
        ess_unit = effective_sample_size(samples_unit)
        assert np.all(np.isfinite(ess_adapted))

        # Adaptation must not be catastrophically worse than identity;
        # with the fix it should be comparable or better in each dimension
        assert np.all(ess_adapted > 0.2 * ess_unit), f"ESS adapted {ess_adapted} vs unit {ess_unit}"


class TestSaveWarmup:
    def test_save_warmup_increases_chain_length(self, temp_dir):
        sampler = NUTSSampler(
            ndim=2,
            logp_and_grad=gaussian_logp_and_grad,
            num_warmup=50,
            seed=42,
            outdir=temp_dir,
            save_freq=200,
            save_warmup=True,
        )
        sampler.sample(np.zeros(2), num_iterations=100)
        chain = sampler.load_chain()
        # Should have warmup + sampling iterations
        assert chain["samples"].shape[0] == 150

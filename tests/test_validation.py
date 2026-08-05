"""Tests for impulse.validation SBC utilities."""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import impulse
from impulse.validation import (
    compute_model_pit,
    compute_sbc_quantile,
    compute_sbc_rank,
    ecdf,
)

# ---------------------------------------------------------------------------
# compute_sbc_rank
# ---------------------------------------------------------------------------


class TestComputeSbcRank:
    def test_known_samples(self):
        """Rank should count samples strictly less than true value."""
        true = np.array([3.0, 5.0])
        posterior = np.array(
            [
                [1.0, 2.0],
                [2.0, 4.0],
                [4.0, 6.0],
                [5.0, 8.0],
            ]
        )
        ranks = compute_sbc_rank(true, posterior)
        # dim 0: 1.0 < 3, 2.0 < 3 -> rank 2
        # dim 1: 2.0 < 5, 4.0 < 5 -> rank 2
        np.testing.assert_array_equal(ranks, [2, 2])

    def test_edges_below(self):
        """True value below all samples -> rank 0."""
        true = np.array([0.0])
        posterior = np.array([[1.0], [2.0], [3.0]])
        ranks = compute_sbc_rank(true, posterior)
        np.testing.assert_array_equal(ranks, [0])

    def test_edges_above(self):
        """True value above all samples -> rank = nsamples."""
        true = np.array([10.0])
        posterior = np.array([[1.0], [2.0], [3.0]])
        ranks = compute_sbc_rank(true, posterior)
        np.testing.assert_array_equal(ranks, [3])

    def test_multidim(self):
        """Each dimension computed independently."""
        rng = np.random.default_rng(42)
        true = np.array([0.0, 0.0, 0.0])
        posterior = rng.standard_normal((100, 3))
        ranks = compute_sbc_rank(true, posterior)
        assert ranks.shape == (3,)
        assert np.all(ranks >= 0)
        assert np.all(ranks <= 100)


# ---------------------------------------------------------------------------
# compute_sbc_quantile
# ---------------------------------------------------------------------------


class TestComputeSbcQuantile:
    def test_normalized(self):
        """Quantile should be rank / nsamples, in [0, 1]."""
        true = np.array([3.0])
        posterior = np.array([[1.0], [2.0], [4.0], [5.0]])
        q = compute_sbc_quantile(true, posterior)
        np.testing.assert_allclose(q, [0.5])  # rank=2, nsamples=4

    def test_bounds(self):
        """Quantile values should lie in [0, 1]."""
        rng = np.random.default_rng(123)
        true = rng.standard_normal(5)
        posterior = rng.standard_normal((200, 5))
        q = compute_sbc_quantile(true, posterior)
        assert np.all(q >= 0)
        assert np.all(q <= 1)


# ---------------------------------------------------------------------------
# compute_model_pit
# ---------------------------------------------------------------------------


class TestComputeModelPit:
    def test_bounds(self):
        """PIT should lie in [P_below, P_below + P_at]."""
        probs = np.array([0.2, 0.5, 0.3])
        rng = np.random.default_rng(0)
        for _ in range(100):
            pit = compute_model_pit(1, probs, rng=rng)
            assert 0.2 <= pit <= 0.7  # P(<1) = 0.2, P(=1) = 0.5

    def test_first_model(self):
        """PIT for model 0: P_below = 0, so PIT in [0, P(0)]."""
        probs = np.array([0.4, 0.3, 0.3])
        rng = np.random.default_rng(42)
        for _ in range(100):
            pit = compute_model_pit(0, probs, rng=rng)
            assert 0.0 <= pit <= 0.4

    def test_last_model(self):
        """PIT for last model: P_below = 1 - P(last), so PIT in [P_below, 1]."""
        probs = np.array([0.3, 0.3, 0.4])
        rng = np.random.default_rng(7)
        for _ in range(100):
            pit = compute_model_pit(2, probs, rng=rng)
            assert 0.6 <= pit <= 1.0

    def test_uniform_distribution(self):
        """With many draws from true posterior, PIT should be Uniform(0,1)."""
        rng = np.random.default_rng(12345)
        n = 10000
        pits = np.zeros(n)
        probs = np.array([0.25, 0.5, 0.25])
        for i in range(n):
            # Draw true model from the same distribution as the posterior
            true_model = rng.choice(3, p=probs)
            pits[i] = compute_model_pit(true_model, probs, rng=rng)
        # KS test against Uniform(0,1)
        stat, pval = stats.kstest(pits, "uniform")
        assert pval > 0.01, f"KS test failed: stat={stat:.4f}, p={pval:.4f}"


# ---------------------------------------------------------------------------
# ecdf
# ---------------------------------------------------------------------------


class TestEcdf:
    def test_sorted(self):
        """ECDF x-values should be sorted."""
        x = np.array([5, 1, 3, 2, 4])
        sx, cdf = ecdf(x)
        np.testing.assert_array_equal(sx, [1, 2, 3, 4, 5])

    def test_cdf_values(self):
        """ECDF should step from 1/n to 1."""
        x = np.array([10, 20, 30, 40])
        sx, cdf = ecdf(x)
        np.testing.assert_allclose(cdf, [0.25, 0.5, 0.75, 1.0])

    def test_single_value(self):
        """Single element should give CDF = 1."""
        sx, cdf = ecdf(np.array([42.0]))
        np.testing.assert_allclose(cdf, [1.0])


# ---------------------------------------------------------------------------
# SBC with exact posterior (validates machinery itself)
# ---------------------------------------------------------------------------


class TestSbcExactPosterior:
    def test_conjugate_normal_uniform_quantiles(self):
        """
        Draw theta ~ prior, data ~ likelihood(theta), theta_post ~ exact posterior.
        SBC ranks should be uniform — validates the SBC computation itself.
        """
        rng = np.random.default_rng(42)
        n_sim = 500
        n_post = 200
        ndim = 2

        # Prior: N(0, sigma_prior^2 I)
        sigma_prior = 2.0
        # Likelihood: data ~ N(theta, sigma_noise^2 I) with n_obs observations
        sigma_noise = 1.0
        n_obs = 5

        # Posterior: N(mu_post, sigma_post^2 I)
        # sigma_post^2 = 1 / (1/sigma_prior^2 + n_obs/sigma_noise^2)
        # mu_post = sigma_post^2 * (n_obs * data_mean / sigma_noise^2)
        var_prior = sigma_prior**2
        var_noise = sigma_noise**2
        var_post = 1.0 / (1.0 / var_prior + n_obs / var_noise)
        sigma_post = np.sqrt(var_post)

        quantiles = np.zeros((n_sim, ndim))
        for i in range(n_sim):
            # Draw from prior
            theta_true = rng.normal(0, sigma_prior, size=ndim)
            # Generate data
            data = rng.normal(theta_true, sigma_noise, size=(n_obs, ndim))
            data_mean = data.mean(axis=0)
            # Exact posterior
            mu_post = var_post * (n_obs * data_mean / var_noise)
            # Draw posterior samples
            post_samples = rng.normal(mu_post, sigma_post, size=(n_post, ndim))
            quantiles[i] = compute_sbc_quantile(theta_true, post_samples)

        # KS test per dimension — should not reject uniformity
        for j in range(ndim):
            stat, pval = stats.kstest(quantiles[:, j], "uniform")
            assert pval > 0.01, (
                f"Dim {j}: KS test rejected uniformity " f"(stat={stat:.4f}, p={pval:.4f})"
            )

    def test_detects_bias(self):
        """
        Shift the 'posterior' by a constant so it's wrong — SBC should detect it.
        """
        rng = np.random.default_rng(99)
        n_sim = 300
        n_post = 200

        sigma_prior = 2.0
        sigma_noise = 1.0
        n_obs = 5
        var_prior = sigma_prior**2
        var_noise = sigma_noise**2
        var_post = 1.0 / (1.0 / var_prior + n_obs / var_noise)
        sigma_post = np.sqrt(var_post)

        bias = 2.0  # deliberately wrong
        quantiles = np.zeros(n_sim)
        for i in range(n_sim):
            theta_true = rng.normal(0, sigma_prior)
            data = rng.normal(theta_true, sigma_noise, size=n_obs)
            data_mean = data.mean()
            mu_post = var_post * (n_obs * data_mean / var_noise) + bias
            post_samples = rng.normal(mu_post, sigma_post, size=n_post)
            quantiles[i] = compute_sbc_quantile(np.array([theta_true]), post_samples[:, None])[0]

        stat, pval = stats.kstest(quantiles, "uniform")
        assert pval < 0.01, (
            f"KS test should reject biased posterior " f"(stat={stat:.4f}, p={pval:.4f})"
        )


# ---------------------------------------------------------------------------
# matplotlib is optional: lazy-import regression test
# ---------------------------------------------------------------------------


class TestLazyMatplotlibImport:
    def test_import_and_numerics_work_without_matplotlib(self):
        """
        With matplotlib blocked, 'import impulse' and the numeric SBC
        functions must work, and the plotting functions must raise an
        ImportError pointing at the [plots] extra.

        Runs in a subprocess so blocking matplotlib cannot interfere with
        other tests in this process.
        """
        code = """
import sys

class _BlockMatplotlib:
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"{name} is blocked for this test")
        return None

sys.meta_path.insert(0, _BlockMatplotlib())
assert "matplotlib" not in sys.modules

import numpy as np
import impulse
import impulse.validation as v

# Numeric SBC functions stay matplotlib-free.
ranks = v.compute_sbc_rank(np.array([0.5]), np.array([[0.1], [0.9]]))
assert ranks[0] == 1, ranks
q = v.compute_sbc_quantile(np.array([0.5]), np.array([[0.1], [0.9]]))
assert q[0] == 0.5, q

# Plotting functions raise a helpful ImportError.
for fn in (
    lambda: v.sbc_ecdf_plot(np.linspace(0.05, 0.95, 10)),
    lambda: v.coverage_plot(np.linspace(0.05, 0.95, 10)),
    lambda: v.rank_histogram(np.arange(10), 20),
):
    try:
        fn()
    except ImportError as e:
        assert "impulse-mcmc[plots]" in str(e), str(e)
    else:
        raise AssertionError("plot function should raise without matplotlib")

print(impulse.__version__)
"""
        repo_root = Path(impulse.__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=120,
        )
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert result.stdout.strip() == impulse.__version__


# ---------------------------------------------------------------------------
# Plotting helpers
#
# These three are part of the supported API and are rendered in the docs, but
# until now the ONLY test that named them asserted they raise without
# matplotlib -- their bodies never executed in any CI job. These exercise the
# real drawing path (Agg backend, no display required).
# ---------------------------------------------------------------------------

# Guarded import, NOT pytest.importorskip: importorskip at module scope raises
# Skipped for the WHOLE file, which would silently skip the numeric SBC tests
# above wherever matplotlib is absent (the floor-versions CI job, any bare
# install). Only the plotting class depends on it.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover - exercised only without the extra
    HAS_MPL = False


@pytest.mark.skipif(
    not HAS_MPL, reason="matplotlib not installed (pip install impulse-mcmc[plots])"
)
class TestPlottingHelpers:
    """Each helper must draw without raising and return usable axes."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")

    @staticmethod
    def _quantiles(rng, n=200, d=1):
        return rng.uniform(size=n) if d == 1 else rng.uniform(size=(n, d))

    def test_sbc_ecdf_plot_1d_returns_axes(self):
        rng = np.random.default_rng(0)
        ax = impulse.sbc_ecdf_plot(self._quantiles(rng))
        assert isinstance(ax, matplotlib.axes.Axes)
        # DKW band + diagonal + one step per parameter
        assert len(ax.lines) >= 2
        assert ax.get_xlim() == (0.0, 1.0)
        plt.close(ax.figure)

    def test_sbc_ecdf_plot_2d_labels_every_parameter(self):
        rng = np.random.default_rng(1)
        ax = impulse.sbc_ecdf_plot(self._quantiles(rng, d=3))
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "param 0" in labels and "param 2" in labels
        plt.close(ax.figure)

    def test_sbc_ecdf_plot_honours_param_names(self):
        rng = np.random.default_rng(2)
        ax = impulse.sbc_ecdf_plot(self._quantiles(rng, d=2), param_names=["alpha", "beta"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "alpha" in labels and "beta" in labels
        plt.close(ax.figure)

    def test_sbc_ecdf_plot_draws_on_supplied_axes(self):
        rng = np.random.default_rng(3)
        fig, ax = plt.subplots()
        out = impulse.sbc_ecdf_plot(self._quantiles(rng), ax=ax)
        assert out is ax
        plt.close(fig)

    def test_coverage_plot_returns_axes(self):
        rng = np.random.default_rng(4)
        ax = impulse.coverage_plot(self._quantiles(rng))
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_coverage_plot_honours_param_names(self):
        rng = np.random.default_rng(5)
        ax = impulse.coverage_plot(self._quantiles(rng, d=2), param_names=["alpha", "beta"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "alpha" in labels and "beta" in labels
        plt.close(ax.figure)

    def test_coverage_plot_recovers_nominal_coverage_for_uniform_ranks(self):
        # Uniform quantiles => actual coverage tracks the nominal diagonal.
        rng = np.random.default_rng(6)
        q = rng.uniform(size=20_000)
        levels = np.array([0.2, 0.5, 0.8])
        ax = impulse.coverage_plot(q, nominal_levels=levels)
        drawn = ax.lines[1].get_ydata()  # lines[0] is the diagonal reference
        assert np.allclose(drawn, levels, atol=0.02), drawn
        plt.close(ax.figure)

    def test_rank_histogram_returns_one_axes_per_parameter(self):
        rng = np.random.default_rng(7)
        axes = impulse.rank_histogram(rng.integers(0, 21, size=(200, 3)), 20)
        assert len(axes) == 3
        plt.close(axes[0].figure)

    def test_rank_histogram_honours_param_names(self):
        rng = np.random.default_rng(8)
        axes = impulse.rank_histogram(
            rng.integers(0, 21, size=(50, 2)), 20, param_names=["alpha", "beta"]
        )
        assert [a.get_title() for a in axes] == ["alpha", "beta"]
        plt.close(axes[0].figure)

    def test_rank_histogram_1d_input_is_promoted(self):
        rng = np.random.default_rng(9)
        axes = impulse.rank_histogram(rng.integers(0, 21, size=100), 20)
        assert len(axes) == 1
        plt.close(axes[0].figure)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: impulse.sbc_ecdf_plot(np.array([])), id="ecdf-empty"),
            pytest.param(lambda: impulse.coverage_plot(np.array([])), id="coverage-empty"),
            pytest.param(
                lambda: impulse.rank_histogram(np.array([], dtype=int), 20), id="rank-empty"
            ),
            pytest.param(lambda: impulse.sbc_ecdf_plot(np.array([0.5])), id="ecdf-single"),
            pytest.param(lambda: impulse.coverage_plot(np.array([0.5])), id="coverage-single"),
            pytest.param(lambda: impulse.rank_histogram(np.array([0, 1, 0]), 1), id="rank-one-bin"),
        ],
    )
    def test_degenerate_input_neither_raises_nor_warns(self, call):
        """Empty/singleton input must not raise, and must not emit numpy
        divide-by-zero or mean-of-empty-slice RuntimeWarnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            call()
        plt.close("all")

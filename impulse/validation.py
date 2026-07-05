"""
Simulation-Based Calibration (SBC) validation utilities.

Provides tools for validating MCMC samplers via SBC:
- Rank and quantile statistics for continuous parameters
- Randomized PIT for discrete model indices
- ECDF, coverage, and rank histogram plots
- Loop runners for continuous and model-selection SBC
"""

import shutil
import tempfile
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _require_matplotlib():
    """
    Import and return matplotlib.pyplot for the plotting helpers.

    matplotlib is an optional dependency: the numeric SBC functions in this
    module work without it, and only the plotting functions need it.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the plotting functions in "
            "impulse.validation but is not installed. Install it with "
            "'pip install impulse-mcmc[plots]' (or 'pip install matplotlib')."
        ) from exc
    return plt


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_sbc_rank(true_value: ArrayLike, posterior_samples: ArrayLike) -> np.ndarray:
    """
    Count posterior samples less than the true value, per dimension.

    Parameters
    ----------
    true_value : array_like, shape (ndim,)
        True parameter values.
    posterior_samples : array_like, shape (nsamples, ndim)
        Posterior samples.

    Returns
    -------
    np.ndarray, shape (ndim,)
        Rank of the true value among posterior samples per dimension.
    """
    true_value = np.asarray(true_value)
    posterior_samples = np.asarray(posterior_samples)
    return np.sum(posterior_samples < true_value, axis=0)


def compute_sbc_quantile(true_value: ArrayLike, posterior_samples: ArrayLike) -> np.ndarray:
    """
    Quantile rank of the true value among posterior samples.

    Parameters
    ----------
    true_value : array_like, shape (ndim,)
        True parameter values.
    posterior_samples : array_like, shape (nsamples, ndim)
        Posterior samples.

    Returns
    -------
    np.ndarray, shape (ndim,)
        Quantile ranks in [0, 1].
    """
    posterior_samples = np.asarray(posterior_samples)
    ranks = compute_sbc_rank(true_value, posterior_samples)
    return ranks / posterior_samples.shape[0]


def compute_model_pit(
    true_model_index: int,
    posterior_probs: ArrayLike,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Randomized probability integral transform for a discrete model index.

    PIT = P(k < k_true) + U * P(k = k_true), where U ~ Uniform(0, 1).
    Under correct calibration, PIT ~ Uniform(0, 1).

    Parameters
    ----------
    true_model_index : int
        True model index.
    posterior_probs : array_like, shape (num_models,)
        Posterior probabilities for each model.
    rng : np.random.Generator, optional
        Random number generator. Created if not provided.

    Returns
    -------
    float
        PIT value in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    posterior_probs = np.asarray(posterior_probs, dtype=float)
    p_below = np.sum(posterior_probs[:true_model_index])
    p_at = posterior_probs[true_model_index]
    u = rng.uniform()
    return p_below + u * p_at


def ecdf(x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical cumulative distribution function.

    Parameters
    ----------
    x : array_like
        Sample values.

    Returns
    -------
    sorted_x : np.ndarray
        Sorted sample values.
    cdf_values : np.ndarray
        ECDF values at each sorted sample point.
    """
    x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(sorted_x)
    cdf_values = np.arange(1, n + 1) / n
    return sorted_x, cdf_values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def sbc_ecdf_plot(
    quantiles: ArrayLike,
    param_names: Optional[Sequence[str]] = None,
    ax: Optional[Any] = None,
    alpha: float = 0.05,
) -> Any:
    """
    ECDF of SBC quantile ranks vs the Uniform(0,1) diagonal.

    Parameters
    ----------
    quantiles : array_like, shape (n_simulations, ndim)
        SBC quantile ranks.
    param_names : list of str, optional
        Parameter names for the legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.
    alpha : float
        Significance level for DKW confidence band.

    Returns
    -------
    matplotlib.axes.Axes
    """
    quantiles = np.asarray(quantiles)
    if quantiles.ndim == 1:
        quantiles = quantiles[:, None]
    n_sim, ndim = quantiles.shape

    if ax is None:
        plt = _require_matplotlib()
        _, ax = plt.subplots()

    # DKW confidence band
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * n_sim))
    t = np.linspace(0, 1, 200)
    ax.fill_between(
        t,
        np.clip(t - epsilon, 0, 1),
        np.clip(t + epsilon, 0, 1),
        color="gray",
        alpha=0.2,
        label=f"{1 - alpha:.0%} DKW band",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)

    for j in range(ndim):
        sx, cdf = ecdf(quantiles[:, j])
        name = param_names[j] if param_names is not None else f"param {j}"
        ax.step(sx, cdf, where="post", label=name)

    ax.set_xlabel("Quantile rank")
    ax.set_ylabel("ECDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    ax.set_title("SBC ECDF")
    return ax


def coverage_plot(
    quantiles: ArrayLike,
    nominal_levels: Optional[ArrayLike] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Actual vs nominal coverage of credible intervals.

    For nominal level p, actual coverage = fraction of simulations where
    |quantile - 0.5| < p/2.

    Parameters
    ----------
    quantiles : array_like, shape (n_simulations, ndim)
        SBC quantile ranks.
    nominal_levels : array_like, optional
        Nominal coverage levels to evaluate. Defaults to np.arange(0.1, 1.0, 0.1).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.

    Returns
    -------
    matplotlib.axes.Axes
    """
    quantiles = np.asarray(quantiles)
    if quantiles.ndim == 1:
        quantiles = quantiles[:, None]
    n_sim, ndim = quantiles.shape

    if nominal_levels is None:
        nominal_levels = np.arange(0.1, 1.0, 0.1)
    nominal_levels = np.asarray(nominal_levels)

    if ax is None:
        plt = _require_matplotlib()
        _, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)

    for j in range(ndim):
        actual = np.array([np.mean(np.abs(quantiles[:, j] - 0.5) < p / 2) for p in nominal_levels])
        ax.plot(nominal_levels, actual, "o-", markersize=4, label=f"param {j}")

    # Binomial 95% CI for the diagonal
    for p in nominal_levels:
        se = 1.96 * np.sqrt(p * (1 - p) / n_sim)
        ax.plot([p, p], [p - se, p + se], color="gray", lw=0.8, alpha=0.5)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Actual coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    ax.set_title("Coverage")
    return ax


def rank_histogram(
    ranks: ArrayLike,
    n_posterior_samples: int,
    param_names: Optional[Sequence[str]] = None,
    axes: Optional[Any] = None,
) -> Any:
    """
    Histogram of SBC ranks (should be approximately flat).

    Parameters
    ----------
    ranks : array_like, shape (n_simulations, ndim)
        SBC ranks (integer counts of samples < true value).
    n_posterior_samples : int
        Number of posterior samples used per simulation.
    param_names : list of str, optional
        Parameter names for subplot titles.
    axes : array of matplotlib.axes.Axes, optional
        Axes to plot on. Created if not provided.

    Returns
    -------
    array of matplotlib.axes.Axes
    """
    ranks = np.asarray(ranks)
    if ranks.ndim == 1:
        ranks = ranks[:, None]
    n_sim, ndim = ranks.shape

    if axes is None:
        plt = _require_matplotlib()
        _, axes = plt.subplots(1, ndim, figsize=(4 * ndim, 3), squeeze=False)
        axes = axes.ravel()

    n_bins = min(20, n_posterior_samples + 1)
    expected = n_sim / n_bins

    for j in range(ndim):
        ax = axes[j]
        ax.hist(
            ranks[:, j], bins=n_bins, range=(0, n_posterior_samples), edgecolor="black", alpha=0.7
        )
        ax.axhline(expected, color="red", ls="--", lw=1, label="expected")
        name = param_names[j] if param_names is not None else f"param {j}"
        ax.set_title(name)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
        ax.legend(fontsize="small")

    return axes


# ---------------------------------------------------------------------------
# SBC loop runners
# ---------------------------------------------------------------------------


def run_sbc_continuous(
    sampler_factory: Callable,
    prior_draw: Callable,
    data_generator: Callable,
    n_simulations: int = 200,
    burn: int = 500,
    thin: int = 1,
    seed: Optional[int] = None,
) -> dict:
    """
    Run SBC for continuous-parameter samplers.

    Parameters
    ----------
    sampler_factory : callable
        ``(data, rng, outdir) -> sampler`` that creates, configures, runs
        ``.sample()``, and returns the sampler.
    prior_draw : callable
        ``(rng) -> theta_true``, shape ``(ndim,)``.
    data_generator : callable
        ``(theta_true, rng) -> data``.
    n_simulations : int
        Number of SBC repetitions.
    burn : int
        Burn-in samples to discard from each chain.
    thin : int
        Thinning factor for posterior samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``ranks`` : shape ``(n_simulations, ndim)``
        ``quantiles`` : shape ``(n_simulations, ndim)``
        ``true_values`` : shape ``(n_simulations, ndim)``
        ``n_posterior`` : int, number of posterior samples per simulation
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(range(n_simulations), desc="SBC continuous")
    except ImportError:
        iterator = range(n_simulations)

    rng = np.random.default_rng(seed)

    all_ranks: list[np.ndarray] = []
    all_quantiles: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    n_posterior: Optional[int] = None

    for i in iterator:
        theta_true = prior_draw(rng)
        data = data_generator(theta_true, rng)

        tmpdir = tempfile.mkdtemp()
        try:
            sim_rng = np.random.default_rng(rng.integers(0, 2**32))
            sampler = sampler_factory(data, sim_rng, tmpdir)
            chain_dict = sampler.load_chain()

            # Extract cold chain samples
            samples = chain_dict["samples"]
            if samples.ndim == 3:
                # PT sampler: (ntemps, nsamples, ndim) -> cold chain
                samples = samples[0]
            # samples is now (nsamples, ndim)
            samples = samples[burn::thin]

            ndim = theta_true.shape[0]
            # Only use the first ndim columns (exclude model index etc.)
            samples = samples[:, :ndim]

            if n_posterior is None:
                n_posterior = samples.shape[0]

            ranks = compute_sbc_rank(theta_true, samples)
            quantiles = compute_sbc_quantile(theta_true, samples)

            all_ranks.append(ranks)
            all_quantiles.append(quantiles)
            all_true.append(theta_true)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "ranks": np.array(all_ranks),
        "quantiles": np.array(all_quantiles),
        "true_values": np.array(all_true),
        "n_posterior": n_posterior,
    }


def run_sbc_model_selection(
    sampler_factory: Callable,
    model_prior_draw: Callable,
    param_prior_draw: Callable,
    data_generator: Callable,
    num_models: int,
    n_simulations: int = 200,
    burn: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """
    Run SBC for model-selection (product-space) samplers.

    Parameters
    ----------
    sampler_factory : callable
        ``(true_nmodel, true_params, data, rng, outdir) -> (sampler, space)``
        Creates, configures, runs ``.sample()``, and returns the sampler
        and ``BirthDeathProductSpace``.
    model_prior_draw : callable
        ``(rng) -> int``, draws a model index from the prior.
    param_prior_draw : callable
        ``(nmodel, rng) -> theta_true`` for active sources.
    data_generator : callable
        ``(true_nmodel, true_params, rng) -> data``.
    num_models : int
        Total number of models.
    n_simulations : int
        Number of SBC repetitions.
    burn : int
        Burn-in samples to discard.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``pit_values`` : shape ``(n_simulations,)``
        ``true_models`` : shape ``(n_simulations,)``
        ``posterior_probs`` : shape ``(n_simulations, num_models)``
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(range(n_simulations), desc="SBC model selection")
    except ImportError:
        iterator = range(n_simulations)

    rng = np.random.default_rng(seed)

    pit_values = np.zeros(n_simulations)
    true_models = np.zeros(n_simulations, dtype=int)
    posterior_probs = np.zeros((n_simulations, num_models))

    for i in iterator:
        true_nmodel = model_prior_draw(rng)
        true_params = param_prior_draw(true_nmodel, rng)
        data = data_generator(true_nmodel, true_params, rng)

        tmpdir = tempfile.mkdtemp()
        try:
            sim_rng = np.random.default_rng(rng.integers(0, 2**32))
            sampler, space = sampler_factory(true_nmodel, true_params, data, sim_rng, tmpdir)
            chain_dict = sampler.load_chain()

            samples = chain_dict["samples"]
            if samples.ndim == 3:
                samples = samples[0]  # cold chain

            probs = space.model_posterior_probs(samples, burn=burn)
            pit = compute_model_pit(true_nmodel, probs, rng=rng)

            pit_values[i] = pit
            true_models[i] = true_nmodel
            posterior_probs[i] = probs
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "pit_values": pit_values,
        "true_models": true_models,
        "posterior_probs": posterior_probs,
    }

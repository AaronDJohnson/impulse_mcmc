"""Convergence and model-selection diagnostics for MCMC chains.

Provides FFT-based autocorrelation-length estimates via initial-positive-
sequence and initial-monotone-sequence estimators
(:func:`autocorr_length_ips_ims`), per-parameter effective sample sizes
(:func:`effective_sample_size`), and the Gelman-Rubin split-R-hat statistic
(:func:`grubin`). For product-space model-selection runs,
:func:`model_visitation_stats` summarizes a chain's model-index column (posterior model
probabilities, visit counts, transition matrix, dwell times) and
:func:`bayes_factor_from_chain` estimates Bayes factors from posterior model
frequencies.
"""

import numpy as np
from numpy.fft import irfft, rfft
from scipy.stats import norm


def _next_fast_len(n: int) -> int:
    """
    Compute optimal FFT padding length for efficient autocorrelation computation.

    Parameters
    ----------
    n : int
        Input sequence length.

    Returns
    -------
    int
        Power-of-two length >= 2*n for efficient FFT computation.

    Examples
    --------
    >>> _next_fast_len(1000)
    2048
    >>> _next_fast_len(2000)
    4096
    """
    m = 1
    while m < 2 * n:
        m <<= 1
    return m


def _acf_fft(x: np.ndarray) -> np.ndarray:
    """
    Normalized autocorrelation ρ(k) for k = 0..T-1 using FFT (Wiener–Khinchin).
    Returns a length-T array with ρ(0)=1. NaNs if var=0.
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    x = x - x.mean()
    var = np.dot(x, x) / T
    if var == 0.0 or not np.isfinite(var):
        return np.full(T, np.nan)

    nfft = _next_fast_len(T)
    fx = rfft(x, n=nfft)
    S = fx * np.conjugate(fx)
    acov = irfft(S, n=nfft)[:T]
    # biased normalization (div by T); then normalize by acov[0] to get ρ
    acov = acov / T
    rho = acov / acov[0]
    return np.real(rho)


def _pair_sums_gamma(rho: np.ndarray) -> np.ndarray:
    """
    γ_m = ρ(2m-1) + ρ(2m), m=1,2,...
    Using 1-based lag indexing for ρ; here rho[0] is lag 0.
    """
    # lags start at 1, so we take rho[1], rho[2], rho[3], rho[4], ...
    tail = rho[1:]
    # Truncate to even length and reshape into pairs
    L = (tail.shape[0] // 2) * 2
    if L == 0:
        return np.array([], dtype=float)
    pairs = tail[:L].reshape(-1, 2)
    gamma = pairs.sum(axis=1)
    return gamma


def _ips_tau_from_gamma(gamma: np.ndarray) -> float:
    """
    IPS estimator:
    τ = 1 + 2 * sum_{m=1}^{M*} γ_m, where M* is last index s.t. all γ_1..γ_{M*} > 0.
    """
    if gamma.size == 0:
        return 1.0
    # Find last index with all previous strictly positive
    pos_mask = gamma > 0.0
    if not np.any(pos_mask):
        return 1.0
    # longest prefix of strictly-positive values
    Mstar = np.argmax(~pos_mask)  # first False index; 0 if gamma[0]<=0
    if not pos_mask.all():
        gamma_use = gamma[:Mstar] if Mstar > 0 else np.array([], dtype=float)
    else:
        gamma_use = gamma
    return 1.0 + 2.0 * np.sum(gamma_use)


def _pava_monotone_nonincreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Pooled Adjacent Violators Algorithm (PAVA) to enforce a nonincreasing sequence.
    Returns the greatest nonincreasing sequence <= the isotonic fit target.
    Here we fit on indices with constraint y_1 >= y_2 >= ... (nonincreasing).
    """
    y = y.astype(float)
    n = y.size
    if n == 0:
        return y
    if w is None:
        w = np.ones(n, dtype=float)

    # Transform to nondecreasing by flipping sign, run standard isotonic (nondecreasing), then flip back
    yy = -y
    # standard PAVA for nondecreasing constraint
    v = yy.copy()
    ww = w.copy()
    k = 0
    # store block averages and weights
    avg = []
    wsum = []

    for i in range(n):
        avg.append(v[i])
        wsum.append(ww[i])
        k += 1
        # merge while violating nondecreasing: last avg < prev avg
        while k >= 2 and avg[k - 2] > avg[k - 1]:
            # pool the last two blocks
            new_w = wsum[k - 2] + wsum[k - 1]
            new_avg = (wsum[k - 2] * avg[k - 2] + wsum[k - 1] * avg[k - 1]) / new_w
            avg[k - 2] = new_avg
            wsum[k - 2] = new_w
            # pop last block
            avg.pop()
            wsum.pop()
            k -= 1

    # expand block-averaged values
    out = np.empty(n, dtype=float)
    idx = 0
    for a, w_ in zip(avg, wsum):
        m = int(round(w_)) if np.allclose(w_, round(w_)) else int(w_)
        out[idx : idx + m] = a
        idx += m

    # Flip sign back to nonincreasing sequence
    return -out


def _ims_tau_from_gamma(gamma: np.ndarray) -> float:
    """
    IMS estimator:
      1) Enforce monotone nonincreasing on γ_m via PAVA.
      2) Truncate at first nonpositive adjusted γ_m.
      3) τ = 1 + 2 * sum of retained adjusted γ_m.
    """
    if gamma.size == 0:
        return 1.0
    gamma_mon = _pava_monotone_nonincreasing(gamma)
    # Truncate at first nonpositive
    pos = np.where(gamma_mon > 0.0)[0]
    if pos.size == 0:
        return 1.0
    Mprime = pos[-1] + 1  # keep indices [0 .. last positive]
    gamma_use = gamma_mon[:Mprime]
    return 1.0 + 2.0 * np.sum(gamma_use)


def autocorr_length_ips_ims(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IPS and IMS integrated autocorrelation times for each dimension.

    Parameters
    ----------
    chain : array_like, shape (T, D)
        MCMC samples over time (axis 0) for D parameters (axis 1).

    Returns
    -------
    tau_ips : ndarray, shape (D,)
    tau_ims : ndarray, shape (D,)
    """
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    T, D = x.shape
    tau_ips = np.empty(D, dtype=float)
    tau_ims = np.empty(D, dtype=float)

    for j in range(D):
        rho = _acf_fft(x[:, j])
        if not np.isfinite(rho).all():
            tau_ips[j] = np.nan
            tau_ims[j] = np.nan
            continue
        gamma = _pair_sums_gamma(rho)
        tau_ips[j] = _ips_tau_from_gamma(gamma)
        tau_ims[j] = _ims_tau_from_gamma(gamma)

    return tau_ips, tau_ims


def effective_sample_size(chain: np.ndarray) -> np.ndarray:
    """
    Compute effective sample size (ESS) for each dimension of the chain.

    Parameters
    ----------
    chain : array_like, shape (T, D)
        MCMC samples over time (axis 0) for D parameters (axis 1).

    Returns
    -------
    ess : ndarray, shape (D,)
        Effective sample size for each parameter.
    """
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    T, D = x.shape
    ess = np.empty(D, dtype=float)

    tau_ips, tau_ims = autocorr_length_ips_ims(x)
    for j in range(D):
        tau = min(tau_ips[j], tau_ims[j])
        if not np.isfinite(tau) or tau <= 0.0:
            ess[j] = np.nan
        else:
            ess[j] = T / tau

    return ess


def grubin(chains: np.ndarray, M=2, threshold=1.01, burn=None):
    """
    Modern Gelman-Rubin R-hat (rank-normalized, folded, split) to assess convergence.

    Implements the Stan/ArviZ recommendation:
      1) Split chains (or a single chain) into M segments of equal length.
      2) Rank-normalize samples with Blom offset ((r - 3/8)/(n + 1/4)), then z = Phi^{-1}(.)
      3) Compute split-Rhat on z.
      4) Compute split-Rhat on folded values |z - median(z)|.
      5) Return max(Rhat_z, Rhat_folded) per parameter.

    Parameters
    ----------
    chains : np.ndarray or list of np.ndarray
        MCMC draws as a 2-D array of shape ``(T, D)`` holding **parameters
        only**, matching :func:`effective_sample_size`. If a list of two
        arrays, they are concatenated along axis 0 before processing.

        This is exactly what :meth:`~impulse.PTSampler.load_chain` returns in
        ``chain["samples"][k]``. If you are reading a raw ``chain_*.txt`` file
        instead, slice off its four trailing bookkeeping columns first::

            data = np.loadtxt("chains/chain_0.txt")
            rhat, idx = grubin(data[:, :ndim])
    M : int, default 2
        Number of segments to split the chain into.
    threshold : float, default 1.01
        Flag parameters with R-hat above this value.
    burn : int, optional
        Number of initial samples to discard. Defaults to 10% of the chain.

    Returns
    -------
    Rhat : np.ndarray, shape (D,)
        Modern R-hat per parameter.
    idx : np.ndarray
        Indices of parameters where R-hat > threshold.

    Raises
    ------
    ValueError
        If ``chains`` is not 2-D.

    Notes
    -----
    Versions before 2.0.0 silently dropped the last two columns of the input.
    That convention matched no format this library produces -- chain files
    carry four trailing columns (lnlike, lnprob, accepted, temperature) and
    ``load_chain`` returns none -- so it either discarded real parameters or
    promoted lnlike/lnprob to parameters, without complaining either way.
    Pass parameters only.
    """
    # ---- ingest & (optionally) concatenate two chains ----
    if isinstance(chains, list) and len(chains) == 2:
        data = np.asarray(np.concatenate([chains[0], chains[1]]), dtype=float)
    else:
        data = np.asarray(chains, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"Expected chains with shape (T, D) holding parameters only, got "
            f"shape {data.shape}. If this is a raw chain file, slice off its "
            "four trailing columns (lnlike, lnprob, accepted, temperature) first."
        )
    if burn is None:  # if no burn is set, burn 10% of the chain
        burn = int(0.1 * data.shape[0])
    X = data[burn:]
    T = X.shape[0]

    # ---- split into M contiguous subchains of equal length ----
    try:
        chunks = np.split(X, M, axis=0)
    except ValueError:
        # make T divisible by M by trimming additional "burn" from the front
        P = int(np.floor(T / M))
        extra = T - M * P
        burn += extra
        X = data[burn:]
        chunks = np.split(X, M, axis=0)

    # data_s: shape (M, N, D) with N = draws per split-chain, D = #params
    data_s = np.asarray(chunks, dtype=float)
    M_s, N, D = data_s.shape

    # ---- helper: split-Rhat on an (M, N, D) array ----
    def split_rhat(arr):
        # arr expected shape (M, N, D)
        # between-chain means per split-chain
        theta_bar_m = np.mean(arr, axis=1)  # (M, D)
        theta_bar = np.mean(theta_bar_m, axis=0)  # (D,)

        # Between-chain variance B (per parameter)
        B = (N / (M_s - 1)) * np.sum((theta_bar_m - theta_bar) ** 2, axis=0)  # (D,)

        # Within-chain variance W (per parameter)
        s2_m = np.sum((arr - theta_bar_m[:, None, :]) ** 2, axis=1) / (N - 1)  # (M, D)
        W = np.mean(s2_m, axis=0)  # (D,)

        # Marginal posterior variance estimator
        var_hat = ((N - 1) / N) * W + (1 / N) * B

        # R-hat
        Rhat = np.sqrt(np.maximum(var_hat / W, 0.0))
        # protect against tiny numerical negatives
        Rhat[~np.isfinite(Rhat)] = np.nan
        return Rhat

    # ---- rank-normalization (Blom) + inverse normal transform ----
    # vectorize across all split-chains and draws for each parameter
    Z = np.empty_like(data_s)  # transformed z-scores
    for j in range(D):
        # flatten MxN for parameter j
        x = data_s[:, :, j].reshape(-1)
        n = x.size

        # ranks in 1..n; positional (ties get deterministic but arbitrary ordering via mergesort)
        ranks = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) + 1.0
        # Blom offset maps strictly inside (0,1)
        u = (ranks - 0.375) / (n + 0.25)
        # inverse normal
        z = norm.ppf(u)
        Z[:, :, j] = z.reshape(M_s, N)

    # ---- R-hat on rank-normalized draws ----
    rhat_z = split_rhat(Z)

    # ---- Folded R-hat: heavy-tail check on |z - median(z)| ----
    Z_med = np.median(Z.reshape(-1, D), axis=0)  # (D,)
    Z_folded = np.abs(Z - Z_med[None, None, :])

    # Rank-normalize the folded values again before R-hat (as in ArviZ/Stan)
    Zf = np.empty_like(Z_folded)
    for j in range(D):
        x = Z_folded[:, :, j].reshape(-1)
        n = x.size
        ranks = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) + 1.0
        u = (ranks - 0.375) / (n + 0.25)
        z = norm.ppf(u)
        Zf[:, :, j] = z.reshape(M_s, N)

    rhat_folded = split_rhat(Zf)

    # ---- Modern R-hat is the maximum of the two ----
    Rhat = np.maximum(rhat_z, rhat_folded)

    idx = np.where(Rhat > threshold)[0]
    return Rhat, idx


# ---------------------------------------------------------------------------
# Model-selection diagnostics
# ---------------------------------------------------------------------------


def model_visitation_stats(chain: np.ndarray, num_models: int, burn: int = 0):
    """
    Compute model visitation statistics from a product-space model-selection chain.

    Parameters
    ----------
    chain : np.ndarray, shape (N, ndim)
        Cold-chain samples with model index in the last column.
    num_models : int
        Total number of models (model index ranges from 0 to ``num_models - 1``).
    burn : int
        Number of initial samples to discard.

    Returns
    -------
    dict
        - ``posterior_probs`` : np.ndarray, shape (num_models,)
            Posterior probability of each model.
        - ``visit_counts`` : np.ndarray, shape (num_models,)
            Number of samples spent in each model.
        - ``transition_matrix`` : np.ndarray, shape (num_models, num_models)
            Row-normalised transition matrix ``T[i,j]`` = fraction of times
            the chain moved from model ``i`` to model ``j``.
        - ``mean_dwell_times`` : np.ndarray, shape (num_models,)
            Average consecutive run length in each model.
    """
    nmodel_samples = np.rint(chain[burn:, -1]).astype(int)
    n = len(nmodel_samples)

    visit_counts = np.bincount(nmodel_samples, minlength=num_models).astype(float)
    posterior_probs = visit_counts / visit_counts.sum()

    # transition matrix
    transition_counts = np.zeros((num_models, num_models))
    for t in range(n - 1):
        transition_counts[nmodel_samples[t], nmodel_samples[t + 1]] += 1
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for unvisited models
    transition_matrix = transition_counts / row_sums

    # mean dwell times (average run length)
    mean_dwell_times = np.zeros(num_models)
    if n > 0:
        current = nmodel_samples[0]
        run_len = 1
        run_lengths: dict[int, list[int]] = {k: [] for k in range(num_models)}
        for t in range(1, n):
            if nmodel_samples[t] == current:
                run_len += 1
            else:
                run_lengths[current].append(run_len)
                current = nmodel_samples[t]
                run_len = 1
        run_lengths[current].append(run_len)  # last run
        for k in range(num_models):
            if run_lengths[k]:
                mean_dwell_times[k] = np.mean(run_lengths[k])

    return {
        "posterior_probs": posterior_probs,
        "visit_counts": visit_counts,
        "transition_matrix": transition_matrix,
        "mean_dwell_times": mean_dwell_times,
    }


def bayes_factor_from_chain(chain: np.ndarray, model_i: int, model_j: int, burn: int = 0) -> float:
    """
    Estimate the Bayes factor B_{ij} from posterior model frequencies.

    Parameters
    ----------
    chain : np.ndarray, shape (N, ndim)
        Cold-chain samples with model index in the last column.
    model_i : int
        Numerator model index.
    model_j : int
        Denominator model index.
    burn : int
        Number of initial samples to discard.

    Returns
    -------
    float
        Estimated Bayes factor ``P(model_i | data) / P(model_j | data)``.
        Returns ``np.inf`` if ``model_j`` was never visited, or ``0.0``
        if ``model_i`` was never visited.
    """
    nmodel_samples = np.rint(chain[burn:, -1]).astype(int)
    count_i = np.sum(nmodel_samples == model_i)
    count_j = np.sum(nmodel_samples == model_j)
    if count_j == 0:
        return np.inf if count_i > 0 else np.nan
    if count_i == 0:
        return 0.0
    return count_i / count_j

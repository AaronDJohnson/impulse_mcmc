"""Convergence diagnostics for MCMC chains."""

import numpy as np
from numpy.fft import irfft, rfft
from scipy.stats import norm


def _next_fast_len(n: int) -> int:
    m = 1
    while m < 2 * n:
        m <<= 1
    return m


def _acf_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    x = x - x.mean()
    var = np.dot(x, x) / T
    if var == 0.0 or not np.isfinite(var):
        return np.full(T, np.nan)
    nfft = _next_fast_len(T)
    fx = rfft(x, n=nfft)
    acov = irfft(fx * np.conjugate(fx), n=nfft)[:T] / T
    return np.real(acov / acov[0])


def _pair_sums_gamma(rho: np.ndarray) -> np.ndarray:
    tail = rho[1:]
    L = (tail.shape[0] // 2) * 2
    if L == 0:
        return np.array([], dtype=float)
    return tail[:L].reshape(-1, 2).sum(axis=1)


def _ips_tau_from_gamma(gamma: np.ndarray) -> float:
    if gamma.size == 0:
        return 1.0
    pos_mask = gamma > 0.0
    if not np.any(pos_mask):
        return 1.0
    m_star = np.argmax(~pos_mask)
    gamma_use = (
        gamma if pos_mask.all() else (gamma[:m_star] if m_star > 0 else np.array([], dtype=float))
    )
    return 1.0 + 2.0 * float(np.sum(gamma_use))


def _pava_monotone_nonincreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = y.astype(float)
    n = y.size
    if n == 0:
        return y
    if w is None:
        w = np.ones(n, dtype=float)

    yy = -y
    avg: list[float] = []
    wsum: list[float] = []
    k = 0
    for i in range(n):
        avg.append(float(yy[i]))
        wsum.append(float(w[i]))
        k += 1
        while k >= 2 and avg[k - 2] > avg[k - 1]:
            new_w = wsum[k - 2] + wsum[k - 1]
            avg[k - 2] = (wsum[k - 2] * avg[k - 2] + wsum[k - 1] * avg[k - 1]) / new_w
            wsum[k - 2] = new_w
            avg.pop()
            wsum.pop()
            k -= 1

    out = np.empty(n, dtype=float)
    idx = 0
    for a, w_ in zip(avg, wsum):
        rw = round(w_)
        m = int(rw) if abs(w_ - rw) <= 1e-8 else int(w_)
        out[idx : idx + m] = a
        idx += m
    return -out


def _ims_tau_from_gamma(gamma: np.ndarray) -> float:
    if gamma.size == 0:
        return 1.0
    gamma_mon = _pava_monotone_nonincreasing(gamma)
    pos = np.where(gamma_mon > 0.0)[0]
    if pos.size == 0:
        return 1.0
    return 1.0 + 2.0 * float(np.sum(gamma_mon[: pos[-1] + 1]))


def autocorr_length_ips_ims(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute IPS and IMS integrated autocorrelation times for each dimension of ``chain`` (T, D)."""
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    _, D = x.shape
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
    """Compute effective sample size (ESS) for each dimension of ``chain`` (T, D)."""
    x = np.asarray(chain, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected chain with shape (T, D).")
    T, D = x.shape
    ess = np.empty(D, dtype=float)
    tau_ips, tau_ims = autocorr_length_ips_ims(x)
    for j in range(D):
        tau = min(tau_ips[j], tau_ims[j])
        ess[j] = np.nan if (not np.isfinite(tau) or tau <= 0.0) else T / tau
    return ess


def _rank_normalize(data_s: np.ndarray, q_table: np.ndarray | None = None) -> np.ndarray:
    M_s, N, D = data_s.shape
    n = M_s * N
    Z = np.empty_like(data_s)
    if n == 0:
        return Z
    if q_table is None:
        q_table = norm.ppf((np.arange(1, n + 1, dtype=float) - 0.375) / (n + 0.25))
    z_col = np.empty(n, dtype=float)
    for j in range(D):
        x = data_s[:, :, j].reshape(-1)
        order = np.argsort(x, kind="mergesort")
        z_col[order] = q_table
        Z[:, :, j] = z_col.reshape(M_s, N)
    return Z


def grubin(chains: np.ndarray, M: int = 2, threshold: float = 1.01, burn: int | None = None):
    """Rank-normalized, folded, split Gelman-Rubin R-hat diagnostic."""
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
    if burn is None:
        burn = int(0.1 * data.shape[0])
    X = data[burn:]
    T = X.shape[0]

    extra = T - M * (T // M)
    if extra:
        X = data[burn + extra :]
    data_s = np.asarray(np.split(X, M, axis=0), dtype=float)
    M_s, N, D = data_s.shape

    def split_rhat(arr: np.ndarray) -> np.ndarray:
        theta_bar_m = np.mean(arr, axis=1)
        theta_bar = np.mean(theta_bar_m, axis=0)
        B = (N / (M_s - 1)) * np.sum((theta_bar_m - theta_bar) ** 2, axis=0)
        s2_m = np.sum((arr - theta_bar_m[:, None, :]) ** 2, axis=1) / (N - 1)
        W = np.mean(s2_m, axis=0)
        var_hat = ((N - 1) / N) * W + (1 / N) * B
        rhat = np.sqrt(np.maximum(var_hat / W, 0.0))
        rhat[~np.isfinite(rhat)] = np.nan
        return rhat

    n = M_s * N
    q_table = norm.ppf((np.arange(1, n + 1, dtype=float) - 0.375) / (n + 0.25)) if n > 0 else None
    Z = _rank_normalize(data_s, q_table)
    rhat_z = split_rhat(Z)
    Z_med = np.median(Z.reshape(-1, D), axis=0)
    Zf = _rank_normalize(np.abs(Z - Z_med[None, None, :]), q_table)
    rhat_folded = split_rhat(Zf)

    Rhat = np.maximum(rhat_z, rhat_folded)
    return Rhat, np.where(Rhat > threshold)[0]

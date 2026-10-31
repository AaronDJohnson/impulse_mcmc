"""Per-chain adaptive statistics and covariance eigendecomposition."""

import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

COV_RIDGE_REL = 1e-10


def svd_groups(
    svd_U: list,
    svd_S: list,
    groups: list,
    sample_cov: np.ndarray,
    proposal_L: list | None = None,
    ridge: float = 0.0,
) -> tuple[list, list, list]:
    """Compute eigendecomposition and scaled square-root ``L = U * sqrt(S)`` per group."""
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, got {ridge}")
    if proposal_L is None:
        proposal_L = [None] * len(groups)

    full_dim = sample_cov.shape[0]
    for i, group in enumerate(groups):
        if len(group) == full_dim and np.array_equal(group, np.arange(full_dim)):
            sub_cov = sample_cov.copy() if ridge > 0.0 else sample_cov
        else:
            sub_cov = sample_cov[np.ix_(group, group)]
        if ridge > 0.0:
            sub_cov.flat[:: sub_cov.shape[0] + 1] += ridge
        # eigh returns NaN for non-finite input rather than raising, and can
        # overflow to inf on finite input near the float64 limit.
        try:
            eigvals, eigvecs = np.linalg.eigh(sub_cov)
            failure = None
            if not (np.all(np.isfinite(eigvals)) and np.all(np.isfinite(eigvecs))):
                failure = "non-finite result"
        except np.linalg.LinAlgError as err:
            failure = str(err)
        if failure is not None:
            k = len(group)
            prev_L = proposal_L[i]
            if (
                prev_L is not None
                and svd_U[i] is not None
                and svd_S[i] is not None
                and np.shape(prev_L) == (k, k)
            ):
                warnings.warn(
                    f"eigendecomposition of the group {i} covariance failed ({failure}); "
                    "keeping the previous proposal scale for that group.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"eigendecomposition of the group {i} covariance failed ({failure}); "
                "using an identity proposal scale for that group until the next update.",
                RuntimeWarning,
                stacklevel=2,
            )
            eigvals = np.ones(k)
            eigvecs = np.eye(k)
        svd_S[i] = eigvals[::-1]
        svd_U[i] = eigvecs[:, ::-1]
        sqrt_s = np.sqrt(np.maximum(svd_S[i], 0.0))
        proposal_L[i] = svd_U[i] * sqrt_s[None, :]

    return svd_U, svd_S, proposal_L


class _HistoryBuffer:
    """Circular sample-history buffer used for DE proposals and covariance updates."""

    @staticmethod
    def select_for_storage(new_samples: np.ndarray, seen_raw: int, thin: int) -> np.ndarray:
        if thin <= 1 or len(new_samples) == 0:
            return new_samples
        offset = (-seen_raw) % thin
        return new_samples[offset::thin]

    @staticmethod
    def append(buffer: np.ndarray, new_samples: np.ndarray) -> np.ndarray:
        n = len(new_samples)
        buf_len = len(buffer)
        if n == 0:
            return buffer
        if n >= buf_len:
            buffer[:] = new_samples[-buf_len:]
        else:
            buffer[:-n] = buffer[n:]
            buffer[-n:] = new_samples
        return buffer

    @staticmethod
    def moments(buffer: np.ndarray, n_valid: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if n_valid < 2:
            return None
        # np.cov's own arithmetic, minus its second pass to compute the mean.
        centered = np.array(buffer[-n_valid:], dtype=np.float64)
        mean = centered.mean(axis=0)
        centered -= mean
        cov = np.dot(centered.T, centered)
        cov *= np.true_divide(1, n_valid - 1)
        if not np.all(np.isfinite(cov)):
            return None
        return mean, cov


@dataclass
class ChainStats:
    """Adaptive statistics and history buffer for a single temperature chain."""

    ndim: int
    pt_state: Any
    chain_index: int
    rng: np.random.Generator
    groups: Optional[list] = None
    svd_U: Optional[list] = None
    svd_S: Optional[list] = None
    sample_cov: Optional[np.ndarray] = None
    proposal_L: Optional[list] = None
    sample_mean: Optional[np.ndarray] = None
    current_sample: Optional[np.ndarray] = None
    sample_total: int = 0
    buffer_size: int = 2_000
    buffer_thin: int = 25

    def __post_init__(self):
        if self.pt_state.ladder is None:
            raise ValueError("pt_state.ladder must be initialized")
        self.temp = self.pt_state.ladder[self.chain_index]

        if self.sample_cov is None:
            self.sample_cov = np.identity(self.ndim)
        else:
            self.sample_cov = np.atleast_2d(np.asarray(self.sample_cov, dtype=float))
            if self.sample_cov.shape != (self.ndim, self.ndim):
                raise ValueError(
                    f"sample_cov must have shape ({self.ndim}, {self.ndim}), got "
                    f"{self.sample_cov.shape}; pass a full covariance matrix "
                    "(np.diag(variances) for a diagonal one)."
                )
            if not np.all(np.isfinite(self.sample_cov)):
                raise ValueError("sample_cov contains non-finite values (NaN or inf).")
        if self.sample_mean is None:
            self.sample_mean = np.zeros(self.ndim)
        else:
            self.sample_mean = np.atleast_1d(np.asarray(self.sample_mean, dtype=float))
        if self.groups is None:
            self.groups = [np.arange(0, self.ndim)]
        else:
            covered = {int(i) for g in self.groups for i in np.atleast_1d(g)}
            out_of_range = sorted(i for i in covered if i < 0 or i >= self.ndim)
            if out_of_range:
                raise ValueError(
                    f"groups contain out-of-range parameter index/indices "
                    f"{out_of_range} for ndim={self.ndim}."
                )
            missing = sorted(set(range(self.ndim)) - covered)
            if missing:
                warnings.warn(
                    f"groups do not cover parameter index/indices {missing} "
                    f"(ndim={self.ndim}). am/scam/de only propose within a group, so "
                    "unless a custom proposal moves them they will stay frozen at their "
                    "initial values and the chain will sample a conditional of the "
                    "posterior, not the marginal.",
                    UserWarning,
                    stacklevel=3,
                )
        if self.svd_U is None:
            self.svd_U = [None for _ in range(len(self.groups))]
        if self.svd_S is None:
            self.svd_S = [None for _ in range(len(self.groups))]
        if self.proposal_L is None:
            self.proposal_L = [None for _ in range(len(self.groups))]

        if self.buffer_thin < 1:
            raise ValueError(f"buffer_thin must be >= 1, got {self.buffer_thin}")
        self._buffer = np.zeros((self.buffer_size, self.ndim))
        self.buffer_full = False
        self._seen_raw = 0

        initial_scale = float(np.mean(np.diag(self.sample_cov)))
        if not np.isfinite(initial_scale) or initial_scale <= 0.0:
            initial_scale = 1.0
        self._cov_ridge = COV_RIDGE_REL * initial_scale

        self._group_specs_ref: Any = None
        self._group_specs: list = []
        self._rebuild_group_specs()

        self.svd_U, self.svd_S, self.proposal_L = svd_groups(
            self.svd_U,
            self.svd_S,
            self.groups,
            self.sample_cov,
            self.proposal_L,
            ridge=self._cov_ridge,
        )

    def _rebuild_group_specs(self) -> list:
        """Build cached per-group indexing and scale metadata for fast proposals."""
        assert self.groups is not None
        specs = []
        arrays = []
        for g in self.groups:
            arr = np.asarray(g, dtype=np.intp).ravel()
            arrays.append(arr)
            neff = int(arr.size)
            sqrt_2neff = math.sqrt(2 * neff) if neff > 0 else 1.0
            is_full = neff == self.ndim and np.array_equal(arr, np.arange(self.ndim))
            if is_full:
                indexer: Any = slice(None)
            elif neff > 0 and np.array_equal(arr, np.arange(int(arr[0]), int(arr[0]) + neff)):
                indexer = slice(int(arr[0]), int(arr[0]) + neff)
            else:
                indexer = arr
            specs.append((indexer, neff, sqrt_2neff, is_full))
        self._group_cache = (self.groups, arrays)
        self._group_specs_ref = self.groups
        self._group_specs = specs
        return specs

    def update_buffer(self, new_samples: np.ndarray) -> None:
        """Append ``new_samples`` to the circular history buffer."""
        self._buffer = _HistoryBuffer.append(self._buffer, new_samples)
        if not self.buffer_full and self.sample_total > self.buffer_size:
            self.buffer_full = True

    def update_from_window(self, sample_num: int, new_samples: np.ndarray) -> None:
        """Append new samples to the history buffer and recompute moments and SVD."""
        if not np.all(np.isfinite(new_samples)):
            return
        if self.sample_cov is None or self.sample_mean is None:
            raise ValueError(
                "sample_cov and sample_mean must be initialized before calling update_from_window"
            )
        if self.svd_U is None or self.svd_S is None:
            raise ValueError(
                "svd_U and svd_S must be initialized before calling update_from_window"
            )
        if self.groups is None:
            raise ValueError("groups must be initialized before calling update_from_window")

        stored = _HistoryBuffer.select_for_storage(new_samples, self._seen_raw, self.buffer_thin)
        self._seen_raw += len(new_samples)
        if len(stored) == 0:
            return
        self.sample_total += len(stored)
        self.update_buffer(stored)
        if sample_num + len(new_samples) < 2:
            return
        moments = _HistoryBuffer.moments(self._buffer, min(self.sample_total, self.buffer_size))
        if moments is None:
            return
        self.sample_mean, self.sample_cov = moments
        self.svd_U, self.svd_S, self.proposal_L = svd_groups(
            self.svd_U,
            self.svd_S,
            self.groups,
            self.sample_cov,
            self.proposal_L,
            ridge=self._cov_ridge,
        )

    recursive_update = update_from_window

    def get_group_U(self, group_idx: int) -> np.ndarray:
        if self.svd_U is None:
            raise ValueError("svd_U is not initialized")
        if self.svd_U[group_idx] is None:
            raise ValueError(f"U for group {group_idx} is not initialized")
        return self.svd_U[group_idx]

    def get_group_S(self, group_idx: int) -> np.ndarray:
        if self.svd_S is None:
            raise ValueError("svd_S is not initialized")
        if self.svd_S[group_idx] is None:
            raise ValueError(f"Singular values for group {group_idx} are not initialized")
        return self.svd_S[group_idx]

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        assert self.groups is not None and self.sample_cov is not None
        assert self.sample_mean is not None
        assert self.svd_U is not None and self.svd_S is not None and self.proposal_L is not None
        arrays: dict = {
            "sample_cov": np.asarray(self.sample_cov),
            "sample_mean": np.asarray(self.sample_mean),
            "buffer": np.asarray(self._buffer),
        }
        meta: dict = {
            "sample_total": int(self.sample_total),
            "seen_raw": int(self._seen_raw),
            "buffer_full": bool(self.buffer_full),
            "groups": [list(map(int, g)) for g in self.groups],
            "has_current_sample": self.current_sample is not None,
            # Always None here; written so impulse v2.0, which reads this key
            # unconditionally, can still resume checkpoints from this version.
            "per_model": None,
        }
        if self.current_sample is not None:
            arrays["current_sample"] = np.asarray(self.current_sample)
        for gi in range(len(self.groups)):
            arrays[f"svd_U.g{gi}"] = np.asarray(self.svd_U[gi])
            arrays[f"svd_S.g{gi}"] = np.asarray(self.svd_S[gi])
            arrays[f"proposal_L.g{gi}"] = np.asarray(self.proposal_L[gi])
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        self.sample_total = int(meta["sample_total"])
        self._seen_raw = int(meta.get("seen_raw", meta["sample_total"]))
        self.buffer_full = bool(meta["buffer_full"])
        self.groups = [np.array(g, dtype=int) for g in meta["groups"]]
        self._rebuild_group_specs()
        self.sample_cov = np.array(arrays["sample_cov"], dtype=float)
        self.sample_mean = np.array(arrays["sample_mean"], dtype=float)
        self._buffer = np.array(arrays["buffer"], dtype=float)
        self.buffer_size = len(self._buffer)
        if meta["has_current_sample"]:
            self.current_sample = np.array(arrays["current_sample"], dtype=float)
        ng = len(self.groups)
        self.svd_U = [np.array(arrays[f"svd_U.g{gi}"]) for gi in range(ng)]
        self.svd_S = [np.array(arrays[f"svd_S.g{gi}"]) for gi in range(ng)]
        self.proposal_L = [np.array(arrays[f"proposal_L.g{gi}"]) for gi in range(ng)]

    def update_sample(self, position: np.ndarray) -> None:
        self.current_sample = position

    def group_index_arrays(self) -> list:
        assert self.groups is not None
        if self._group_specs_ref is not self.groups:
            self._rebuild_group_specs()
        return self._group_cache[1]


@dataclass
class MultiChainStats:
    """Container managing ``ChainStats`` across all temperature chains."""

    chain_stats: List[ChainStats]

    @property
    def ntemps(self) -> int:
        return len(self.chain_stats)

    @property
    def ndim(self) -> int:
        return self.chain_stats[0].ndim

    @property
    def sample_total(self) -> int:
        return self.chain_stats[0].sample_total

    def update_from_window(self, new_samples: np.ndarray) -> None:
        for i, cs in enumerate(self.chain_stats):
            cs.update_from_window(cs.sample_total, new_samples[i])

    recursive_update = update_from_window

    def get_group_U(self, chain_idx: int, group_idx: int) -> np.ndarray:
        return self.chain_stats[chain_idx].get_group_U(group_idx)

    def get_group_S(self, chain_idx: int, group_idx: int) -> np.ndarray:
        return self.chain_stats[chain_idx].get_group_S(group_idx)

    def set_current_sample(self, state: Any) -> None:
        for i, cs in enumerate(self.chain_stats):
            cs.update_sample(state.positions[i])

    update_sample = set_current_sample

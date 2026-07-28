"""Per-chain statistics that drive the adaptive proposals.

:class:`ChainStats` tracks, for one temperature chain, the running sample
mean/covariance, the per-group SVD used to orient AM/SCAM jumps, and the
circular sample-history buffer consumed by the differential-evolution moves;
``update_sample`` refreshes the view of the current sample before each
proposal call. For product-space model selection, per-model statistics (:class:`_PerModelState`)
can be enabled so that within-model proposals use covariances and buffers
learned separately for each model index. :class:`MultiChainStats` holds one
:class:`ChainStats` per temperature and applies batched recursive updates
across the ladder.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from impulse.online_updates import svd_groups
from impulse.product_space import ParameterLayout
from impulse.sampler_state import PTState, SamplerState
from impulse.utils import shift_array

#: Ridge added to the proposal covariance, as a fraction of the mean initial
#: variance (Haario et al. 2001's ``epsilon``). Small enough to be irrelevant to
#: a healthy adaptive covariance, large enough that a fully collapsed one still
#: yields a proposal that can move. See :func:`~impulse.online_updates.svd_groups`.
COV_RIDGE_REL = 1e-10


class _HistoryBuffer:
    """Tail-filled rolling window of recent chain states.

    This is the single owner of the sample history that feeds two very
    different consumers: the differential-evolution move (which wants *diverse*
    pairs to form difference vectors) and the adaptive covariance (which wants a
    long horizon). Both the global chain statistics and each per-model state own
    one of these, which is what keeps their update logic from drifting apart --
    the two paths previously carried separate copies of the same append,
    fill-tracking, and moment-estimation code.

    Rows are stored at the TAIL: ``buffer[-n_filled:]`` is the valid window and
    everything before it is zero padding, so ``buffer[-k:]`` is always the k most
    recent states regardless of how full the buffer is.

    The counters (``sample_total``, ``buffer_full``, ``buffer_size``) stay on the
    owning object because they are part of its constructor and checkpoint
    surface; this class owns the array and the *logic*, which is the part that
    both paths duplicated and the single place a future storage policy (e.g.
    retaining thinned rather than consecutive states) has to change.
    """

    @staticmethod
    def select_for_storage(new_samples: np.ndarray, seen_raw: int, thin: int) -> np.ndarray:
        """Rows of ``new_samples`` to retain when storing every ``thin``-th state.

        The phase is taken from ``seen_raw``, the number of states seen before
        this batch, so the retained rows are exactly the global iterations
        divisible by ``thin`` no matter how the run is chunked into batches.
        Slicing each batch with ``[::thin]`` instead would restart the phase
        every batch and retain a non-uniformly spaced set.

        ``thin <= 1`` returns the batch unchanged.
        """
        if thin <= 1:
            return new_samples
        offset = (-seen_raw) % thin
        return new_samples[offset::thin]

    @staticmethod
    def append(buffer: np.ndarray, new_samples: np.ndarray) -> np.ndarray:
        """Return ``buffer`` with ``new_samples`` written at the tail.

        Rows live at the TAIL: ``buffer[-n_filled:]`` is the valid window and
        anything before it is zero padding, so ``buffer[-k:]`` is always the k
        most recent states however full the buffer is.
        """
        n = len(new_samples)
        buffer = shift_array(buffer, -n)
        buffer[-n:] = new_samples
        return buffer

    @staticmethod
    def moments(buffer: np.ndarray, n_filled: int) -> Optional[tuple]:
        """``(mean, cov)`` over the filled tail, or None if too few samples.

        ``atleast_2d`` because np.cov returns a 0-d scalar for a single column
        (ndim == 1), which would break the (ndim, ndim) contract svd_groups
        relies on.
        """
        if n_filled < 2:
            return None
        buf = buffer[-n_filled:]
        return np.mean(buf, axis=0), np.atleast_2d(np.cov(buf, rowvar=False, ddof=1))


@dataclass
class _PerModelState:
    """Per-model statistics for adaptive proposals in product-space model selection.

    Holds the groups, covariance, SVD, and DE buffer for a single
    model dimension so that within-model proposals use model-specific
    learned statistics.
    """

    groups: list
    sample_cov: np.ndarray
    sample_mean: np.ndarray
    svd_U: list
    svd_S: list
    proposal_L: list
    buffer: np.ndarray
    buffer_full: bool = False
    sample_total: int = 0  # STORED rows (differs from seen_raw when thinning)
    seen_raw: int = 0  # this model's raw visits; carries the thinning phase


@dataclass
class ChainStats:
    """
    Statistics tracker for adaptive MCMC proposals on a single temperature chain.

    Maintains running estimates of sample covariance, means, and sample history
    needed for adaptive Metropolis, differential evolution, and other sophisticated
    proposal mechanisms.

    Parameters
    ----------
    ndim : int
        Dimensionality of parameter space.
    pt_state : PTState
        Parallel tempering state containing temperature information.
    chain_index : int
        Index of this chain in the temperature ladder.
    rng : np.random.Generator
        Random number generator for this chain.
    groups : list, optional
        Parameter groups for block updates. Default: single group [0, 1, ..., ndim-1].
    sample_cov : np.ndarray, optional
        Initial covariance matrix estimate. Default: identity matrix.
    svd_U : list of np.ndarray, optional
        Left singular vectors for each parameter group.
    svd_S : list of np.ndarray, optional
        Singular values for each parameter group.
    sample_mean : np.ndarray, optional
        Initial mean estimate. Default: zero vector.
    current_sample : np.ndarray, optional
        Current parameter position.
    sample_total : int, default 0
        Total number of samples processed.
    buffer_size : int, default 50000
        Size of circular buffer for differential evolution proposals.

    Examples
    --------
    >>> ptstate = PTState(ndim=3, ntemps=5)
    >>> rng = np.random.default_rng(42)
    >>> stats = ChainStats(ndim=3, pt_state=ptstate, chain_index=0, rng=rng)
    >>> # Used internally by proposal functions
    >>> new_pos, log_ratio = am(stats)

    Notes
    -----
    - Automatically maintains SVD decomposition for efficient proposals
    - The circular history buffer fills from the tail; the DE move
      (:func:`impulse.proposals.de`) starts proposing difference moves
      once the buffer holds ``min_fill`` samples
    - Temperature-specific statistics help with parallel tempering adaptation
    """

    ndim: int
    pt_state: PTState
    chain_index: int
    rng: np.random.Generator
    groups: Optional[list] = None
    sample_cov: Optional[np.ndarray] = None
    svd_U: List[Optional[np.ndarray]] | None = None  # U in the SVD of samples
    svd_S: List[Optional[np.ndarray]] | None = None  # Sigma in the SVD of samples
    proposal_L: List[Optional[np.ndarray]] | None = None  # Precomputed U * sqrt(S)
    sample_mean: Optional[np.ndarray] = None
    current_sample: Optional[np.ndarray] = None

    # DEBuffer pieces:
    sample_total: int = 0
    buffer_size: int = 50_000
    buffer_thin: int = 1

    def __post_init__(self):
        if self.pt_state.ladder is None:
            raise ValueError("pt_state.ladder must be initialized")
        self.temp = self.pt_state.ladder[self.chain_index]
        if self.sample_cov is None:
            self.sample_cov = np.identity(self.ndim)
        else:
            # A user-supplied sample_cov is normalized here rather than trusted.
            # The obvious way to produce one for a 1-parameter model,
            # np.cov(pilot, rowvar=False), returns a 0-d scalar, which breaks
            # the (ndim, ndim) contract svd_groups relies on.
            self.sample_cov = np.atleast_2d(np.asarray(self.sample_cov, dtype=float))
        if self.sample_mean is None:
            self.sample_mean = np.zeros(self.ndim)
        else:
            self.sample_mean = np.atleast_1d(np.asarray(self.sample_mean, dtype=float))
        if self.groups is None:
            self.groups = [np.arange(0, self.ndim)]
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
        # Raw states seen, as opposed to states STORED (which is sample_total).
        # These differ exactly when buffer_thin > 1, and this counter carries the
        # global thinning phase across batches and across a resume.
        self._seen_raw = 0

        # Ridge (Haario et al. 2001 `epsilon * I`) keeping the adaptive proposal
        # from degenerating into an absorbing state -- see svd_groups. Scaled to
        # the INITIAL covariance so it is meaningful for the problem's units: a
        # fixed absolute constant would be either useless on a tiny-scale
        # posterior or a large perturbation on a huge-scale one. The initial
        # covariance is the only scale information available before sampling.
        initial_scale = float(np.mean(np.diag(self.sample_cov)))
        if not np.isfinite(initial_scale) or initial_scale <= 0.0:
            initial_scale = 1.0
        self._cov_ridge = COV_RIDGE_REL * initial_scale

        self.svd_U, self.svd_S, self.proposal_L = svd_groups(
            self.svd_U,
            self.svd_S,
            self.groups,
            self.sample_cov,
            self.proposal_L,
            ridge=self._cov_ridge,
        )

    def update_buffer(self, new_samples: np.ndarray) -> None:
        """
        Add new samples to circular buffer.

        Updates the internal circular buffer with new samples, maintaining
        a rolling window of recent samples for differential evolution proposals.

        Parameters
        ----------
        new_samples : np.ndarray
            New samples to add to buffer, shape (n_new, ndim).

        Examples
        --------
        >>> import numpy as np
        >>> new_samples = np.array([[1.0, 2.0], [1.1, 2.1]])
        >>> stats.update_buffer(new_samples)
        >>> # Buffer now contains the new samples in most recent positions
        """
        self._buffer = _HistoryBuffer.append(self._buffer, new_samples)
        if not self.buffer_full:
            if self.sample_total > self.buffer_size:
                self.buffer_full = True

    def recursive_update(self, sample_num: int, new_samples: np.ndarray) -> None:
        """
        Update all statistics with new samples using online algorithms.

        Performs comprehensive update of sample count, buffer, mean, covariance,
        and SVD decompositions using numerically stable online methods.

        When per-model statistics are active, samples are partitioned by
        their model index and each model's statistics are updated
        independently.

        Parameters
        ----------
        sample_num : int
            Current total sample count before adding new samples.
        new_samples : np.ndarray
            New samples to incorporate, shape (n_new, ndim).
        """
        # Refuse to poison stats with non-finite samples. Guards against a
        # single bad accepted position propagating into proposal_L forever.
        if not np.all(np.isfinite(new_samples)):
            return

        # Per-model path: partition samples by nmodel
        if hasattr(self, "_per_model") and self._per_model is not None:
            nmodels_arr = np.rint(new_samples[:, self._nmodel_idx]).astype(int)
            for k, pm in self._per_model.items():
                mask = nmodels_arr == k
                if not np.any(mask):
                    continue
                model_samples = new_samples[mask]
                # Thinning phase is per model, counted in that model's own
                # visits, so each model's buffer spans buffer_thin * capacity
                # of ITS OWN samples.
                stored = _HistoryBuffer.select_for_storage(
                    model_samples, pm.seen_raw, self.buffer_thin
                )
                pm.seen_raw += len(model_samples)
                if len(stored) == 0:
                    continue
                pm.sample_total += len(stored)
                pm.buffer = _HistoryBuffer.append(pm.buffer, stored)
                if not pm.buffer_full and pm.sample_total > self.buffer_size:
                    pm.buffer_full = True
                moments = _HistoryBuffer.moments(pm.buffer, min(pm.sample_total, self.buffer_size))
                if moments is None:
                    continue
                pm.sample_mean, pm.sample_cov = moments
                pm.svd_U, pm.svd_S, pm.proposal_L = svd_groups(
                    pm.svd_U,
                    pm.svd_S,
                    pm.groups,
                    pm.sample_cov,
                    pm.proposal_L,
                    ridge=self._cov_ridge,
                )
            return

        if self.sample_cov is None or self.sample_mean is None:
            raise ValueError(
                "sample_cov and sample_mean must be initialized before calling recursive_update"
            )
        if self.svd_U is None or self.svd_S is None:
            raise ValueError("svd_U and svd_S must be initialized before calling recursive_update")
        if self.groups is None:
            raise ValueError("groups must be initialized before calling recursive_update")

        # update buffer
        # Retain every buffer_thin-th state. sample_total counts STORED rows, not
        # raw iterations, because `de` derives its window as
        # min(sample_total, buffer_size) and indexes buffer[-n_filled:]; if
        # sample_total counted raw iterations under thinning, that window would
        # reach past the stored rows into the zero padding and DE would build
        # difference vectors out of zeros.
        stored = _HistoryBuffer.select_for_storage(new_samples, self._seen_raw, self.buffer_thin)
        self._seen_raw += len(new_samples)
        if len(stored) == 0:
            return
        self.sample_total += len(stored)
        self.update_buffer(stored)
        # need at least 2 total samples for a meaningful covariance update
        if sample_num + len(new_samples) < 2:
            return
        # Recompute mean and covariance from the filled portion of the buffer
        moments = _HistoryBuffer.moments(self._buffer, min(self.sample_total, self.buffer_size))
        if moments is None:
            return
        self.sample_mean, self.sample_cov = moments
        # new SVD on groups
        self.svd_U, self.svd_S, self.proposal_L = svd_groups(
            self.svd_U,
            self.svd_S,
            self.groups,
            self.sample_cov,
            self.proposal_L,
            ridge=self._cov_ridge,
        )

    def get_group_U(self, group_idx: int) -> np.ndarray:
        """Return U for group `group_idx` (shape (k, k))."""
        if self.svd_U is None:
            raise ValueError("svd_U is not initialized")
        u = self.svd_U[group_idx]
        if u is None:
            raise ValueError(f"U for group {group_idx} is not initialized")
        return u

    def get_group_S(self, group_idx: int) -> np.ndarray:
        """Return singular values for group `group_idx` (shape (k,))."""
        if self.svd_S is None:
            raise ValueError("svd_S is not initialized")
        s = self.svd_S[group_idx]
        if s is None:
            raise ValueError(f"Singular values for group {group_idx} are not initialized")
        return s

    def enable_per_model(
        self, num_models: int, num_params: int, layout: Optional[ParameterLayout] = None
    ) -> None:
        """Activate per-model adaptive statistics for product-space model selection.

        Creates independent covariance, SVD, and DE-buffer state for each
        model index so that within-model proposals use model-specific
        learned statistics.

        Parameters
        ----------
        num_models : int
            Maximum number of models (e.g. ``product_space.num_models``).
        num_params : int
            Number of continuous parameters per source.
        layout : ParameterLayout, optional
            Product-space parameter layout (e.g. ``product_space.layout``);
            when given it is the single source of truth and the scalar
            arguments are ignored. Without it, one is built from the
            scalars.
        """
        # __post_init__ guarantees these are set on any constructed instance
        assert (
            self.groups is not None and self.sample_cov is not None and self.sample_mean is not None
        )
        if layout is None:
            layout = ParameterLayout(num_params=num_params, num_models=num_models)
        self._num_models = layout.num_models
        self._num_params = layout.num_params
        self._nmodel_idx = layout.nmodel_index  # last element of position

        all_groups = self.groups  # full list, one group per source slot

        self._per_model: dict[int, _PerModelState] = {}
        for k in range(self._num_models):
            model_groups = [list(g) for g in all_groups[: k + 1]]
            model_svd_U: list = [None] * len(model_groups)
            model_svd_S: list = [None] * len(model_groups)
            model_proposal_L: list = [None] * len(model_groups)
            model_svd_U, model_svd_S, model_proposal_L = svd_groups(
                model_svd_U,
                model_svd_S,
                model_groups,
                self.sample_cov,
                model_proposal_L,
                ridge=self._cov_ridge,
            )
            self._per_model[k] = _PerModelState(
                groups=model_groups,
                sample_cov=self.sample_cov.copy(),
                sample_mean=self.sample_mean.copy(),
                svd_U=model_svd_U,
                svd_S=model_svd_S,
                proposal_L=model_proposal_L,
                buffer=np.zeros((self.buffer_size, self.ndim)),
            )

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle, recomputing proposal_L for old checkpoints."""
        self.__dict__.update(state)
        if not hasattr(self, "proposal_L") or self.proposal_L is None:
            # any pickled instance went through __post_init__, so these are set
            assert self.groups is not None and self.svd_U is not None and self.svd_S is not None
            self.proposal_L = [None] * len(self.groups)
            for ct, group in enumerate(self.groups):
                svd_u = self.svd_U[ct]
                svd_s = self.svd_S[ct]
                assert svd_u is not None and svd_s is not None
                sqrt_s = np.sqrt(np.maximum(svd_s, 0.0))
                self.proposal_L[ct] = svd_u * sqrt_s[None, :]
        # Recompute proposal_L for per-model states
        if hasattr(self, "_per_model") and self._per_model is not None:
            for pm in self._per_model.values():
                if pm.proposal_L is None:
                    pm.proposal_L = [None] * len(pm.groups)
                for ct, group in enumerate(pm.groups):
                    if pm.svd_U[ct] is not None and pm.svd_S[ct] is not None:
                        sqrt_s = np.sqrt(np.maximum(pm.svd_S[ct], 0.0))
                        pm.proposal_L[ct] = pm.svd_U[ct] * sqrt_s[None, :]

    def get_checkpoint_state(self) -> tuple[dict, dict]:
        """Serialize the adaptive state to ``(arrays, meta)`` for the checkpoint.

        Captures everything that steers future proposals: running
        mean/covariance, the per-group SVD factors and precomputed
        ``proposal_L``, the DE history buffer and its fill counters, and (when
        per-model statistics are active) the full per-model cache.  Array
        keys are local; the sampler prefixes them (e.g. ``cs.c3.m1.buffer``).
        """
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
            "seen_raw": int(getattr(self, "_seen_raw", self.sample_total)),
            "buffer_full": bool(self.buffer_full),
            "groups": [list(map(int, g)) for g in self.groups],
            "has_current_sample": self.current_sample is not None,
        }
        if self.current_sample is not None:
            arrays["current_sample"] = np.asarray(self.current_sample)
        for gi in range(len(self.groups)):
            arrays[f"svd_U.g{gi}"] = np.asarray(self.svd_U[gi])
            arrays[f"svd_S.g{gi}"] = np.asarray(self.svd_S[gi])
            arrays[f"proposal_L.g{gi}"] = np.asarray(self.proposal_L[gi])
        per_model = getattr(self, "_per_model", None)
        if per_model is not None:
            models = []
            for k, pm in per_model.items():
                arrays[f"m{k}.sample_cov"] = np.asarray(pm.sample_cov)
                arrays[f"m{k}.sample_mean"] = np.asarray(pm.sample_mean)
                arrays[f"m{k}.buffer"] = np.asarray(pm.buffer)
                for gi in range(len(pm.groups)):
                    arrays[f"m{k}.svd_U.g{gi}"] = np.asarray(pm.svd_U[gi])
                    arrays[f"m{k}.svd_S.g{gi}"] = np.asarray(pm.svd_S[gi])
                    arrays[f"m{k}.proposal_L.g{gi}"] = np.asarray(pm.proposal_L[gi])
                models.append(
                    {
                        "k": int(k),
                        "sample_total": int(pm.sample_total),
                        "seen_raw": int(getattr(pm, "seen_raw", pm.sample_total)),
                        "buffer_full": bool(pm.buffer_full),
                        "groups": [list(map(int, g)) for g in pm.groups],
                    }
                )
            meta["per_model"] = {
                "num_models": int(self._num_models),
                "num_params": int(self._num_params),
                "nmodel_idx": int(self._nmodel_idx),
                "models": models,
            }
        else:
            meta["per_model"] = None
        return arrays, meta

    def set_checkpoint_state(self, arrays: dict, meta: dict) -> None:
        """Restore adaptive state from :meth:`get_checkpoint_state` output."""
        self.sample_total = int(meta["sample_total"])
        # Checkpoints written before buffer_thin existed have no "seen_raw";
        # there stored == raw, so sample_total is the correct phase.
        self._seen_raw = int(meta.get("seen_raw", meta["sample_total"]))
        self.buffer_full = bool(meta["buffer_full"])
        self.groups = [np.array(g, dtype=int) for g in meta["groups"]]
        self.sample_cov = np.array(arrays["sample_cov"], dtype=float)
        self.sample_mean = np.array(arrays["sample_mean"], dtype=float)
        self._buffer = np.array(arrays["buffer"], dtype=float)
        # Keep the len(_buffer) == buffer_size invariant. Restoring the buffer
        # without its size leaves the differential-evolution proposal drawing
        # indices from a range that does not match the array it indexes: it
        # raises IndexError when the buffer grew across the resume and silently
        # mis-scales the adaptation history when it shrank.
        self.buffer_size = len(self._buffer)
        if meta["has_current_sample"]:
            self.current_sample = np.array(arrays["current_sample"], dtype=float)
        ng = len(self.groups)
        self.svd_U = [np.array(arrays[f"svd_U.g{gi}"]) for gi in range(ng)]
        self.svd_S = [np.array(arrays[f"svd_S.g{gi}"]) for gi in range(ng)]
        self.proposal_L = [np.array(arrays[f"proposal_L.g{gi}"]) for gi in range(ng)]
        pm_meta = meta["per_model"]
        if pm_meta is not None:
            self._num_models = int(pm_meta["num_models"])
            self._num_params = int(pm_meta["num_params"])
            self._nmodel_idx = int(pm_meta["nmodel_idx"])
            self._per_model = {}
            for m in pm_meta["models"]:
                k = int(m["k"])
                groups = [np.array(g, dtype=int) for g in m["groups"]]
                ngm = len(groups)
                self._per_model[k] = _PerModelState(
                    groups=groups,
                    sample_cov=np.array(arrays[f"m{k}.sample_cov"], dtype=float),
                    sample_mean=np.array(arrays[f"m{k}.sample_mean"], dtype=float),
                    svd_U=[np.array(arrays[f"m{k}.svd_U.g{gi}"]) for gi in range(ngm)],
                    svd_S=[np.array(arrays[f"m{k}.svd_S.g{gi}"]) for gi in range(ngm)],
                    proposal_L=[np.array(arrays[f"m{k}.proposal_L.g{gi}"]) for gi in range(ngm)],
                    buffer=np.array(arrays[f"m{k}.buffer"], dtype=float),
                    buffer_full=bool(m["buffer_full"]),
                    sample_total=int(m["sample_total"]),
                    seen_raw=int(m.get("seen_raw", m["sample_total"])),
                )

    def update_sample(self, position: np.ndarray):
        """
        Update current parameter position.

        If per-model statistics are active, swaps in the groups, SVD,
        and DE buffer for the current model index so that proposals
        transparently use model-specific learned statistics.

        Parameters
        ----------
        position : np.ndarray
            New parameter position, shape (ndim,).
        """
        self.current_sample = position
        if hasattr(self, "_per_model") and self._per_model is not None:
            nmodel = int(np.rint(position[self._nmodel_idx]))
            nmodel = max(0, min(nmodel, self._num_models - 1))
            pm = self._per_model[nmodel]
            self.groups = pm.groups
            self.proposal_L = pm.proposal_L
            self.svd_U = pm.svd_U
            self.svd_S = pm.svd_S
            self._buffer = pm.buffer
            self.buffer_full = pm.buffer_full
            self.sample_total = pm.sample_total


@dataclass
class MultiChainStats:
    """
    Container for statistics tracking across multiple temperature chains.

    Manages ChainStats objects for all temperature chains in parallel tempering,
    providing vectorized operations and coordination between chains.

    Parameters
    ----------
    chain_stats : list of ChainStats
        Statistics objects for each individual temperature chain.

    Attributes
    ----------
    ntemps : int
        Number of temperature chains.
    ndim : int
        Dimensionality of parameter space.
    sample_total : int
        Total number of samples processed across all chains.

    Examples
    --------
    >>> import numpy as np
    >>> from impulse.sampler_state import PTState
    >>> ptstate = PTState(ndim=2, ntemps=3)
    >>> rngs = [np.random.default_rng(i) for i in range(3)]
    >>> chain_list = [ChainStats(2, ptstate, i, rngs[i]) for i in range(3)]
    >>> multi_stats = MultiChainStats(chain_list)
    >>> print(f"Managing {multi_stats.ntemps} chains")

    Notes
    -----
    - Provides unified interface for operations across all temperature chains
    - Enables vectorized updates and queries
    - Essential component of parallel tempering sampling infrastructure
    """

    chain_stats: List["ChainStats"]

    @property
    def ntemps(self) -> int:
        return len(self.chain_stats)

    @property
    def ndim(self) -> int:
        return self.chain_stats[0].ndim

    @property
    def sample_total(self) -> int:
        return self.chain_stats[0].sample_total

    def recursive_update(self, new_samples: np.ndarray) -> None:
        """
        Update statistics for all temperature chains simultaneously.

        Parameters
        ----------
        new_samples : np.ndarray
            New samples for all chains, shape (ntemps, n_new, ndim).

        Examples
        --------
        >>> import numpy as np
        >>> new_samples = np.random.randn(5, 100, 3)  # 5 chains, 100 new samples, 3 dimensions
        >>> multi_stats.recursive_update(new_samples)
        >>> # All chain statistics updated with new samples
        """
        for i, cs in enumerate(self.chain_stats):
            cs.recursive_update(cs.sample_total, new_samples[i])

    def get_group_U(self, chain_idx: int, group_idx: int) -> np.ndarray:
        """Return U for chain `chain_idx` and group `group_idx` (shape (k, k))."""
        return self.chain_stats[chain_idx].get_group_U(group_idx)

    def get_group_S(self, chain_idx: int, group_idx: int) -> np.ndarray:
        """Return singular values for chain `chain_idx` and group `group_idx` (shape (k,))."""
        return self.chain_stats[chain_idx].get_group_S(group_idx)

    def enable_per_model(
        self, num_models: int, num_params: int, layout: Optional[ParameterLayout] = None
    ) -> None:
        """Activate per-model adaptive statistics on every chain.

        Parameters
        ----------
        num_models : int
            Maximum number of models.
        num_params : int
            Number of continuous parameters per source.
        layout : ParameterLayout, optional
            Product-space parameter layout (single source of truth); the
            scalars are ignored when it is given.
        """
        for cs in self.chain_stats:
            cs.enable_per_model(num_models, num_params, layout=layout)

    def update_sample(self, state: SamplerState):
        """
        Update current positions for all temperature chains.

        Parameters
        ----------
        state : SamplerState
            Sampler state containing positions for all chains.

        Examples
        --------
        >>> multi_stats.update_sample(current_state)
        >>> # All chains updated with their current positions
        """
        for i, cs in enumerate(self.chain_stats):
            cs.update_sample(state.positions[i])

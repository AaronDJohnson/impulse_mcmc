from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from impulse.online_updates import update_covariance, svd_groups
from impulse.sampler_state import SamplerState, PTState
from impulse.utils import shift_array

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
    - Buffer fills gradually and enables DE proposals when full
    - Temperature-specific statistics help with parallel tempering adaptation
    """
    ndim: int
    pt_state: PTState
    chain_index: int
    rng: np.random.Generator
    groups: Optional[list] = None
    sample_cov: Optional[np.ndarray] = None
    svd_U: List[Optional[np.ndarray]]|None = None  # U in the SVD of samples
    svd_S: List[Optional[np.ndarray]]|None = None  # Sigma in the SVD of samples
    sample_mean: Optional[np.ndarray] = None
    current_sample: Optional[np.ndarray] = None

    # DEBuffer pieces:
    sample_total: int = 0
    buffer_size: int = 50_000

    def __post_init__(self):
        if self.pt_state.ladder is None:
            raise ValueError("pt_state.ladder must be initialized")
        self.temp = self.pt_state.ladder[self.chain_index]
        if self.sample_cov is None:
            self.sample_cov = np.identity(self.ndim)
        if self.sample_mean is None:
            self.sample_mean = np.zeros(self.ndim)
        if self.groups is None:
            self.groups = [np.arange(0, self.ndim)]
        if self.svd_U is None:
            self.svd_U = [None for _ in range(len(self.groups))]
        if self.svd_S is None:
            self.svd_S = [None for _ in range(len(self.groups))]

        self._buffer = np.zeros((self.buffer_size, self.ndim))
        self.buffer_full = False

        self.svd_U, self.svd_S = svd_groups(self.svd_U, self.svd_S, self.groups, self.sample_cov)

    def update_buffer(self,
                      new_samples: np.ndarray
                      ) -> None:
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
        self._buffer = shift_array(self._buffer, -len(new_samples))
        self._buffer[-len(new_samples):] = new_samples
        if not self.buffer_full:
            if self.sample_total > self.buffer_size:
                self.buffer_full = True

    def recursive_update(self,
                         sample_num: int,
                         new_samples: np.ndarray
                         ) -> None:
        """
        Update all statistics with new samples using online algorithms.

        Performs comprehensive update of sample count, buffer, mean, covariance,
        and SVD decompositions using numerically stable online methods.

        Parameters
        ----------
        sample_num : int
            Current total sample count before adding new samples.
        new_samples : np.ndarray
            New samples to incorporate, shape (n_new, ndim).

        Examples
        --------
        >>> import numpy as np
        >>> new_samples = np.array([[0.5, 1.5], [0.8, 1.2]])
        >>> stats.recursive_update(1000, new_samples)
        >>> # All statistics updated with new samples
        """
        if self.sample_cov is None or self.sample_mean is None:
            raise ValueError("sample_cov and sample_mean must be initialized before calling recursive_update")
        if self.svd_U is None or self.svd_S is None:
            raise ValueError("svd_U and svd_S must be initialized before calling recursive_update")
        if self.groups is None:
            raise ValueError("groups must be initialized before calling recursive_update")

        # update buffer
        self.sample_total += len(new_samples)
        self.update_buffer(new_samples)
        # need at least 2 total samples for a meaningful covariance update
        if sample_num + len(new_samples) < 2:
            return
        # get new sample mean and covariance
        self.sample_mean, self.sample_cov = update_covariance(sample_num, self.sample_cov, self.sample_mean, new_samples)
        # new SVD on groups
        self.svd_U, self.svd_S = svd_groups(self.svd_U, self.svd_S, self.groups, self.sample_cov)

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

    def update_sample(self, position: np.ndarray):
        """
        Update current parameter position.

        Parameters
        ----------
        position : np.ndarray
            New parameter position, shape (ndim,).

        Examples
        --------
        >>> import numpy as np
        >>> new_position = np.array([1.2, -0.8])
        >>> stats.update_sample(new_position)
        >>> # Current position updated for next proposal
        """
        self.current_sample = position

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
    chain_stats: List['ChainStats']

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


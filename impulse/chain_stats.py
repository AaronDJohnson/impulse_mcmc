from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from impulse.online_updates import update_covariance, svd_groups
from impulse.sampler_state import SamplerState, PTState
from impulse.utils import shift_array

@dataclass
class ChainStats:
    """
    Data to be used to propose new samples
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
    current_state: Optional[np.ndarray] = None

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
        self._buffer = shift_array(self._buffer, -len(new_samples))
        self._buffer[-len(new_samples):] = new_samples
        if not self.buffer_full:
            if self.sample_total > self.buffer_size:
                self.buffer_full = True

    def recursive_update(self,
                         sample_num: int,
                         new_samples: np.ndarray
                         ) -> None:
        if self.sample_cov is None or self.sample_mean is None:
            raise ValueError("sample_cov and sample_mean must be initialized before calling recursive_update")
        if self.svd_U is None or self.svd_S is None:
            raise ValueError("svd_U and svd_S must be initialized before calling recursive_update")
        if self.groups is None:
            raise ValueError("groups must be initialized before calling recursive_update")

        # update buffer
        self.sample_total += len(new_samples)
        self.update_buffer(new_samples)
        # get new sample mean and covariance
        self.sample_mean, self.sample_cov = update_covariance(sample_num, self.sample_cov, self.sample_mean, self._buffer)
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
        self.current_sample = position

    # def update_temp(self, temperature: float):
    #     self.temp = temperature

@dataclass
class MultiChainStats:
    """
    Holds ChainStats for multiple chains (e.g., for PT)
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
        for i, cs in enumerate(self.chain_stats):
            cs.recursive_update(self.sample_total, new_samples[i])

    def get_group_U(self, chain_idx: int, group_idx: int) -> np.ndarray:
        """Return U for chain `chain_idx` and group `group_idx` (shape (k, k))."""
        return self.chain_stats[chain_idx].get_group_U(group_idx)

    def get_group_S(self, chain_idx: int, group_idx: int) -> np.ndarray:
        """Return singular values for chain `chain_idx` and group `group_idx` (shape (k,))."""
        return self.chain_stats[chain_idx].get_group_S(group_idx)

    def update_sample(self, state: SamplerState):
        for i, cs in enumerate(self.chain_stats):
            cs.update_sample(state.positions[i])

    # def update_temp(self, state: SamplerState):
    #     for i, cs in enumerate(self.chain_stats):
    #         cs.update_temp(state.temps[i])

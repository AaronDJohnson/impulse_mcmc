import numpy as np
from dataclasses import dataclass

@dataclass
class SamplerState:
    """
    Batched sampler state for all temperature chains.

    Fields are always arrays with leading dimension = ntemps:
      - positions: (ntemps, ndim)
      - lnlike:    (ntemps,)
      - lnprior:   (ntemps,)
      - lnprob:    (ntemps,)
      - accepted:  (ntemps,) int or bool
      - temp:      (ntemps,)
    """
    positions: np.ndarray  # shape (ntemps, ndim)
    lnlikes: np.ndarray  # shape (ntemps,)
    lnpriors: np.ndarray  # shape (ntemps,)
    lnprobs: np.ndarray  # shape (ntemps,)
    accepted: np.ndarray  # shape (ntemps,)
    temps: np.ndarray  # shape (ntemps,)

    @property
    def ntemps(self) -> int:
        return int(self.positions.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.positions.shape[1])

@dataclass
class PTState():
    """
    Parallel Tempering State
    Contains the PT ladder, the swap acceptance rate, and the number of swaps
    """
    ndim: int
    ntemps: int
    swap_steps: int = 1
    min_temp: float = 1.0
    max_temp: float|None = None
    temp_step: float|None = None
    nswaps: int = 1  # start at 1 to avoid divide by zero errors
    ladder: np.ndarray|None = None
    inf_temp: bool = False
    # adaptive temperature ladder parameters:
    adapt_t0: float = 100
    adapt_nu: float = 10

    def __post_init__(self):
        if self.ladder is None:
            self.ladder = self.compute_temp_ladder()
        self.swap_accept = np.zeros(self.ntemps - 1)  # swap acceptance between chains

    def compute_accept_ratio(self):
        return self.swap_accept / self.nswaps

    def compute_temp_ladder(self):
        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25% temperature swap acceptance rate
        on a multi-variate Gaussian.
        """
        if self.inf_temp:
            self.ntemps -= 1  # remove top value from ladder
        if self.temp_step is None and self.max_temp is None:
            self.temp_step = 1 + np.sqrt(2 / self.ndim)
        elif self.temp_step is None and self.max_temp is not None:
            self.temp_step = np.exp(np.log(self.max_temp / self.min_temp) / (self.ntemps - 1))
        temp_idxs = np.arange(self.ntemps)

        if self.temp_step is None:
            raise ValueError("temp_step is not initialized")

        if self.inf_temp:
            self.ntemps += 1 # add empty top value back to ladder
            ladder = self.min_temp * self.temp_step**temp_idxs  # compute ladder
            ladder = np.concatenate([ladder, [np.inf]])  # add inf value as top of ladder
        else:
            ladder = self.min_temp * self.temp_step**temp_idxs
        return ladder

    def adapt_ladder(self):
        """
        Adapt temperatures according to arXiv:1501.05823 <https://arxiv.org/abs/1501.05823>.
        """
        if self.ladder is None:
            raise ValueError("PTState ladder is not initialized")

        # Temperature adjustments with a hyperbolic decay.
        decay = self.adapt_t0 / (self.nswaps + self.adapt_t0)  # t0 / (t + t0)
        kappa = decay / self.adapt_nu  # 1 / nu
        # Construct temperature adjustments.
        accept_ratio = self.compute_accept_ratio()
        dscaled_accept = kappa * (accept_ratio[:-1] - accept_ratio[1:])  # delta acceptance ratios for chains
        # Compute new ladder (hottest and coldest chains don't move).
        delta_temps = np.diff(self.ladder[:-1])
        delta_temps *= np.exp(dscaled_accept)
        self.ladder[1:-1] = (np.cumsum(delta_temps) + self.ladder[0])

@dataclass
class ModelState:
    """
    Model state for reversible jump MCMC (not implemented yet)
    """
    pass

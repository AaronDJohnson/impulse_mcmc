import numpy as np
from dataclasses import dataclass
from impulse.mhsampler import MHState
from typing import Callable

@dataclass
class PTState():
    """
    Parallel Tempering State
    Contains the PT ladder, the swap acceptance rate, and the number of swaps
    """
    ndim: int
    ntemps: int
    swap_steps: int = 100
    min_temp: float = 1.0
    max_temp: float = None
    temp_step: float = None
    nswaps: int = 1  # start at 1 to avoid divide by zero errors
    ladder: np.ndarray = None
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


def pt_step(mhstates: list[MHState],
              ptstate: PTState,
              lnlike_fn: Callable,
              lnprior_fn: Callable,
              rng: np.random.Generator
              ) -> MHState:
    # set up map to help keep track of swaps
    ladder = ptstate.ladder
    swap_map = list(range(len(ladder)))
    log_likes = [mhstates[ii].lnlike for ii in range(len(ladder))]
    positions = [mhstates[ii].position for ii in range(len(ladder))]

    # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
    # and keep track of results in swap_map
    for swap_chain in reversed(range(len(ladder) - 1)):
        log_acc_ratio = -log_likes[swap_map[swap_chain]] / ladder[swap_chain]
        log_acc_ratio += -log_likes[swap_map[swap_chain + 1]] / ladder[swap_chain + 1]
        log_acc_ratio += log_likes[swap_map[swap_chain + 1]] / ladder[swap_chain]
        log_acc_ratio += log_likes[swap_map[swap_chain]] / ladder[swap_chain + 1]

        if np.log(rng.uniform()) <= log_acc_ratio:
            swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
            ptstate.swap_accept[swap_chain] += 1
            ptstate.nswaps += 1

        else:
            ptstate.nswaps += 1

    new_states = []
    # loop through the chains and record the new samples and log_Ls
    for ii in range(len(ladder)):
        new_position = positions[swap_map[ii]]
        new_loglike = log_likes[swap_map[ii]]
        new_logprior = lnprior_fn(new_position)
        new_lnprob = 1 / ladder[ii] * new_loglike + new_logprior
        new_state = MHState(new_position, new_loglike, new_logprior, new_lnprob, accepted=1, temp=ladder[ii])
        new_states.append(new_state)
    return new_states

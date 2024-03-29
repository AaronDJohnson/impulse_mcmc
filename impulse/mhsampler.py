from dataclasses import dataclass
from typing import Callable
import numpy as np

# Use MHState for everything:
# We can swap between different MHStates for parallel tempering
# 

@dataclass
class MHState:
    """
    Metropolis-Hastings State
    """
    position: np.ndarray
    lnlike: float
    lnprior: float
    lnprob: float
    accepted: int = 1
    temp: float = 1.0

def mh_kernel(state: MHState,
              prop_fn: Callable,
              lnlike_fn: Callable,
              lnprior_fn: Callable,
              rng: np.random.Generator,
              ) -> MHState:
    """
    Generate a MH kernel step
    """
    # propose a move
    x_star, qxy = prop_fn(state)

    # compute hastings ratio
    lnprior_star = lnprior_fn(x_star)

    if lnprior_star == -np.inf:  # if outside the prior, reject
        lnprob_star = -np.inf
    else:
        lnlike_star = lnlike_fn(x_star)
        lnprob_star = 1 / state.temp * lnlike_star + lnprior_star

    hastings_ratio = lnprob_star - (state.lnprob) + qxy
    rand_num = rng.uniform()
    # accept/reject step
    if np.log(rand_num) < hastings_ratio:
        return MHState(x_star, lnlike_star, lnprior_star, 1 / state.temp * lnlike_star + lnprior_star, accepted=1, temp=state.temp)
    else:
        return MHState(state.position, state.lnlike, state.lnprior, state.lnprob, accepted=0, temp=state.temp)

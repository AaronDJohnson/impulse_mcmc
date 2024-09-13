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


def mh_step(state: MHState,
              prop_fn: Callable,
              lnlike_fn: Callable,
              lnprior_fn: Callable,
              rng: np.random.Generator,
              ) -> MHState:
    """
    Generate a Metropolis Hastings step
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


def vectorized_mh_step(states: list[MHState],
                        prop_fn: Callable,
                        lnlike_fn: Callable,
                        lnprior_fn: Callable,
                        rng: np.random.Generator,
                        ) -> list[MHState]:
    """
    Generate a MH kernel step with the likelihood and prior vectorized

    This means that the likelihood and prior functions return VectorizedMHState.num_temps values
    """
    # propose a move
    results = [prop_fn(state) for state in states]
    x_stars, qxys = zip(*results)
    temperatures = np.array([state.temp for state in states])
    old_lnprobs = np.array([state.lnprob for state in states])

    # compute hastings ratio
    x_star_array = np.hstack(x_stars)  # TODO: hstack or vstack?
    print(x_star_array.shape)
    lnlike_stars = lnlike_fn(x_star_array)
    lnprior_stars = lnprior_fn(x_star_array)
    
    lnprob_stars = np.where(lnprior_stars == -np.inf, -np.inf, 1 / temperatures * lnlike_stars + lnprior_stars)

    hastings_ratios = lnprob_stars - (old_lnprobs) + qxys
    rand_num = rng.uniform(size=len(states))

    # accept/reject step
    return np.where(
        np.log(rand_num) < hastings_ratios,
        [MHState(x_star, lnlike_star, lnprior_star, 1 / state.temp * lnlike_star + lnprior_star, accepted=1, temp=state.temp) for x_star, lnlike_star, lnprior_star, state in zip(x_stars, lnlike_stars, lnprior_stars, states)],
        states
    )

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
                        prop_fns: list[Callable],
                        lnlike_fn: Callable,
                        lnprior_fn: Callable,
                        rng: np.random.Generator,
                        ndim: int,
                        ) -> list[MHState]:
    """
    Generate a MH kernel step with the likelihood and prior vectorized

    This means that the likelihood and prior functions return VectorizedMHState.num_temps values
    """
    # propose a move
    results = [prop_fns[ii](states[ii]) for ii in range(len(states))]
    x_stars, qxys = zip(*results)
    x_stars, qxys = np.array(x_stars).flatten(), np.array(qxys)

    temperatures = np.array([state.temp for state in states])
    old_lnprobs = np.array([float(state.lnprob) for state in states])

    # compute hastings ratio
    # x_star_array = np.array(x_stars)  # TODO: hstack or vstack?

    lnlike_stars = lnlike_fn(x_stars)
    lnprior_stars = lnprior_fn(x_stars)

    lnprob_stars = np.where(lnprior_stars == -np.inf, -np.inf, 1 / temperatures * lnlike_stars + lnprior_stars)

    # draw random numbers for accpetance step
    hastings_ratios = lnprob_stars - (old_lnprobs) + qxys
    log_rand_nums = np.log(rng.uniform(size=len(states)))

    # reshape to the original shape
    x_stars = x_stars.reshape(-1, ndim)

    # for hastings_ratio in hastings_ratios:
    #     print("hastings =", hastings_ratio)

    # accept/reject step
    return [MHState(x_star, lnlike_star, lnprior_star, 1 / state.temp * lnlike_star + lnprior_star, accepted=1, temp=state.temp) if (log_rand_num < hastings_ratio) else MHState(state.position, state.lnlike, state.lnprior, state.lnprob, accepted=0, temp=state.temp) for x_star, lnlike_star, lnprior_star, state, log_rand_num, hastings_ratio in zip(x_stars, lnlike_stars, lnprior_stars, states, log_rand_nums, hastings_ratios)]

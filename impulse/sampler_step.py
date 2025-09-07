from typing import Callable
import numpy as np
from impulse.sampler_state import SamplerState, PTState
from impulse.proposals import ProposalBundle


def vectorized_mh_step(state: SamplerState,
                       prop_fn: ProposalBundle,
                       lnlike_fn: Callable,
                       lnprior_fn: Callable,
                       rng: np.random.Generator,
                       ) -> SamplerState:
    """
    Generate a Metropolis Hastings step for all temperatures
    """
    # propose a set of new proposals
    x_stars, qxys = prop_fn(state)

    # compute new prior
    lnprior_stars = lnprior_fn(x_stars)
    lnlike_stars = lnlike_fn(x_stars)
    lnprob_stars = 1 / state.temps * lnlike_stars + lnprior_stars

    probability_ratios = lnprob_stars - (state.lnprobs) + qxys
    rand_num = rng.uniform(size=len(state.temps))
    # accept/reject step
    accepts = np.log(rand_num) < probability_ratios

    new_positions = np.where(accepts[:, None], x_stars, state.positions)
    new_lnlikes = np.where(accepts, lnlike_stars, state.lnlikes)
    new_lnpriors = np.where(accepts, lnprior_stars, state.lnpriors)
    new_lnprobs = np.where(accepts, lnprob_stars, state.lnprobs)
    new_accepted = accepts.astype(int)
    return SamplerState(new_positions, new_lnlikes, new_lnpriors, new_lnprobs, new_accepted, state.temps)


def pt_step(state: SamplerState,
              ptstate: PTState,
              lnlike_fn: Callable,
              lnprior_fn: Callable,
              rng: np.random.Generator
              ) -> SamplerState:
    # set up map to help keep track of swaps
    ladder = ptstate.ladder
    if ladder is None:
        raise ValueError("PTState ladder is not initialized")
    swap_map = list(range(len(ladder)))
    log_likes = state.lnlikes
    positions = state.positions

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

    # loop through the chains and record the new samples and log_Ls
    new_positions = positions[swap_map]
    new_loglikes = log_likes[swap_map]
    new_logpriors = lnprior_fn(new_positions)
    new_lnprobs = 1 / ladder * new_loglikes + new_logpriors
    new_accepted = np.ones(len(ladder), dtype=int)  # all ones for this one (PT swaps are handled separately)
    return SamplerState(new_positions, new_loglikes, new_logpriors, new_lnprobs, new_accepted, ladder)

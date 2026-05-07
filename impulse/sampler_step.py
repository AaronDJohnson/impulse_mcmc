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
    Execute one Metropolis-Hastings step for all temperature chains simultaneously.

    Generates proposals for all temperature chains, evaluates likelihoods and priors,
    computes acceptance probabilities, and updates the sampler state.

    Parameters
    ----------
    state : SamplerState
        Current state containing positions, log-likelihoods, log-priors, and temperatures.
    prop_fn : ProposalBundle
        Collection of proposal distributions for generating new candidate positions.
    lnlike_fn : callable
        Log-likelihood function wrapped with _function_wrapper.
    lnprior_fn : callable
        Log-prior function wrapped with _function_wrapper.
    rng : np.random.Generator
        Random number generator for acceptance decisions.

    Returns
    -------
    SamplerState
        Updated sampler state with new positions and statistics.

    Examples
    --------
    >>> # Typically called within the main sampling loop
    >>> new_state = vectorized_mh_step(current_state, proposals, 
    ...                               likelihood_fn, prior_fn, rng)
    >>> acceptance_rate = new_state.accepted.mean()

    Notes
    -----
    - Proposals are generated simultaneously for all temperature chains
    - Acceptance probabilities account for temperature scaling: β = 1/T
    - Proposal ratios (qxy) are included in acceptance calculation
    - Only accepted proposals update positions and log-probability values
    """
    # propose a set of new positions
    x_stars, qxys = prop_fn(state)

    # compute new prior first; skip the likelihood for out-of-bounds rows
    lnprior_stars = lnprior_fn(x_stars)
    finite = np.isfinite(lnprior_stars)
    lnprior_stars = np.where(finite, lnprior_stars, -np.inf)

    lnlike_stars = np.full(len(x_stars), -np.inf)
    if np.any(finite):
        lnlike_stars[finite] = lnlike_fn(x_stars[finite])

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
    """
    Perform parallel tempering swap attempts between adjacent temperature chains.

    Proposes swaps between neighboring temperature chains and accepts/rejects
    based on the detailed balance condition for parallel tempering.

    Parameters
    ----------
    state : SamplerState
        Current state with positions and log-likelihoods for all chains.
    ptstate : PTState
        Parallel tempering state containing temperature ladder and swap statistics.
    lnlike_fn : callable
        Log-likelihood function (used to recompute likelihoods after swaps).
    lnprior_fn : callable
        Log-prior function (used to recompute priors after swaps).
    rng : np.random.Generator
        Random number generator for swap acceptance decisions.

    Returns
    -------
    SamplerState
        Updated state with potentially swapped chain configurations.

    Examples
    --------
    >>> # Called periodically during sampling
    >>> if iteration % swap_steps == 0:
    ...     state = pt_step(state, ptstate, likelihood_fn, prior_fn, rng)
    >>> swap_acceptance = ptstate.swap_accept.sum() / ptstate.nswaps

    Notes
    -----
    - Swaps are proposed starting from the hottest chain down to coldest
    - Acceptance probability: min(1, exp(Δβ × ΔE)) where Δβ = 1/T_i - 1/T_j
    - Swap statistics are updated in ptstate for monitoring efficiency
    - All chain positions are reordered after successful swaps
    """
    # set up map to help keep track of swaps
    ladder = ptstate.ladder
    if ladder is None:
        raise ValueError("PTState ladder is not initialized")
    swap_map = list(range(len(ladder)))
    log_likes = state.lnlikes
    log_priors = state.lnpriors
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

    # increment once per sweep, not per pair
    ptstate.nswaps += 1

    # permute positions, likelihoods, and priors according to swap map
    new_positions = positions[swap_map]
    new_loglikes = log_likes[swap_map]
    new_logpriors = log_priors[swap_map]
    new_lnprobs = 1 / ladder * new_loglikes + new_logpriors
    new_accepted = np.ones(len(ladder), dtype=int)  # all ones for this one (PT swaps are handled separately)
    return SamplerState(new_positions, new_loglikes, new_logpriors, new_lnprobs, new_accepted, ladder)

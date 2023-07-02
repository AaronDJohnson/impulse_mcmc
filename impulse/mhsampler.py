from dataclasses import dataclass
import numpy as np

@dataclass
class MHState():
    position: np.ndarray
    lnlike: float
    lnprior: float




def mh_step(x0, lnlike0, lnprob0, lnlike_fn, lnprior_fn,
            prop_fn, rng, temp=1.):
    """
    Parallel tempered Metropolis Hastings step (temp=1 => standard MH step).

    x0: vector of length ndim
    lnlike_fn: log likelihood function
    lnprior_fn: log prior function
    """
    # propose a move
    x_star, qxy = prop_fn(x0, temp)

    # compute hastings ratio
    lnprior_star = lnprior_fn(x_star)

    if lnprior_star == -np.inf:  # if outside the prior, reject
        lnprob_star = -np.inf
    else:
        lnlike_star = lnlike_fn(x_star)
        lnprob_star = 1 / temp * lnlike_star + lnprior_star

    hastings_ratio = lnprob_star - lnprob0 + qxy
    rand_num = rng.uniform()
    # accept/reject step
    if np.log(rand_num) < hastings_ratio:
        accepted = 1
        return x_star, lnlike_star, lnprob_star, accepted
    else:
        accepted = 0
        return x0, lnlike0, lnprob0, accepted

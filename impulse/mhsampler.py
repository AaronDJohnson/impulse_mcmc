import os
import numpy as np
from impulse.random_nums import rng


class MHSampler(object):
    def __init__(self, ndim, lnlike_fn, lnprior_fn, prop_fn, lnlike_kwargs={},
                 lnprior_kwargs={}, iterations=1000):
        """
        x0: vector of length ndim
        lnlike_fn: log likelihood function
        lnpost_fn: log prior function

        """
        self.ndim = ndim

        self.lnlike_fn = lnlike_fn
        self.lnprior_fn = lnprior_fn
        self.prop_fn = prop_fn
        self.lnlike_kwargs = lnlike_kwargs
        self.lnprior_kwargs = lnprior_kwargs
        self.iterations = iterations
        self.num_samples = 0  # running total of all samples

        # initialize chain, acceptance rate, and lnprob
        self.chain = np.zeros((self.iterations, self.ndim))
        self.lnlike = np.zeros(iterations)
        self.lnprob = np.zeros(iterations)
        self.accept_rate = np.zeros(iterations)
        self.naccept = 0

    def sample(self, x0, temp=1.):
        # first sample
        self.chain[0] = x0
        lnlike0 = 1 / temp * self.lnlike_fn(x0, **self.lnlike_kwargs)
        lnprior0 = self.lnprior_fn(x0, **self.lnlike_kwargs)
        self.lnprob0 = lnlike0 + lnprior0
        self.lnprob[0] = self.lnprob0
        self.lnlike[0] = lnlike0
        self.num_samples += 1

        for ii in range(1, self.iterations):
            self.num_samples += 1
            # propose a move
            x_star, factor = self.prop_fn(x0)
            # x_star = x_star
            # draw random number
            rand_num = rng.uniform()

            # compute hastings ratio
            lnprior_star = self.lnprior_fn(x_star, **self.lnprior_kwargs)
            if np.isinf(lnprior_star):
                lnprob_star = -np.inf
                lnlike_star = self.lnlike[ii - 1]
            else:
                lnlike_star = 1 / temp * self.lnlike_fn(x_star, **self.lnlike_kwargs)
                lnprob_star = lnprior_star + lnlike_star

            hastings_ratio = lnprob_star - self.lnprob0 + factor

            # accept/reject step
            if np.log(rand_num) < hastings_ratio:
                x0 = x_star
                self.lnprob0 = lnprob_star
                self.naccept += 1

            # update chain
            self.chain[ii] = x0
            self.lnprob[ii] = self.lnprob0
            self.lnlike[ii] = lnlike_star
            self.accept_rate[ii] = self.naccept / self.num_samples

        return self.chain, self.lnlike, self.lnprob, self.accept_rate, x0

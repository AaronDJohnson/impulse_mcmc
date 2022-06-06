import os
import numpy as np
from impulse.random_nums import rng


class MHSampler(object):
    def __init__(self, x0, lnlike_fn, lnprior_fn, prop_fn, lnlike_kwargs={},
                 lnprior_kwargs={}, iterations=1000, init_temp=1.):
        """
        x0: vector of length ndim
        lnlike_fn: log likelihood function
        lnpost_fn: log prior function

        """
        self.x0 = x0
        self.ndim = len(x0)
        self.temp = init_temp

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

        # pickup sample
        lnlike0 = self.lnlike_fn(x0, **self.lnlike_kwargs)
        lnprior0 = self.lnprior_fn(x0, **self.lnlike_kwargs)
        self.lnprob0 = 1 / self.temp * lnlike0 + lnprior0
        self.num_samples += 1

    def set_x0(self, x0, lnprob0, temp=1.):
        # set x0 and temp!
        self.temp = temp
        self.x0 = x0
        self.lnprob0 = lnprob0
        self.lnprob0 = 1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0)
        # print(1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0))
        # print(self.lnprob0)
        # print(np.abs(1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0) - self.lnprob0) / self.lnprob0)

    def sample(self):

        for ii in range(self.iterations):
            self.num_samples += 1
            # propose a move
            x_star, factor = self.prop_fn(self.x0, self.temp)
            # x_star = x_star
            # draw random number
            rand_num = rng.uniform()

            # compute hastings ratio
            lnprior_star = self.lnprior_fn(x_star, **self.lnprior_kwargs)
            if np.isinf(lnprior_star):
                lnprob_star = -np.inf
                lnlike_star = self.lnlike[ii - 1]
            else:
                lnlike_star = self.lnlike_fn(x_star, **self.lnlike_kwargs)
                lnprob_star = 1 / self.temp * lnlike_star + lnprior_star

            hastings_ratio = lnprob_star - self.lnprob0 + factor

            # accept/reject step
            if np.log(rand_num) < hastings_ratio:
                self.x0 = x_star
                self.lnprob0 = lnprob_star
                self.naccept += 1

            # update chain
            self.chain[ii] = self.x0
            self.lnprob[ii] = self.lnprob0
            self.lnlike[ii] = lnlike_star
            self.accept_rate[ii] = self.naccept / self.num_samples

        return self.chain, self.lnlike, self.lnprob, self.accept_rate
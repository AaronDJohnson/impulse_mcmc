import os
import numpy as np
from impulse.random_nums import rng


class MHSampler(object):
    def __init__(self, x0, lnlike_fn, lnprior_fn, prop_fn, lnlike_kwargs={},
                 lnprior_kwargs={}, prop_kwargs={}, iterations=1000):
        """
        x0: vector of length ndim
        lnlike_fn: log likelihood function
        lnpost_fn: log prior function
        num_iters: number of iterations to perform
        """
        self.ndim = len(x0)

        # self.cov = cov
        # if cov is None:
        #     self.cov = np.diag(np.ones(self.ndim) * 0.01**2)
        self.lnlike_fn = lnlike_fn
        self.lnprior_fn = lnprior_fn
        self.prop_fn = prop_fn
        self.lnlike_kwargs = lnlike_kwargs
        self.lnprior_kwargs = lnprior_kwargs
        self.prop_kwargs = prop_kwargs
        self.iterations = iterations
        self.x0 = x0
        self.num_runs = 0

        # initialize chain, acceptance rate, and lnprob
        self.chain = np.zeros((self.iterations, self.ndim))
        self.lnprob = np.zeros(iterations)
        self.accept_rate = np.zeros(iterations)

        # first sample
        self.chain[0] = x0
        lnlike0 = self.lnlike_fn(x0, **self.lnlike_kwargs)
        lnprior0 = self.lnprior_fn(x0, **self.lnlike_kwargs)
        self.lnprob0 = lnlike0 + lnprior0
        self.lnprob[0] = self.lnprob0

    def sample(self):
        naccept = 0
        x0 = self.chain[self.num_runs * self.iterations]
        self.num_runs += 1
        for ii in range(1, self.iterations):

            # propose a move
            x_star, factor = self.prop_fn(x0, **self.prop_kwargs)
            # x_star = x_star
            # draw random number
            rand_num = rng.uniform()

            # compute hastings ratio
            lnprior_star = self.lnprior_fn(x_star, **self.lnprior_kwargs)
            if np.isinf(lnprior_star):
                lnprob_star = -np.inf
            else:
                lnlike_star = self.lnlike_fn(x_star, **self.lnlike_kwargs)
                lnprob_star = lnprior_star + lnlike_star

            hastings_ratio = lnprob_star - self.lnprob0 + factor

            # accept/reject step
            if np.log(rand_num) < hastings_ratio:
                x0 = x_star
                self.lnprob0 = lnprob_star
                naccept += 1

            # update chain
            self.chain[ii] = x0
            self.lnprob[ii] = self.lnprob0
            self.accept_rate[ii] = naccept / ii

        return self.chain, self.accept_rate, self.lnprob

    def save_samples(self, outdir):
        # make directory if it doesn't exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = outdir + '/chain_1.txt'
        data = np.column_stack((self.chain, self.lnprob, self.accept_rate))
        with open(filename, 'a+') as fname:
            np.savetxt(fname, data)



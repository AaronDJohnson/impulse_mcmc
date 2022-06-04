import os
import numpy as np
from impulse.random_nums import rng


class PTSampler(object):
    def __init__(self, x0, lnlike_fn, lnprior_fn, prop_fn, temp, lnlike_kwargs={},
                 lnprior_kwargs={}, prop_kwargs={}, iterations=1000):
        """
        x0: vector of length ndim
        lnlike_fn: log likelihood function
        lnpost_fn: log prior function
        num_iters: number of iterations to perform
        """
        self.ndim = len(x0)
        self.temp = temp

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
        self.lnlike = np.zeros(iterations)
        self.lnprob = np.zeros(iterations)

        # first sample
        self.chain[0] = x0
        lnlike0 = 1 / self.temp * self.lnlike_fn(x0, **self.lnlike_kwargs)
        lnprior0 = self.lnprior_fn(x0, **self.lnlike_kwargs)
        self.lnprob0 = lnlike0 + lnprior0
        self.lnprob[0] = self.lnprob0
        self.lnlike[0] = lnlike0

    def sample(self):
        naccept = 0
        # x0 = self.chain[self.num_runs * self.iterations]
        self.num_runs += 1
        for ii in range(1, self.iterations):

            # propose a move
            x_star, factor = self.prop_fn(self.x0, **self.prop_kwargs)
            # x_star = x_star
            # draw random number
            rand_num = rng.uniform()

            # compute hastings ratio
            lnprior_star = self.lnprior_fn(x_star, **self.lnprior_kwargs)
            if np.isinf(lnprior_star):
                lnprob_star = -np.inf
            else:
                lnlike_star = 1 / self.temp * self.lnlike_fn(x_star, **self.lnlike_kwargs)
                lnprob_star = lnprior_star + lnlike_star

            hastings_ratio = lnprob_star - self.lnprob0 + factor

            # accept/reject step
            if np.log(rand_num) < hastings_ratio:
                self.x0 = x_star
                self.lnprob0 = lnprob_star
                naccept += 1

            # update chain
            self.chain[ii] = self.x0
            self.lnprob[ii] = self.lnprob0
            self.lnlike[ii] = lnlike_star
            # self.accept_rate[ii] = naccept / ii

        return self.chain, self.lnlike

    def save_samples(self, outdir, filename='/chain_1.txt'):
        # make directory if it doesn't exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = outdir + filename
        data = np.column_stack((self.chain, self.lnprob, self.lnlike))
        with open(filename, 'a+') as fname:
            np.savetxt(fname, data)


def temp_ladder(tmin, ndim, ntemps, tmax=None, tstep=None):
    """
    Method to compute temperature ladder. At the moment this uses
    a geometrically spaced temperature ladder with a temperature
    spacing designed to give 25 % temperature swap acceptance rate.
    """

    if tstep is None and tmax is None:
        tstep = 1 + np.sqrt(2 / ndim)
    elif tstep is None and tmax is not None:
        tstep = np.exp(np.log(tmax / tmin) / (ntemps - 1))
    ii = np.arange(ntemps)
    ladder = tmin * tstep**ii

    return ladder


def propose_swaps(chain, lnlike, ladder):
    lnchainswap = (1 / ladder[:-1] - 1 / ladder[1:]) * (lnlike[-1, :-1] - lnlike[-1, 1:])
    nums = np.log(rng.random(size=len(ladder)))
    for idx in np.where(lnchainswap > nums)[0]:  # this could be done without a for loop
        chain[-1, :, [idx - 1, idx]] = chain[-1, :, [idx, idx - 1]]
    return chain




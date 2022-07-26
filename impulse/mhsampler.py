import numpy as np
from impulse.random_nums import rng
import ray


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

    def set_x0(self, x0, lnprob0, temp=1.):
        # set x0 and temp!
        self.temp = temp
        self.x0 = x0
        self.lnprob0 = 1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0)
        # print(1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0))
        # print(self.lnprob0)
        # print(np.abs(1 / self.temp * self.lnlike_fn(x0) + self.lnprior_fn(x0) - self.lnprob0) / self.lnprob0)

    def sample(self):

        for ii in range(self.iterations):
            self.num_samples += 1
            # propose a move
            x_star, factor = self.prop_fn(self.x0, self.temp)
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

    def get_num_samples(self):
        return self.num_samples


def mh_sample_step(lnlike_fn, lnprior_fn, prop_fn, x0, temp,
                   iterations, lnprob0, chain, lnlike, lnprob,
                   accept_rate, lnlike_kwargs={}, lnprior_kwargs={}):
    # TODO (Aaron): Fix accept_rate
    chain = np.copy(chain)
    lnprob = np.copy(lnprob)
    lnlike = np.copy(lnlike)
    accept_rate = np.copy(accept_rate)
    lnprob0 = np.copy(lnprob0)
    x0 = np.copy(x0)

    naccept = 0
    num_samples = 0
    for ii in range(iterations):
        num_samples += 1
        # propose a move
        x_star, factor = prop_fn(x0, temp)
        # draw random number
        rand_num = rng.uniform()

        # compute hastings ratio
        lnprior_star = lnprior_fn(x_star, **lnprior_kwargs)
        if np.isinf(lnprior_star):
            lnprob_star = -np.inf
            lnlike_star = lnlike[ii - 1]
        else:
            lnlike_star = lnlike_fn(x_star, **lnlike_kwargs)
            lnprob_star = 1 / temp * lnlike_star + lnprior_star

        hastings_ratio = lnprob_star - lnprob0 + factor

        # accept/reject step
        if np.log(rand_num) < hastings_ratio:
            x0 = x_star
            lnprob0 = lnprob_star
            naccept += 1

        # update chain
        chain[ii] = x0
        lnprob[ii] = lnprob0
        lnlike[ii] = lnlike_star
        accept_rate[ii] = naccept / num_samples
    return chain, lnlike, lnprob, accept_rate


@ray.remote(num_cpus=1)
def parallel_mh_sample_step(lnlike_fn, lnprior_fn, prop_fn, x0, temp,
                            iterations, lnprob0, chain, lnlike, lnprob,
                            accept_rate, lnlike_kwargs={}, lnprior_kwargs={}):
    return mh_sample_step(lnlike_fn, lnprior_fn, prop_fn, x0, temp,
                          iterations, lnprob0, chain, lnlike, lnprob,
                          accept_rate, lnlike_kwargs=lnlike_kwargs,
                          lnprior_kwargs=lnprior_kwargs)

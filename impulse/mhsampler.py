import numpy as np

# class MHSampler():
#     def __init__(self, x0, jump, logl, logp, rng, temp=1.):
#         self.rng = rng
#         self.sample_count = 1
#         self.x0 = x0
#         self.jump = jump
#         self.logl = logl
#         self.logp = logp
#         self.temp = temp
#         self.lnlike0 = logl(self.x0)
#         self.lnprob0 = 1 / self.temp * self.lnlike0 + self.logp(self.x0)

#         self.lnlike_arr = np.zeros(num_samples)
#         self.lnprob_arr = np.zeros(num_samples)
#         self.x_arr = np.zeros((num_samples, len(self.x0)))
#         self.accept = np.zeros(num_samples)

#     def set_sample(self, x0, temp=1.):
#         # set x0 and temp!
#         # update lnprob0 and lnlike0
#         self.temp = temp
#         self.x0 = x0
#         self.lnlike0 = self.logl(self.x0)
#         self.lnprob0 = 1 / self.temp * self.lnlike0 + self.logp(self.x0)

#     def sample(self, num_samples):
#         for ii in range(num_samples):
#             self.x0, self.lnlike0, self.lnprob0, self.accepted = mh_step(self.x0, self.lnlike0, self.lnprob0, self.logl,
#                                                                          self.logp, self.jump, self.rng, self.temp)
#             self.lnlike_arr[ii] = self.lnlike0
#             self.lnprob_arr[ii] = self.lnprob0
#             self.x_arr[ii] = self.x0
#             self.accept[ii] = self.accept
#             self.sample_count += 1
#         return self.x_arr, self.lnlike_arr, self.lnprob_arr, self.accept


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

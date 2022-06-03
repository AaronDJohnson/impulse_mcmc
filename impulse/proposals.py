import numpy as np
from numpy.linalg import LinAlgError

from impulse.batch_updates import update_covariance, svd_groups
from impulse.random_nums import rng


class JumpProposals():
    def __init__(self, ndim, buf_size=1000, groups=None, cov=None, mean=None):
        """
        ndim (int): number of dimensions in the parameter space
        """
        self.prop_list = []
        self.prop_weights = []
        self.prop_probs = []
        self.ndim = ndim
        self._buffer = np.zeros((buf_size, ndim))
        self._fullbuffer = False  # check if enough samples for DE jumps

        # setup sampling groups
        self.groups = groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]

        # set up sample covariance matrix
        self.cov = cov
        if cov is None:
            self.cov = np.identity(ndim) * 0.01**2
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)

        # do SVD on param groups
        self.U, self.S = svd_groups(self.U, self.S, self.groups, self.cov)

        self.mean = mean
        if mean is None:
            self.mean = 0

    def add_jump(self, jump, weight):
        self.prop_list.append(jump)
        self.prop_weights.append(weight)
        self.prop_probs = self.prop_weights / sum(self.prop_weights)

    def recursive_update(self, sample_num, new_chain):
        # update buffer
        if sample_num > len(self._buffer):
            self._fullbuffer = True
        self._buffer = new_chain

        # get new sample mean and covariance
        self.mean, self.cov = update_covariance(sample_num, self.cov, self.mean, self._buffer)

        # new SVD on groups
        self.U, self.S = svd_groups(self.U, self.S, self.groups, self.cov)

    def __call__(self, x):
        if sum(self.prop_probs) != 1:
            self.prop_probs /= sum(self.prop_probs)  # normalize the probabilities
        proposal = rng.choice(self.prop_list, p=self.prop_probs)
        return proposal(x)


# class ProposalMix():
#     def __init__(self, ndim, prop_list, cov=None, mean=None):
#         """
#         ndim (int): number of dimensions in the parameter space
#         prop_list (list of tuples): (Proposal, weight)
#         """
#         self.ndim = ndim
#         self.cov = cov
#         self.mean = mean
#         if cov is None:
#             self.cov = np.identity(ndim) * 0.01**2
#         if mean is None:
#             self.mean = 0
#         self.prop_list = list(map(list, zip(*prop_list)))

#     def update(self, old_length, new_chain, **kwargs):
#         chain = kwargs.get('full_chain', None)
#         self.mean, self.cov = update_covariance(old_length, self.cov, self.mean, new_chain)
#         for proposal in self.prop_list[0]:
#             proposal.update(cov=self.cov, chain=chain)

#     def __call__(self, x):
#         proposal = rng.choice(self.prop_list[0], p=self.prop_list[1])
#         return proposal(x)


# class GaussianProposal():
#     def __init__(self, sigma=0.1):
#         self.sigma = sigma

#     def update(self, **kwargs):
#         self.sigma = kwargs.get('sigma', 0.1)

#     def __call__(self, x):
#         # draw x_star
#         x_star = x + rng.standard_normal(len(x)) * self.sigma

#         # proposal ratio factor is 1 (symmetric jump)
#         factor = 1

#         return x_star, factor


# class PriorProposal():
#     """
#     Are you feeling lucky? Uniform prior draws.
#     """
#     def __init__(self, prior_min, prior_max, rng=rng):
#         """
#         prior_min: vector containing mins of priors
#         prior_max: vector containing maxes of priors
#         """
#         self.rng = rng
#         self.prior_min = prior_min
#         self.prior_max = prior_max

#     def update(self, **kwargs):
#         return None

#     def __call__(self, x):
#         x_star = rng.uniform(self.prior_min, self.prior_max)
#         return x_star, 0


def am(x, U, S, groups, temp, buffer):
    """
    Adaptive Jump Proposal. This function will occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(groups))
    # jumpind = np.random.randint(0, len(groups))

    # adjust step size
    prob = rng.random()
    # prob = np.random.rand()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:
    #    scale = 0.5

    # standard medium jump
    else:
        scale = 1.0

    # adjust scale based on temperature
    if temp <= 100:
        scale *= np.sqrt(temp)

    # get parmeters in new diagonalized basis
    y = np.dot(U[jumpind].T, x[groups[jumpind]])

    # make correlated componentwise adaptive jump
    ind = np.arange(len(groups[jumpind]))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    y[ind] = y[ind] + rng.standard_normal(neff) * cd * np.sqrt(S[jumpind][ind])
    q[groups[jumpind]] = np.dot(U[jumpind], y)

    return q, qxy


def scam(x, U, S, groups, temp, buffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(groups))
    ndim = len(groups[jumpind])

    # adjust step size
    # prob = np.random.rand()
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:

    # standard medium jump
    else:
        scale = 1.0

    # scale = np.random.uniform(0.5, 10)

    # adjust scale based on temperature
    if temp <= 100:
        scale *= np.sqrt(temp)

    # get parmeters in new diagonalized basis
    # y = np.dot(self.U.T, x[self.covinds])

    # make correlated componentwise adaptive jump
    # ind = np.unique(np.random.randint(0, ndim, 1))
    ind = np.unique(rng.integers(0, ndim, 1))

    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    # y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
    # q[self.covinds] = np.dot(self.U, y)
    q[groups[jumpind]] += (
        rng.standard_normal() * cd * np.sqrt(S[jumpind][ind]) * U[jumpind][:, ind].flatten()
    )

    return q, qxy


def de(x, U, S, groups, temp, buffer):
    """
    Differential Evolution Jump. This function will  occasionally
    use different jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain

    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    # get old parameters
    q = x.copy()
    qxy = 0

    # choose group
    jumpind = np.random.randint(0, len(groups))
    ndim = len(groups[jumpind])

    bufsize = len(buffer)

    # draw a random integer from 0 - iter
    # mm = np.random.randint(0, bufsize)
    # nn = np.random.randint(0, bufsize)
    mm = rng.integers(0, bufsize)
    nn = rng.integers(0, bufsize)

    # make sure mm and nn are not the same iteration
    while mm == nn:
        nn = rng.integers(0, bufsize)

    # get jump scale size
    prob = rng.random()

    # mode jump
    if prob > 0.5:
        scale = 1.0

    else:
        scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(temp)

    for ii in range(ndim):

        # jump size
        sigma = buffer[mm, groups[jumpind][ii]] - buffer[nn, groups[jumpind][ii]]

        # jump
        q[groups[jumpind][ii]] += scale * sigma

    return q, qxy

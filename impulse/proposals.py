import numpy as np

from impulse.batch_updates import update_covariance, svd_groups
from impulse.random_nums import rng
from loguru import logger


def shift_array(arr, num, fill_value=0.):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class JumpProposals():
    def __init__(self, ndim, buf_size=50000, groups=None, cov=None,
                 mean=None):
        """
        ndim (int): number of dimensions in the parameter space
        """
        self.nsamples = 0  # sample count
        self.buf_size = buf_size
        self.prop_list = []
        self.prop_weights = []
        self.prop_probs = []
        self.ndim = ndim
        self._buffer = np.zeros((buf_size, ndim))

        # setup sampling groups
        self.groups = groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]

        # set up sample covariance matrix
        self.cov = cov
        if cov is None:
            self.cov = np.identity(ndim)
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)
        # logger.debug('groups: {}'.format(self.groups))

        # do SVD on param groups
        self.U, self.S = svd_groups(self.U, self.S, self.groups, self.cov)
        # self.U, self.S = svd_groups(self.groups, self.cov)

        self.mean = mean
        if mean is None:
            self.mean = 0

    def add_jump(self, jump, weight):
        self.prop_list.append(jump)
        self.prop_weights.append(weight)
        self.prop_probs = np.array(self.prop_weights) / sum(self.prop_weights)  # normalize probabilities

    def recursive_update(self, sample_num, new_chain):
        # update buffer
        self._buffer = shift_array(self._buffer, -len(new_chain))
        self._buffer[-len(new_chain):] = new_chain

        # get new sample mean and covariance
        self.mean, self.cov = update_covariance(sample_num, self.cov, self.mean, self._buffer)

        # new SVD on groups
        self.U, self.S = svd_groups(self.U, self.S, self.groups, self.cov)
        # self.U, self.S = svd_groups(self.groups, self.cov)

    def __call__(self, x, temp=1.):
        self.nsamples += 1
        proposal = rng.choice(self.prop_list, p=self.prop_probs)
        # don't let DE jumps happen until after buffer is full
        while proposal.__name__ == 'de' and self.nsamples < self.buf_size:
            proposal = rng.choice(self.prop_list, p=self.prop_probs)
        return proposal(x, self.U, self.S, self.groups, temp, self._buffer)


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

import numpy as np
from numpy.linalg import LinAlgError

from impulse.batch_updates import update_covariance, svd_groups
from impulse.random_nums import rng


class JumpProposals():
    def __init__(self, ndim, buf_size=10000, groups=None, cov=None, mean=None):
        """
        ndim (int): number of dimensions in the parameter space
        """
        self.prop_list = []
        self.ndim = ndim
        self._buffer = np.zeros((buf_size, ndim))

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
        self.prop_list.append()




class ProposalMix():
    def __init__(self, ndim, prop_list, cov=None, mean=None):
        """
        ndim (int): number of dimensions in the parameter space
        prop_list (list of tuples): (Proposal, weight)
        """
        self.ndim = ndim
        self.cov = cov
        self.mean = mean
        if cov is None:
            self.cov = np.identity(ndim) * 0.01**2
        if mean is None:
            self.mean = 0
        self.prop_list = list(map(list, zip(*prop_list)))

    def update(self, old_length, new_chain, **kwargs):
        chain = kwargs.get('full_chain', None)
        self.mean, self.cov = update_covariance(old_length, self.cov, self.mean, new_chain)
        for proposal in self.prop_list[0]:
            proposal.update(cov=self.cov, chain=chain)

    def __call__(self, x):
        proposal = rng.choice(self.prop_list[0], p=self.prop_list[1])
        return proposal(x)


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


class PriorProposal():
    """
    Are you feeling lucky? Uniform prior draws.
    """
    def __init__(self, prior_min, prior_max, rng=rng):
        """
        prior_min: vector containing mins of priors
        prior_max: vector containing maxes of priors
        """
        self.rng = rng
        self.prior_min = prior_min
        self.prior_max = prior_max

    def update(self, **kwargs):
        return None

    def __call__(self, x):
        x_star = rng.uniform(self.prior_min, self.prior_max)
        return x_star, 0


# AM jump
class AMProposal():

    def covarianceJumpProposalAM(self, x, iter, beta):
        """
        Adaptive Jump Proposal. This function will occasionally
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # choose group
        jumpind = np.random.randint(0, len(self.groups))

        # adjust step size
        prob = np.random.rand()

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
        if self.temp <= 100:
            scale *= np.sqrt(self.temp)

        # get parmeters in new diagonalized basis
        y = np.dot(self.U[jumpind].T, x[self.groups[jumpind]])

        # make correlated componentwise adaptive jump
        ind = np.arange(len(self.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[jumpind][ind])
        q[self.groups[jumpind]] = np.dot(self.U[jumpind], y)

        return q, qxy


class AMProposal():
    """
    Adaptive Metropolis
    """
    def __init__(self, ndim, cov=None, alpha=None):
        self.cov = cov
        self.alpha = alpha
        if cov is None:
            self.cov = np.identity(ndim) * 0.01**2
        if alpha is None:
            self.alpha = 2.38**2 / ndim
        self.cov_chol = np.linalg.cholesky(self.cov)
        self.ndim = ndim
        
    def update(self, **kwargs):
        cov = kwargs['cov']
        try:
            self.cov_chol = np.linalg.cholesky(cov)
        except LinAlgError:
            self.cov_chol = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * 5e-15)

    def __call__(self, x):
        # draw x_star
        u = rng.standard_normal(len(x))
        x_star = x + np.sqrt(self.alpha) * self.cov_chol @ u
        # proposal log ratio factor is 1(symmetric jump)
        factor = 0

        return x_star, factor


class SCAMProposal():
    """
    Single Component Adaptive Metropolis
    """
    def __init__(self, ndim, cov=None, alpha=None):
        self.cov = cov
        self.alpha = alpha
        if cov is None:
            self.cov = np.identity(ndim) * 0.01**2
        if alpha is None:
            self.alpha = 2.38**2
        self.cov_evals, self.cov_evec = np.linalg.eig(self.cov)
        
    def update(self, **kwargs):
        cov = kwargs['cov']
        self.cov_evals, self.cov_evec = np.linalg.eig(cov)

    def __call__(self, x):
        idx = rng.integers(low=0, high=len(x))
        uj = rng.standard_normal(len(x)) * self.cov_evals[idx]
        Dj = self.cov_evec[:, idx]
        x_star = x + np.sqrt(self.alpha) * Dj @ uj
        factor = 0

        return x_star, factor


class DEProposal():
    """
    Differential Evolution
    Use after some burn-in period!
    """
    def __init__(self, ndim, chain, alpha=None):
        self.alpha = alpha
        if alpha is None:
            self.alpha = 2.38**2 / ndim
        self.chain = chain

    def update(self, **kwargs):
        chain = kwargs['chain']
        if chain is None:
            return None
        self.chain = chain

    def __call__(self, x):
        temp = rng.choice(self.chain, size=2, replace=True)
        jump_vec = temp[1, :] - temp[0, :]

        x_star = x + self.alpha * jump_vec

        factor = 0
        return x_star, factor


def pre_proposals(ndim):
    adapt = AMProposal(ndim)
    scam = SCAMProposal(ndim)
    mix = ProposalMix(ndim, [(adapt, 0.5), (scam, 0.5)])
    return mix


def stock_proposals(ndim, chain):
    adapt = AMProposal(ndim)
    scam = SCAMProposal(ndim)
    de = DEProposal(ndim, chain)
    mix = ProposalMix(ndim, [(adapt, 0.25), (scam, 0.15), (de, 0.60)])
    return mix
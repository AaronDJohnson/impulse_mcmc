import numpy as np
from impulse.batch_updates import update_covariance
from impulse.random import rng


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
        self.cov_chol = np.linalg.cholesky(cov)

    def __call__(self, x):
        # draw x_star
        u = rng.standard_normal(len(x))
        x_star = x + np.sqrt(self.alpha) * self.cov_chol @ u

        # proposal ratio factor is 0 (symmetric jump)
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
    def __init__(self, ndim, chain, alpha=None):  # try not to have to call cov_chol here....
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
    mix = ProposalMix(ndim, [(adapt, 0.15), (scam, 0.35), (de, 0.5)])
    return mix
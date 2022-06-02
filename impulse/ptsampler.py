import numpy as np
import os


class PTSampler(object):

    """
    Parallel Tempering Markov Chain Monte-Carlo (PTMCMC) sampler.
    This implementation uses an adaptive jump proposal scheme
    by default using both standard and single component Adaptive
    Metropolis (AM) and Differential Evolution (DE) jumps.

    This implementation also makes use of MPI (mpi4py) to run
    the parallel chains.

    Along with the AM and DE jumps, the user can add custom
    jump proposals with the ``addProposalToCycle`` fuction.

    @param ndim: number of dimensions in problem
    @param logl: log-likelihood function
    @param logp: log prior function (must be normalized for evidence evaluation)
    @param cov: Initial covariance matrix of model parameters for jump proposals
    @param covinds: Indices of parameters for which to perform adaptive jumps
    @param loglargs: any additional arguments (apart from the parameter vector) for
    log likelihood
    @param loglkwargs: any additional keyword arguments (apart from the parameter vector)
    for log likelihood
    @param logpargs: any additional arguments (apart from the parameter vector) for
    log like prior
    @param logl_grad: log-likelihood function, including gradients
    @param logp_grad: prior function, including gradients
    @param logpkwargs: any additional keyword arguments (apart from the parameter vector)
    for log prior
    @param outDir: Full path to output directory for chain files (default = ./chains)
    @param verbose: Update current run-status to the screen (default=True)
    @param resume: Resume from a previous chain (still in testing so beware) (default=False)

    """

    def __init__(
        self,
        ndim,
        logl,
        logp,
        ladder=None,
        Tmin=1,
        Tmax=None,
        Tskip=100,
        cov=None,
        groups=None,
        loglargs=[],
        loglkwargs={},
        logpargs=[],
        logpkwargs={},
        ncores=1,
        outDir="./chains",
        verbose=True,
        resume=False,
        covUpdate=1000,
        buffer_size=1000,
        thin=10,
        writeHotChains=True,
        hotChain=False,):

        self.ndim = ndim
        self.logl = _function_wrapper(logl, loglargs, loglkwargs)
        self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        self.outDir = outDir
        self.verbose = verbose
        self.resume = resume

        # setup output file
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass

        # find indices for which to perform adaptive jumps
        self.groups = groups
        if groups is None:
            self.groups = [np.arange(0, self.ndim)]

        # set up covariance matrix
        self.cov = cov
        self.U = [[]] * len(self.groups)
        self.S = [[]] * len(self.groups)

        self.U, self.S = svd_groups(self.U, self.S, self.groups, self.cov)

        self.M2 = np.zeros((ndim, ndim))
        self.mu = np.zeros(ndim)

        # initialize proposal cycle
        self.propCycle = []
        self.jumpDict = {}

        # indicator for auxilary jumps
        self.aux = []










def svd_groups(U, S, groups, cov):
    # do svd on parameter groups
    for ct, group in enumerate(groups):
        covgroup = np.zeros((len(group), len(group)))
        for ii in range(len(group)):
            for jj in range(len(group)):
                covgroup[ii, jj] = cov[group[ii], group[jj]]

        U[ct], S[ct], v = np.linalg.svd(covgroup)
    return U, S







class _function_wrapper(object):

    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)
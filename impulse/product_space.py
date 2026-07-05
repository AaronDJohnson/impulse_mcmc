from typing import Callable

import numpy as np


class ProductSpace:
    """
    Product space for trans-dimensional MCMC with multiple competing models.

    Handles sampling across different model structures by embedding all
    possible parameters in a joint space and using a model index to
    determine which parameters are active.

    Parameters
    ----------
    model_names : list of str
        Names of competing models.
    loglikelihoods : list of callable
        Log-likelihood functions for each model.
    logpriors : list of callable
        Log-prior functions for each model.
    param_names : list of list of str
        Parameter names for each model.

    Attributes
    ----------
    num_models : int
        Number of competing models.
    all_params : list of str
        All parameter names across models plus model index.
    ndim : int
        Total dimensionality of joint parameter space.

    Examples
    --------
    >>> model_names = ['linear', 'quadratic']
    >>> likelihoods = [linear_loglike, quad_loglike]
    >>> priors = [linear_logprior, quad_logprior]
    >>> param_names = [['a', 'b'], ['a', 'b', 'c']]
    >>> space = ProductSpace(model_names, likelihoods, priors, param_names)
    >>> # Now can sample over both model structures
    """

    def __init__(
        self, model_names: list, loglikelihoods: list, logpriors: list, param_names: list[list[str]]
    ):
        self.model_names = model_names
        self.loglikelihoods = loglikelihoods
        self.logpriors = logpriors
        self.num_models = len(self.model_names)
        self.nmodels = np.arange(self.num_models)

        # get unique parameters for output
        self.all_params = []
        for jj, param_list in enumerate(param_names):
            for i in range(len(param_list)):
                self.all_params.append(param_list[i] + "_" + model_names[jj])
        self.all_params.append("nmodel")
        self.ndim = len(self.all_params)

        # get indices for each model
        self.model_params = []
        for i in range(self.num_models):
            self.model_params.append([])
            for param in param_names[i]:
                self.model_params[i].append(self.all_params.index(param + "_" + model_names[i]))

    def loglikelihood(self, x):
        """
        Evaluate log-likelihood for active model.

        Parameters
        ----------
        x : array_like
            Full parameter vector including model index.

        Returns
        -------
        float
            Log-likelihood value for the active model, or -inf if the
            model index is out of range.
        """
        nmodel = int(np.rint(x[-1]))
        if nmodel not in self.nmodels:
            return -np.inf
        return self.loglikelihoods[nmodel](x[self.model_params[nmodel]])

    def logprior(self, x):
        """
        Evaluate log-prior for active model.

        Only the active model's prior is evaluated. Inactive model
        parameters are unconstrained.

        Parameters
        ----------
        x : array_like
            Full parameter vector including model index.

        Returns
        -------
        float
            Log-prior value for the active model, or -inf if the
            model index is out of range.
        """
        nmodel = int(np.rint(x[-1]))
        if nmodel not in self.nmodels:
            return -np.inf
        return self.logpriors[nmodel](x[self.model_params[nmodel]])


class NestedProductSpace:
    """
    Product space for nested models. The likelihood and prior functions are capable of taking variable number of sources.

    Parameters
    ----------
    loglikelihood : Callable
        Loglikelihood function

    logprior : Callable
        Logprior function

    num_sources : int
        Number of sources

    num_params : int
        Number of parameters for each source
    """

    def __init__(
        self, loglikelihood: Callable, logprior: Callable, num_sources: int, num_params: int
    ):

        self.loglikelihood = loglikelihood
        self.logprior = logprior
        self.num_models = num_sources
        self.nmodels = np.arange(self.num_models)
        self.num_params = num_params
        # number of parameters for each source + 1 for model index
        self.ndim = num_sources * num_params + 1

    def get_loglikelihood(self, params):
        """
        Evaluate log-likelihood for current number of active sources.

        Only the first ``(nmodel + 1) * num_params`` parameters are passed
        to the likelihood function.

        Parameters
        ----------
        params : array_like
            Parameter vector with model index in last position.

        Returns
        -------
        float
            Log-likelihood value using only active parameters, or -inf
            if the model index is out of range.

        Examples
        --------
        >>> # For 2 sources, only first 2*num_params elements are used
        >>> loglike_val = space.get_loglikelihood(full_params)
        """
        nmodel = int(np.rint(params[-1]))
        if nmodel not in self.nmodels:
            return -np.inf
        return self.loglikelihood(params[: (nmodel + 1) * self.num_params])

    def get_logprior(self, params):
        """
        Evaluate log-prior for active source parameters.

        Only the first ``(nmodel + 1) * num_params`` parameters are passed
        to the prior function, matching the behavior of ``get_loglikelihood``.

        Parameters
        ----------
        params : array_like
            Parameter vector with model index in last position.

        Returns
        -------
        float
            Log-prior value, or -inf if model index is invalid.

        Examples
        --------
        >>> # Returns -inf if nmodel is outside valid range
        >>> logprior_val = space.get_logprior(full_params)
        """
        nmodel = int(np.rint(params[-1]))
        if nmodel not in self.nmodels:
            return -np.inf
        return self.logprior(params[: (nmodel + 1) * self.num_params])

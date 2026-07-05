"""Product-space embeddings for trans-dimensional (model-selection) MCMC.

Competing models are embedded in one fixed-dimension parameter vector whose
last coordinate is an integer model index, so a standard MCMC sampler can
move between models by changing that index. :class:`ProductSpace` handles
arbitrary model lists with per-model likelihoods/priors;
:class:`NestedProductSpace` specializes to nested "N identical sources"
models with a shared per-source parameterization, routing only the active
sources' parameters to the likelihood AND the prior. Note the contrast with
:class:`impulse.rjmcmc.BirthDeathProductSpace`, which overrides ``get_logprior``
to evaluate the prior on ALL source slots (active and inactive), as required
for exact reversible-jump birth/death moves.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ParameterLayout:
    """
    Single source of truth for the nested product-space parameter layout.

    The layout convention is: ``num_models`` contiguous per-source blocks of
    ``num_params`` continuous parameters, followed by one trailing model
    index, so the full vector has ``num_models * num_params + 1`` entries
    and the model index lives at position ``nmodel_index`` (== ``-1`` for a
    full-length vector).  Every component that slices source blocks or reads
    the model index consumes this object instead of re-deriving the
    convention.

    Parameters
    ----------
    num_params : int
        Number of continuous parameters per source.
    num_models : int
        Maximum number of sources (model index ranges over
        ``0 .. num_models - 1``).

    Attributes
    ----------
    nmodel_index : int
        Position of the model index: ``num_models * num_params``.

    Examples
    --------
    >>> layout = ParameterLayout(num_params=3, num_models=2)
    >>> layout.total_dim
    7
    >>> layout.active_slice(0)
    slice(0, 3, None)
    >>> layout.source_slice(1)
    slice(3, 6, None)
    """

    num_params: int
    num_models: int
    nmodel_index: int = field(init=False)

    def __post_init__(self):
        # frozen dataclass: assign the derived field via object.__setattr__
        object.__setattr__(self, "nmodel_index", self.num_models * self.num_params)

    @classmethod
    def from_total_dim(cls, num_params: int, total_dim: int) -> "ParameterLayout":
        """Rebuild a layout from the per-source size and the full vector length.

        Inverse of :attr:`total_dim`; used by components that carry only
        ``num_params`` (e.g. legacy-constructed source-swap proposals) to
        recover the layout from a concrete parameter vector.

        Parameters
        ----------
        num_params : int
            Number of continuous parameters per source.
        total_dim : int
            Length of the full parameter vector
            (``num_models * num_params + 1``).

        Returns
        -------
        ParameterLayout
        """
        return cls(num_params, (total_dim - 1) // num_params)

    @property
    def total_dim(self) -> int:
        """Full parameter-vector length: ``nmodel_index + 1``."""
        return self.nmodel_index + 1

    def model_index_of(self, params) -> int:
        """Read the model index from a parameter vector.

        Uses the canonical rint semantics: ``int(np.rint(...))`` of the
        TRAILING coordinate, which for a full-length vector is position
        :attr:`nmodel_index`.  Reading ``params[-1]`` (rather than the fixed
        index) is deliberate: legacy call paths — e.g. the NUTS prior hook,
        which evaluates ``get_logprior`` on the model-index-free source
        block — pass truncated vectors, and the historical behavior in that
        case was to rint whatever the trailing entry holds.

        Parameters
        ----------
        params : array_like
            Parameter vector; full length :attr:`total_dim` in normal use.

        Returns
        -------
        int
        """
        return int(np.rint(params[-1]))

    def model_indices_of(self, samples: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`model_index_of` for a batch of samples.

        Reads the trailing column (see :meth:`model_index_of`).

        Parameters
        ----------
        samples : np.ndarray, shape (n, total_dim)
            Batch of full parameter vectors.

        Returns
        -------
        np.ndarray of int, shape (n,)
        """
        return np.rint(samples[:, -1]).astype(int)

    @staticmethod
    def set_model_index(params: np.ndarray, nmodel) -> None:
        """Write ``nmodel`` into the model-index coordinate of ``params``, in place.

        Writes the TRAILING coordinate — mirroring :meth:`model_index_of` —
        which for a full-length vector is position :attr:`nmodel_index`.  A
        static method because the position does not depend on the block
        sizes; callers without a layout instance (legacy-unpickled
        proposals) invoke it as ``ParameterLayout.set_model_index(...)``.
        """
        params[-1] = nmodel

    def active_slice(self, nmodel: int) -> slice:
        """Slice covering the active source blocks for model index ``nmodel``.

        Parameters
        ----------
        nmodel : int
            Model index (``nmodel = k`` means ``k + 1`` active sources).

        Returns
        -------
        slice
            ``slice(0, (nmodel + 1) * num_params)``.
        """
        return slice(0, (nmodel + 1) * self.num_params)

    def source_slice(self, k: int) -> slice:
        """Slice covering source slot ``k``'s parameter block.

        Parameters
        ----------
        k : int
            Source slot index (0-based).

        Returns
        -------
        slice
            ``slice(k * num_params, (k + 1) * num_params)``.
        """
        return slice(k * self.num_params, (k + 1) * self.num_params)

    def active_indices(self, nmodel: int) -> np.ndarray:
        """Index array covering the active source blocks (see :meth:`active_slice`).

        Parameters
        ----------
        nmodel : int
            Model index.

        Returns
        -------
        np.ndarray of int
            ``np.arange((nmodel + 1) * num_params)``.
        """
        return np.arange((nmodel + 1) * self.num_params)


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

    Notes
    -----
    This class is intentionally NOT exported from the top-level
    ``impulse`` namespace and is not used by the samplers themselves:
    the RJMCMC machinery is built on :class:`NestedProductSpace` /
    :class:`impulse.rjmcmc.BirthDeathProductSpace` ("N identical sources").
    It is kept as a standalone utility for the heterogeneous case — a
    fixed list of structurally different models, each with its own
    likelihood/prior/parameter names — which the nested classes cannot
    express, and it is exercised directly by the test suite.  Import it
    explicitly with ``from impulse.product_space import ProductSpace``.
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
        self.model_params: list[list[int]] = []
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

    Attributes
    ----------
    layout : ParameterLayout
        The product-space parameter layout (single source of truth for the
        source-block slices and the model-index position).
    """

    def __init__(
        self, loglikelihood: Callable, logprior: Callable, num_sources: int, num_params: int
    ):

        self.loglikelihood = loglikelihood
        self.logprior = logprior
        layout = ParameterLayout(num_params=num_params, num_models=num_sources)
        self.num_models = layout.num_models
        self.nmodels = np.arange(self.num_models)
        self.num_params = layout.num_params
        # number of parameters for each source + 1 for model index
        self.ndim = layout.total_dim

    @property
    def layout(self) -> ParameterLayout:
        """Product-space parameter layout derived from the stored scalars.

        A property rather than a stored attribute so the pickled attribute
        set (checkpoint serialization format) is unchanged and instances
        restored from pre-layout checkpoints get it for free.
        """
        return ParameterLayout(num_params=self.num_params, num_models=self.num_models)

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
        layout = self.layout
        nmodel = layout.model_index_of(params)
        if nmodel not in self.nmodels:
            return -np.inf
        return self.loglikelihood(params[layout.active_slice(nmodel)])

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
        layout = self.layout
        nmodel = layout.model_index_of(params)
        if nmodel not in self.nmodels:
            return -np.inf
        return self.logprior(params[layout.active_slice(nmodel)])

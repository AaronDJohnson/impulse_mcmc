"""
RJMCMCProductSpace — user-facing class for reversible-jump MCMC
in the product-space embedding.
"""

import numpy as np
from typing import Callable, Optional

from impulse.product_space import NestedProductSpace
from impulse.proposals import make_source_swap_proposal
from impulse.rjmcmc_proposals import (
    default_birth_death_probs,
    make_birth_proposal,
    make_death_proposal,
    make_nmodel_jump,
)


class RJMCMCProductSpace(NestedProductSpace):
    """
    Product space for reversible-jump MCMC with birth/death proposals.

    Extends :class:`NestedProductSpace` with factory methods for
    birth, death, model-index jump, and source-swap proposals so users
    don't have to compute Hastings ratios by hand.

    The prior is evaluated on **all** source parameters (active and inactive),
    not just the active ones.  This ensures that newly activated parameters
    are within bounds and that the implicit Occam's razor from the prior
    volume ratio is correct.

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood accepting the first ``(nmodel+1)*num_params`` params.
    logprior : callable
        Log-prior that receives ``num_sources * num_params`` parameters
        and must check bounds on **all** of them.
    num_sources : int
        Maximum number of sources.
    num_params : int
        Number of parameters per source.
    source_prior_draw : callable
        ``source_prior_draw(rng) -> np.ndarray`` of shape ``(num_params,)``.
        Draws one source's parameters from the prior (or a proposal
        distribution; see ``source_prior_logpdf``).
    source_prior_logpdf : callable, optional
        ``source_prior_logpdf(params) -> float``.  Log-density of the
        *prior* on a single source's parameters.  Required only when
        ``source_prior_draw`` samples from a distribution other than the
        prior.
    source_proposal_logpdf : callable, optional
        ``source_proposal_logpdf(params) -> float``.  Log-density of the
        distribution that ``source_prior_draw`` actually uses.  Required
        only when it differs from the prior.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.
        Defaults to :func:`default_birth_death_probs`.

    Examples
    --------
    >>> space = RJMCMCProductSpace(
    ...     loglikelihood=my_loglike,
    ...     logprior=my_logprior,
    ...     num_sources=3,
    ...     num_params=3,
    ...     source_prior_draw=lambda rng: rng.uniform(0, 5, size=3),
    ... )
    >>> from impulse import PTSampler
    >>> sampler = PTSampler.from_rjmcmc(space)
    """

    def __init__(
        self,
        loglikelihood: Callable,
        logprior: Callable,
        num_sources: int,
        num_params: int,
        source_prior_draw: Callable,
        source_prior_logpdf: Optional[Callable] = None,
        source_proposal_logpdf: Optional[Callable] = None,
        prob_schedule: Optional[Callable] = None,
    ):
        super().__init__(loglikelihood, logprior, num_sources, num_params)
        self.source_prior_draw = source_prior_draw
        self.source_prior_logpdf = source_prior_logpdf
        self.source_proposal_logpdf = source_proposal_logpdf
        self.prob_schedule = prob_schedule or default_birth_death_probs

    # ------------------------------------------------------------------
    # Prior: evaluate on ALL source params (active + inactive)
    # ------------------------------------------------------------------

    def get_logprior(self, params):
        """
        Evaluate log-prior on **all** source parameters.

        Unlike :meth:`NestedProductSpace.get_logprior`, this always passes
        ``num_sources * num_params`` parameters to the prior function,
        ensuring that inactive source parameters remain within bounds.

        Parameters
        ----------
        params : array_like
            Full parameter vector with model index in last position.

        Returns
        -------
        float
            Log-prior value, or ``-inf`` if any parameter is out of bounds
            or the model index is invalid.
        """
        nmodel = int(np.rint(params[-1]))
        if nmodel not in self.nmodels:
            return -np.inf
        return self.logprior(params[:self.num_models * self.num_params])

    # ------------------------------------------------------------------
    # Proposal factories
    # ------------------------------------------------------------------

    def get_birth_proposal(self) -> Callable:
        """Return a birth proposal closure wired to this space's prior draw."""
        return make_birth_proposal(
            self.num_params,
            self.num_models,
            self.source_prior_draw,
            log_proposal_density=self.source_proposal_logpdf,
            log_prior_density=self.source_prior_logpdf,
            prob_schedule=self.prob_schedule,
        )

    def get_death_proposal(self) -> Callable:
        """Return a death proposal closure wired to this space."""
        return make_death_proposal(
            self.num_params,
            self.num_models,
            log_proposal_density=self.source_proposal_logpdf,
            log_prior_density=self.source_prior_logpdf,
            prob_schedule=self.prob_schedule,
        )

    def get_nmodel_jump(self) -> Callable:
        """Return a uniform model-index jump proposal."""
        return make_nmodel_jump(self.num_models)

    def get_source_swap_proposal(self) -> Callable:
        """Return a source-swap proposal for label switching."""
        return make_source_swap_proposal(self.num_params)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_default_groups(self) -> list[list[int]]:
        """
        Parameter groups: one per source, model index excluded.

        Returns
        -------
        list of list of int
            Each inner list holds the indices for one source's parameters.
        """
        groups = []
        for i in range(self.num_models):
            groups.append(list(range(i * self.num_params,
                                     (i + 1) * self.num_params)))
        return groups

    def model_posterior_probs(
        self, chain: np.ndarray, burn: int = 0
    ) -> np.ndarray:
        """
        Estimate posterior model probabilities from cold-chain samples.

        Parameters
        ----------
        chain : np.ndarray, shape (N, ndim)
            Samples from the cold chain.
        burn : int
            Number of initial samples to discard.

        Returns
        -------
        np.ndarray, shape (num_models,)
            Posterior probability for each model (0-indexed).
        """
        nmodel_samples = np.rint(chain[burn:, -1]).astype(int)
        counts = np.bincount(nmodel_samples, minlength=self.num_models)
        return counts / counts.sum()

    def draw_initial_position(
        self, rng: np.random.Generator, nmodel: int = 0
    ) -> np.ndarray:
        """
        Draw a valid initial position from the prior.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.
        nmodel : int
            Initial model index (number of active sources minus 1).

        Returns
        -------
        np.ndarray, shape (ndim,)
            Initial parameter vector with all source slots filled from the
            prior and the model index set.
        """
        x0 = np.zeros(self.ndim)
        for i in range(self.num_models):
            x0[i * self.num_params:(i + 1) * self.num_params] = (
                self.source_prior_draw(rng)
            )
        x0[-1] = nmodel
        return x0

"""
BirthDeathProductSpace — user-facing class for product-space (composite
model space) model selection with birth/death moves.

This is a product-space / saturated-space sampler: the state has fixed
dimension (all source slots plus a model index), the likelihood reads only
the active slots, and the prior is evaluated on all slots. Model moves are
ordinary Metropolis-Hastings birth/death (plus a direct model-index jump
and a source swap) on the fixed-dimension state — not dimension-changing
reversible jump in the Green (1995) sense, and no trans-dimensional
Jacobian appears. It is the product-space member of the trans-model MCMC
family (Carlin & Chib 1995; Godsill 2001). ``RJMCMCProductSpace`` is kept
as a deprecated alias.
"""

from typing import Callable, Optional

import numpy as np

from impulse.birth_death_proposals import (
    default_birth_death_probs,
    make_birth_death_proposal,
    make_birth_proposal,
    make_death_proposal,
    make_nmodel_jump,
)
from impulse.product_space import NestedProductSpace
from impulse.proposals import make_source_swap_proposal


class BirthDeathProductSpace(NestedProductSpace):
    """
    Product-space model selection with birth/death moves.

    Extends :class:`NestedProductSpace` with factory methods for
    birth, death, model-index jump, and source-swap proposals so users
    don't have to compute Hastings ratios by hand.

    This is a **product-space** (composite-model-space) sampler, *not*
    dimension-changing reversible jump: the state has fixed dimension
    (every source slot plus a model index), and birth/death change only the
    model index while re-drawing the affected slot. There is no
    trans-dimensional Jacobian; the newly activated parameters enter through
    an ordinary Metropolis-Hastings proposal-density ratio.

    The prior is evaluated on **all** source parameters (active and inactive),
    not just the active ones.  This keeps newly activated parameters within
    bounds and supplies the Occam penalty: the inactive slots carry their
    prior in the target, so marginalizing them out yields the correct model
    posterior (the product-space analogue of the reversible-jump prior-volume
    ratio).

    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood accepting the first ``(nmodel+1)*num_params`` params.
    logprior : callable
        Log-prior that receives ``num_sources * num_params`` parameters
        and must check bounds on **all** of them.  When
        ``source_prior_logpdf`` is omitted, ``logprior`` is additionally
        used as the per-source prior density for the birth/death moves'
        slot-density terms: it must then also accept a single source's
        ``num_params``-length vector and be additive across per-source
        slots (``logprior(all slots) == sum of per-slot values``).  This
        holds for the usual independent per-source priors and is probed at
        proposal construction (a failed probe raises); supply
        ``source_prior_logpdf`` explicitly otherwise.
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
    >>> space = BirthDeathProductSpace(
    ...     loglikelihood=my_loglike,
    ...     logprior=my_logprior,
    ...     num_sources=3,
    ...     num_params=3,
    ...     source_prior_draw=lambda rng: rng.uniform(0, 5, size=3),
    ... )
    >>> from impulse import PTSampler
    >>> sampler = PTSampler.from_product_space(space)
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
        layout = self.layout
        nmodel = layout.model_index_of(params)
        if nmodel not in self.nmodels:
            return -np.inf
        return self.logprior(params[: layout.nmodel_index])

    # ------------------------------------------------------------------
    # Proposal factories
    # ------------------------------------------------------------------

    def _resolve_source_prior_logpdf(self) -> Callable:
        """
        Per-source prior log-density for the birth/death slot-density terms.

        Returns ``source_prior_logpdf`` when supplied.  Otherwise falls back
        to the full product-space ``logprior`` evaluated on a single source's
        ``num_params``-length vector, which equals the per-source density only
        for priors that are additive across per-source slots.  The fallback is
        probed here on several independent sets of prior draws: ``logprior``
        is evaluated on each slot and on the concatenated full vector.  A
        probe call that fails raises :class:`TypeError`, and a failed
        additivity check raises :class:`ValueError` — proceeding would
        silently corrupt the birth/death Hastings ratios.

        A probe draw *outside* the prior support also raises
        :class:`ValueError`: this fallback only runs when neither
        ``source_prior_logpdf`` nor ``source_proposal_logpdf`` was supplied,
        a configuration in which ``source_prior_draw`` is assumed to sample
        the prior itself — an out-of-support draw proves it does not, so the
        birth/death draw-density terms (which would silently use the prior
        density for a non-prior draw distribution) are provably wrong.  A
        defensively-written ``logprior`` that returns ``-inf`` for a
        single-source ``num_params``-length vector (e.g. a shape check on
        its input) instead of raising trips the same branch; the error
        message names both causes, and the remedy in either case is to
        supply ``source_prior_logpdf``.

        Returns
        -------
        callable
            ``f(params) -> float`` taking one source's parameters.
        """
        if self.source_prior_logpdf is not None:
            return self.source_prior_logpdf
        if self.num_models == 1:
            # Single-slot product space: the full prior IS the per-source
            # prior, so there is no additivity to probe.
            return self.logprior
        rng = np.random.default_rng(0)
        num_probe_sets = 3
        probes = []
        try:
            for _ in range(num_probe_sets):
                slots = [
                    np.asarray(self.source_prior_draw(rng), dtype=float)
                    for _ in range(self.num_models)
                ]
                per_slot = [float(self.logprior(s)) for s in slots]
                total = float(self.logprior(np.concatenate(slots)))
                probes.append((total, per_slot))
        except Exception as err:
            raise TypeError(
                "source_prior_logpdf was not supplied, so logprior is used as "
                "the per-source prior density for the birth/death moves — but "
                "evaluating logprior on a single source's "
                f"{self.num_params}-parameter vector failed ({err!r}). "
                "Supply source_prior_logpdf explicitly."
            ) from err
        for total, per_slot in probes:
            if not all(np.isfinite(v) for v in per_slot):
                raise ValueError(
                    "logprior returned -inf for a single-source probe draw. "
                    "Either source_prior_draw samples outside the prior "
                    "support (so it cannot be the prior), or logprior "
                    "returns -inf for a single-source "
                    f"{self.num_params}-parameter vector instead of raising "
                    "(e.g. a defensive shape check on its input).  With "
                    "neither source_proposal_logpdf nor source_prior_logpdf "
                    "supplied, the birth/death moves would use the prior "
                    "density as the draw density, which is provably wrong "
                    "in the first case and unevaluable in the second.  "
                    "Supply source_prior_logpdf (and source_proposal_logpdf "
                    "if the draw distribution is not the prior)."
                )
            per_slot_sum = float(sum(per_slot))
            if not (
                np.isfinite(total) and abs(total - per_slot_sum) <= 1e-6 * max(1.0, abs(total))
            ):
                raise ValueError(
                    "source_prior_logpdf was not supplied and logprior is "
                    "not additive across per-source slots "
                    f"(logprior(full vector) = {total}, sum over slots = "
                    f"{per_slot_sum}).  The birth/death slot-density terms "
                    "would be wrong; supply source_prior_logpdf explicitly."
                )
        return self.logprior

    def _source_draw_density_args(self):
        """
        Resolve the ``(log_proposal_density, log_prior_density)`` pair for
        the birth/death proposals.

        The proposals' Hastings terms use the log-density of the distribution
        ``source_prior_draw`` actually samples: ``source_proposal_logpdf``
        when supplied, otherwise the per-source prior density (resolved, and
        probed for additivity, by :meth:`_resolve_source_prior_logpdf`).
        When a proposal density is supplied the per-source prior density is
        not needed, so the probe is skipped.
        """
        if self.source_proposal_logpdf is not None:
            return self.source_proposal_logpdf, self.source_prior_logpdf
        return None, self._resolve_source_prior_logpdf()

    def get_birth_proposal(self) -> Callable:
        """Return a birth proposal closure wired to this space's prior draw.

        .. warning::
            Do not register this standalone with a constant selection weight;
            use :meth:`get_birth_death_proposal` (see
            :class:`~impulse.birth_death_proposals.BirthProposal`).
        """
        log_proposal, log_prior = self._source_draw_density_args()
        return make_birth_proposal(
            self.num_params,
            self.num_models,
            self.source_prior_draw,
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
            prob_schedule=self.prob_schedule,
        )

    def get_birth_death_proposal(self) -> Callable:
        """Return the combined birth-death kernel wired to this space.

        This is the trans-dimensional kernel to register with the sampler:
        it selects birth vs death internally with the ``prob_schedule``
        probabilities, which is what makes the sub-proposals' Hastings
        ratios exact under constant-weight jump selection.
        """
        log_proposal, log_prior = self._source_draw_density_args()
        return make_birth_death_proposal(
            self.num_params,
            self.num_models,
            self.source_prior_draw,
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
            prob_schedule=self.prob_schedule,
        )

    def get_death_proposal(self) -> Callable:
        """Return a death proposal closure wired to this space.

        .. warning::
            Do not register this standalone with a constant selection weight;
            use :meth:`get_birth_death_proposal` (see
            :class:`~impulse.birth_death_proposals.DeathProposal`).
        """
        log_proposal, log_prior = self._source_draw_density_args()
        return make_death_proposal(
            self.num_params,
            self.num_models,
            self.source_prior_draw,
            log_proposal_density=log_proposal,
            log_prior_density=log_prior,
            prob_schedule=self.prob_schedule,
        )

    def get_nmodel_jump(self) -> Callable:
        """Return a uniform model-index jump proposal."""
        return make_nmodel_jump(self.num_models, layout=self.layout)

    def get_source_swap_proposal(self) -> Callable:
        """Return a source-swap proposal for label switching."""
        return make_source_swap_proposal(self.num_params, layout=self.layout)

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
            sl = self.layout.source_slice(i)
            groups.append(list(range(sl.start, sl.stop)))
        return groups

    def model_posterior_probs(self, chain: np.ndarray, burn: int = 0) -> np.ndarray:
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
        nmodel_samples = self.layout.model_indices_of(chain[burn:])
        counts = np.bincount(nmodel_samples, minlength=self.num_models)
        return counts / counts.sum()

    def draw_initial_position(self, rng: np.random.Generator, nmodel: int = 0) -> np.ndarray:
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
        layout = self.layout
        for i in range(self.num_models):
            x0[layout.source_slice(i)] = self.source_prior_draw(rng)
        layout.set_model_index(x0, nmodel)
        return x0


# Deprecated alias. This class was named ``RJMCMCProductSpace`` through
# 2.0.0-dev, but it is a product-space (composite-model-space) sampler, not
# dimension-changing reversible jump; the accurate name is
# ``BirthDeathProductSpace``. The alias is retained for backward
# compatibility and may be removed in a future release.
RJMCMCProductSpace = BirthDeathProductSpace

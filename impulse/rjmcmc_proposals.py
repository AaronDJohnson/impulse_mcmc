"""
RJMCMC birth/death proposals in the product space.

All proposals are picklable callable classes so that sampler checkpointing
works correctly.
"""

import numpy as np
from typing import Callable, Optional


def default_birth_death_probs(nmodel: int, max_sources: int):
    """
    Default probability schedule for choosing birth vs death moves.

    Parameters
    ----------
    nmodel : int
        Current model index (0-based: ``nmodel=k`` means ``k+1`` active sources).
    max_sources : int
        Maximum number of sources (``nmodel`` ranges from 0 to ``max_sources - 1``).

    Returns
    -------
    p_birth : float
        Probability of proposing a birth move.
    p_death : float
        Probability of proposing a death move.
    """
    if nmodel == 0:
        return 1.0, 0.0
    if nmodel == max_sources - 1:
        return 0.0, 1.0
    return 0.5, 0.5


class BirthProposal:
    """
    Birth proposal that adds a new source to the product space.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    draw_from_prior : callable
        ``draw_from_prior(rng) -> np.ndarray`` of shape ``(num_params,)``.
    log_proposal_density : callable, optional
        Log-density of the proposal distribution. Required only when the
        proposal distribution differs from the prior.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters. Required
        only when ``log_proposal_density`` is provided.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.
    """

    # give a __name__ so JumpProposals.add_jump deduplication works
    __name__ = "birth_proposal"

    def __init__(
        self,
        num_params: int,
        max_sources: int,
        draw_from_prior: Callable,
        log_proposal_density: Optional[Callable] = None,
        log_prior_density: Optional[Callable] = None,
        prob_schedule: Optional[Callable] = None,
    ):
        self.num_params = num_params
        self.max_sources = max_sources
        self.draw_from_prior = draw_from_prior
        self.log_proposal_density = log_proposal_density
        self.log_prior_density = log_prior_density
        self.prob_schedule = prob_schedule or default_birth_death_probs

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        nmodel = int(np.rint(q[-1]))

        if nmodel >= self.max_sources - 1:
            return q, 0.0

        new_params = self.draw_from_prior(rng)

        slot = nmodel + 1
        q[slot * self.num_params:(slot + 1) * self.num_params] = new_params
        q[-1] = nmodel + 1

        p_birth_k, _ = self.prob_schedule(nmodel, self.max_sources)
        _, p_death_k1 = self.prob_schedule(nmodel + 1, self.max_sources)

        qxy = np.log(p_death_k1) - np.log(nmodel + 2) - np.log(p_birth_k)

        if self.log_proposal_density is not None and self.log_prior_density is not None:
            qxy += self.log_prior_density(new_params) - self.log_proposal_density(new_params)

        return q, qxy


class DeathProposal:
    """
    Death proposal that removes an active source from the product space.

    The killed source's parameters are swapped to the last active slot to
    maintain contiguity.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    log_proposal_density : callable, optional
        Log-density of the birth proposal distribution.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.
    """

    __name__ = "death_proposal"

    def __init__(
        self,
        num_params: int,
        max_sources: int,
        log_proposal_density: Optional[Callable] = None,
        log_prior_density: Optional[Callable] = None,
        prob_schedule: Optional[Callable] = None,
    ):
        self.num_params = num_params
        self.max_sources = max_sources
        self.log_proposal_density = log_proposal_density
        self.log_prior_density = log_prior_density
        self.prob_schedule = prob_schedule or default_birth_death_probs

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        nmodel = int(np.rint(q[-1]))

        if nmodel <= 0:
            return q, 0.0

        kill_idx = rng.integers(0, nmodel + 1)
        killed_params = q[kill_idx * self.num_params:(kill_idx + 1) * self.num_params].copy()

        if kill_idx != nmodel:
            last_params = q[nmodel * self.num_params:(nmodel + 1) * self.num_params].copy()
            q[kill_idx * self.num_params:(kill_idx + 1) * self.num_params] = last_params
            q[nmodel * self.num_params:(nmodel + 1) * self.num_params] = killed_params

        q[-1] = nmodel - 1

        _, p_death_k = self.prob_schedule(nmodel, self.max_sources)
        p_birth_km1, _ = self.prob_schedule(nmodel - 1, self.max_sources)

        qxy = np.log(p_birth_km1) + np.log(nmodel + 1) - np.log(p_death_k)

        if self.log_proposal_density is not None and self.log_prior_density is not None:
            qxy += self.log_proposal_density(killed_params) - self.log_prior_density(killed_params)

        return q, qxy


class NmodelJump:
    """
    Uniform model-index jump proposal.

    Parameters
    ----------
    max_sources : int
        Maximum number of sources.
    """

    __name__ = "nmodel_jump"

    def __init__(self, max_sources: int):
        self.max_sources = max_sources

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        q[-1] = rng.integers(0, self.max_sources)
        return q, 0.0


# ---------------------------------------------------------------------------
# Factory functions (thin wrappers for backward compatibility and
# consistency with the plan's API)
# ---------------------------------------------------------------------------

def make_birth_proposal(
    num_params: int,
    max_sources: int,
    draw_from_prior: Callable,
    log_proposal_density: Optional[Callable] = None,
    log_prior_density: Optional[Callable] = None,
    prob_schedule: Optional[Callable] = None,
) -> BirthProposal:
    """
    Create a birth proposal that adds a new source.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    draw_from_prior : callable
        ``draw_from_prior(rng) -> np.ndarray`` of shape ``(num_params,)``.
    log_proposal_density : callable, optional
        Log-density of the proposal distribution used by ``draw_from_prior``.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.

    Returns
    -------
    BirthProposal
    """
    return BirthProposal(
        num_params, max_sources, draw_from_prior,
        log_proposal_density, log_prior_density, prob_schedule,
    )


def make_death_proposal(
    num_params: int,
    max_sources: int,
    log_proposal_density: Optional[Callable] = None,
    log_prior_density: Optional[Callable] = None,
    prob_schedule: Optional[Callable] = None,
) -> DeathProposal:
    """
    Create a death proposal that removes an active source.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    log_proposal_density : callable, optional
        Log-density of the birth proposal distribution.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.

    Returns
    -------
    DeathProposal
    """
    return DeathProposal(
        num_params, max_sources,
        log_proposal_density, log_prior_density, prob_schedule,
    )


def make_nmodel_jump(max_sources: int) -> NmodelJump:
    """
    Create a uniform model-index jump proposal.

    Parameters
    ----------
    max_sources : int
        Maximum number of sources.

    Returns
    -------
    NmodelJump
    """
    return NmodelJump(max_sources)

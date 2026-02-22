import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple
from impulse.chain_stats import ChainStats
from impulse.sampler_state import SamplerState


class JumpProposals:
    """
    Manages weighted collection of proposal distributions for a single temperature chain.

    Maintains a portfolio of different MCMC proposal types (e.g., adaptive Metropolis,
    differential evolution) and selects between them probabilistically based on
    specified weights.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics and state information for the associated temperature chain.
    proposal_list : list, default empty
        List of proposal functions to use.
    proposal_weights : list, default empty
        Relative weights for each proposal type.
    proposal_probs : np.ndarray, optional
        Normalized probabilities (computed automatically from weights).

    Examples
    --------
    >>> from impulse.proposals import am, de, scam
    >>> proposals = JumpProposals(chain_stats)
    >>> proposals.add_jump(am, weight=15)     # Adaptive Metropolis
    >>> proposals.add_jump(de, weight=50)     # Differential Evolution  
    >>> proposals.add_jump(scam, weight=30)   # Single Component AM
    >>> # Now proposals will be selected with probabilities [15/95, 50/95, 30/95]

    Notes
    -----
    - Weights are automatically normalized to probabilities
    - Differential evolution proposals are disabled until sample buffer is full
    - Each proposal function should return (new_sample, log_proposal_ratio)
    """
    def __init__(self,
                 chain_stats: ChainStats,
                 proposal_list: list|None = None,
                 proposal_weights: list|None = None,
                 proposal_probs: np.ndarray|None = None):
        self.chain_stats = chain_stats
        self.proposal_list = proposal_list if proposal_list is not None else []
        self.proposal_weights = proposal_weights if proposal_weights is not None else []
        self.proposal_probs = proposal_probs

    def add_jump(self,
                 jump: Callable,
                 weight: float
                 ) -> None:
        """
        Add or update a proposal type with specified weight.

        Parameters
        ----------
        jump : callable
            Proposal function that takes ChainStats and returns (sample, log_ratio).
        weight : float
            Relative weight for this proposal type. Higher weights increase selection probability.

        Examples
        --------
        >>> proposals.add_jump(am, weight=20)      # 20% weight
        >>> proposals.add_jump(de, weight=60)      # 60% weight  
        >>> proposals.add_jump(scam, weight=20)    # 20% weight
        """
        if jump not in self.proposal_list:
            self.proposal_list.append(jump)
            self.proposal_weights.append(weight)
        elif weight != self.proposal_weights[self.proposal_list.index(jump)]:
            self.proposal_weights[self.proposal_list.index(jump)] = weight
        total = sum(self.proposal_weights)
        if total > 0:
            self.proposal_probs = np.array(self.proposal_weights) / total
        else:
            self.proposal_probs = np.ones(len(self.proposal_weights)) / len(self.proposal_weights)

    def __call__(self,
                 state: SamplerState
                 ) -> Tuple[np.ndarray, float]:
        old_sample = state.positions[self.chain_stats.chain_index]
        self.chain_stats.update_sample(old_sample)
        rng = self.chain_stats.rng
        proposal = rng.choice(self.proposal_list, p=self.proposal_probs)
        # DE requires a filled sample buffer; fall back to gaussian if unavailable
        if proposal.__name__ == 'de' and not self.chain_stats.buffer_full:
            proposal = gaussian
        new_sample, qxy = proposal(self.chain_stats)
        return new_sample, qxy

@dataclass
class ProposalBundle:
    """
    Collection of JumpProposals objects for all temperature chains.

    Coordinates proposal generation across all temperature chains in parallel
    tempering, ensuring each chain uses its own statistics and random state.

    Parameters
    ----------
    jump_proposals : list of JumpProposals
        One JumpProposals instance for each temperature chain.

    Examples
    --------
    >>> # Typically created by setup_standard_jumps()
    >>> bundle = ProposalBundle(jump_proposals_list)
    >>> new_positions, log_ratios = bundle(current_state)
    >>> print(f"Proposed {new_positions.shape[0]} new positions")

    Notes
    -----
    - Generates proposals for all chains simultaneously
    - Each chain maintains independent proposal statistics
    - Used by the main sampling loop for vectorized proposal generation
    """
    jump_proposals: List['JumpProposals']

    def get_new_position(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        new_samples = np.zeros_like(state.positions)
        qxys = np.zeros((len(self.jump_proposals),))
        for i, jp in enumerate(self.jump_proposals):
            new_samples[i], qxys[i] = jp(state)
        return new_samples, qxys

    def add_jump(self, jump: Callable, weight: float) -> None:
        for jp in self.jump_proposals:
            jp.add_jump(jump, weight)

    def __call__(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_new_position(state)

def am(chain_stats: ChainStats) -> Tuple[np.ndarray, float]:
    """
    Adaptive Metropolis proposal using empirical covariance matrix.

    Generates proposals from a multivariate Gaussian distribution using
    the current empirical covariance estimate. Occasionally uses different
    scale factors to improve mixing and exploration.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics object containing current position, covariance estimate,
        and random number generator.

    Returns
    -------
    tuple of (np.ndarray, float)
        new_position : np.ndarray
            Proposed parameter values.
        log_proposal_ratio : float
            Log of forward/backward proposal probability ratio (always 0 for symmetric proposals).

    Examples
    --------
    >>> # Typically called automatically by JumpProposals
    >>> new_pos, log_ratio = am(chain_stats)
    >>> # log_ratio is 0 because AM proposals are symmetric

    Notes
    -----
    - Uses adaptive scale factors: 10x (3% chance), 0.2x (7% chance), 1x (90% chance)
    - Proposal covariance: 2.38²/d × empirical_covariance (optimal for Gaussians)
    - Proposals are symmetric, so log_proposal_ratio = 0
    - Falls back to identity matrix if covariance is not yet reliable
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    # jumpind = np.random.randint(0, len(groups))

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # get parameters in new diagonalized basis
    y = np.dot(chain_stats.svd_U[jumpind].T, chain_stats.current_sample[chain_stats.groups[jumpind]])

    # make correlated componentwise adaptive jump
    ind = np.arange(len(chain_stats.groups[jumpind]))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    y[ind] = y[ind] + rng.standard_normal(neff) * cd * np.sqrt(chain_stats.svd_S[jumpind][ind])
    q[chain_stats.groups[jumpind]] = np.dot(chain_stats.svd_U[jumpind], y)

    return q, qxy


def scam(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Single Component Adaptive Metropolis proposal.

    Updates one parameter at a time using the marginal variance from the
    empirical covariance matrix. More efficient than full AM for high-dimensional
    problems and helps with parameter correlations.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics object containing current position, covariance estimate,
        and random number generator.

    Returns
    -------
    tuple of (np.ndarray, float)
        new_position : np.ndarray
            Proposed parameter values with one component updated.
        log_proposal_ratio : float
            Always 0 due to proposal symmetry.

    Examples
    --------
    >>> # Called automatically during sampling
    >>> new_pos, log_ratio = scam(chain_stats)
    >>> # Only one parameter component will differ from current position

    Notes
    -----
    - Updates single randomly-chosen parameter using marginal variance
    - Scale factors: 10x (3% chance), 0.2x (7% chance), 1x (90% chance)
    - More efficient than full AM for high-dimensional spaces
    - Complementary to AM and DE proposals in the proposal mix
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # make correlated componentwise adaptive jump
    ind = rng.integers(0, ndim, size=1)

    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    q[chain_stats.groups[jumpind]] += (
        rng.standard_normal() * cd * np.sqrt(chain_stats.svd_S[jumpind][ind]) * chain_stats.svd_U[jumpind][:, ind].flatten()
    )

    return q, qxy


def de(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Differential Evolution proposal using historical samples.

    Generates proposals by combining differences between randomly selected
    past samples, which helps escape local modes and improves mixing in
    multimodal distributions.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics object containing sample history buffer and random generator.

    Returns
    -------
    tuple of (np.ndarray, float)
        new_position : np.ndarray
            Proposed parameter values based on sample history.
        log_proposal_ratio : float
            Always 0 due to proposal symmetry.

    Examples
    --------
    >>> # Only called when sample buffer is sufficiently full
    >>> if chain_stats.buffer_full:
    ...     new_pos, log_ratio = de(chain_stats)

    Notes
    -----
    - Requires sufficient sample history (buffer_full = True)
    - Formula: X_new = X_current + γ(X_a - X_b) for random past samples X_a, X_b
    - Scale factors: 1.0 (50% chance, mode jump) or random × 2.4/√(2d) (50% chance)
    - Particularly effective for multimodal and highly correlated distributions
    - Based on differential evolution optimization algorithm principles
    """
    rng = chain_stats.rng
    # get old parameters
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # Use actual filled buffer size, not maximum buffer size
    if chain_stats.buffer_full:
        bufsize = chain_stats.buffer_size
    else:
        bufsize = min(chain_stats.sample_total, chain_stats.buffer_size)

    # draw a random integer from 0 - iter
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
        scale = rng.random() * 2.4 / np.sqrt(2 * ndim)

    for ii in range(ndim):

        # jump size
        sigma = (chain_stats._buffer[mm, chain_stats.groups[jumpind][ii]] -
                 chain_stats._buffer[nn, chain_stats.groups[jumpind][ii]])

        # jump
        q[chain_stats.groups[jumpind][ii]] += scale * sigma

    return q, qxy

def gaussian(chain_stats: ChainStats) -> tuple[np.ndarray, float]:
    """
    Simple Gaussian random walk proposal.

    Generates proposals using independent Gaussian steps for each parameter
    group. Occasionally uses different step sizes to improve exploration
    and mixing properties.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics object containing current position, parameter groups,
        and random number generator.

    Returns
    -------
    tuple of (np.ndarray, float)
        new_position : np.ndarray
            Proposed parameter values.
        log_proposal_ratio : float
            Always 0 due to proposal symmetry.

    Examples
    --------
    >>> # Typically used as part of proposal mix
    >>> new_pos, log_ratio = gaussian(chain_stats)
    >>> # log_ratio is 0 because Gaussian proposals are symmetric

    Notes
    -----
    - Scale factors: 1.0 (50% chance, large mode jump) or random × 2.4/√(2d) (50% chance)
    - Proposals are symmetric, so log_proposal_ratio = 0
    - Simpler than adaptive proposals but useful for initial exploration
    """
    rng = chain_stats.rng
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # get jump scale size
    prob = rng.random()

    # mode jump
    if prob > 0.5:
        scale = 1.0

    else:
        scale = rng.random() * 2.4 / np.sqrt(2 * ndim)

    # make jump
    q[chain_stats.groups[jumpind]] += rng.standard_normal(ndim) * scale

    return q, qxy

class SourceSwapProposal:
    """
    Proposal that swaps parameter blocks between two randomly selected sources.

    Parameters
    ----------
    num_params : int, default 3
        Number of parameters per source.

    Notes
    -----
    - Model index is assumed to be in the last position of the parameter vector
    - Only attempts swap if number of sources > 0
    - Returns original position unchanged if only one source or same source selected
    - Proposal is symmetric, so log_proposal_ratio = 0
    """

    __name__ = "source_swap_proposal"

    def __init__(self, num_params: int = 3):
        self.num_params = num_params

    def __call__(self, chain_stats: ChainStats) -> tuple[np.ndarray, float]:
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        qxy = 0
        nmodel = int(np.rint(q[-1]))
        if nmodel == 0:
            return q, qxy
        swap_source_1 = rng.integers(0, nmodel + 1)
        swap_source_2 = rng.integers(0, nmodel + 1)
        if swap_source_1 == swap_source_2:
            return q, qxy

        num_params = self.num_params
        x = q[num_params * swap_source_1:num_params * (swap_source_1 + 1)].copy()
        y = q[num_params * swap_source_2:num_params * (swap_source_2 + 1)].copy()

        q[num_params * swap_source_1:num_params * (swap_source_1 + 1)] = y
        q[num_params * swap_source_2:num_params * (swap_source_2 + 1)] = x
        return q, qxy


def make_source_swap_proposal(num_params: int = 3) -> SourceSwapProposal:
    """
    Create a reversible-jump proposal for swapping parameters between sources.

    Parameters
    ----------
    num_params : int, default 3
        Number of parameters per source.

    Returns
    -------
    SourceSwapProposal
        Picklable callable with signature ``(ChainStats) -> (np.ndarray, float)``.

    Examples
    --------
    >>> sampler.add_custom_jump(make_source_swap_proposal(5), weight=25)
    """
    return SourceSwapProposal(num_params)


# Backward-compatible default: 3 parameters per source
source_swap_proposal = make_source_swap_proposal(3)

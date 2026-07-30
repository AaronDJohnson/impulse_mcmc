"""MCMC proposal distributions and the containers that mix them.

Proposal contract: every proposal is a callable
``proposal(chain_stats: ChainStats) -> (new_sample: np.ndarray, qxy: float)``
where ``qxy`` is the log proposal-density ratio

    qxy = log q(x | y) - log q(y | x),

with ``x`` the CURRENT sample and ``y`` the PROPOSED sample (``q(a | b)`` is
the density of proposing ``a`` from ``b``). ``qxy`` is ADDED to the
log-posterior ratio in the Metropolis-Hastings acceptance, so positive
``qxy`` favors acceptance and symmetric proposals return ``0.0``. Checkpoints
store no code, so proposals are not serialized: to resume, re-register the same
proposals in the same order with the same weights (the sampler verifies this).
Use module-level functions or callable classes, and give callable classes a
``__name__`` attribute (it keys acceptance-rate reports and the resume check).
The module
provides the adaptive kernels :func:`am`, :func:`scam`, and the
min-fill-gated :func:`de` (:class:`DEProposal` registers it with a
non-default ``min_fill``; :class:`EarlyDE` / :func:`make_early_de` are
backward-compatibility aliases), the :func:`gaussian` random walk, the
product-space label-switching :class:`SourceSwapProposal`, and the per-chain
:class:`JumpProposals` / per-ladder :class:`ProposalBundle` containers
that select among registered proposals by weight.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from impulse.chain_stats import ChainStats
from impulse.product_space import ParameterLayout
from impulse.sampler_state import SamplerState

_SQRT2_INV = 2.4 / math.sqrt(2)  # constant for SCAM (neff is always 1)

# Module-level default for the DE activation threshold: minimum number of
# history-buffer samples before :func:`de` runs its difference move.
DE_MIN_FILL = 100


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
    - The selected proposal always runs as registered; proposals gate
      themselves (e.g. :func:`de` returns the current position unchanged
      until its history buffer holds ``min_fill`` samples)
    - Each proposal function should return (new_sample, log_proposal_ratio)
    """

    def __init__(
        self,
        chain_stats: ChainStats,
        proposal_list: list | None = None,
        proposal_weights: list | None = None,
        proposal_probs: np.ndarray | None = None,
    ):
        self.chain_stats = chain_stats
        self.proposal_list = proposal_list if proposal_list is not None else []
        self.proposal_weights = proposal_weights if proposal_weights is not None else []
        self.proposal_probs = proposal_probs
        self._last_proposal_idx = -1
        n = len(self.proposal_list)
        self._proposal_calls = np.zeros(n, dtype=np.int64)
        self._proposal_accepts = np.zeros(n, dtype=np.int64)
        self._proposal_cumprobs = (
            np.cumsum(self.proposal_probs) if self.proposal_probs is not None else np.zeros(0)
        )

    def add_jump(self, jump: Callable, weight: float) -> None:
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
            self._proposal_calls = np.append(self._proposal_calls, 0)
            self._proposal_accepts = np.append(self._proposal_accepts, 0)
        elif weight != self.proposal_weights[self.proposal_list.index(jump)]:
            self.proposal_weights[self.proposal_list.index(jump)] = weight
        total = sum(self.proposal_weights)
        if total > 0:
            self.proposal_probs = np.array(self.proposal_weights) / total
        else:
            self.proposal_probs = np.ones(len(self.proposal_weights)) / len(self.proposal_weights)
        self._refresh_cumprobs()

    def _refresh_cumprobs(self) -> None:
        """Cache the cumulative selection probabilities for :meth:`__call__`.

        Kept in sync with ``proposal_probs`` by every writer of the weights.
        An empty cache is correct for a bundle with no proposals registered
        yet -- ``__call__`` cannot run in that state anyway.
        """
        probs = self.proposal_probs
        self._proposal_cumprobs = np.zeros(0) if probs is None else np.cumsum(probs)

    def _select_proposal_index(self, rng) -> int:
        """Draw a proposal index from the weight distribution.

        Equivalent in distribution to ``rng.choice(k, p=probs)``, but ~5x
        cheaper: ``Generator.choice`` with a ``p=`` argument validates and
        normalizes the probability vector on every call, and this runs once per
        (iteration x temperature) -- it measured 4.53 us/draw versus 0.94 us for
        the inverse-CDF form, roughly a fifth of per-chain proposal cost.

        The clamp matters: ``cumsum`` may end a few ULP below 1.0, so a
        ``random()`` draw above that would index one past the end.
        """
        idx = int(np.searchsorted(self._proposal_cumprobs, rng.random(), side="right"))
        return min(idx, len(self.proposal_list) - 1)

    def __call__(self, state: SamplerState) -> Tuple[np.ndarray, float]:
        chain_stats = self.chain_stats
        chain_stats.update_sample(state.positions[chain_stats.chain_index])
        idx = self._select_proposal_index(chain_stats.rng)
        self._last_proposal_idx = idx
        self._proposal_calls[idx] += 1
        return self.proposal_list[idx](chain_stats)

    def report_accept(self, accepted: bool) -> None:
        """Report whether the last proposal was accepted."""
        if accepted and self._last_proposal_idx >= 0:
            self._proposal_accepts[self._last_proposal_idx] += 1

    def acceptance_rates(self) -> dict:
        """Return per-proposal acceptance statistics.

        Returns
        -------
        dict
            ``{proposal_name: {calls, accepts, rate}}`` for each proposal.
        """
        result = {}
        for i, proposal in enumerate(self.proposal_list):
            calls = int(self._proposal_calls[i])
            accepts = int(self._proposal_accepts[i])
            rate = accepts / calls if calls > 0 else 0.0
            result[proposal.__name__] = {"calls": calls, "accepts": accepts, "rate": rate}
        return result

    def __setstate__(self, state):
        """Restore from pickle, initializing counters for old checkpoints."""
        self.__dict__.update(state)
        n = len(self.proposal_list)
        if not hasattr(self, "_last_proposal_idx"):
            self._last_proposal_idx = -1
        if not hasattr(self, "_proposal_calls"):
            self._proposal_calls = np.zeros(n, dtype=np.int64)
        if not hasattr(self, "_proposal_accepts"):
            self._proposal_accepts = np.zeros(n, dtype=np.int64)
        # Derived cache, absent from checkpoints written before the
        # inverse-CDF selection landed. Rebuild rather than store it.
        if not hasattr(self, "_proposal_cumprobs"):
            self._refresh_cumprobs()


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

    jump_proposals: List["JumpProposals"]

    def get_new_position(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        """Propose one new position per temperature chain.

        This loop is serial by construction: each chain draws from its OWN
        generator, so the chains cannot be batched into single array
        operations without merging their RNG streams -- which would change
        every chain and break the per-chain reproducibility the resume
        contract depends on. The per-iteration allocations are hoisted
        instead; see :meth:`JumpProposals.__call__` for the per-chain cost.
        """
        jump_proposals = self.jump_proposals
        # empty_like, not zeros_like: every row is overwritten below, so
        # zero-filling first is pure waste at this call rate.
        new_samples = np.empty_like(state.positions)
        qxys = np.empty(len(jump_proposals))
        for i, jp in enumerate(jump_proposals):
            new_samples[i], qxys[i] = jp(state)
        return new_samples, qxys

    def add_jump(self, jump: Callable, weight: float) -> None:
        for jp in self.jump_proposals:
            jp.add_jump(jump, weight)

    def __call__(self, state: SamplerState) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_new_position(state)

    def report_accepts(self, accepts: np.ndarray) -> None:
        """Report acceptance results to each chain's JumpProposals.

        Parameters
        ----------
        accepts : np.ndarray
            Boolean/int array of length ``nchains``.
        """
        for i, jp in enumerate(self.jump_proposals):
            jp.report_accept(bool(accepts[i]))

    def acceptance_report(self) -> dict:
        """Aggregate per-proposal acceptance statistics across all chains.

        Returns
        -------
        dict
            ``{name: {calls, accepts, rate, per_chain: [...]}}``
        """
        # Collect per-chain stats
        chain_reports = [jp.acceptance_rates() for jp in self.jump_proposals]
        # Gather all proposal names (preserving order from first chain)
        all_names = list(chain_reports[0].keys()) if chain_reports else []
        result = {}
        for name in all_names:
            total_calls = 0
            total_accepts = 0
            per_chain = []
            for cr in chain_reports:
                if name in cr:
                    total_calls += cr[name]["calls"]
                    total_accepts += cr[name]["accepts"]
                    per_chain.append(cr[name])
            rate = total_accepts / total_calls if total_calls > 0 else 0.0
            result[name] = {
                "calls": total_calls,
                "accepts": total_accepts,
                "rate": rate,
                "per_chain": per_chain,
            }
        return result

    def chain_acceptance_rates(self) -> list:
        """Overall MH acceptance rate for each chain, across all proposals.

        Returns
        -------
        list of dict
            One entry per chain, in temperature order:
            ``{calls, accepts, rate, per_proposal: {name: rate}}``.
        """
        out = []
        for jp in self.jump_proposals:
            rates = jp.acceptance_rates()
            total_calls = int(sum(r["calls"] for r in rates.values()))
            total_accepts = int(sum(r["accepts"] for r in rates.values()))
            out.append(
                {
                    "calls": total_calls,
                    "accepts": total_accepts,
                    "rate": (total_accepts / total_calls) if total_calls > 0 else 0.0,
                    "per_proposal": {name: r["rate"] for name, r in rates.items()},
                }
            )
        return out


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
    # ChainStats.__post_init__ / update_sample guarantee these are set
    assert chain_stats.current_sample is not None
    assert chain_stats.groups is not None and chain_stats.proposal_L is not None
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    # jumpind = np.random.randint(0, len(groups))

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10.0

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # make correlated adaptive jump using precomputed L = U * sqrt(S)
    group = chain_stats.groups[jumpind]
    neff = len(group)
    cd = 2.4 / math.sqrt(2 * neff) * scale
    proposal_L = chain_stats.proposal_L[jumpind]
    assert proposal_L is not None
    q[group] += cd * (proposal_L @ rng.standard_normal(neff))

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
    # ChainStats.__post_init__ / update_sample guarantee these are set
    assert chain_stats.current_sample is not None
    assert chain_stats.groups is not None and chain_stats.proposal_L is not None
    q = chain_stats.current_sample.copy()
    qxy = 0

    # choose group
    jumpind = rng.integers(0, len(chain_stats.groups))
    ndim = len(chain_stats.groups[jumpind])

    # adjust step size
    prob = rng.random()

    # large jump
    if prob > 0.97:
        scale = 10.0

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # standard medium jump
    else:
        scale = 1.0

    # make correlated componentwise adaptive jump
    ind = rng.integers(0, ndim)
    cd = _SQRT2_INV * scale

    proposal_L = chain_stats.proposal_L[jumpind]
    assert proposal_L is not None
    q[chain_stats.groups[jumpind]] += rng.standard_normal() * cd * proposal_L[:, ind]

    return q, qxy


def de(chain_stats: ChainStats, min_fill: int = DE_MIN_FILL) -> tuple[np.ndarray, float]:
    """
    Differential-evolution proposal gated on a minimum buffer fill.

    Runs the difference move ``x' = x + gamma * (b_mm - b_nn)`` on one
    parameter group, with ``b_mm``, ``b_nn`` rows of the sample-history
    buffer.  The move activates as soon as the current buffer holds at
    least ``min_fill`` samples (module default :data:`DE_MIN_FILL`, 100);
    below the threshold the proposal returns the current position
    unchanged (identity kernel) — nothing is substituted for it.  Rows
    are drawn from the TAIL of the circular buffer (``ChainStats`` fills
    its buffer from the tail, so the head of a partially filled buffer
    is zero padding).

    The min-fill gate matters most for product-space model selection: per-model buffers split
    the run's samples across all model indices, so at realistic run
    lengths no model's buffer ever fills completely, yet the mixture
    still needs this history-based, ridge-following move after only
    ``min_fill`` within-model samples.

    Parameters
    ----------
    chain_stats : ChainStats
        Statistics object containing the sample-history buffer and random
        generator.  When per-model statistics are active (product-space model selection),
        ``update_sample`` has already swapped in the CURRENT model's
        buffer and sample count, so the gate applies per model.
    min_fill : int, default DE_MIN_FILL (100)
        Minimum number of buffered samples before the difference move
        runs.  Values below 2 are treated as 2 (two distinct rows are
        needed).  To register the move with a non-default threshold use
        :class:`DEProposal` (the samplers' ``de_min_fill`` argument does
        this for you).

    Returns
    -------
    tuple of (np.ndarray, float)
        new_position : np.ndarray
            Proposed parameter values (the current position, unchanged,
            while fewer than ``min_fill`` samples are buffered).
        log_proposal_ratio : float
            Always 0; see Notes for the derivation.

    Examples
    --------
    >>> new_pos, log_ratio = de(chain_stats)
    >>> # identity below min_fill, difference move above it

    Notes
    -----
    ``qxy`` derivation: the move is fixed-dimension, ``x' = x + gamma *
    (b_mm - b_nn)`` on one group, where the group index, the row pair
    ``(mm, nn)``, and the scale ``gamma`` are drawn from distributions that
    do not depend on ``x``, and the buffer is frozen within the iteration.
    The reverse move ``x' -> x`` is generated by the same kernel with the
    row pair swapped to ``(nn, mm)`` at the same ``gamma`` and group, which
    is drawn with identical probability, so ``q(x'|x) = q(x|x')`` and
    ``qxy = log[q(x|x') / q(x'|x)] = 0``.  Below ``min_fill`` the kernel is
    the identity, which trivially satisfies detailed balance (``qxy = 0``).

    Scale schedule: ``gamma = 1`` (mode jump, 50% of draws) or
    ``uniform(0, 1) * 2.4 / sqrt(2 d)`` with ``d`` the group size.

    The proposal only ever modifies entries of one continuous-parameter
    group.  In RJ configurations the groups exclude the model index (both
    ``get_default_groups`` and the per-model groups swapped in by
    ``ChainStats.update_sample``), so ``nmodel`` is never touched and the
    trans-dimensional birth/death kernel's exactness is unaffected.

    Using the chain's own history makes this an adaptive proposal; the
    buffer update cadence satisfies diminishing adaptation in the usual
    way.
    """
    rng = chain_stats.rng
    # ChainStats.__post_init__ / update_sample guarantee these are set
    assert chain_stats.current_sample is not None and chain_stats.groups is not None
    q = chain_stats.current_sample.copy()

    # When per-model statistics are active, update_sample() has swapped
    # in the CURRENT model's buffer and sample_total.
    n_filled = min(chain_stats.sample_total, chain_stats.buffer_size)
    if n_filled < max(min_fill, 2):
        return q, 0.0

    # tail of the circular buffer = the filled portion
    buf = chain_stats._buffer[-n_filled:]

    # choose group. Cached index arrays: rebuilding a Python list here on
    # every call was measurable at once-per-iteration-per-temperature, and
    # the indices are constant for a given groups object.
    groups = chain_stats.group_index_arrays()
    jumpind = rng.integers(0, len(groups))
    group = groups[jumpind]
    ndim = group.size

    # two distinct history rows
    mm = rng.integers(0, n_filled)
    nn = rng.integers(0, n_filled)
    while mm == nn:
        nn = rng.integers(0, n_filled)

    # get jump scale size
    if rng.random() > 0.5:
        scale = 1.0  # mode jump
    else:
        scale = rng.random() * 2.4 / np.sqrt(2 * ndim)

    q[group] += scale * (buf[mm, group] - buf[nn, group])
    return q, 0.0


class DEProposal:
    """
    Picklable differential-evolution proposal with a configurable ``min_fill``.

    Callable-class wrapper around :func:`de` for registering the
    difference move with a non-default activation threshold.  A class
    rather than a closure so the threshold is inspectable state and the
    object carries a stable ``__name__`` for acceptance reports and the
    resume check.  ``__name__`` is ``'de'``, so acceptance-rate reports key
    it identically to the module-level :func:`de`.

    Parameters
    ----------
    min_fill : int, default DE_MIN_FILL (100)
        Minimum number of buffered samples before the difference move
        runs.  Must be at least 2 (two distinct rows are needed).

    Examples
    --------
    >>> sampler.add_custom_jump(DEProposal(min_fill=200), weight=15)
    """

    __name__ = "de"

    def __init__(self, min_fill: int = DE_MIN_FILL):
        if min_fill < 2:
            raise ValueError(
                f"min_fill must be >= 2 (two distinct buffer rows are "
                f"needed for a difference move), got {min_fill}"
            )
        self.min_fill = int(min_fill)

    def __call__(self, chain_stats: ChainStats) -> tuple[np.ndarray, float]:
        return de(chain_stats, min_fill=self.min_fill)


class EarlyDE(DEProposal):
    """
    Backward-compatibility alias for :class:`DEProposal`.

    Historically the stock :func:`de` required a completely FULL buffer
    (more than ``buffer_size`` samples, 50,000 by default — never reached
    at realistic run lengths) and ``JumpProposals`` silently substituted
    :func:`gaussian` for it; ``EarlyDE`` was the separate min-fill-gated
    variant that restored the difference move.  The min-fill gate now
    lives in :func:`de` itself and the substitution is gone, so this
    class remains only for existing
    ``add_custom_jump(make_early_de(...), ...)`` call sites and for
    resuming old checkpoints.  It reports under the historical
    ``'early_de'`` name: old RJ configurations register both a weight-0
    ``de`` and an ``EarlyDE``, and keeping the names distinct avoids
    colliding acceptance-report keys on resume.
    """

    __name__ = "early_de"


def make_early_de(min_fill: int = DE_MIN_FILL) -> EarlyDE:
    """
    Create a min-fill-gated differential-evolution proposal (compat alias).

    Backward-compatibility factory for :class:`EarlyDE`.  New code should
    rely on the min-fill-gated :func:`de` that the samplers register
    (``de_weight`` / ``de_min_fill``) or register :class:`DEProposal`
    directly.

    Parameters
    ----------
    min_fill : int, default DE_MIN_FILL (100)
        Minimum buffer fill before the difference move activates; see
        :func:`de`.

    Returns
    -------
    EarlyDE
        Picklable callable with signature ``(ChainStats) -> (np.ndarray, float)``.

    Examples
    --------
    >>> sampler.add_custom_jump(make_early_de(200), weight=15)
    """
    return EarlyDE(min_fill)


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
    # ChainStats.__post_init__ / update_sample guarantee these are set
    assert chain_stats.current_sample is not None and chain_stats.groups is not None
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
    layout : ParameterLayout, optional
        Product-space parameter layout (single source of truth for the
        source-block slices and the model-index position).  When omitted
        (legacy construction/checkpoints), the layout is recovered from
        ``num_params`` and the length of the parameter vector at call time.

    Notes
    -----
    - Model index is assumed to be in the last position of the parameter vector
    - Only attempts swap if number of sources > 0
    - Returns original position unchanged if only one source or same source selected
    - Proposal is symmetric, so log_proposal_ratio = 0
    """

    __name__ = "source_swap_proposal"

    def __init__(self, num_params: int = 3, layout: Optional[ParameterLayout] = None):
        self.num_params = num_params
        self.layout = layout

    def __getattr__(self, name):
        """Back-fill ``layout`` on instances unpickled from pre-layout checkpoints."""
        if name == "layout":
            self.layout = None
            return None
        raise AttributeError(name)

    def __call__(self, chain_stats: ChainStats) -> tuple[np.ndarray, float]:
        rng = chain_stats.rng
        # ChainStats.__post_init__ / update_sample guarantee this is set
        assert chain_stats.current_sample is not None
        q = chain_stats.current_sample.copy()
        qxy = 0
        layout = self.layout
        if layout is None:
            # Legacy-constructed instance: recover the layout from the
            # per-source size and the concrete vector length.
            layout = ParameterLayout.from_total_dim(self.num_params, q.shape[0])
        nmodel = layout.model_index_of(q)
        if nmodel == 0:
            return q, qxy
        swap_source_1 = int(rng.integers(0, nmodel + 1))
        swap_source_2 = int(rng.integers(0, nmodel + 1))
        if swap_source_1 == swap_source_2:
            return q, qxy

        slice_1 = layout.source_slice(swap_source_1)
        slice_2 = layout.source_slice(swap_source_2)
        x = q[slice_1].copy()
        y = q[slice_2].copy()

        q[slice_1] = y
        q[slice_2] = x
        return q, qxy


def make_source_swap_proposal(
    num_params: int = 3, layout: Optional[ParameterLayout] = None
) -> SourceSwapProposal:
    """
    Create a source-swap proposal for swapping parameters between sources.

    Parameters
    ----------
    num_params : int, default 3
        Number of parameters per source.
    layout : ParameterLayout, optional
        Product-space parameter layout; without it the layout is recovered
        from ``num_params`` and the vector length at call time.

    Returns
    -------
    SourceSwapProposal
        Picklable callable with signature ``(ChainStats) -> (np.ndarray, float)``.

    Examples
    --------
    >>> sampler.add_custom_jump(make_source_swap_proposal(5), weight=25)
    """
    return SourceSwapProposal(num_params, layout=layout)


# Backward-compatible default: 3 parameters per source
source_swap_proposal = make_source_swap_proposal(3)

"""
RJMCMC birth/death proposals in the product space.

All proposals are picklable callable classes so that sampler checkpointing
works correctly.
"""

import warnings
from typing import Callable, Optional

import numpy as np

from impulse.product_space import ParameterLayout


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

    The new source is drawn from the prior into the next free slot and the
    model index is incremented.  The acceptance ratio is the birth/death
    move-selection ratio only; the change this move makes to the (full) product
    space prior is cancelled in ``qxy`` so that — for a source drawn from the
    prior — the move is an exact reverse of :class:`DeathProposal` and leaves
    the model-index posterior unbiased.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    draw_from_prior : callable
        ``draw_from_prior(rng) -> np.ndarray`` of shape ``(num_params,)``.
    log_proposal_density : callable, optional
        Log-density of the distribution ``draw_from_prior`` actually samples
        from.  Only needed when that distribution differs from the prior.
    log_prior_density : callable, optional
        ``log_prior_density(params) -> float``, the log prior density of one
        source's parameters.  Used as the draw density (which cancels the new
        slot's contribution to the full product-space prior) when
        ``log_proposal_density`` is not given.  If ``None`` the prior is
        assumed flat over the (in-bounds) source parameters; supply it for
        any non-flat prior.
        :class:`~impulse.rjmcmc.BirthDeathProductSpace` wires this automatically.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.

    Notes
    -----
    ``qxy`` uses the log-density of the distribution ``draw_from_prior``
    *actually* samples: ``log_proposal_density`` when supplied, otherwise
    ``log_prior_density`` (``None`` meaning a flat draw density whose terms
    cancel).  Together with the matching :class:`DeathProposal` terms this
    makes the birth/death pair *exactly* balanced pointwise — including
    when the draw distribution differs from the prior — because the
    kill-last death is the exact inverse state transition of this
    append-last birth and re-fills the vacated slot from the same draw
    distribution.  See :class:`DeathProposal` for the derivation.

    .. warning::
        Do **not** register this proposal standalone in a constant-weight
        mixture (e.g. via ``add_custom_jump``).  Its ``qxy`` assumes birth vs
        death is *selected* with the ``prob_schedule`` probabilities; a
        constant-weight mixture selects it with a state-independent
        probability instead, which violates detailed balance and biases the
        model posterior (boundary model indices get twice the interior
        weight under the default schedule).  Use it only as a component of
        :class:`BirthDeathProposal`, which performs the schedule-driven
        selection internally.
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

    @property
    def layout(self) -> ParameterLayout:
        """Product-space parameter layout derived from the stored scalars.

        A property rather than a stored attribute so the pickled attribute
        set (checkpoint serialization format) is unchanged and instances
        restored from pre-layout checkpoints get it for free.
        """
        return ParameterLayout(num_params=self.num_params, num_models=self.max_sources)

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        layout = self.layout
        q = chain_stats.current_sample.copy()
        nmodel = layout.model_index_of(q)

        if nmodel >= self.max_sources - 1:
            return q, 0.0

        slot = layout.source_slice(nmodel + 1)
        old_params = q[slot].copy()
        new_params = self.draw_from_prior(rng)
        q[slot] = new_params
        layout.set_model_index(q, nmodel + 1)

        p_birth_k, _ = self.prob_schedule(nmodel, self.max_sources)
        _, p_death_k1 = self.prob_schedule(nmodel + 1, self.max_sources)

        # Move-selection ratio.  The previous ``-log(nmodel + 2)`` source-count
        # term is unmatched by the *uniform* model-index prior and biased the
        # posterior toward fewer sources; see
        # ``tests/test_rjmcmc_detailed_balance.py``.
        qxy = np.log(p_death_k1) - np.log(p_birth_k)

        # Slot re-fill ratio: the forward move draws ``new_params`` with the
        # draw density q, and the reverse death move re-fills the vacated slot
        # (i.e. re-creates ``old_params``) from the same draw distribution, so
        # the proposal-density ratio for the overwritten slot is
        # ``q(old) / q(new)``.  The per-source *prior* densities of the slot
        # values are not part of ``qxy``: they enter through the target itself
        # because the sampler evaluates the prior on ALL slots.  See
        # :class:`DeathProposal` for the detailed-balance derivation.
        draw_logpdf = self.log_proposal_density or self.log_prior_density
        if draw_logpdf is not None:
            qxy += draw_logpdf(old_params) - draw_logpdf(new_params)

        return q, qxy


class DeathProposal:
    """
    Death proposal that removes the *last* active source from the product space.

    The victim is always the last active slot (index ``nmodel``), which makes
    the move the exact pointwise reverse of :class:`BirthProposal` (which
    appends into that same slot); no other slot moves.  The vacated slot is
    **refreshed with a fresh draw** from the same distribution the birth move
    samples.  The refresh is essential: keeping the killed source's
    (posterior-distributed) parameters in the now-inactive slot leaves the
    *joint* distribution of active and inactive slots wrong, and any move
    that later re-activates a stored slot with ``qxy = 0`` (e.g.
    :class:`NmodelJump`) then biases the model posterior toward more sources.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    draw_from_prior : callable
        ``draw_from_prior(rng) -> np.ndarray`` of shape ``(num_params,)``.
        Must be the same sampler the paired :class:`BirthProposal` uses; it
        re-fills the vacated slot.
    log_proposal_density : callable, optional
        Log-density of the distribution ``draw_from_prior`` actually samples
        from.  Only needed when that distribution differs from the prior.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters (used as the
        draw density when ``log_proposal_density`` is not given).  ``None``
        means the draw density is flat, so the slot re-fill terms cancel.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.

    Notes
    -----
    Let ``p`` be the per-source prior density (a factor of the full
    product-space target for *every* slot, active or inactive) and ``q`` the
    density of ``draw_from_prior`` (``q = p`` unless ``log_proposal_density``
    is given).  A death at model index ``m`` deactivates the last active
    slot ``m`` (parameters ``x_m``) and refreshes it with a fresh draw
    ``w ~ q``.  The reverse move is a birth at model index ``m - 1`` that
    redraws ``x_m ~ q`` into slot ``m``, overwriting ``w`` — the exact
    inverse state transition::

        forward proposal density:   p_death(m) * q(w)
        reverse proposal density:   p_birth(m - 1) * q(x_m)
        full-space target ratio:    [L_{m-1} / L_m] * p(w) / p(x_m)

    Pointwise detailed balance then fixes::

        qxy = log p_birth(m-1) - log p_death(m)     (move-selection ratio)
            + log q(x_m) - log q(w)                 (slot re-fill ratio)

    There is no kill-choice factor in ``qxy`` because there is no kill
    choice: the death has a single channel that pairs one-to-one with the
    append-last birth.  An earlier variant instead killed a victim chosen
    *uniformly* among the ``m + 1`` active sources (moving the last active
    source into the hole) while using this same ``qxy``, on the argument
    that the ``1 / (m + 1)`` kill choice cancels against the ``(m + 1)``
    orderings mapping to the same multiset of active sources.  That kernel
    is **not** the reverse of the append-last birth: the multiset
    cancellation balances only one of the two flow pairings, so the
    missing kill-choice factor leaves a genuine detailed-balance violation.
    Exact finite-state enumeration of the production kernels showed a
    residual stationary bias toward fewer sources for parameter-dependent
    exchangeable likelihoods (marginal maxerr ~3e-3 at ``max_sources = 4``),
    while the kill-last kernel with the identical ``qxy`` is exactly
    invariant (residual at machine precision).  See
    ``tests/test_rjmcmc_detailed_balance.py``.

    Because kill-last only ever removes the LAST active slot per move,
    trans-dimensional mixing across slots relies on the label-permuting and
    within-model moves in the mixture: ``SourceSwapProposal`` (registered
    by default in ``from_rjmcmc``) exchanges slot contents so any active
    source can reach the last slot, and the within-model moves relocate the
    continuous parameters.  Keep the swap move registered whenever
    birth/death is in use.

    The prior densities ``p(w)`` and ``p(x_m)`` never appear in ``qxy``: they
    enter through the target itself, because the sampler evaluates the prior
    on all slots.  When ``q = p`` the re-fill ratio is the prior-density
    ratio ``log p(killed) - log p(fresh)`` (and vanishes for a flat prior);
    when ``q != p`` it contains the death-side proposal Hastings term whose
    omission silently biased the model posterior.

    .. warning::
        Do **not** register this proposal standalone in a constant-weight
        mixture; see the warning on :class:`BirthProposal`.  Use it only as a
        component of :class:`BirthDeathProposal`.
    """

    __name__ = "death_proposal"

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

    @property
    def layout(self) -> ParameterLayout:
        """Product-space parameter layout derived from the stored scalars.

        A property rather than a stored attribute so the pickled attribute
        set (checkpoint serialization format) is unchanged and instances
        restored from pre-layout checkpoints get it for free.
        """
        return ParameterLayout(num_params=self.num_params, num_models=self.max_sources)

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        layout = self.layout
        q = chain_stats.current_sample.copy()
        nmodel = layout.model_index_of(q)

        if nmodel <= 0:
            return q, 0.0

        # Kill the LAST active source (slot ``nmodel``): the single-channel
        # exact reverse of the append-last birth.  (A uniformly chosen victim
        # compacted into the hole, with this same qxy, is NOT the reverse
        # move and leaves a residual bias toward fewer sources; see class
        # Notes.)  Mixing across slots is provided by SourceSwapProposal and
        # the within-model moves.
        slot = layout.source_slice(nmodel)
        killed_params = q[slot].copy()

        # Refresh the vacated slot with a fresh draw so inactive slots stay
        # draw-distributed rather than remembering the killed source (see
        # class Notes); the reverse birth overwrites this value.
        fresh_params = self.draw_from_prior(rng)
        q[slot] = fresh_params

        layout.set_model_index(q, nmodel - 1)

        _, p_death_k = self.prob_schedule(nmodel, self.max_sources)
        p_birth_km1, _ = self.prob_schedule(nmodel - 1, self.max_sources)

        # Move-selection ratio only from the schedule: the ``+log(nmodel+1)``
        # source-count term of previous versions is unmatched by the uniform
        # model-index prior and biased the posterior toward fewer sources.
        qxy = np.log(p_birth_km1) - np.log(p_death_k)

        # Slot re-fill ratio (see class Notes): forward draws ``fresh_params``
        # and the reverse birth redraws ``killed_params``, both with the draw
        # density q.
        draw_logpdf = self.log_proposal_density or self.log_prior_density
        if draw_logpdf is not None:
            qxy += draw_logpdf(killed_params) - draw_logpdf(fresh_params)

        return q, qxy


class BirthDeathProposal:
    """
    Combined birth-death kernel with schedule-driven move selection.

    Wraps a :class:`BirthProposal` and a :class:`DeathProposal` and, on each
    call, selects between them with the ``prob_schedule`` probabilities
    evaluated at the *current* model index.  This is the kernel that must be
    registered with the sampler: the sub-proposals' ``qxy`` terms contain the
    schedule ratio ``log[p_reverse / p_forward]``, which is only the correct
    Hastings factor when the forward move really is selected with probability
    ``p_forward``.  Registering birth and death as separate jumps in a
    constant-weight mixture selects them with state-independent probabilities
    and violates detailed balance (see the warnings on the sub-proposals).

    Parameters
    ----------
    birth : BirthProposal
        Birth move; must be constructed with the same ``prob_schedule``.
    death : DeathProposal
        Death move; must be constructed with the same ``prob_schedule``.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.
        Defaults to ``birth.prob_schedule``.  The returned pair is normalized
        for the selection draw, but the sub-proposals use the *raw* values in
        their Hastings ratios, so a custom schedule must be consistently
        normalized across model indices (the default schedule sums to 1).

    Raises
    ------
    ValueError
        If the birth and death proposals disagree on ``max_sources`` or
        ``num_params``; if ``max_sources < 2`` (a single model admits no
        trans-dimensional move — the default schedule would then always
        select a birth that can only no-op as a silently accepted move);
        or if the schedules fail validation: the selection, birth, and
        death schedules must agree pointwise at every model index, return
        non-negative probabilities, and keep ``p_birth + p_death``
        constant across model indices (a varying normalization would
        corrupt the sub-proposals' Hastings ratios).

    Warns
    -----
    UserWarning
        If ``birth.draw_from_prior is not death.draw_from_prior``: the
        death move re-fills the vacated slot from the same draw
        distribution the birth samples, and identity is the only check
        this constructor can make.  Distinct-but-equal callables trip the
        warning too; build the pair with
        :func:`make_birth_death_proposal`, which shares one callable, to
        avoid it.
    """

    __name__ = "birth_death"

    def __init__(
        self,
        birth: BirthProposal,
        death: DeathProposal,
        prob_schedule: Optional[Callable] = None,
    ):
        if birth.max_sources != death.max_sources:
            raise ValueError("birth and death proposals disagree on max_sources")
        if birth.num_params != death.num_params:
            raise ValueError("birth and death proposals disagree on num_params")
        if birth.max_sources < 2:
            raise ValueError(
                "BirthDeathProposal requires max_sources >= 2: with a single "
                "model there is no trans-dimensional move to make (the "
                "default schedule would always select a birth that can only "
                "no-op as a silently accepted move)"
            )
        if birth.draw_from_prior is not death.draw_from_prior:
            warnings.warn(
                "birth and death proposals hold distinct draw_from_prior "
                "callables (checked by identity): the death move must "
                "re-fill the vacated slot from the SAME draw distribution "
                "the birth samples, or the slot re-fill Hastings terms are "
                "wrong.  If the two callables do sample the same "
                "distribution, build the pair with "
                "make_birth_death_proposal, which wires one shared callable "
                "into both moves and avoids this warning.",
                UserWarning,
            )
        self.birth = birth
        self.death = death
        self.max_sources = birth.max_sources
        self.prob_schedule = prob_schedule or birth.prob_schedule
        self._validate_schedules()

    def _validate_schedules(self):
        """Check the selection/birth/death schedules are mutually consistent.

        The sub-proposals put the *raw* schedule values in their Hastings
        ratios while this kernel selects birth vs death with the *normalized*
        probabilities, so the ratios are exact only if all three callables
        agree pointwise and ``p_birth + p_death`` is the same at every model
        index (the normalization then cancels from the selection ratio).
        """
        total = None
        for k in range(self.max_sources):
            ref = self.prob_schedule(k, self.max_sources)
            for name, schedule in (
                ("birth", self.birth.prob_schedule),
                ("death", self.death.prob_schedule),
            ):
                vals = schedule(k, self.max_sources)
                if not np.allclose(vals, ref, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"the {name} proposal's prob_schedule disagrees with "
                        f"the selection schedule at nmodel={k}: {vals} != "
                        f"{ref}; construct all three with the same schedule"
                    )
            p_birth, p_death = ref
            if p_birth < 0.0 or p_death < 0.0:
                raise ValueError(
                    f"prob_schedule returned a negative probability at " f"nmodel={k}: {ref}"
                )
            if total is None:
                total = p_birth + p_death
            elif abs((p_birth + p_death) - total) > 1e-9:
                raise ValueError(
                    "p_birth + p_death must be constant across model indices "
                    f"(got {p_birth + p_death} at nmodel={k} but {total} at "
                    "nmodel=0): the sub-proposals use the raw schedule values "
                    "in their Hastings ratios, so a varying normalization "
                    "would bias the acceptance ratio"
                )

    def __call__(self, chain_stats):
        # The birth proposal carries the layout (with legacy-checkpoint
        # back-fill), so read the model index through it.
        nmodel = self.birth.layout.model_index_of(chain_stats.current_sample)
        p_birth, p_death = self.prob_schedule(nmodel, self.max_sources)
        total = p_birth + p_death
        if total <= 0.0:
            # Degenerate schedule: neither move is available from this state.
            # Reject outright rather than silently accepting a no-op.
            return chain_stats.current_sample.copy(), -np.inf
        if chain_stats.rng.random() < p_birth / total:
            return self.birth(chain_stats)
        return self.death(chain_stats)


class NmodelJump:
    """
    Uniform model-index jump proposal.

    Parameters
    ----------
    max_sources : int
        Maximum number of sources.
    layout : ParameterLayout, optional
        Product-space parameter layout locating the model index.  The
        model-index write itself is position-independent
        (:meth:`ParameterLayout.set_model_index` targets the trailing
        coordinate), so the jump behaves identically without one (legacy
        construction/checkpoints).
    """

    __name__ = "nmodel_jump"

    def __init__(self, max_sources: int, layout: Optional[ParameterLayout] = None):
        self.max_sources = max_sources
        self.layout = layout

    def __getattr__(self, name):
        """Back-fill ``layout`` on instances unpickled from pre-layout checkpoints."""
        if name == "layout":
            self.layout = None
            return None
        raise AttributeError(name)

    def __call__(self, chain_stats):
        rng = chain_stats.rng
        q = chain_stats.current_sample.copy()
        # Static: the model-index position is layout-independent (trailing
        # coordinate), so this also covers legacy instances whose layout
        # back-fills to None.
        ParameterLayout.set_model_index(q, rng.integers(0, self.max_sources))
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
        num_params,
        max_sources,
        draw_from_prior,
        log_proposal_density,
        log_prior_density,
        prob_schedule,
    )


def make_death_proposal(
    num_params: int,
    max_sources: int,
    draw_from_prior: Callable,
    log_proposal_density: Optional[Callable] = None,
    log_prior_density: Optional[Callable] = None,
    prob_schedule: Optional[Callable] = None,
) -> DeathProposal:
    """
    Create a death proposal that removes the last active source.

    Parameters
    ----------
    num_params : int
        Number of parameters per source.
    max_sources : int
        Maximum number of sources.
    draw_from_prior : callable
        ``draw_from_prior(rng) -> np.ndarray`` of shape ``(num_params,)``.
        Re-fills the vacated slot; must be the same sampler the paired birth
        proposal uses.
    log_proposal_density : callable, optional
        Log-density of the distribution ``draw_from_prior`` samples from,
        when it differs from the prior.
    log_prior_density : callable, optional
        Log-density of the prior on a single source's parameters.
    prob_schedule : callable, optional
        ``prob_schedule(nmodel, max_sources) -> (p_birth, p_death)``.

    Returns
    -------
    DeathProposal
    """
    return DeathProposal(
        num_params,
        max_sources,
        draw_from_prior,
        log_proposal_density,
        log_prior_density,
        prob_schedule,
    )


def make_birth_death_proposal(
    num_params: int,
    max_sources: int,
    draw_from_prior: Callable,
    log_proposal_density: Optional[Callable] = None,
    log_prior_density: Optional[Callable] = None,
    prob_schedule: Optional[Callable] = None,
) -> BirthDeathProposal:
    """
    Create the combined birth-death kernel.

    Builds a :class:`BirthProposal` / :class:`DeathProposal` pair sharing one
    ``prob_schedule`` and wraps them in a :class:`BirthDeathProposal` that
    selects between them with the schedule probabilities.  This is the kernel
    to register with the sampler; see :class:`BirthDeathProposal`.

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
    BirthDeathProposal
    """
    birth = BirthProposal(
        num_params,
        max_sources,
        draw_from_prior,
        log_proposal_density,
        log_prior_density,
        prob_schedule,
    )
    death = DeathProposal(
        num_params,
        max_sources,
        draw_from_prior,
        log_proposal_density,
        log_prior_density,
        prob_schedule,
    )
    return BirthDeathProposal(birth, death)


def make_nmodel_jump(max_sources: int, layout: Optional[ParameterLayout] = None) -> NmodelJump:
    """
    Create a uniform model-index jump proposal.

    Parameters
    ----------
    max_sources : int
        Maximum number of sources.
    layout : ParameterLayout, optional
        Product-space parameter layout locating the model index; without it
        the proposal writes the trailing coordinate (same position).

    Returns
    -------
    NmodelJump
    """
    return NmodelJump(max_sources, layout=layout)


# ---------------------------------------------------------------------------
# Legacy checkpoint migration
# ---------------------------------------------------------------------------


def _rebuild_combined_from_legacy_birth(legacy_birth) -> BirthDeathProposal:
    """Reconstruct the combined birth-death kernel from a legacy birth proposal.

    Checkpoints written before the detailed-balance fix pickled SEPARATE
    ``BirthProposal``/``DeathProposal`` instances.  The legacy
    ``BirthProposal`` carried everything needed to build the correct
    combined kernel (``num_params``, ``max_sources``, ``draw_from_prior``
    and the optional densities/schedule); the legacy ``DeathProposal`` had
    NO ``draw_from_prior`` and therefore can never be reused — the current
    death move re-fills the vacated slot from the birth draw distribution.
    Attribute access is defensive (``getattr``) because unpickled legacy
    instances bypass ``__init__`` and may predate any given attribute.

    Parameters
    ----------
    legacy_birth : object
        The unpickled legacy birth proposal instance (attribute layout of
        the pre-fix ``BirthProposal``).

    Returns
    -------
    BirthDeathProposal
        A freshly constructed combined kernel sharing one
        ``draw_from_prior`` between its birth and death moves.

    Raises
    ------
    TypeError
        If a required attribute is missing or of an unusable type, or if
        the legacy birth carries NEITHER ``log_proposal_density`` NOR
        ``log_prior_density``.  At the pre-fix HEAD both densities defaulted
        to ``None`` unless the user supplied one, but the current kernel's
        correctness requires the true per-source draw density: rebuilding
        with both ``None`` silently assumes a flat draw density, which gives
        wrong acceptance ratios for non-flat priors.  Migration is refused
        so the warn-only fallback fires and tells the user what to do.
    ValueError
        Propagated from :class:`BirthDeathProposal` validation (e.g.
        ``max_sources < 2`` or an inconsistent ``prob_schedule``).
    """
    num_params = getattr(legacy_birth, "num_params", None)
    max_sources = getattr(legacy_birth, "max_sources", None)
    draw_from_prior = getattr(legacy_birth, "draw_from_prior", None)
    if not isinstance(num_params, (int, np.integer)):
        raise TypeError("legacy birth proposal has no usable num_params")
    if not isinstance(max_sources, (int, np.integer)):
        raise TypeError("legacy birth proposal has no usable max_sources")
    if not callable(draw_from_prior):
        raise TypeError("legacy birth proposal has no usable draw_from_prior")
    log_proposal_density = getattr(legacy_birth, "log_proposal_density", None)
    log_prior_density = getattr(legacy_birth, "log_prior_density", None)
    if not callable(log_proposal_density):
        log_proposal_density = None
    if not callable(log_prior_density):
        log_prior_density = None
    if log_proposal_density is None and log_prior_density is None:
        # Cannot verify the draw density is flat; a silently wrong density
        # would bias the rebuilt kernel for non-flat priors.
        raise TypeError(
            "legacy birth proposal carries neither log_proposal_density nor "
            "log_prior_density; the combined kernel needs the per-source "
            "draw density, so migration is refused"
        )
    prob_schedule = getattr(legacy_birth, "prob_schedule", None)
    return make_birth_death_proposal(
        int(num_params),
        int(max_sources),
        draw_from_prior,
        log_proposal_density=log_proposal_density,
        log_prior_density=log_prior_density,
        prob_schedule=prob_schedule if callable(prob_schedule) else None,
    )


def migrate_legacy_birth_death(jump_proposals) -> Optional[BirthDeathProposal]:
    """Best-effort in-place migration of pre-fix separate birth/death jumps.

    Checkpoints written before the detailed-balance fix registered
    ``birth_proposal`` and ``death_proposal`` as separate constant-weight
    jumps, a wiring that violates detailed balance and biases the model
    posterior toward fewer sources.  This function replaces the pair, in
    EVERY per-chain :class:`~impulse.proposals.JumpProposals`, with one
    freshly reconstructed combined :class:`BirthDeathProposal` (built via
    :func:`make_birth_death_proposal` from the unpickled legacy birth
    proposal) whose selection weight is the SUM of the two legacy weights.
    All other proposals, their weights, and the total weight are left
    untouched, so the mixture normalization of the untouched proposals is
    unchanged.  Per-proposal call/accept counters for the pair are merged
    into the combined entry.

    The migration is atomic: every chain is validated and its replacement
    lists are fully built before any chain is mutated, so a failure at any
    point leaves the sampler exactly as it was loaded (never
    half-migrated) and this function returns ``None`` instead of raising.

    Parameters
    ----------
    jump_proposals : list of JumpProposals
        Per-chain proposal collections restored from a checkpoint (e.g.
        ``sampler.proposal_bundle.jump_proposals``).

    Returns
    -------
    BirthDeathProposal or None
        The reconstructed combined kernel — the SAME instance is
        registered in every chain, mirroring ``add_custom_jump`` — on
        success; ``None`` if reconstruction was not possible, in which
        case no chain was modified.
    """
    try:
        combined = None
        commits = []
        for jp in jump_proposals:
            names = [getattr(p, "__name__", "") for p in jp.proposal_list]
            # Require exactly the unambiguous legacy pair in this chain;
            # anything else (missing half, duplicates) is not migratable.
            if names.count("birth_proposal") != 1 or names.count("death_proposal") != 1:
                return None
            bi = names.index("birth_proposal")
            di = names.index("death_proposal")
            if combined is None:
                combined = _rebuild_combined_from_legacy_birth(jp.proposal_list[bi])
            keep, drop = min(bi, di), max(bi, di)
            new_list = list(jp.proposal_list)
            new_weights = [float(w) for w in jp.proposal_weights]
            if len(new_weights) != len(new_list):
                return None
            new_weights[keep] = new_weights[bi] + new_weights[di]
            new_list[keep] = combined
            del new_list[drop]
            del new_weights[drop]
            total = sum(new_weights)
            if total > 0:
                new_probs = np.asarray(new_weights) / total
            else:
                new_probs = np.ones(len(new_weights)) / len(new_weights)
            new_calls = _merge_counter(
                getattr(jp, "_proposal_calls", None), len(names), keep, drop, bi, di
            )
            new_accepts = _merge_counter(
                getattr(jp, "_proposal_accepts", None), len(names), keep, drop, bi, di
            )
            commits.append((jp, new_list, new_weights, new_probs, new_calls, new_accepts))
        if combined is None:
            return None
        # Commit phase: plain attribute assignment only, so nothing below
        # can fail after the first chain has been mutated.
        for jp, plist, weights, probs, calls, accepts in commits:
            jp.proposal_list = plist
            jp.proposal_weights = weights
            jp.proposal_probs = probs
            jp._proposal_calls = calls
            jp._proposal_accepts = accepts
            # A pickled last-proposal index may point at a removed entry.
            jp._last_proposal_idx = -1
        return combined
    except Exception:
        return None


def _merge_counter(counter, n_old, keep, drop, bi, di) -> np.ndarray:
    """Merge a per-proposal counter array across the migrated pair.

    Parameters
    ----------
    counter : np.ndarray or None
        Counter aligned with the pre-migration proposal list, or ``None``
        (or a stale length) for checkpoints predating the counters.
    n_old : int
        Pre-migration proposal count.
    keep, drop : int
        Indices of the retained (combined) and removed entries.
    bi, di : int
        Indices of the legacy birth and death entries.

    Returns
    -------
    np.ndarray
        Counter aligned with the post-migration proposal list; the
        combined entry carries the pair's summed counts.
    """
    if counter is None or len(counter) != n_old:
        return np.zeros(n_old - 1, dtype=np.int64)
    merged = np.asarray(counter, dtype=np.int64).copy()
    merged[keep] = merged[bi] + merged[di]
    return np.delete(merged, drop)

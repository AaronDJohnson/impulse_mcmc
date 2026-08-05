"""Wiring shared by the product-space (birth-death) sampler constructors.

These helpers previously lived in the core PT engine, which meant the supported
:class:`impulse.PTSampler` imported model-selection code it never uses. They
belong with the experimental product-space machinery.
"""

import warnings

import numpy as np

from impulse.experimental.birth_death_proposals import migrate_legacy_birth_death


def _expand_product_space_cov_mean(product_space, kwargs: dict) -> tuple:
    """Expand per-source ``sample_cov`` / ``sample_mean`` to the product space.

    Shared by :func:`~impulse.experimental.make_product_space_sampler` and
    ``HybridPTSampler.from_product_space``:
    pops ``sample_cov`` / ``sample_mean`` out of ``kwargs`` (so they are not
    forwarded twice) and, when they are shaped for a SINGLE source block,
    tiles them block-diagonally / block-wise across all model slots of the
    full product space (with unit variance for the trailing model index).
    Values already shaped for the full space pass through unchanged.

    Parameters
    ----------
    product_space : BirthDeathProductSpace
        Configured birth-death product space.
    kwargs : dict
        Keyword arguments destined for the sampler constructor; mutated in
        place (``sample_cov`` / ``sample_mean`` are removed).

    Returns
    -------
    tuple
        ``(sample_cov, sample_mean)`` expanded (or passed through / None).
    """
    layout = product_space.layout
    sample_cov = kwargs.pop("sample_cov", None)
    if sample_cov is not None:
        sample_cov = np.asarray(sample_cov)
        if sample_cov.shape == (layout.num_params, layout.num_params):
            full_cov = np.zeros((layout.total_dim, layout.total_dim))
            for i in range(layout.num_models):
                sl = layout.source_slice(i)
                full_cov[sl, sl] = sample_cov
            full_cov[layout.nmodel_index, layout.nmodel_index] = 1.0  # model index
            sample_cov = full_cov
    sample_mean = kwargs.pop("sample_mean", None)
    if sample_mean is not None:
        sample_mean = np.asarray(sample_mean)
        if sample_mean.shape == (layout.num_params,):
            full_mean = np.zeros(layout.total_dim)
            for i in range(layout.num_models):
                full_mean[layout.source_slice(i)] = sample_mean
            sample_mean = full_mean
    return sample_cov, sample_mean


def _register_model_selection_jumps(
    sampler,
    product_space,
    *,
    birth_weight: float,
    death_weight: float,
    nmodel_weight: float,
    swap_weight: float,
) -> None:
    """Register the model-move jump set on a freshly constructed sampler.

    Shared tail of both product-space constructors: registers the ONE
    combined birth-death kernel (separate constant-weight birth/death jumps
    violate detailed balance), the model-index jump, and the source-swap
    proposal for multi-model spaces, and enables per-model chain
    statistics.  The min-fill-gated :func:`~impulse.proposals.de` move is
    registered by the sampler constructor itself (``de_weight`` /
    ``de_min_fill``), so no extra DE registration happens here.
    """
    # Trans-dimensional and label-permuting jumps only exist for
    # multi-model spaces: with a single model there is no birth/death
    # move to make, no other model index to jump to, and no second
    # source slot to swap with (BirthDeathProposal itself rejects
    # max_sources < 2), so only the standard continuous jumps are
    # registered.
    if product_space.num_models > 1:
        if birth_weight != death_weight:
            warnings.warn(
                "birth_weight != death_weight has no effect on the birth/death "
                "split: birth and death form one combined kernel selected with "
                "weight birth_weight + death_weight, and the split is governed "
                "by the space's prob_schedule.",
                UserWarning,
            )
        # Birth and death must be one kernel with schedule-driven selection;
        # separate constant-weight jumps violate detailed balance (see
        # impulse.experimental.birth_death_proposals.BirthDeathProposal).
        if birth_weight + death_weight > 0:
            sampler.add_custom_jump(
                product_space.get_birth_death_proposal(), birth_weight + death_weight
            )
        sampler.add_custom_jump(product_space.get_nmodel_jump(), nmodel_weight)
        sampler.add_custom_jump(product_space.get_source_swap_proposal(), swap_weight)
    sampler.multi_chain_stats.enable_per_model(
        product_space.num_models,
        product_space.num_params,
        layout=product_space.layout,
    )


def migrate_or_warn_legacy_birth_death(sampler) -> None:
    """Migrate resumed pre-fix birth/death wiring, or warn loudly.

    Checkpoints written before the detailed-balance fix register
    ``birth_proposal`` and ``death_proposal`` as SEPARATE
    constant-weight jumps. That wiring violates detailed balance and
    biases the model posterior toward fewer sources; resuming it
    unchanged reproduces the bias. Detection starts from the proposal
    ``__name__``\\ s over the restored proposal lists, then inspects the
    attribute layout: CURRENT-code standalone registrations carry the
    same ``__name__``\\ s, but the current ``DeathProposal`` stores
    ``draw_from_prior`` (it re-fills the vacated slot) while the legacy
    one never did.  A pair whose death proposals all carry
    ``draw_from_prior`` is therefore NOT migrated — it is not legacy —
    and an accurate warning is emitted instead (standalone birth/death
    registration violates detailed balance; use the combined kernel).

    When a true legacy pair is found, a best-effort migration
    (:func:`impulse.experimental.birth_death_proposals.migrate_legacy_birth_death`)
    reconstructs the combined ``birth_death`` kernel from the
    unpickled legacy birth proposal and replaces the pair in every
    chain with their summed selection weight, then warns that
    PRE-resume samples remain biased. If reconstruction fails the
    checkpoint is left untouched and the historical loud warning is
    emitted instead.
    """
    props = [prop for jp in sampler.proposal_bundle.jump_proposals for prop in jp.proposal_list]
    names = {getattr(prop, "__name__", "") for prop in props}
    if "birth_proposal" not in names and "death_proposal" not in names:
        return
    deaths = [p for p in props if getattr(p, "__name__", "") == "death_proposal"]
    if deaths and all(callable(getattr(p, "draw_from_prior", None)) for p in deaths):
        warnings.warn(
            "Resumed checkpoint registers separate standalone "
            "'birth_proposal'/'death_proposal' jumps whose attribute "
            "layout matches current-code standalone registrations (the "
            "death proposal carries draw_from_prior), not a pre-fix "
            "legacy checkpoint; no migration was attempted. Standalone "
            "birth/death registration in a constant-weight mixture "
            "violates detailed balance and biases the model posterior "
            "toward fewer sources: register the ONE combined "
            "birth-death kernel (make_birth_death_proposal or "
            "make_product_space_sampler) instead.",
            UserWarning,
        )
        return
    migrated = migrate_legacy_birth_death(sampler.proposal_bundle.jump_proposals)
    if migrated is not None:
        warnings.warn(
            "Resumed checkpoint registered separate 'birth_proposal'/"
            "'death_proposal' jumps (pre-detailed-balance-fix wiring). "
            "The checkpoint was migrated automatically: the pair was "
            "replaced by the combined 'birth_death' kernel with their "
            "summed selection weight, so sampling continues from a "
            "detailed-balance-correct kernel. Model posteriors built "
            "from PRE-resume samples remain biased toward fewer "
            "sources and should be discarded.",
            UserWarning,
        )
        return
    warnings.warn(
        "Resumed checkpoint registers separate 'birth_proposal'/"
        "'death_proposal' jumps: it predates the detailed-balance "
        "fix and carries the biased birth/death wiring, so model "
        "posteriors will remain biased toward fewer sources. Start "
        "a fresh run (or re-register the combined birth-death "
        "kernel) for correct model posteriors.",
        UserWarning,
    )

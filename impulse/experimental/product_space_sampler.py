"""Constructor for a product-space (birth-death) model-selection sampler.

This was :meth:`impulse.PTSampler.from_product_space`. It moved here when the
product-space machinery became experimental, so that the supported
:class:`impulse.PTSampler` carries no knowledge of model-selection proposals.
"""

from impulse.experimental._product_space_setup import (
    _expand_product_space_cov_mean,
    _register_model_selection_jumps,
)
from impulse.proposals import DE_MIN_FILL


def make_product_space_sampler(
    product_space,
    sampler_cls=None,
    birth_weight: float = 15,
    death_weight: float = 15,
    nmodel_weight: float = 10,
    swap_weight: float = 15,
    am_weight: float = 15,
    scam_weight: float = 15,
    de_weight: float = 15,
    de_min_fill: int = DE_MIN_FILL,
    **kwargs,
):
    """Build a sampler wired for product-space (birth-death) model selection.

    Parameters
    ----------
    product_space : BirthDeathProductSpace
        Configured birth-death product space object.
    sampler_cls : type, optional
        Sampler class to construct. Defaults to :class:`impulse.PTSampler`;
        pass :class:`~impulse.experimental.HybridPTSampler` (with
        ``lnlike_grad=...``) to interleave NUTS transitions.
    birth_weight : float
        Contribution to the combined birth-death kernel's selection weight.
        Birth and death are registered as ONE kernel whose selection weight is
        ``birth_weight + death_weight``; the split between them is governed by
        the space's ``prob_schedule``. Registering them as separate
        constant-weight jumps violates detailed balance.
    death_weight : float
        Contribution to the combined birth-death kernel's selection weight;
        see ``birth_weight``.
    nmodel_weight : float
        Relative weight for uniform model-index jumps.
    swap_weight : float
        Relative weight for source-swap proposals.
    am_weight, scam_weight, de_weight : float
        Relative weights for the standard continuous proposals.
    de_min_fill : int
        Minimum per-model buffer fill before the DE difference move activates;
        see :func:`impulse.proposals.de`.
    **kwargs
        Passed through to the sampler constructor (e.g. ``ntemps``, ``seed``,
        ``outdir``, and ``lnlike_grad`` for the hybrid sampler).

    Returns
    -------
    The constructed sampler, with the birth-death kernel, model-index jump and
    source-swap proposal registered on top of the standard continuous jumps.

    Notes
    -----
    For a single-model space (``product_space.num_models == 1``) the
    birth-death kernel, the model-index jump and the source-swap proposal are
    all skipped -- none is meaningful with one model, and the birth-death
    kernel itself rejects ``max_sources < 2`` -- so only the standard
    continuous jumps are registered.

    Examples
    --------
    >>> from impulse.experimental import (
    ...     BirthDeathProductSpace, make_product_space_sampler)
    >>> space = BirthDeathProductSpace(loglike, logprior, 3, 3, draw_fn)
    >>> sampler = make_product_space_sampler(space, ntemps=15, seed=42)
    >>> x0 = space.draw_initial_position(np.random.default_rng(42))
    >>> sampler.sample(x0, num_iterations=50000)
    """
    if sampler_cls is None:
        from impulse.samplers import PTSampler

        sampler_cls = PTSampler

    # Expand per-source sample_cov / sample_mean to the full product space
    sample_cov, sample_mean = _expand_product_space_cov_mean(product_space, kwargs)

    sampler = sampler_cls(
        ndim=product_space.ndim,
        lnlike=product_space.get_loglikelihood,
        lnprior=product_space.get_logprior,
        groups=product_space.get_default_groups(),
        # get_default_groups deliberately omits the model index: it is moved by
        # the birth/death kernel and nmodel_jump, never by am/scam/de. Declare
        # that so ChainStats does not warn about an uncovered index.
        unmanaged_indices=[product_space.ndim - 1],
        sample_cov=sample_cov,
        sample_mean=sample_mean,
        am_weight=am_weight,
        scam_weight=scam_weight,
        de_weight=de_weight,
        de_min_fill=de_min_fill,
        **kwargs,
    )
    _register_model_selection_jumps(
        sampler,
        product_space,
        birth_weight=birth_weight,
        death_weight=death_weight,
        nmodel_weight=nmodel_weight,
        swap_weight=swap_weight,
    )
    return sampler

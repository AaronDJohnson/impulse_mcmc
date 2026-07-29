"""Experimental samplers and model-selection machinery.

Everything here works and is tested, but it is **not** covered by the 2.x API
stability policy that applies to :class:`impulse.PTSampler`: names, signatures
and defaults may change in a minor release.

What lives here and why:

- :class:`HybridPTSampler` -- parallel tempering with NUTS transitions
  interleaved, plus product-space model moves.
- :class:`BirthDeathProductSpace` and the birth/death proposals -- product-space
  ("RJ"-style) model selection. The trans-dimensional kernel itself is proven
  exact by finite-state enumeration
  (``tests/test_birth_death_detailed_balance.py``), but the surrounding
  integration is where this package's defects have historically concentrated.
- :func:`make_product_space_sampler` -- formerly
  ``PTSampler.from_product_space``. It moved so that the supported sampler
  carries no knowledge of model-selection proposals.

The supported core is :class:`impulse.PTSampler` together with the adaptive
proposals, diagnostics and checkpointing in the top-level ``impulse`` namespace.

Examples
--------
>>> from impulse.experimental import (
...     BirthDeathProductSpace, make_product_space_sampler)
>>> space = BirthDeathProductSpace(loglike, logprior, 3, 2, draw_fn)
>>> sampler = make_product_space_sampler(space, ntemps=8, seed=1)
"""

from impulse.experimental.birth_death import BirthDeathProductSpace
from impulse.experimental.birth_death_proposals import (
    BirthDeathProposal,
    NmodelJump,
    default_birth_death_probs,
    make_birth_death_proposal,
    make_nmodel_jump,
)
from impulse.experimental.hybrid_sampler import HybridPTSampler
from impulse.experimental.product_space_sampler import make_product_space_sampler
from impulse.resume import load_hybrid_checkpoint

__all__ = [
    "BirthDeathProductSpace",
    "BirthDeathProposal",
    "HybridPTSampler",
    "NmodelJump",
    "default_birth_death_probs",
    "load_hybrid_checkpoint",
    "make_birth_death_proposal",
    "make_nmodel_jump",
    "make_product_space_sampler",
]

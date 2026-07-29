Experimental
============

.. warning::

   Everything on this page lives in ``impulse.experimental`` and is **not**
   covered by the 2.x API stability policy that applies to the rest of the
   package. Names, signatures and defaults here may change in a minor release.
   It is tested and it works — the trans-dimensional kernel is proven exact by
   finite-state enumeration — but the surrounding integration is where this
   package's defects have historically concentrated, so it is versioned
   separately from the supported core.

   Import it explicitly::

       from impulse.experimental import HybridPTSampler, BirthDeathProductSpace

.. currentmodule:: impulse.experimental

.. autosummary::

   HybridPTSampler
   BirthDeathProductSpace
   make_product_space_sampler
   make_birth_death_proposal
   make_nmodel_jump
   default_birth_death_probs
   load_hybrid_checkpoint

Samplers
--------

.. autoclass:: HybridPTSampler
   :members:

Model spaces
------------

.. autoclass:: BirthDeathProductSpace
   :members:

Construction
------------

.. autofunction:: make_product_space_sampler

Proposals
---------

.. autoclass:: BirthDeathProposal
   :members:

.. autoclass:: NmodelJump
   :members:

.. autofunction:: make_birth_death_proposal

.. autofunction:: make_nmodel_jump

.. autofunction:: default_birth_death_probs

Checkpointing
-------------

.. autofunction:: load_hybrid_checkpoint

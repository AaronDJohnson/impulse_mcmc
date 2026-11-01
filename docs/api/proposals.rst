Proposals
=========

Every proposal is a callable ``proposal(chain_stats) -> (new_sample, qxy)``;
see :doc:`/user_guide/custom-proposals` for the contract and the ``qxy``
convention. ``PTSampler`` registers ``am``, ``scam`` and ``de`` on every
temperature chain by default (weights ``am_weight``, ``scam_weight`` and
``de_weight``); the others are added with ``PTSampler.add_custom_jump``.

.. currentmodule:: impulse

.. autosummary::

   am
   scam
   de
   gaussian
   DEProposal
   NormalizingFlowProposal
   JumpProposals
   ProposalBundle
   IndependentComponentProposal
   RJLayout

Built-in moves
--------------

.. autofunction:: am

.. autofunction:: scam

.. autofunction:: de

.. autoclass:: DEProposal

.. autofunction:: gaussian

Normalizing-flow proposal
-------------------------

Requires the optional ``coppuccino`` package
(``pip install "impulse-mcmc[flow]"``, Python 3.11+); constructing one without
it raises ``ImportError`` with an install hint.

.. autoclass:: NormalizingFlowProposal
   :members: freeze_adaptation, set_flow

Proposal containers
-------------------

``PTSampler`` holds one ``JumpProposals`` per temperature chain inside a
``ProposalBundle`` (``sampler.proposal_bundle``). Each chain picks its own
proposal independently on every step, from its own random stream.

.. autoclass:: JumpProposals

.. autoclass:: ProposalBundle

Reversible-jump building blocks
-------------------------------

``RJPTSampler`` draws new components from a component proposal: any object
with ``draw(rng, x, k)`` and ``logpdf(u, x, k)``, or an
``IndependentComponentProposal`` wrapping state-free functions. Custom moves
find the row layout in ``chain_stats.layout``. Draws of a periodic component
coordinate must lie inside its declared interval, and ``logpdf`` must be the
density of that (wrapped) value; out-of-interval draws raise ``ValueError``. See
:doc:`/user_guide/reversible-jump`.

.. autoclass:: IndependentComponentProposal

.. autoclass:: RJLayout
   :members: k, params, shared, components, slot, active_mask, active_mask_rows, n_active, ndim, width,
              check_k, make_row, insert, remove

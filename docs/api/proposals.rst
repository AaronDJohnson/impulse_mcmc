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

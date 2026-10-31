API reference
=============

The supported API is everything in ``impulse.__all__``, grouped below exactly
as in ``impulse.__init__``. It is covered by the 2.x stability policy.

.. currentmodule:: impulse

.. autosummary::

   PTSampler
   am
   scam
   de
   gaussian
   DEProposal
   NormalizingFlowProposal
   JumpProposals
   ProposalBundle
   grubin
   effective_sample_size
   autocorr_length_ips_ims
   CheckpointMismatchError
   TornCheckpointError
   checkpoint_sampler
   save_state_checkpoint
   load_state_checkpoint
   restore_state_checkpoint
   check_for_checkpoint

.. toctree::
   :maxdepth: 2

   samplers
   proposals
   diagnostics
   checkpoint

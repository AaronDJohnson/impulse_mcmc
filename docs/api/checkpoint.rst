Checkpoint / resume
===================

Checkpoints are a **no-code-execution** pair -- ``sampler_checkpoint.npz``
(arrays) plus a schema-versioned ``sampler_checkpoint.json`` -- so loading one
executes no code. Resume is *reconstruct then restore*: rebuild the sampler the
same way, then ``resume=True`` verifies and restores state into it. See
:doc:`/user_guide/checkpointing` for the full contract.

``resume=True`` calls these for you; use them directly only for manual control.

.. currentmodule:: impulse

.. autosummary::

   checkpoint_sampler
   save_state_checkpoint
   load_state_checkpoint
   restore_state_checkpoint
   check_for_checkpoint
   CheckpointMismatchError
   TornCheckpointError

.. autofunction:: checkpoint_sampler

.. autofunction:: save_state_checkpoint

.. autofunction:: load_state_checkpoint

.. autofunction:: restore_state_checkpoint

.. autofunction:: check_for_checkpoint

.. autoexception:: CheckpointMismatchError

.. autoexception:: TornCheckpointError

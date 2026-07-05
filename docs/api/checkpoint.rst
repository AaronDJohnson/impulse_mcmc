Checkpoint / resume
===================

The default checkpoint format is a **no-code-execution** pair —
``sampler_checkpoint.npz`` (arrays) plus a schema-versioned
``sampler_checkpoint.json`` — so loading a checkpoint executes no code. Resume
is *reconstruct then restore*: rebuild the sampler the same way, then
``resume=True`` verifies and restores state into it. Legacy
``sampler_checkpoint.pkl`` checkpoints still load (with a security/deprecation
warning), but unpickling one can execute arbitrary code — read the
:doc:`trust boundary </user_guide/checkpointing>` before resuming anything you
did not write yourself.

.. currentmodule:: impulse

.. autosummary::

   checkpoint_sampler
   load_checkpoint
   load_nuts_checkpoint
   load_hybrid_checkpoint
   check_for_checkpoint

.. autofunction:: checkpoint_sampler

.. autofunction:: load_checkpoint

.. autofunction:: load_nuts_checkpoint

.. autofunction:: load_hybrid_checkpoint

.. autofunction:: check_for_checkpoint

New-format helpers
------------------

Lower-level entry points for the no-code-execution format (used internally by
``resume=True``; useful for advanced or manual control).

.. currentmodule:: impulse.resume

.. autosummary::

   save_state_checkpoint
   restore_state_checkpoint
   load_state_checkpoint

.. autofunction:: save_state_checkpoint

.. autofunction:: restore_state_checkpoint

.. autofunction:: load_state_checkpoint

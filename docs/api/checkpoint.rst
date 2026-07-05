Checkpoint / resume
===================

Pickle-based checkpointing. Checkpoints can execute arbitrary code when
loaded — read the :doc:`trust boundary </user_guide/checkpointing>` before
resuming anything you did not write yourself.

.. currentmodule:: impulse

.. autosummary::

   checkpoint_sampler
   load_checkpoint
   load_nuts_checkpoint
   load_rjpt_checkpoint
   check_for_checkpoint

.. autofunction:: checkpoint_sampler

.. autofunction:: load_checkpoint

.. autofunction:: load_nuts_checkpoint

.. autofunction:: load_rjpt_checkpoint

.. autofunction:: check_for_checkpoint

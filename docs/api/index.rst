API reference
=============

The public API is everything in ``impulse.__all__``, grouped below exactly
as in ``impulse.__init__``.

.. currentmodule:: impulse

.. autosummary::

   PTSampler
   HybridPTSampler
   NUTSSampler
   BirthDeathProductSpace
   NestedProductSpace
   model_visitation_stats
   bayes_factor_from_chain
   grubin
   effective_sample_size
   autocorr_length_ips_ims
   checkpoint_sampler
   load_checkpoint
   load_nuts_checkpoint
   load_hybrid_checkpoint
   check_for_checkpoint
   compose_logp_and_grad
   make_logp_and_grad_numerical
   MassMatrix
   MassMatrixType
   compute_sbc_rank
   compute_sbc_quantile
   compute_model_pit
   ecdf
   sbc_ecdf_plot
   coverage_plot
   rank_histogram
   run_sbc_continuous
   run_sbc_model_selection

.. toctree::
   :maxdepth: 2

   samplers
   model-spaces
   diagnostics
   checkpoint
   nuts-utilities
   validation

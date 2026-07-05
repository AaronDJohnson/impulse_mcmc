Validation / SBC
================

Simulation-based calibration utilities. The numeric functions need only
NumPy/SciPy; the plotting helpers (``sbc_ecdf_plot``, ``coverage_plot``,
``rank_histogram``) require matplotlib, available via
``pip install "impulse-mcmc[plots]"``.

.. currentmodule:: impulse

.. autosummary::

   compute_sbc_rank
   compute_sbc_quantile
   compute_model_pit
   ecdf
   sbc_ecdf_plot
   coverage_plot
   rank_histogram
   run_sbc_continuous
   run_sbc_model_selection

.. autofunction:: compute_sbc_rank

.. autofunction:: compute_sbc_quantile

.. autofunction:: compute_model_pit

.. autofunction:: ecdf

.. autofunction:: sbc_ecdf_plot

.. autofunction:: coverage_plot

.. autofunction:: rank_histogram

.. autofunction:: run_sbc_continuous

.. autofunction:: run_sbc_model_selection

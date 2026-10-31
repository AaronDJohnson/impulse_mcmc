Diagnostics
===========

Convergence diagnostics for finished (or running) chains. All three take
parameters only, shape ``(T, D)`` -- exactly ``sampler.load_chain()["samples"][k]``.

.. currentmodule:: impulse

.. autosummary::

   grubin
   effective_sample_size
   autocorr_length_ips_ims

.. autofunction:: grubin

.. autofunction:: effective_sample_size

.. autofunction:: autocorr_length_ips_ims

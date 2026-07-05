NUTS utilities
==============

Gradient helpers and mass-matrix containers for the NUTS-based samplers.
Mass matrices follow Stan's convention (:math:`M = \Sigma^{-1}`); see the
:doc:`NUTS user guide </user_guide/nuts>`.

.. currentmodule:: impulse

.. autosummary::

   compose_logp_and_grad
   make_logp_and_grad_numerical
   MassMatrix
   MassMatrixType

.. autofunction:: compose_logp_and_grad

.. autofunction:: make_logp_and_grad_numerical

.. autoclass:: MassMatrix
   :members:

.. autoclass:: MassMatrixType
   :members:
   :undoc-members:

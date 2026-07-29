Model spaces
============

Product-space embeddings for product-space (birth-death) model selection.
See the :doc:`user guide </user_guide/model-selection>` for the contract each
callable must satisfy.

.. currentmodule:: impulse

.. autosummary::

   NestedProductSpace

NestedProductSpace
------------------

.. autoclass:: NestedProductSpace
   :members:

.. seealso::

   :class:`impulse.experimental.BirthDeathProductSpace` -- the "N identical
   sources" birth-death space, and
   :func:`impulse.experimental.make_product_space_sampler` which wires a
   sampler for it. Both live in :doc:`experimental`.

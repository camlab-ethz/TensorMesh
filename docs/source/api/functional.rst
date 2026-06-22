tensormesh.functional
=====================

.. py:module:: tensormesh.functional

ops
---

.. automodule:: tensormesh.functional.ops
    :members:
    :show-inheritance:
  
elasticity
----------

.. automodule:: tensormesh.functional.elasticity
    :members:
    :show-inheritance:
  

plastic
-------

Plastic constitutive primitives, including the pure Drucker-Prager return mapping
(:func:`~tensormesh.functional.plastic.drucker_prager_return_mapping`, returning a
:class:`~tensormesh.functional.plastic.DruckerPragerReturn`) used by the built-in
:class:`~tensormesh.assemble.DruckerPragerPlasticity` assembler.

.. automodule:: tensormesh.functional.plastic
    :members:
    :show-inheritance:
  
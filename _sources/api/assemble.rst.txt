tensormesh.assemble
===================

.. py:module:: tensormesh.assemble

Element Assembler
-----------------

.. autoclass:: tensormesh.ElementAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation


Facet Assembler
---------------

.. autoclass:: tensormesh.FacetAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation, facet_mask

Node Assembler
--------------

.. autoclass:: tensormesh.NodeAssembler
    :members:
    :show-inheritance:
    :exclude-members: dimension, edges, element_types, elements, n_points, projector, transformation


Built-in Assemblers
-------------------

.. autoclass:: tensormesh.LaplaceElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.MassElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.LinearElasticityElementAssembler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.NeoHookeanModel
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.J2Plasticity
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.assemble.ContactAssembler
    :members:
    :show-inheritance:

.. autofunction:: tensormesh.const_node_assembler

.. autofunction:: tensormesh.func_node_assembler


Projector
---------

Internal helper used by the assemblers to scatter element-local
contributions onto global degrees of freedom. Two concrete subclasses
(``ReduceProjector``, ``SparseProjector``) exist; both share the same
interface documented here.

.. autoclass:: tensormesh.assemble.projector.Projector
    :members:
    :show-inheritance:

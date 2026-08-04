tensormesh.element
==================

.. py:module:: tensormesh.element

The :mod:`tensormesh.element` module is organised in three tiers, each
aimed at a different audience.

1. :ref:`api-element-transformation` — the only thing most users call
   directly. If you are writing a new assembler or computing shape
   values / gradients / ``J×W`` at quadrature points, this is your
   interface.
2. :ref:`api-element-catalog` — the seven reference shapes and the
   string-↔-class registry. Read this when you need to know *which* type
   string corresponds to which class, what its facet shape is, or how its
   node ordering compares to Gmsh / VTK.
3. :ref:`api-element-internals` — basis builders, quadrature factories,
   facet-normal helpers, and the :class:`~tensormesh.element.Polynomial`
   machinery. Read this only if you are adding a new element type or
   quadrature rule.


.. _api-element-transformation:

Transformation
--------------

The workhorse: given a mesh's points + connectivity + element type, a
:class:`~tensormesh.Transformation` caches the per-element Jacobian,
shape-function values, shape-function gradients, quadrature
points/weights, and their facet analogues. Every built-in assembler in
:mod:`tensormesh.assemble` ultimately reads from a ``Transformation``;
custom assemblers should too.

.. autoclass:: tensormesh.Transformation
    :members:
    :show-inheritance:
    :exclude-members: basis_order, element, elements, points, quadrature_order


.. _api-element-catalog:

Reference-shape catalog
-----------------------

TensorMesh ships seven reference shapes. The class objects double as
*types* (you name them in :meth:`~tensormesh.Element.reorder` calls, in
:func:`~tensormesh.element_type2element` results, as the value of
:attr:`~tensormesh.Transformation.element`, …) and as a small public
surface for shape introspection — basis-node coordinates, quadrature
rules, facet type, Gmsh/VTK permutation. Users rarely call methods on
subclasses directly; the methods listed below are the ones referenced
from the :doc:`user guide </user_guide/elements_and_quadrature>` and the
:doc:`example gallery </example_gallery/basics>`, plus the extension
hooks an element-type author overrides.

.. autoclass:: tensormesh.Element
    :members: get_basis, get_basis_fns, get_basis_grad_fns,
              get_polynomial, get_quadrature, get_facet, get_facet_type,
              get_facet_quadrature, reorder, get_gmsh_permutation
    :show-inheritance:

The seven concrete shapes inherit the methods above; the subclass entries
below mainly document their reference geometry (vertex coordinates, facet
shape, type-string family).

.. autoclass:: tensormesh.Line
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Triangle
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Quadrilateral
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Tetrahedron
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Hexahedron
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Prism
    :show-inheritance:
    :no-members:

.. autoclass:: tensormesh.Pyramid
    :show-inheritance:
    :no-members:


Type-string registry
~~~~~~~~~~~~~~~~~~~~

Lookup tables and helpers that map between element type strings
(``"triangle"``, ``"tetra10"``, …) and the corresponding element class,
spatial dimension, and polynomial order.

.. py:currentmodule:: tensormesh

.. autofunction:: tensormesh.element_type2element

.. py:data:: element_types
   :type: list[str]

   List of every element type string the library understands —
   first-order shapes (``"line"``, ``"triangle"``, ``"quad"``,
   ``"tetra"``, ``"hexahedron"``, ``"wedge"``, ``"pyramid"``) plus their
   higher-order counterparts (``"triangle6"``, ``"quad9"``,
   ``"tetra10"``, ``"hexahedron27"``, ``"triangle10"``, …).

.. py:data:: element_type2dimension
   :type: dict[str, int]

   Map from element type string to spatial dimension
   (``"line": 1``, ``"triangle": 2``, ``"tetra": 3``, …).

.. py:data:: element_type2order
   :type: dict[str, int]

   Map from element type string to polynomial order
   (``"triangle": 1``, ``"triangle6": 2``, ``"triangle10": 3``, …).

.. py:currentmodule:: tensormesh.element


.. _api-element-internals:

Extension internals
-------------------

These layers exist for people adding a **new element type** or a **new
quadrature rule**. None of them are part of the stable user-facing API:
their signatures may evolve between releases and their docstrings live in
the source rather than here.

**Polynomial space.** :class:`tensormesh.element.Polynomial` represents
a single multivariate polynomial; :class:`tensormesh.element.Polynomials`
represents an arbitrarily-shaped batch. They are the building blocks of
every shape-function set:
:meth:`~tensormesh.Element.get_basis_fns` solves a small Vandermonde
system to obtain the Lagrange-interpolation coefficients on top of a
chosen polynomial space (full ``poly_exp``, tensor-product ``tens_exp``,
or the pyramid / prism ad-hoc spaces). To add a new element you
typically override :meth:`~tensormesh.Element.get_polynomial` to return
the right polynomial space and let the base class produce the basis
functions for you.

.. autoclass:: tensormesh.element.Polynomial
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.element.Polynomials
    :members:
    :show-inheritance:
    :exclude-members: device, dtype, n_polys, n_terms, n_vars, shape

**Basis-node and quadrature factories.** The reference-element
interpolation-node coordinates and quadrature rules live in three
internal files; read the source if you need to understand exactly how
:meth:`~tensormesh.Element.get_basis` or
:meth:`~tensormesh.Element.get_quadrature` are implemented for each
shape:

* ``tensormesh/element/basis.py`` — ``lin_basis``, ``tri_basis``,
  ``quad_basis``, ``tet_basis``, ``hex_basis``, ``pyr_basis``,
  ``pri_basis`` and the facet-basis index helpers.
* ``tensormesh/element/quadrature.py`` — ``lin_quadrature``,
  ``tri_quadrature``, ``quad_quadrature``, ``tet_quadrature``,
  ``hex_quadrature``, ``pyr_quadrature``, ``pri_quadrature`` plus the
  facet-quadrature variants.
* ``tensormesh/element/normal.py`` — ``outwards_normal_2d`` /
  ``outwards_normal_3d``, used internally to compute the outward facet
  normals consumed by :attr:`~tensormesh.Transformation.nanson_scale`.

The supported extension path is to subclass :class:`~tensormesh.Element`
and override its basis / quadrature / facet hooks. Use the existing
seven shapes as templates — each subclass is short and self-contained.

Geomechanics: elastic strip footing
===================================

This example adds a small geomechanics boundary-value problem to the solid
mechanics example family.  A linear-elastic soil block is loaded by a centered
strip footing, then solved with TensorMesh's standard direct linear-elasticity
pipeline.

The example is deliberately modest: it demonstrates geotechnical boundary
conditions, a footing load patch, settlement reporting, and reaction/load
balance without adding a public geomechanics API.

Problem
-------

The model is a two-dimensional plane-strain-style soil block with unit
out-of-plane thickness.  The displacement field satisfies small-strain linear
elasticity,

.. math::

   \nabla \cdot \boldsymbol{\sigma} = \mathbf{0},
   \qquad
   \boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon},
   \qquad
   \boldsymbol{\varepsilon}
   = \tfrac12(\nabla\mathbf{u} + \nabla\mathbf{u}^T).

The domain is loaded by a compression-positive footing pressure on a limited
top-surface patch.  Internally, the load is applied in the negative
``y`` direction.  For geomechanics reporting, settlement is shown as positive
downward,

.. math::

   s = -u_y.

Boundary conditions
-------------------

The example uses a simple roller setup:

* bottom boundary: vertical displacement fixed, ``u_y = 0``;
* left and right boundaries: horizontal displacement fixed, ``u_x = 0``;
* top boundary: free except for the loaded footing patch.

The footing pressure is lumped over the top-surface nodes inside the footing
patch.  This keeps the example close to the existing solid-mechanics direct
solve examples, while the sanity check verifies that the total reaction balances
the total applied load.

Sanity checks
-------------

The script reports the total applied vertical load, the vertical reaction at
fixed vertical degrees of freedom, the reaction/load relative error, and the
maximum settlement.

The associated test checks that:

* the soil settles downward under the footing;
* the vertical reaction balances the applied vertical load;
* doubling the footing pressure approximately doubles the settlement.

.. figure:: /_static/solid_mechanics/elastic_footing.png
   :alt: Elastic strip-footing settlement contour and surface settlement profile
   :width: 100%

   Output of ``elastic_footing.py``.  The left panel shows the deformed
   soil mesh colored by settlement ``-u_y``; red arrows mark the footing
   load, blue markers show vertical supports on the base, and purple markers
   show horizontal roller constraints on the side boundaries.  The right
   panel shows the settlement profile along the ground surface.  The largest
   settlement occurs beneath the centered footing, and the deformation is
   exaggerated for visibility.

Running it
----------

.. code-block:: bash

   cd examples/solid/geomechanics/elastic_footing
   python elastic_footing.py

For a fast numerical-only run without writing the plot:

.. code-block:: bash

   python elastic_footing.py --no-plot --chara-length 0.5

Core implementation
-------------------

The full driver is in
``examples/solid/geomechanics/elastic_footing/elastic_footing.py``.  It reuses
the same direct-solve components as the existing linear-elastic solid examples:
``LinearElasticityElementAssembler``, ``Condenser``, and ``SparseMatrix.solve``.

.. literalinclude:: ../../../../examples/solid/geomechanics/elastic_footing/elastic_footing.py
   :language: python
   :start-after: def solve_footing
   :end-before: def _triangles_from_mesh

What's next
-----------

This example is elastic and intentionally small.  A later follow-up could reuse
the same footing geometry with the example-local Drucker-Prager material model,
or promote a stabilized geomechanics assembler into ``tensormesh/assemble/`` if
the API direction becomes clear.

Geomechanics: Drucker-Prager triaxial compression
=================================================

This example introduces a small geomechanics application inside the solid-mechanics
example family.  It implements an example-only Drucker-Prager plasticity assembler
for a pressure-dependent soil or weak-rock material and drives it through a simple
triaxial-compression strain path.

The goal is deliberately modest: demonstrate TensorMesh conventions for
geomechanics without adding a public API.  The local assembler lives in
``examples/solid/geomechanics/drucker_prager_triaxial/drucker_prager_triaxial.py``.

Problem
-------

The model is small-strain, associated Drucker-Prager plasticity with linear
isotropic hardening.  TensorMesh keeps the internal solid-mechanics convention
stress tension-positive.  For geomechanics reporting, the script prints axial
stress and mean pressure as compression-positive quantities.

The yield function is written internally as

.. math::

   f(\sigma, \alpha) = q + \eta I_1 - (k + H\alpha) \le 0,

where

.. math::

   I_1 = \mathrm{tr}(\sigma), \qquad
   q = \sqrt{\frac{3}{2}\,s:s}, \qquad
   p = -\frac{I_1}{3}.

Because compression gives negative ``I1`` in the tension-positive convention,
higher confinement lowers ``f`` and delays yielding.

History variables
-----------------

The example follows the same pattern as the existing J2 plasticity example:

* per-quadrature history variables are stored in ``self.history[etype]``;
* previous-step ``eps_p`` and ``alpha`` are passed through ``element_data``;
* ``update_state(u)`` is called after each converged load step under
  ``torch.no_grad()``.

The example remains local to the script so the public API can be discussed after
the convention-setting example has been reviewed.

Sanity check
------------

Two confinement levels are run:

* ``p0 = 0 kPa``;
* ``p0 = 100 kPa``.

The script checks that the higher-confinement case reaches the elastic trial
yield surface later and that the committed plastic history variable is monotonic.

Running it
----------

.. code-block:: bash

   cd examples/solid/geomechanics/drucker_prager_triaxial
   python drucker_prager_triaxial.py

For a fast numerical-only run without writing the plot:

.. code-block:: bash

   python drucker_prager_triaxial.py --no-plot --steps 16

The default run writes ``drucker_prager_triaxial.png`` with axial stress and
plastic-history curves.

Core implementation
-------------------

.. literalinclude:: ../../../../examples/solid/geomechanics/drucker_prager_triaxial/drucker_prager_triaxial.py
   :language: python
   :start-after: class DruckerPragerPlasticity
   :end-before: def affine_displacement
   :dedent: 0

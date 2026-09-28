Geomechanics: Drucker-Prager triaxial compression
=================================================

This example introduces a small geomechanics application inside the solid-mechanics
example family.  It uses TensorMesh's public Drucker-Prager API — the
:class:`~tensormesh.assemble.DruckerPragerPlasticity` assembler, the
:class:`~tensormesh.material.FrictionalMaterial` container, and the constitutive
primitive :func:`~tensormesh.functional.plastic.drucker_prager_yield_value` — to drive a
pressure-dependent soil or weak-rock material through a simple triaxial-compression
strain path.

The goal is deliberately modest: demonstrate TensorMesh conventions for
geomechanics on a minimal driver.  The example script lives in
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

The built-in assembler follows the same lifecycle as the J2 plasticity model:

* per-quadrature history variables are stored in ``model.history[etype]``;
* previous-step ``eps_p`` and ``alpha`` are passed through ``element_data``
  (via ``model.element_data_from_history()``);
* ``update_state(u)`` is called after each converged load step under
  ``torch.no_grad()``.

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

.. figure:: /_static/solid_mechanics/drucker_prager_triaxial.png
   :alt: Drucker-Prager triaxial compression response showing axial stress and plastic history for two confinement levels
   :width: 100%

   Output of ``drucker_prager_triaxial.py``. The left panel shows the
   compression-positive axial stress response; the right panel shows the
   committed plastic history variable. The higher-confinement case yields
   later, matching the pressure-dependent Drucker-Prager sanity check.

Core implementation
-------------------

The driver builds a :class:`~tensormesh.material.FrictionalMaterial`, assembles
the built-in :class:`~tensormesh.assemble.DruckerPragerPlasticity` model, and
steps the strain path while committing history with ``update_state``:

.. literalinclude:: ../../../../examples/solid/geomechanics/drucker_prager_triaxial/drucker_prager_triaxial.py
   :language: python
   :pyobject: run_case

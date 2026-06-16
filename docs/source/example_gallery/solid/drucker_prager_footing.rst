Geomechanics: Drucker-Prager strip footing
==========================================

This example combines the two existing example-only geomechanics building
blocks into one nonlinear boundary-value problem: the local Drucker-Prager
triaxial constitutive driver and the elastic strip-footing setup.  A rectangular
soil block is loaded by a centered strip footing, the footing pressure is ramped
in load steps, and the per-quadrature Drucker-Prager history is committed after
each converged step.

It is deliberately example-only.  It does not add a public geomechanics API; the
constitutive code is copied locally (Torch only) so the script is self-contained.
TensorMesh keeps the internal solid-mechanics convention stress tension-positive.
For geomechanics reporting, settlement is shown positive downward,

.. math::

   s = -u_y.

This is a compact educational example, not a foundation-design method.

Problem
-------

The model is a two-dimensional plane-strain-style soil block with unit
out-of-plane thickness.  The constitutive model is small-strain, associated
Drucker-Prager plasticity with linear isotropic hardening, written internally
with the tension-positive yield function

.. math::

   f(\sigma, \alpha) = q + \eta I_1 - (k + H\alpha) \le 0,
   \qquad
   I_1 = \mathrm{tr}(\sigma),
   \qquad
   q = \sqrt{\tfrac{3}{2}\,s:s}.

Boundary conditions
-------------------

The example reuses the elastic footing roller setup:

* bottom boundary: vertical displacement fixed, ``u_y = 0``;
* left and right boundaries: horizontal displacement fixed, ``u_x = 0``;
* top boundary: free except for the loaded footing patch.

The footing pressure is lumped over the top-surface nodes inside the footing
patch.

Solver
------

Because the problem is now path-dependent, it is solved by nonlinear energy
minimization with load stepping.  At each load step the total potential energy
``internal - external`` is minimized over the free displacement DOFs with
L-BFGS.  The example follows the same per-quadrature history lifecycle as the
``drucker_prager_triaxial`` example:

* per-quadrature history variables are stored in ``self.history[etype]``;
* previous-step ``eps_p`` and ``alpha`` are passed through ``element_data``;
* ``update_state(u)`` is called after each converged load step under
  ``torch.no_grad()``.

Sanity checks
-------------

The script reports the final footing settlement, the maximum settlement, the
maximum and mean committed plastic history, and the plastic centroid.

The associated test checks that the footing settles downward, that the committed
plastic history and settlement grow monotonically with load, that the plastic
zone localizes under the footing, that a low-load / high-cohesion case stays
essentially elastic and matches the elastic footing settlement, and that a
higher-cohesion soil develops less plasticity.

.. figure:: /_static/solid_mechanics/drucker_prager_footing.png
   :alt: Drucker-Prager strip-footing settlement, plastic history, and load-settlement curve
   :width: 100%

   Output of ``drucker_prager_footing.py``. The left panel shows the
   deformed soil mesh colored by settlement ``-u_y``. The middle panel shows
   the committed Drucker-Prager plastic history variable, which develops
   beneath the footing. The right panel shows the load-settlement curve,
   including nonlinear growth as plasticity accumulates. Deformations are
   exaggerated for visibility.

Running it
----------

.. code-block:: bash

   cd examples/solid/geomechanics/drucker_prager_footing
   python drucker_prager_footing.py

For a fast numerical-only run without writing the plot:

.. code-block:: bash

   python drucker_prager_footing.py --no-plot --steps 6 --chara-length 0.60

Core implementation
-------------------

The load-stepped nonlinear solver is the heart of the example.  It reuses the
local Drucker-Prager assembler's ``energy``, ``element_data_from_history`` and
``update_state`` methods.

.. literalinclude:: ../../../../examples/solid/geomechanics/drucker_prager_footing/drucker_prager_footing.py
   :language: python
   :pyobject: solve_drucker_prager_footing

What's next
-----------

This example stays intentionally example-only.  A natural follow-up would be to
promote a stabilized geomechanics assembler into ``tensormesh/assemble/`` once
the model surface and public API direction are agreed.

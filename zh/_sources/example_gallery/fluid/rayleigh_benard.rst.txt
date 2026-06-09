Rayleigh-Bénard Convection
==========================

The Rayleigh-Bénard problem couples incompressible fluid flow with
heat transport: a horizontal cavity heated from below, cooled from
above, develops convective rolls once the buoyancy force exceeds
the dissipative effects of viscosity and thermal diffusion. The
script ``examples/fluid/rayleigh_benard/rayleigh_benard.py`` solves
the steady-state Boussinesq approximation in a 2:1 aspect-ratio
rectangular cavity at a configurable Rayleigh number.

This is the only example in :doc:`index` that involves a
**multi-physics coupling** — momentum, continuity, and energy in
one monolithic system.


Problem
-------

The Boussinesq equations,

.. math::

   \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u}
   - \rho\, g\, \beta\, T\, \hat{\mathbf{e}}_y,
   \qquad
   \nabla \cdot \mathbf{u} = 0,

   \rho\, c_p\, (\mathbf{u} \cdot \nabla T)
   \;=\; \kappa\, \Delta T,

on :math:`\Omega = [0, 2] \times [0, 1]`, with

* velocity: no-slip on all walls,
* temperature: :math:`T = 1` on the bottom (:math:`y = 0`),
  :math:`T = 0` on the top (:math:`y = 1`), no-flux on the side
  walls.

The Rayleigh number

.. math::

   \mathrm{Ra} \;=\;
   \frac{g\, \beta\, \Delta T\, L^3}{\nu\, \alpha}

controls the regime: below :math:`\mathrm{Ra}_c \approx 1708`
heat transfer is purely conductive; above it convective rolls
appear; at higher :math:`\mathrm{Ra}` the rolls become unsteady
and eventually chaotic. The script defaults to
:math:`\mathrm{Ra} = 10^4`, comfortably in the steady-convective
regime.


Coupled assembler
-----------------

The unknowns at each node are :math:`(u, v, p, T)` — four DOFs
per node — and the assembler returns a :math:`4 \times 4` block
per quadrature point:

.. code-block:: python
   :caption: examples/fluid/rayleigh_benard/rayleigh_benard.py (essence)

   class RayleighBenardAssembler(ElementAssembler):
       def __post_init__(self, rho=1.0, mu=0.01, kappa=0.01,
                         g=9.81, beta=0.1, tau=0.1):
           self.rho, self.mu, self.kappa = rho, mu, kappa
           self.g, self.beta, self.tau   = g, beta, tau

       def forward(self, u, v, gradu, gradv, w_prev, T_prev):
           # row 0: x-momentum    [convection+diffusion, 0,    -dxp,   0   ]
           # row 1: y-momentum    [0,    convection+diffusion, -dyp,   buoyancy ]
           # row 2: continuity    [dxv,    dyv,                 PSPG,    0   ]
           # row 3: energy        [0,      0,                     0,    convection+diffusion]
           ...

The buoyancy block at row 1 / col 3 is what couples temperature
back into the momentum equation:

.. math::

   K_{1,3} \;=\; -\rho\, g\, \beta\, v\, u,

where here :math:`u` and :math:`v` are the trial- and test-shape
function values (note the unfortunate name collision with the
*velocity* :math:`v` field — see :doc:`../../user_guide/forms`
for the dispatch convention). The temperature equation (row 3)
re-uses the convection / diffusion block but with thermal
conductivity :math:`\kappa` in place of viscosity :math:`\mu`.

The previous-iterate velocity ``w_prev`` and temperature ``T_prev``
arrive as ``point_data`` kwargs, exactly the same pattern as the
single-physics cavity example.


Stabilization
-------------

The script uses **PSPG only** (no SUPG), with a constant
:math:`\tau = 0.05 / n_\text{grid}`. At
:math:`\mathrm{Ra} = 10^4` the velocities are mild enough that
SUPG is not strictly required; for higher Rayleigh numbers, swap
in the adaptive :math:`\tau` from :doc:`cylinder_flow`.


Picard iteration
----------------

The same fixed-point loop as :doc:`cavity`. Each iteration:

1. Compute :math:`\mathbf{w}^n, T^n` from the current solution
   vector.
2. Reassemble :math:`K` with the new ``point_data``.
3. Apply the :class:`~tensormesh.Condenser` for the velocity and
   temperature Dirichlet BCs (different masks per component!) and
   solve.
4. Check convergence against the previous iterate.

Typical convergence: ~30 iterations at :math:`\mathrm{Ra} = 10^4`.

.. figure:: /_static/fluid/rayleigh_benard.png
   :alt: Rayleigh-Bénard temperature and velocity magnitude
   :width: 100%

   Output of ``rayleigh_benard.py``. Left: temperature field —
   the warm bottom wall sends a rising plume up the centerline
   that splits at the cold top into two cool downward plumes
   along the side walls. Right: velocity magnitude — two
   counter-rotating convection rolls, with peak speeds along
   the rising / sinking columns and stagnation at the roll
   centers.


Output
------

``rayleigh_benard.png`` shows two panels: temperature (with the
characteristic warm rising plumes / cool sinking plumes) and
velocity magnitude (with the two-roll structure). For a 2:1
cavity at moderate :math:`\mathrm{Ra}` the steady solution
features two counter-rotating rolls.


Running it
----------

.. code-block:: bash

   cd examples/fluid/rayleigh_benard
   python rayleigh_benard.py     # writes rayleigh_benard.png

Edit the ``ra=`` argument inside the script to sweep the Rayleigh
number; values much above :math:`10^5` will need a finer mesh and
likely a transient driver.


What's next
-----------

* :doc:`cavity` — the same Picard recipe without temperature
  coupling.
* :doc:`taylor_green` — a transient incompressible flow with an
  exact solution for verification.
* :doc:`../diffusion` — the heat equation in isolation.
* :doc:`../../user_guide/forms` — multi-component ``forward``
  return signatures.

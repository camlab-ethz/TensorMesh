Fluid Mechanics
===============

A family of worked incompressible-flow examples in ``examples/fluid/``,
from a manufactured Stokes problem with a convergence study to
Rayleigh-Bénard convection coupling momentum and energy. They are all
built on **Taylor-Hood P2-P1 mixed assembly**: a quadratic velocity and
a linear pressure declared on one
:class:`~tensormesh.MixedElementAssembler`, so the pair is inf-sup
(LBB) stable and the weak form *is* the whole discretization — no
SUPG/PSPG stabilization, no ``tau`` tuning.

The recipe is shared across the family (read
:doc:`stokes_taylor_hood` first, then the rest are physics and
boundary-condition variations):

* declare ``Field(trial="u", test="v", order=2, components=d)`` and
  ``Field(trial="p", test="q", order=1)``, write the scalar integrand
  in ``forward``;
* steady problems linearize convection with **Picard** (the lagged
  velocity enters as data), transient ones add a backward-Euler mass
  term and a ``forward_vector`` load;
* boundary conditions go through ``assembler.layout`` masks and the
  unchanged :class:`~tensormesh.Condenser`.

On generated order-2 meshes the velocity nodes coincide with the mesh
points; on linear gmsh/:class:`~tensormesh.MeshGen` meshes (cylinder,
obstacles) the quadratic space is created **topologically** — no
re-meshing at order 2. See :doc:`../../user_guide/mixed_assembly` for
the machinery.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Taylor-Hood Stokes
      :link: stokes_taylor_hood
      :link-type: doc
      :img-top: /_static/fluid/stokes_taylor_hood_convergence.png

      Manufactured Stokes solution + convergence study — the mixed-assembly tutorial.

   .. grid-item-card:: Lid-Driven Cavity
      :link: cavity
      :link-type: doc
      :img-top: /_static/fluid/cavity_results.png

      Steady NS at Re=100, Picard iteration — in 2D and 3D with one
      dimension-generic weak form.

   .. grid-item-card:: Cylinder Flow (Vortex Shedding)
      :link: cylinder_flow
      :link-type: doc
      :img-top: /_static/fluid/vortex_street.gif

      Transient DFG benchmark: backward Euler, topological P2 on a gmsh mesh,
      vorticity via mixed load vectors.

   .. grid-item-card:: Flow Past Multiple Obstacles
      :link: flow_obstacles
      :link-type: doc
      :img-top: /_static/fluid/flow_obstacles.png

      Steady channel flow at Re=150 around six circular obstacles via MeshGen CSG.

   .. grid-item-card:: Rayleigh-Bénard Convection
      :link: rayleigh_benard
      :link-type: doc
      :img-top: /_static/fluid/rayleigh_benard.png

      Three-field Boussinesq system — velocity, pressure, and temperature
      in one block matrix.

   .. grid-item-card:: Taylor-Green Vortex
      :link: taylor_green
      :link-type: doc
      :img-top: /_static/fluid/taylor_green_vortex_final.png

      Decaying vortex with exact solution — the convergence-study showcase.


.. toctree::
   :hidden:
   :maxdepth: 1

   stokes_taylor_hood
   cavity
   cylinder_flow
   flow_obstacles
   rayleigh_benard
   taylor_green

Fluid Mechanics
===============

A family of worked Navier-Stokes examples in ``examples/fluid/``,
from the lid-driven cavity benchmark (2D and 3D) to a Rayleigh-Bénard
convection problem coupling momentum and energy. They all share the
same underlying recipe — a custom ``NavierStokesAssembler`` (or its
transient cousin) defined in the example file, SUPG/PSPG
stabilization for equal-order P1-P1 elements, Picard linearization
for steady solves, implicit Euler for transient ones — so once you
have read :doc:`cavity` the rest are essentially geometry and
boundary-condition variations.

The ``NavierStokesAssembler`` itself lives **in the example
folder**, not in ``tensormesh.assemble``. It is intentionally
example-grade: the production-quality version will move into the
core library when the assembler API for vector-valued problems
stabilizes.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Lid-Driven Cavity
      :link: cavity
      :link-type: doc
      :img-top: /_static/fluid/cavity_results.png

      Steady NS at Re=100, SUPG/PSPG stabilization, Picard iteration —
      in 2D and 3D with one dimension-generic assembler.

   .. grid-item-card:: Cylinder Flow (Vortex Shedding)
      :link: cylinder_flow
      :link-type: doc
      :img-top: /_static/fluid/vortex_street.gif

      Transient DFG benchmark, implicit Euler, drag/lift/Strouhal post-processing.

   .. grid-item-card:: Flow Past Multiple Obstacles
      :link: flow_obstacles
      :link-type: doc
      :img-top: /_static/fluid/flow_obstacles.png

      Steady channel flow at Re=150 around six circular obstacles via MeshGen CSG.

   .. grid-item-card:: Rayleigh-Bénard Convection
      :link: rayleigh_benard
      :link-type: doc
      :img-top: /_static/fluid/rayleigh_benard.png

      Boussinesq-coupled momentum + heat transport, buoyancy-driven flow.

   .. grid-item-card:: Taylor-Green Vortex
      :link: taylor_green
      :link-type: doc
      :img-top: /_static/fluid/taylor_green_vortex_final.png

      Decaying vortex with exact solution — the convergence-study showcase.


.. toctree::
   :hidden:
   :maxdepth: 1

   cavity
   cylinder_flow
   flow_obstacles
   rayleigh_benard
   taylor_green

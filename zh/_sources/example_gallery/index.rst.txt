Example Gallery
===============

Hands-on, runnable examples — from FEM building-block visualizations,
through canonical PDE solves, to research-grade fluid, solid, and
electromagnetic problems. Each page maps the underlying mathematics to
a concrete TensorMesh script you can run.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Basics & Visualization
      :link: basics
      :link-type: doc
      :img-top: /_static/gallery/basics_hybrid_mesh2d.png

      Basis points, shape functions, element ordering, and a mesh-generation gallery.

   .. grid-item-card:: Poisson Equation
      :link: poisson
      :link-type: doc
      :img-top: /_static/gallery/poisson_3d_half_from_cut.png

      Elliptic PDE in 2D / 3D, batched RHS, and h-adaptive refinement on the L-shape.

   .. grid-item-card:: Diffusion
      :link: diffusion
      :link-type: doc
      :img-top: /_static/gallery/diffusion_allen_cahn.gif

      Transient heat equation and nonlinear Allen-Cahn phase-field evolution.

   .. grid-item-card:: Wave Equation
      :link: wave
      :link-type: doc
      :img-top: /_static/gallery/wave.gif

      Hyperbolic PDE with explicit central-difference time stepping.

   .. grid-item-card:: Solid Mechanics
      :link: solid/index
      :link-type: doc
      :img-top: /_static/solid_mechanics/hyperelastic_rubber.png

      Linear elasticity, hyperelasticity, contact, and plasticity — a progressive solver ladder.

   .. grid-item-card:: Fluid Mechanics
      :link: fluid/index
      :link-type: doc
      :img-top: /_static/fluid/cavity_results.png

      Incompressible Navier-Stokes: cavity, cylinder flow, Rayleigh-Bénard, and more.

   .. grid-item-card:: Magnetostatics
      :link: maxwell
      :link-type: doc
      :img-top: /_static/maxwell/magnetostatic_field.png

      3D Maxwell: magnetic field around a wire via a stabilized nodal curl-curl formulation.

   .. grid-item-card:: Inverse Design & Identification
      :link: inverse_design
      :link-type: doc
      :img-top: /_static/gallery/inverse_optimization.gif

      Differentiable FEM: coefficient-field identification and density-based topology optimization.

   .. grid-item-card:: Physics-Informed Learning
      :link: physics_informed
      :link-type: doc
      :img-top: /_static/physics_informed/poisson_galerkin_loss.png

      Train a neural network to minimize the assembled Galerkin residual ``||K u - F||²``.

   .. grid-item-card:: ML Datasets
      :link: dataset
      :link-type: doc
      :img-top: /_static/gallery/dataset_poisson.jpg

      Batch generation of heat and wave snapshots for training neural operators.

   .. grid-item-card:: Distributed FEM
      :link: distributed
      :link-type: doc
      :img-top: /_static/distributed/graph_partition_exploded.png

      Graph coloring, spectral partitioning, and multi-GPU assembly benchmarks.


.. toctree::
   :hidden:
   :maxdepth: 2

   basics
   poisson
   diffusion
   wave
   solid/index
   fluid/index
   maxwell
   inverse_design
   physics_informed
   dataset
   distributed

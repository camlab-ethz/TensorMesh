User Guide
==========

In-depth, topic-oriented guides covering TensorMesh's core concepts,
design choices, and how to wield each component.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Concepts
      :link: concepts
      :link-type: doc

      What TensorMesh is, the FEM pipeline, and how the modules fit together.

   .. grid-item-card:: Meshes
      :link: meshes
      :link-type: doc

      Build, inspect, and load/save meshes; per-node and per-cell data.

   .. grid-item-card:: Elements & Quadrature
      :link: elements_and_quadrature
      :link-type: doc

      The element zoo, basis functions, quadrature rules, and ordering conventions.

   .. grid-item-card:: Forms
      :link: forms
      :link-type: doc

      Write weak forms in pure Python via the three assembler base classes.

   .. grid-item-card:: Boundary Conditions
      :link: boundary_conditions
      :link-type: doc

      Apply Dirichlet BCs via static condensation; handle Neumann naturally.

   .. grid-item-card:: Sparse Solvers
      :link: linear_solvers
      :link-type: doc

      Linear and nonlinear sparse solves via torch-sla — five backends, batched RHS, Newton / Picard / Anderson.

   .. grid-item-card:: Time Integration
      :link: time_integration
      :link-type: doc

      Explicit and implicit-linear Runge-Kutta schemes for transient problems.

   .. grid-item-card:: Batched Workflows
      :link: batched_workflows
      :link-type: doc

      Three axes of batching: memory chunking, batched RHS, and ML datasets.

   .. grid-item-card:: Differentiability
      :link: differentiability
      :link-type: doc

      End-to-end gradients through assemble → solve, for inverse problems and topology opt.


.. toctree::
   :maxdepth: 2
   :hidden:

   concepts
   meshes
   elements_and_quadrature
   forms
   boundary_conditions
   linear_solvers
   time_integration
   batched_workflows
   differentiability

API Reference
=============

Complete API documentation for all TensorMesh modules.


Core Modules
------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Mesh
      :link: mesh
      :link-type: doc
      
      Mesh data structures and operations.

   .. grid-item-card:: Element
      :link: element
      :link-type: doc
      
      Finite element definitions and quadrature.

   .. grid-item-card:: Assemble
      :link: assemble
      :link-type: doc
      
      Matrix and vector assembly routines.


Linear Algebra
--------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Operator
      :link: operator
      :link-type: doc

      Differential operators and boundary conditions.

   .. grid-item-card:: Sparse
      :link: sparse
      :link-type: doc

      Sparse matrix operations and solvers.


Solvers & Integration
---------------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: ODE
      :link: ode
      :link-type: doc

      Time integration and ODE solvers.

   .. grid-item-card:: Functional
      :link: functional
      :link-type: doc

      Functional forms and variational methods.

   .. grid-item-card:: Optimizer
      :link: optimizer
      :link-type: doc

      Optimization algorithms for inverse problems.


Physics & Parallelism
---------------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Material
      :link: material
      :link-type: doc

      Isotropic material model and library presets.

   .. grid-item-card:: Distributed
      :link: distributed
      :link-type: doc

      Multi-GPU mesh partitioning and parallel assembly.


Deep Learning & Data
--------------------

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card:: Neural Network
      :link: nn
      :link-type: doc
      
      Neural network modules for physics-informed learning.

   .. grid-item-card:: Dataset
      :link: dataset
      :link-type: doc
      
      Built-in datasets and data loaders.


Visualization
-------------

.. grid:: 1 2 2 2
   :gutter: 4

   .. grid-item-card:: Visualization
      :link: visualization
      :link-type: doc
      
      Plotting and visualization tools.


.. toctree::
   :maxdepth: 2
   :hidden:

   mesh
   element
   assemble
   operator
   sparse
   ode
   functional
   optimizer
   material
   distributed
   nn
   dataset
   visualization


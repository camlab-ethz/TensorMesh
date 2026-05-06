Benchmark
=========

This page compares TensorMesh against several established FEM frameworks on
two **forward** problems (3D Poisson and 3D linear elasticity) and one
**inverse** problem (compliance-minimization topology optimization).

All benchmark scripts and reference data live in a separate repository:
`camlab-ethz/tensormesh-bench
<https://github.com/camlab-ethz/tensormesh-bench>`_.

Test environment
----------------

.. list-table::
   :widths: 25 75

   * - **CPU**
     - 8 cores from a dual-socket AMD EPYC 9005-series (Zen 5 "Turin") node
   * - **GPU**
     - 1× NVIDIA RTX PRO 6000 Blackwell Server Edition
   * - **Memory**
     - 64 GB DDR5 ECC (8 cores × 8 GB)
   * - **OS**
     - Ubuntu 22.04 LTS (x86_64)

Compared frameworks
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 12 12 36

   * - Framework
     - Backend
     - CPU
     - CUDA
     - Notes
   * - `FEniCS (DOLFINx) <https://fenicsproject.org/>`_
     - C++/Python
     - ✅
     - ❌
     - 8-rank MPI
   * - `Firedrake <https://www.firedrakeproject.org/>`_
     - Python/PETSc
     - ✅
     - ❌
     - 8-rank MPI
   * - `MFEM <https://mfem.org/>`_
     - C++
     - ✅
     - ❌
     - 8-rank MPI
   * - `scikit-fem <https://scikit-fem.readthedocs.io/>`_
     - Python
     - ✅
     - ❌
     - Lightweight, single process
   * - `JAX-FEM <https://deepmodeling.github.io/jax-fem/>`_
     - JAX
     - ✅
     - ✅
     - Differentiable
   * - `torch-fem <https://github.com/meyer-nils/torch-fem>`_
     - PyTorch
     - ✅
     - ✅
     - Differentiable
   * - **TensorMesh**
     - PyTorch
     - ✅
     - ✅
     - Differentiable

Solver configuration
--------------------

To keep the comparison controlled, every framework uses the same iterative
linear solver and stopping criteria:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value
   * - Iterative method
     - BiCGSTAB
   * - Preconditioner
     - Jacobi (diagonal scaling)
   * - Relative tolerance
     - :math:`10^{-10}`
   * - Absolute tolerance
     - :math:`10^{-10}`
   * - Maximum iterations
     - 10,000

.. note::

   The only exception is the CUDA path of **torch-fem**, which uses
   conjugate gradient (CG) instead of BiCGSTAB — its CUDA backend does not
   ship a BiCGSTAB implementation.

Forward problems
----------------

3D Poisson equation
~~~~~~~~~~~~~~~~~~~

On the unit cube :math:`\Omega = [0,1]^3` we solve

.. math::

   \begin{cases}
       -\Delta u(\mathbf{x}) = f(\mathbf{x}) & \text{in } \Omega, \\
       u(\mathbf{x}) = 0 & \text{on } \partial\Omega,
   \end{cases}

with constant source :math:`f = 1` and homogeneous Dirichlet boundary
conditions on every face.

**Solve time and residual convergence.** Wall-clock time is plotted against
problem size; residual histories show that every framework reaches the
same target tolerance.

.. list-table::
   :widths: 50 50

   * - **Solve time**
     - **Residual convergence**
   * - .. image:: ../_static/benchmark/all_frameworks_time_combined_poisson_3d.png
          :width: 100%
          :alt: 3D Poisson solve time across frameworks
     - .. image:: ../_static/benchmark/all_frameworks_residual_combined_poisson_3d.png
          :width: 100%
          :alt: 3D Poisson residual convergence across frameworks

**Solution visualizations.** Spot-check that each framework produces the
same field — useful as a sanity check before reading speed numbers.

.. list-table::
   :widths: 25 25 25 25

   * - **FEniCS (DOLFINx)**
     - **JAX-FEM**
     - **scikit-fem**
     - **TensorMesh**
   * - .. image:: ../_static/benchmark/vis/fenics_poisson_3d_solution.png
          :width: 100%
          :alt: FEniCS Poisson 3D solution
     - .. image:: ../_static/benchmark/vis/jaxfem_poisson_3d_solution.png
          :width: 100%
          :alt: JAX-FEM Poisson 3D solution
     - .. image:: ../_static/benchmark/vis/skfem_poisson_3d_solution.png
          :width: 100%
          :alt: scikit-fem Poisson 3D solution
     - .. image:: ../_static/benchmark/vis/tensormesh_poisson_3d_solution.png
          :width: 100%
          :alt: TensorMesh Poisson 3D solution

3D linear elasticity
~~~~~~~~~~~~~~~~~~~~

We solve the static equilibrium of an isotropic linear elastic body:

.. math::

   \begin{cases}
       -\nabla \cdot \boldsymbol{\sigma}(\mathbf{x}) = \mathbf{f}(\mathbf{x}) & \text{in } \Omega, \\
       \mathbf{u}(\mathbf{x}) = \mathbf{0} & \text{on } \partial\Omega,
   \end{cases}

where :math:`\mathbf{u} : \Omega \to \mathbb{R}^3` is the displacement and
the Cauchy stress satisfies Hooke's law

.. math::

   \boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}) \, \mathbf{I}
   + 2\mu \, \boldsymbol{\varepsilon},
   \qquad
   \boldsymbol{\varepsilon} = \tfrac{1}{2}\bigl(\nabla \mathbf{u} + (\nabla \mathbf{u})^\top\bigr).

We set Young's modulus :math:`E = 1` and Poisson's ratio :math:`\nu = 0.3`,
giving

.. math::

   \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)},
   \qquad
   \mu = \frac{E}{2(1+\nu)}.

A constant body force :math:`\mathbf{f} = (1,1,1)^\top` is applied. To
introduce geometric complexity we use a hollow cube domain

.. math::

   \Omega = [0,1]^3 \setminus (0.25,\, 0.75)^3.

**Solve time and residual convergence.**

.. list-table::
   :widths: 50 50

   * - **Solve time**
     - **Residual convergence**
   * - .. image:: ../_static/benchmark/all_frameworks_time_combined_elasticity_3d.png
          :width: 100%
          :alt: 3D elasticity solve time across frameworks
     - .. image:: ../_static/benchmark/all_frameworks_residual_combined_elasticity_3d.png
          :width: 100%
          :alt: 3D elasticity residual convergence across frameworks

**Solution visualizations.**

.. list-table::
   :widths: 25 25 25 25

   * - **FEniCS (DOLFINx)**
     - **JAX-FEM**
     - **scikit-fem**
     - **TensorMesh**
   * - .. image:: ../_static/benchmark/vis/fenics_elasticity_3d_solution.png
          :width: 100%
          :alt: FEniCS elasticity 3D solution
     - .. image:: ../_static/benchmark/vis/jaxfem_elasticity_3d_solution.png
          :width: 100%
          :alt: JAX-FEM elasticity 3D solution
     - .. image:: ../_static/benchmark/vis/skfem_elasticity_3d_solution.png
          :width: 100%
          :alt: scikit-fem elasticity 3D solution
     - .. image:: ../_static/benchmark/vis/tensormesh_elasticity_3d_solution.png
          :width: 100%
          :alt: TensorMesh elasticity 3D solution

Inverse problem: topology optimization
--------------------------------------

The inverse-problem benchmark exercises the **differentiability** of
TensorMesh on a classical topology-optimization task: compliance
minimization of a 2D cantilever beam using the Solid Isotropic Material
with Penalization (SIMP) method. We compare against JAX-FEM, the only
other framework in the list above that supports end-to-end automatic
differentiation through the FEM pipeline.

Problem setup
~~~~~~~~~~~~~

The computational domain is a rectangle
:math:`\Omega = [0, L_x] \times [0, L_y]` with :math:`L_x = 60` and
:math:`L_y = 30` (dimensionless), discretized into :math:`60 \times 30`
bilinear quadrilateral (QUAD4) elements — 1,891 nodes and 1,800
elements. Homogeneous Dirichlet conditions are imposed on the left edge,

.. math::

   \mathbf{u} = \mathbf{0} \quad \text{on } \Gamma_D = \{(x,y) : x = 0\},

and a distributed traction is applied to the lower portion of the right
boundary,

.. math::

   \mathbf{t} = (0,\, -100)^\top \;\text{N/m}
   \quad \text{on } \Gamma_N = \{(x,y) : x = L_x,\; 0 \leq y \leq 0.1\, L_y\}.

.. image:: ../_static/inverse/boundary_conditions.png
   :width: 60%
   :align: center
   :alt: Cantilever boundary conditions

The element stiffness is parameterized by the SIMP interpolation

.. math::

   E(\rho) = E_{\min} + \rho^p \,(E_{\max} - E_{\min}),

where :math:`\rho \in [\rho_{\min}, 1]` is the element-wise density (the
design variable), :math:`p = 3` penalizes intermediate densities,
:math:`E_{\max} = 70{,}000` MPa is the solid stiffness, and
:math:`E_{\min} = 70` MPa keeps the stiffness matrix non-singular. The
Poisson's ratio is :math:`\nu = 0.3`.

The optimization minimizes structural compliance subject to a volume
constraint:

.. math::

   \begin{aligned}
       \min_{\boldsymbol{\rho}} \quad
           & C(\boldsymbol{\rho})
             = \mathbf{u}^\top \mathbf{K}(\boldsymbol{\rho}) \mathbf{u}
             = \mathbf{F}^\top \mathbf{u} \\
       \text{s.t.} \quad
           & \frac{1}{|\Omega|} \int_{\Omega} \rho \, d\Omega \leq \bar{v} \\
           & \mathbf{K}(\boldsymbol{\rho}) \mathbf{u} = \mathbf{F} \\
           & \rho_{\min} \leq \rho_e \leq 1, \quad \forall e
   \end{aligned}

with target volume fraction :math:`\bar{v} = 0.5` and
:math:`\rho_{\min} = 10^{-3}`. We use the Method of Moving Asymptotes (MMA)
with move limit :math:`\Delta\rho_{\max} = 0.1` and a sensitivity filter
of radius :math:`r_{\min} = 1.5\,h` (where :math:`h` is the element size)
to suppress checkerboard patterns. The optimization runs for 51
iterations.

Sensitivity via automatic differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classical adjoint-method sensitivity for SIMP has the closed form

.. math::

   \frac{\partial C}{\partial \rho_e}
   = -p\, \rho_e^{p-1} (E_{\max} - E_{\min})\,
     \mathbf{u}_e^\top \mathbf{K}_0^e \mathbf{u}_e,

where :math:`\mathbf{K}_0^e` is the element stiffness at unit Young's
modulus. In TensorMesh this gradient is **not implemented manually** —
it is obtained by backpropagating through the differentiable assembly
and sparse solve via PyTorch's reverse-mode autograd. The closed-form
expression above is reproduced only for reference and as a consistency
check against the autograd output.

Final design comparison
~~~~~~~~~~~~~~~~~~~~~~~

Both frameworks converge to the same canonical truss-like topology, and
TensorMesh runs noticeably faster end-to-end thanks to GPU-resident
assembly and solve.

.. image:: ../_static/inverse/final_comparison.png
   :width: 100%
   :align: center
   :alt: Final compliance and topology comparison

Optimization animations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50 50

   * - **JAX-FEM**
     - **TensorMesh**
   * - .. image:: ../_static/inverse/jaxfem_optimization.gif
          :width: 100%
          :alt: JAX-FEM topology-optimization animation
     - .. image:: ../_static/inverse/tensormesh_optimization.gif
          :width: 100%
          :alt: TensorMesh topology-optimization animation

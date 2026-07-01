Concepts
========

TensorMesh is a finite-element library written from the ground up for
PyTorch. A :class:`~tensormesh.Mesh` is an :class:`torch.nn.Module`,
weak forms are plain ``forward`` methods that receive basis tensors,
and every linear solve is a differentiable op. The same code that
solves a Poisson problem on a laptop CPU also runs on a GPU and
backpropagates through to a learnable parameter — without changing
the FEM logic.

This page is the mental model: how the modules fit together and the
design principles behind those choices.


The FEM workflow
----------------

Solving a PDE in TensorMesh follows one canonical pipeline:

.. code-block:: text

   Mesh  →  Assembler  →  SparseMatrix  →  Condenser  →  Solve

1. **Mesh** discretizes the domain into points and cells.
2. **Assembler** turns a weak form (``a(u, v)`` or ``l(v)``) into a
   :class:`~tensormesh.sparse.SparseMatrix` or load vector.
3. **Condenser** applies Dirichlet boundary conditions by static
   condensation, producing a reduced system on the interior DOFs.
4. **Solve** dispatches the reduced system to a sparse-linear-algebra
   backend (via the :doc:`torch-sla <linear_solvers>` package).

The :doc:`../getting_started/quickstart` walks through this pipeline
end-to-end in about 30 lines of Python. Each subsequent chapter of
this guide zooms in on one stage.


Module map
----------

The library splits cleanly along the pipeline. The arrows show data
flow, not import direction:

.. code-block:: text

   ┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────┐
   │   Mesh   │ →  │ Assembler  │ →  │ SparseMatrix │ →  │ Condenser │ →  │  Solve  │
   │  (nn.    │    │ (Element / │    │ (torch_sla.  │    │ (Dirichlet│    │ (torch- │
   │  Module) │    │  Node /    │    │  SparseTensor│    │  static   │    │  sla    │
   │          │    │  Facet)    │    │  + spmm / @) │    │  cond.)   │    │ .solve) │
   └──────────┘    └────────────┘    └──────────────┘    └───────────┘    └─────────┘
        ↑                ↑                                                       │
        │                │                                                       ▼
   ┌──────────┐    ┌────────────┐                                          ┌───────────┐
   │ element  │    │ functional │                                          │ Postproc  │
   │ (Triangle│    │ (voigt,    │                                          │ visualize │
   │  Hex,…)  │    │  strain,…) │                                          │  ode step │
   └──────────┘    └────────────┘                                          └───────────┘

What lives in each module:

* :mod:`tensormesh.mesh` — :class:`~tensormesh.Mesh` and its built-in
  generators (``gen_rectangle``, ``gen_circle``, ``gen_cube``, …);
  meshio I/O; adjacency, partitioning, and graph coloring helpers.
* :mod:`tensormesh.element` — reference shapes
  (:class:`~tensormesh.Triangle`, :class:`~tensormesh.Quadrilateral`,
  :class:`~tensormesh.Tetrahedron`, :class:`~tensormesh.Hexahedron`,
  :class:`~tensormesh.Prism`, :class:`~tensormesh.Pyramid`,
  :class:`~tensormesh.Line`), basis evaluation, quadrature rules, and
  the Gmsh/VTK ↔ TensorMesh ordering convention.
* :mod:`tensormesh.assemble` — the three weak-form base classes
  :class:`~tensormesh.ElementAssembler`,
  :class:`~tensormesh.NodeAssembler`,
  :class:`~tensormesh.FacetAssembler`, plus built-ins for the most
  common forms (Laplace, mass, linear elasticity, Neo-Hookean, …).
* :mod:`tensormesh.sparse` — :class:`~tensormesh.sparse.SparseMatrix`
  (subclass of ``torch_sla.SparseTensor``), so linear systems are
  solved by ``K.solve(b)`` and nonlinear systems by
  ``K.nonlinear_solve(residual, u0, *params)``, dispatched through
  ``torch-sla``. The in-tree ``spsolve`` /
  :func:`~tensormesh.sparse.nonlinear_solve` free functions are
  legacy entry points scheduled for removal.
* :mod:`tensormesh.operator` — :class:`~tensormesh.Condenser` for
  Dirichlet BCs via static condensation.
* :mod:`tensormesh.ode` — explicit and implicit-linear time
  integrators (Euler, midpoint, Runge-Kutta) for transient problems.
* :mod:`tensormesh.functional` — Voigt elasticity helpers, strain /
  stress, and other tensor utilities used inside ``forward`` methods.
* :mod:`tensormesh.dataset` — :class:`~tensormesh.MeshGen` and pre-built
  multi-frequency equation classes for generating training datasets.
* :mod:`tensormesh.material` —
  :class:`~tensormesh.material.IsotropicMaterial` and library
  presets (Steel, Aluminum, Rubber, Glass).
* :mod:`tensormesh.optimizer` — :class:`~tensormesh.optimizer.OCOptimizer`
  (Optimality Criteria) for compliance-based topology optimization.
* :mod:`tensormesh.visualization` — matplotlib (2D) and PyVista (3D)
  backends; lazily imported by :meth:`~tensormesh.Mesh.plot`.
* :mod:`tensormesh.distributed` — graph-partitioned distributed
  assembly across multiple ranks (advanced; see the example gallery).

The sparse-linear-algebra stack (``SparseMatrix``, ``.solve`` /
``.nonlinear_solve``, gradient-aware adjoint backward) is delegated to
a separate package,
`torch-sla <https://www.torchsla.com/>`_, and shared with
other projects in the same ecosystem.


Design principles
-----------------

**PyTorch-native.** :class:`~tensormesh.Mesh` extends
:class:`torch.nn.Module`. Its ``points`` and per-element connectivity
are buffers; per-node fields are buffers too. Assemblers are also
``nn.Module`` s. There is no separate "FEM kernel" abstraction layer —
everything is a tensor that flows through familiar PyTorch machinery
(``.to(device)``, ``.double()``, ``state_dict``, autograd, JIT
tracing).

**Weak forms in pure Python.** The only PDE-specific code a user
writes is a ``forward`` method that returns the integrand at the
quadrature points:

.. code-block:: python

   class LaplaceAssembler(ElementAssembler):
       def forward(self, gradu, gradv):
           return gradu @ gradv

The library handles reference-element evaluation, geometry,
quadrature weights, and the global assemble-into-sparse step. The
same pattern works for load vectors (:class:`~tensormesh.NodeAssembler`)
and boundary integrals (:class:`~tensormesh.FacetAssembler`).

**Tensorized assembly.** There is no Python-level loop over
elements. Inside ``__call__``, basis functions and quadrature points
are evaluated once for the whole mesh; the user's ``forward`` runs
on a tensor that already has element and quadrature dimensions
broadcast-ready; the global assemble is a sparse scatter. The
result: assembly is a single GPU kernel.

**Differentiable by construction.** ``SparseMatrix.solve`` is a
:class:`torch.autograd.Function` with a custom backward (an adjoint
sparse solve). Gradients therefore flow end-to-end from a loss back
through the linear solve, the assembly, and any parameter that
touched either — be it a material coefficient, a Dirichlet value, or
a neural network's prediction. See :doc:`differentiability`.

**Modular linear algebra.** The solver layer is a separate package,
``torch-sla``. The same FEM code retargets between SciPy (CPU),
Eigen (CPU), native PyTorch (CPU/GPU), cuDSS (GPU), and CuPy (GPU)
by changing one keyword argument. PETSc and Hypre are on the
``torch-sla`` roadmap; until they ship, fallback paths in
:mod:`tensormesh.sparse` provide best-effort support if those
libraries are already installed locally.


What's next
-----------

* :doc:`meshes` — build, inspect, and load meshes; per-node and
  per-cell data; meshio round-tripping.
* :doc:`elements_and_quadrature` — the element zoo and the basis
  / quadrature interface.
* :doc:`forms` — write your own weak form against the
  :class:`~tensormesh.ElementAssembler` /
  :class:`~tensormesh.NodeAssembler` /
  :class:`~tensormesh.FacetAssembler` contract.
* :doc:`../getting_started/quickstart` — the same pipeline as a
  complete worked example.

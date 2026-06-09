Sparse Solvers
==============

After assembly and boundary-condition condensation you have a sparse
system — linear or nonlinear — to solve. TensorMesh delegates this work
to `torch-sla <https://www.torchsla.com/>`_, a standalone differentiable
sparse-linear-algebra library. The same FEM code retargets between CPU
and GPU, and between iterative and direct solvers, by changing one
keyword argument; nonlinear systems are driven by the same backend
through a Newton / Picard / Anderson interface.


Install ``torch-sla``
---------------------

.. important::

   ``torch-sla`` is a **hard, import-time** dependency of
   :mod:`tensormesh.sparse`. The module will not import without it.
   It is also the library where all current and future solver work
   lands — TensorMesh tracks ``torch-sla`` releases as the canonical
   sparse-linear-algebra layer and we recommend keeping it up to date.

The base wheel ships the CPU stack (SciPy / native PyTorch).
GPU backends are opt-in extras — you can install one or both, depending
on which CUDA solver you want:

.. code-block:: bash

   pip install torch-sla              # CPU stack only
   pip install "torch-sla[cupy]"      # + CuPy backend (iterative + SuperLU)
   pip install "torch-sla[cudss]"     # + cuDSS backend (fastest GPU direct)
   pip install "torch-sla[all]"       # both GPU backends + dev tooling

TensorMesh also exposes the GPU extras at its own install layer
(``pip install "tensormesh-fem[cupy]"`` etc.) — see
:doc:`/getting_started/installation`.

Inspect available backends
~~~~~~~~~~~~~~~~~~~~~~~~~~

After installing, ``torch_sla.show_backends()`` prints a status table
showing which backends are usable on the current machine and how to
install any that are missing:

.. code-block:: python

   >>> import torch_sla
   >>> torch_sla.show_backends()
   torch-sla backend status (CUDA: available)
     scipy    [CPU]      available
     eigen    [CPU]      not available — JIT-compiled C++ extension (requires a C++ compiler)
     pytorch  [CPU/CUDA] available
     cupy     [CUDA]     not available — pip install torch-sla[cupy]
     cudss    [CUDA]     not available — pip install torch-sla[cudss]

What ``torch-sla`` gives TensorMesh:

* :class:`~tensormesh.sparse.SparseTensor` — the COO data type that
  :class:`~tensormesh.sparse.SparseMatrix` extends. Element assembly
  hands you a :class:`~tensormesh.sparse.SparseMatrix` directly, so every operator method
  on the parent class (``@``, ``.solve``, ``.nonlinear_solve``,
  ``.is_symmetric``, …) is available without any extra wrapping.
* A unified ``solve`` op with a custom backward pass (an adjoint
  sparse solve), so gradients flow through every linear system.
* A pluggable backend layer covering CPU, GPU, iterative and direct
  solvers behind a single dispatch entry point.


Solving ``A x = b``: the ``.solve()`` method
--------------------------------------------

The canonical entry point is ``SparseMatrix.solve`` — inherited
unchanged from ``torch_sla.SparseTensor``. Assembly already returns a
:class:`~tensormesh.sparse.SparseMatrix`, so this is the only call you normally need:

.. code-block:: python

   K = LaplaceElementAssembler.from_mesh(mesh)()
   x = K.solve(b)                              # auto backend + method
   x = K.solve(b, backend="cudss")             # force NVIDIA GPU direct
   x = K.solve(b, method="bicgstab")           # force a specific iterative method

**Symmetry / positive-definiteness is auto-detected.** On every call,
``torch-sla`` computes ``is_symmetric()`` and
``is_positive_definite()`` on the matrix, then picks CG / Cholesky
for SPD systems and BiCGStab / LU otherwise. You do **not** pass an
``is_spd`` hint; passing one would be ignored.

Key keyword arguments (full list in the ``torch-sla`` reference):

* ``backend``: ``"auto"`` (CPU → SciPy, CUDA → cuDSS when available,
  else CuPy / native PyTorch), or one of ``"scipy"``, ``"eigen"``,
  ``"pytorch"``, ``"cupy"``, ``"cudss"``.
* ``method``: ``"auto"`` (chosen from matrix properties + backend), or
  an iterative method (``"cg"``, ``"bicgstab"``, ``"minres"``,
  ``"gmres"``, ``"lgmres"``) or a direct factorization (``"lu"``,
  ``"umfpack"``, ``"cholesky"``, ``"ldlt"``). Not every backend
  supports every method — see the table below.
* ``atol`` (default ``1e-10``), ``tol`` (default ``1e-12``),
  ``maxiter`` (default ``10000``) for iterative convergence.
* ``verbose=True`` prints a one-line summary of the auto-selected
  backend / method and the detected matrix properties — handy when
  you want to confirm that ``"auto"`` picked the path you expected:

  .. code-block:: pycon

     >>> x = K.solve(b, verbose=True)
     [torch-sla] solve: n=1024, nnz=7168, dtype=float64, device=cpu, symmetric=True, spd=True, backend=scipy, method=lu


Supported backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 16 46

   * - Backend
     - String
     - Device
     - Direct / Iter
     - Methods
   * - SciPy
     - ``"scipy"``
     - CPU
     - both
     - ``lu``, ``umfpack``, ``cg``, ``bicgstab``, ``gmres``,
       ``lgmres``, ``minres``, ``qmr``. Default on CPU.
   * - Eigen
     - ``"eigen"``
     - CPU
     - iterative
     - ``cg``, ``bicgstab``. C++ Eigen via pybind11.
   * - Native PyTorch
     - ``"pytorch"``
     - CPU / GPU
     - iterative
     - ``cg``, ``bicgstab`` with Jacobi preconditioning. Pure
       torch — fully autograd-traced.
   * - CuPy
     - ``"cupy"``
     - GPU
     - both
     - ``lu``, ``cg``, ``cgs``, ``gmres``, ``minres``, ``lsqr``,
       ``lsmr``.
   * - cuDSS
     - ``"cudss"``
     - GPU
     - direct
     - ``lu``, ``cholesky``, ``ldlt``. NVIDIA cuDSS — fastest GPU
       path; default on CUDA when memory allows.


Batched right-hand sides
------------------------

When ``b`` has shape ``[n_dof, n_batch]``, ``torch-sla`` automatically
amortizes the factorization across the batch — one factor pass,
``n_batch`` back-substitutions — instead of running an iterative
solver independently per column. This is the workhorse of the
:mod:`tensormesh.dataset` ML workflow.

.. code-block:: python

   B = torch.randn(K.shape[0], 64)             # 64 right-hand sides
   X = K.solve(B)                              # one factorization

See :doc:`batched_workflows` for the broader story on batching.


Nonlinear systems: ``.nonlinear_solve()``
-----------------------------------------

For problems where the residual ``F(u, params) = 0`` is nonlinear in
``u`` — hyperelasticity, plasticity, phase-field — use
``SparseMatrix.nonlinear_solve`` (inherited from
``torch_sla.SparseTensor``). The matrix is passed automatically into
the residual closure, the Jacobian is obtained through autograd, and
the backward pass uses the adjoint method.

.. code-block:: python

   from torch_sla import SparseTensor  # SparseMatrix also works

   # Nonlinear PDE: A @ u + u**2 = f
   def residual(u, A, f):
       return A @ u + u**2 - f

   f  = torch.randn(n, requires_grad=True)
   u0 = torch.zeros(n)

   u = A.nonlinear_solve(residual, u0, f, method='newton')

   # Gradients flow via the implicit-function theorem
   loss = u.sum()
   loss.backward()
   print(f.grad)        # dL/df

Key keyword arguments:

* ``method``: ``"newton"`` (default, Newton-Raphson with Armijo line
  search), ``"picard"`` (fixed-point), or ``"anderson"`` (Anderson
  acceleration).
* ``line_search`` (default ``True``) — toggle Armijo backtracking.
* ``tol`` / ``atol`` / ``max_iter`` for the outer nonlinear loop.
* ``linear_solver`` / ``linear_method`` to pick the backend used for
  each inner linear solve.

The forward pass runs the chosen nonlinear iteration; the backward
pass solves a single adjoint system, so gradients of ``u`` w.r.t. any
``requires_grad`` parameter (including ``A.values``) cost roughly one
extra linear solve regardless of how many nonlinear iterations the
forward took.


Lower-level entry points (legacy)
---------------------------------

.. deprecated:: 0.x

   :func:`tensormesh.sparse.spsolve` and
   :func:`tensormesh.sparse.nonlinear_solve` are TensorMesh-internal
   fallbacks that pre-date the ``torch-sla`` integration. They are
   **scheduled for removal**; the canonical solver path is the
   :class:`~tensormesh.sparse.SparseMatrix` (i.e. ``torch_sla.SparseTensor``) methods
   above.

:func:`tensormesh.sparse.spsolve` is a free function that operates on
raw COO arrays ``(edata, row, col, shape, b)`` instead of a
:class:`~tensormesh.sparse.SparseMatrix` object. In practice this is rarely useful —
assembly produces a :class:`~tensormesh.sparse.SparseMatrix`, condensation produces a
:class:`~tensormesh.sparse.SparseMatrix`, and ``K.solve(b)`` covers every FEM workflow
in this guide. ``tensormesh.sparse.spsolve`` will be retired once the
remaining call sites have moved over to ``SparseMatrix.solve`` /
``torch_sla.spsolve``.

:func:`tensormesh.sparse.nonlinear_solve` is the in-tree Newton-Raphson
helper. It requires the user to supply an explicit Jacobian closure
and lacks line search, Picard / Anderson modes, and the autograd-based
Jacobian assembly that ``torch_sla.SparseTensor.nonlinear_solve``
provides. Migrate to ``A.nonlinear_solve(residual, u0, *params)``.


What's next
-----------

* :doc:`time_integration` — wrap a linear solve in a time-stepping
  loop for transient problems.
* :doc:`batched_workflows` — three different axes of batching, and
  when each one applies.
* :doc:`differentiability` — how the adjoint backward works and how
  to wire a learnable parameter through a sparse solve.

Verify Install
==============

A self-contained smoke test that confirms TensorMesh, ``torch-sla``, and
your PyTorch build are wired up correctly. The script:

* prints the core versions (TensorMesh, PyTorch, torch-sla, CUDA),
* solves a tiny Poisson problem on the **CPU** (always),
* repeats the solve on the **GPU** if CUDA is available, and
* reports the ``torch-sla`` sparse-solver backends available on your
  machine.

The checker ships with the package, so after installing just run
``python -m tensormesh.verify_install`` — there is no file to save. Total
runtime is a couple of seconds; the full source is reproduced below for
reference.


The script
----------

.. code-block:: python

   """Verify a TensorMesh install: print the core versions, solve a tiny
   Poisson problem on CPU (and on GPU if available), and report the
   torch-sla sparse-solver backends available on this machine."""

   import math
   import time

   import torch
   import torch_sla
   import tensormesh


   def solve_poisson(device):
       from tensormesh import ElementAssembler, NodeAssembler, Mesh, Condenser

       mesh = Mesh.gen_rectangle(chara_length=0.1).to(device)

       class Laplace(ElementAssembler):
           def forward(self, gradu, gradv):
               return gradu @ gradv

       class Source(NodeAssembler):
           def forward(self, v, f):
               return f * v

       x, y = mesh.points[:, 0], mesh.points[:, 1]
       f_vals = 2 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)

       K = Laplace.from_mesh(mesh)()
       b = Source.from_mesh(mesh)(point_data={"f": f_vals})

       cond = Condenser(mesh.boundary_mask)
       K_, b_ = cond(K, b)
       u = cond.recover(K_.solve(b_))

       u_exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)
       return float((u - u_exact).norm() / u_exact.norm())


   def main():
       print("TensorMesh smoke test")
       print("=" * 40)
       print(f"tensormesh : {tensormesh.__version__}")
       print(f"torch      : {torch.__version__}")
       print(f"torch-sla  : {torch_sla.__version__}")
       print(f"cuda       : {torch.version.cuda or 'not available'}")
       print()

       t0 = time.perf_counter()
       err = solve_poisson("cpu")
       print(f"[CPU ] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter() - t0:.2f} s")

       if torch.cuda.is_available():
           t0 = time.perf_counter()
           err = solve_poisson("cuda")
           print(f"[CUDA] Poisson 2D ... OK   L2 error = {err:.3e}   {time.perf_counter() - t0:.2f} s")
       else:
           print("[CUDA] not available, skipping GPU test")

       print()
       # Every sparse-solver backend is provided by torch-sla; let it report them.
       torch_sla.show_backends()

       print()
       print("All required checks passed.")


   if __name__ == "__main__":
       main()


Expected output
---------------

On a CUDA machine (a Linux box with an NVIDIA GPU and a CUDA-enabled
PyTorch build), you should see something close to:

.. code-block:: text

   TensorMesh smoke test
   ========================================
   tensormesh : 0.1.0
   torch      : 2.10.0+cu128
   torch-sla  : 0.2.1
   cuda       : 12.8

   [CPU ] Poisson 2D ... OK   L2 error = 1.185e-02   0.05 s
   [CUDA] Poisson 2D ... OK   L2 error = 1.185e-02   0.88 s

   torch-sla backend status (CUDA: available)
     scipy    [CPU]       available
     eigen    [CPU]       not available — JIT-compiled C++ extension (requires a C++ compiler)
     pytorch  [CPU/CUDA]  available
     cupy     [CUDA]      available
     cudss    [CUDA]      available

   All required checks passed.

On a CPU-only machine (e.g. a macOS / Linux laptop) the ``cuda`` line
reads ``not available``, the ``[CUDA]`` solve is skipped, and the
backend-table header reads ``(CUDA: not available)`` with ``cupy`` /
``cudss`` listed as optional extras to install.

The exact L2 error depends on the mesh, but should be on the order of
:math:`10^{-2}` — anything more than a few percent indicates a numerical
problem.

.. note::

   **Why is the GPU run sometimes *slower* than the CPU run?**
   The smoke-test mesh is intentionally tiny (:math:`\sim` 100 DOFs). At
   this scale, the CUDA solve is dominated by one-time overheads —
   context creation, JIT kernel compilation, host↔device transfers, and
   ``cuSPARSE``/``cuSOLVER`` workspace allocation — rather than by
   actual floating-point work. The first GPU call also pays for CUDA
   driver initialization. Run a real-sized problem
   (:math:`\geq 10^4`–:math:`10^5` DOFs) to see the GPU pull ahead;
   see :doc:`../performance/index` for benchmarks.

.. note::

   **Which solver does ``A.solve(b)`` actually use by default?**
   ``SparseMatrix.solve`` is inherited from ``torch_sla.SparseTensor``
   and called with ``backend="auto"``, ``method="auto"``. torch-sla's
   auto-selector then picks, based on device and problem size:

   * **CPU** → SciPy / **SuperLU** (``backend="scipy"``, ``method="lu"``) —
     a direct factorization, fast and machine-precision for the sizes
     reachable on CPU.
   * **CUDA, DOF < 2M** → **cuDSS** if available
     (``method="cholesky"`` when SPD, else ``ldlt`` / ``lu``); falls
     back to CuPy, and finally to the PyTorch-native iterative CG.
   * **CUDA, DOF ≥ 2M** → PyTorch-native iterative solver
     (``backend="pytorch"``, ``method="cg"`` for SPD or
     ``bicgstab`` otherwise) with Jacobi preconditioning, to stay
     within GPU memory.

   So the smoke test above runs **SuperLU on CPU** and **cuDSS Cholesky
   on GPU** (Poisson is SPD), not an iterative solver. To pick a
   different solver explicitly, pass ``backend=...`` / ``method=...``
   to ``solve`` — see :doc:`../user_guide/linear_solvers`.

Every sparse-solver backend is provided by ``torch-sla`` — TensorMesh
adds none of its own. Backends reported as ``not available`` are not a
failure; they are optional ``torch-sla`` extras you can install when you
need them:

* **scipy** ``[CPU]`` — SciPy / SuperLU direct and iterative solvers; the
  default CPU path, always available.
* **pytorch** ``[CPU/CUDA]`` — torch-native iterative solvers (CG /
  BiCGSTAB), always available and fully autograd-friendly.
* **eigen** ``[CPU]`` — a JIT-compiled C++ direct solver; needs a C++
  compiler on the machine.
* **cupy** ``[CUDA]`` / **cudss** ``[CUDA]`` — GPU sparse-direct solvers
  (``pip install torch-sla[cupy]`` / ``pip install torch-sla[cudss]``).

Run ``torch_sla.show_backends()`` at any time to re-check status. See
:doc:`../user_guide/linear_solvers` for guidance on choosing among them.


Troubleshooting
---------------

**``ModuleNotFoundError: No module named 'tensormesh'``** — the package
isn't installed in the active Python environment. Re-check
:doc:`installation`.

**``ModuleNotFoundError: No module named 'torch_sla'``** — install it
with ``pip install "torch-sla>=0.2.1"``. torch-sla is a hard,
import-time dependency: :mod:`tensormesh.sparse` refuses to import
without it, and every sparse-solver backend is provided through it.

**Stuck on ``[CPU ] Poisson 2D``** — your PyTorch build may be downloading
``gmsh`` cache files on first use; subsequent runs are instant.

**``L2 error`` ≫ ``1e-2``** — likely an issue with the mesh or PyTorch
build. Open an issue at
`github.com/camlab-ethz/TensorMesh/issues
<https://github.com/camlab-ethz/TensorMesh/issues>`_ with the full output
of this script.

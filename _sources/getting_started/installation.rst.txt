Installation
============

TensorMesh runs on Linux, macOS, and Windows. The only hard requirements are
**Python ≥ 3.10** and **PyTorch ≥ 2.0**; everything else (NumPy, SciPy, meshio,
``torch-sla``, ...) is pulled in automatically by ``pip``.

If you plan to use the GPU backend, install a CUDA-enabled PyTorch build *before*
installing TensorMesh — follow the official `PyTorch installation selector
<https://pytorch.org/get-started/locally/>`_ for the right command on your
platform / CUDA version.


Install via PyPI
----------------

The recommended way to install TensorMesh is from PyPI:

.. code-block:: bash

    pip install tensormesh-fem             # CPU stack (SciPy + PyTorch Krylov)
    pip install "tensormesh-fem[gpu]"      # + all PyPI solver backends (cuDSS + PyAMG)

Use the second form if you have an NVIDIA GPU and want the fastest CUDA
direct solver — the ``[gpu]`` extra pulls in cuDSS (plus PyAMG) through
``torch-sla``. The quotes are needed because ``[...]`` is a shell glob
character; see `Sparse solvers and GPU acceleration`_ for the per-backend
breakdown if you only want one of them.

Either form pulls in all required dependencies, including
`torch-sla <https://www.torchsla.com/>`_, the differentiable sparse linear
algebra library that powers TensorMesh's solvers. The base
``pip install tensormesh-fem`` installs the **CPU** sparse stack
(SciPy + a device-agnostic native-PyTorch Krylov backend); see
`Sparse solvers and GPU acceleration`_ below for the full extras matrix
(``[cudss]`` / ``[pyamg]`` / ``[gpu]``) and how to verify which backends
are usable on your machine.


Install from source
-------------------

For development work, or to get the latest unreleased changes, clone the
repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/camlab-ethz/TensorMesh.git
    cd TensorMesh
    pip install -e .

The ``-e`` (editable) flag means edits to the source tree are picked up
without reinstalling. To install with the test dependencies as well:

.. code-block:: bash

    pip install -e ".[test]"


Sparse solvers and GPU acceleration
-----------------------------------

The sparse linear-algebra layer that powers TensorMesh's solvers lives in a
standalone library, `torch-sla <https://www.torchsla.com/>`_, so that it can
evolve independently and serve other projects. ``torch-sla`` is a **hard,
import-time** dependency — :mod:`tensormesh.sparse` will not import without it
— and is the canonical entry point for both CPU and GPU sparse solves. All
current and future solver work lands in ``torch-sla`` first; we recommend
keeping it up to date.

The base ``tensormesh-fem`` wheel pulls the **CPU** stack
(SciPy + native PyTorch Krylov — the latter also runs on CUDA/ROCm
devices out of the box). To add another solver backend, install one of
the mirrored extras:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Install command
     - Adds backend
     - When to pick this
   * - ``pip install tensormesh-fem``
     - CPU stack
     - Default; ``scipy`` direct/iterative + device-agnostic
       ``pytorch`` Krylov.
   * - ``pip install "tensormesh-fem[cudss]"``
     - cuDSS (CUDA)
     - Fastest GPU direct solver (LU / Cholesky / LDLT).
   * - ``pip install "tensormesh-fem[pyamg]"``
     - PyAMG (CPU)
     - Algebraic-multigrid V-cycles for large PDE systems.
   * - ``pip install "tensormesh-fem[gpu]"``
     - All PyPI backends
     - Convenience extra — installs ``torch-sla[all]``
       (cuDSS + PyAMG).

These mirror the upstream ``torch-sla`` extras (``[cudss]`` /
``[pyamg]`` / ``[all]``) — installing ``tensormesh-fem[gpu]`` is exactly
equivalent to ``pip install tensormesh-fem torch-sla[all]``, just spelled
in one step. Two further backends — ``strumpack`` (portable multifrontal
direct, CPU/CUDA/ROCm) and ``amgx`` (NVIDIA AmgX) — are compiled
extensions shipped as prebuilt wheels on `torch-sla's GitHub Releases
<https://github.com/sparsexlab/torch-sla>`_, not PyPI.

Inspect what's installed
~~~~~~~~~~~~~~~~~~~~~~~~

After installing, you can see which backends are usable on the current
machine — and a one-line install hint for any that are not:

.. code-block:: python

    >>> import torch_sla
    >>> torch_sla.show_backends()
    torch-sla backend status (CUDA: available)
      scipy      [CPU]           available
      pytorch    [CPU/CUDA]      available
      cudss      [CUDA]          available
      pyamg      [CPU]           available
      amgx       [CUDA]          not available — pip install torch-sla[amgx]
      strumpack  [CPU/CUDA/ROCm] not available — pip install torch-strumpack

See :doc:`/user_guide/linear_solvers` for the full backend / method matrix
and how to pick a non-default backend at solve time.


Other extras
~~~~~~~~~~~~

A couple of smaller extras for optional functionality:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Extra
     - Install command
   * - Plotly for example notebooks
     - ``pip install "tensormesh-fem[example]"``
   * - Test suite (pytest, pytest-cov)
     - ``pip install "tensormesh-fem[test]"``

Two further packages are commonly useful but are *not* declared as extras —
install them directly when needed:

.. code-block:: bash

    pip install gmsh       # external mesh generation / .msh I/O
    pip install pyvista    # interactive 3D visualization


Next steps
----------

Once installed, head to :doc:`verify_install` to run a smoke test, or jump
straight into :doc:`quickstart` for a 2D Poisson walkthrough.

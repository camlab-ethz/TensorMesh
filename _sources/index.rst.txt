:github_url: https://github.com/camlab-ethz/TensorMesh

:tensor-blue:`Tensor`\ :mesh-teal:`Mesh` Documentation
======================================================

.. code-block:: none

    ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗███████╗██╗  ██╗
    ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║
       ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██╔████╔██║█████╗  ███████╗███████║
       ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║
       ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

A fast, differentiable, JIT-free, and debugging-friendly finite element
library — bringing modern PyTorch ergonomics to scientific computing.

TensorMesh is a finite element method (FEM) library built on PyTorch,
designed to solve partial differential equations (PDEs) with elegance and
efficiency. By integrating natively with the PyTorch ecosystem, it provides
automatic differentiation, GPU acceleration, and an intuitive Pythonic
interface — without sacrificing the rigor of classical FEM.


Core Strengths
--------------

- **GPU-Native & Differentiable**: Built on PyTorch from the ground up.
  Moving an entire FEM workflow to the GPU takes a single line of code —
  every downstream assembly, solve, and gradient inherits the device
  automatically, with no separate backend or data-marshalling step.
  Native autograd flows seamlessly through assembly and solve, enabling
  end-to-end differentiable PDE pipelines.

- **High-Performance Tensorized Assembly**: A fully tensorized Map-Reduce
  algorithm powered by `TensorGalerkin <https://arxiv.org/abs/2602.05052>`_,
  which fuses element-wise operations into monolithic GPU kernels,
  eliminating Python-level loops and delivering order-of-magnitude
  speedups over CPU-based FEM stacks.

- **JIT-Free & Debugging-Friendly**: Eager execution with no compilation
  overhead. Dynamic meshes, adaptive refinement, and interactive
  workflows just work — no recompilation latency, no opaque traces.

- **Comprehensive Element & Mesh Support**: Triangular, tetrahedral,
  pyramid, and prismatic elements with automated mesh generation for
  common geometries and seamless Gmsh / VTK-HDF5 I/O.

- **Flexible Solvers**: Powered by `torch-sla <https://www.torchsla.com/>`_,
  our companion library for differentiable sparse linear algebra. Linear,
  nonlinear, and eigenvalue solvers run across multiple backends on CPU and
  GPU, with full autograd support, batched solves, and distributed multi-GPU
  scaling.

- **Pythonic API**: Custom weak forms in pure Python — no separate DSL,
  no form compiler. If you can write PyTorch, you can write FEM.


Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 12 14 12 12 14

   * - Feature
     - FEniCS
     - scikit-fem
     - JAX-FEM
     - torch-fem
     - **TensorMesh**
   * - Custom Weak Forms (Pythonic)
     - ⚠️
     - ✅
     - ❌
     - ❌
     - ✅
   * - Easy Install
     - ❌
     - ✅
     - ⚠️
     - ✅
     - ✅
   * - Easy Debug
     - ❌
     - ✅
     - ❌
     - ✅
     - ✅
   * - Easy IO
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - Large Mesh
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
   * - GPU Support
     - ✅
     - ❌
     - ✅
     - ✅
     - ✅
   * - Efficiency
     - ✅
     - ❌
     - ✅
     - ⚠️
     - ✅
   * - End-to-End Autograd
     - ⚠️
     - ❌
     - ✅
     - ✅
     - ✅
   * - DL Integration (PyTorch)
     - ❌
     - ❌
     - ⚠️
     - ✅
     - ✅
   * - Maturity
     - ✅
     - ✅
     - ⚠️
     - ⚠️
     - ⚠️

.. note::

   **Custom Weak Forms (Pythonic)** — supports user-defined bilinear / linear
   forms directly in Python, without a separate DSL such as UFL.
   **End-to-End Autograd** — gradients flow natively through the entire
   pipeline; FEniCS supports this via the external ``dolfin-adjoint`` package.
   **Maturity** — reflects project age, ecosystem size, and production
   deployments.


Citation
--------

TensorMesh is the FEM solver component of the **TensorGalerkin** framework.
If you use TensorMesh in your research, please cite the TensorGalerkin paper:

.. code-block:: bibtex

   @article{wen2026tensorgalerkin,
     title   = {Learning, Solving and Optimizing PDEs with {TensorGalerkin}:
                an Efficient High-Performance Galerkin Assembly Algorithm},
     author  = {Wen, Shizheng and Chi, Mingyuan and Yu, Tianwei and
                Moseley, Ben and Michelis, Mike Yan and Ren, Pu and
                Sun, Hao and Mishra, Siddhartha},
     journal = {arXiv preprint arXiv:2602.05052},
     year    = {2026}
   }

If your work also relies on torch-sla (TensorMesh's solver backend),
please additionally cite:

.. code-block:: bibtex

   @article{chi2026torchsla,
     title   = {torch-sla: Differentiable Sparse Linear Algebra with Adjoint
                Solvers and Sparse Tensor Parallelism for PyTorch},
     author  = {Chi, Mingyuan and Wen, Shizheng},
     journal = {arXiv preprint arXiv:2601.13994},
     year    = {2026}
   }


Contact Us
----------

TensorMesh is released under the `Apache License 2.0
<https://github.com/camlab-ethz/TensorMesh/blob/main/LICENSE>`_. For
collaborations and partnerships, please contact Shizheng Wen at
`shizheng.wen@sam.math.ethz.ch <mailto:shizheng.wen@sam.math.ethz.ch>`_.

.. grid:: 3
   :gutter: 3
   :class-container: affiliations
   :margin: 2 0 0 0

   .. grid-item::
      :child-align: center

      .. image:: _static/affiliations/CAMLab_logo.png
         :target: https://camlab.ethz.ch/
         :alt: CAMLab — Computational and Applied Mathematics Laboratory, ETH Zürich
         :class: affiliation-logo

   .. grid-item::
      :child-align: center

      .. image:: _static/affiliations/eth_ai_center_logo.png
         :target: https://ai.ethz.ch/
         :alt: ETH AI Center
         :class: affiliation-logo

   .. grid-item::
      :child-align: center

      .. image:: _static/affiliations/eth-logo-pos.png
         :target: https://ethz.ch/
         :alt: ETH Zürich
         :class: affiliation-logo eth-wordmark


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   getting_started/index
   user_guide/index
   example_gallery/index
   api/index
   performance/index
   community/index

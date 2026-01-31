:github_url: https://github.com/walkerchi/tensormesh 

Tensormesh Document 
===================


.. code-block:: none

    ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗███████╗██╗  ██╗
    ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║
       ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██╔████╔██║█████╗  ███████╗███████║
       ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║
       ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

TensorMesh: The Modern FEM Library 🚀
-------------------------------------

A fast 🚀, differentiable 🎯, cross-platform 💻, JIT-free 📌, and debugging-friendly 🚨 finite element library that prioritizes user experience through clean, Pythonic APIs 🤗

TensorMesh is a modern finite element method (FEM) library built on PyTorch, designed to solve partial differential equations (PDEs) with elegance and efficiency. By seamlessly integrating with PyTorch's ecosystem, it provides automatic differentiation and GPU acceleration while maintaining an intuitive, Pythonic interface.


Core Strengths
--------------

- **Easy to Use**: Clean, intuitive Pythonic APIs that make FEM accessible to both beginners and experts
- **Easy to Debug**: Clear error messages and straightforward execution flow for painless debugging
- **Cross Platform**: Works seamlessly across Windows, Linux, and macOS without complex dependencies
- **Seamless PyTorch Integration**: Leverage PyTorch's powerful automatic differentiation and GPU acceleration
- **Comprehensive Element Support**: Work with a wide range of elements including triangular, tetrahedral, pyramid, and prismatic types
- **High-Performance Assembly**: Optimized element assembly operations for both CPU and GPU architectures
- **Advanced Solvers**: Efficient sparse matrix solvers with flexible backend options (PETSc, PyTorch)
- **Rich Visualization**: Integrated tools for mesh and solution visualization
- **Smart Mesh Generation**: Automated mesh generation for common geometries with intelligent defaults

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15

   * - Feature
     - FEniCS
     - scikit-fem
     - JAX-FEM
     - TensorMesh
   * - Flexibility
     - ❌
     - ✅
     - ❌
     - ✅
   * - Easy Install
     - ❌
     - ✅
     - ✅
     - ✅
   * - Easy Debug
     - ❌
     - ✅
     - ❌
     - ✅
   * - Easy IO
     - ❌
     - ❌
     - ❌
     - ✅
   * - Large Mesh
     - ✅
     - ✅
     - ❌
     - ✅
   * - GPU Support
     - ✅
     - ❌
     - ✅
     - ✅
   * - Efficiency
     - ✅
     - ❌
     - ✅
     - ✅
   * - Auto-diff
     - ✅
     - ❌
     - ✅
     - ✅
   * - DL Integration
     - ❌
     - ❌
     - ✅
     - ✅


.. toctree::
   :maxdepth: 2
   :hidden:

   get_started/index
   examples/index
   api_reference/index
   
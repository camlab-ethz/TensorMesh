![TensorMesh Logo](assets/logo.png)

```bash

    ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗███████╗██╗  ██╗
    ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║
       ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██╔████╔██║█████╗  ███████╗███████║
       ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║
       ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
```                                                                              

# TensorMesh🚀

   A fast🚀, differentiable🎯, cross-platform💻, jit-free📌, debugging-friendly🚨 FEM library
         
   We only provide pythonic api for user-friendly use  🤗


## Why use TensorMesh 🤗

| Feature      | FEniCS  | scikit-fem | JAX-FEM  | TensorMesh |
|--------------|---------|------------|----------|------------|
| Flexibility  | ❌      | ✅         | ❌        | ✅         |
| Easy Install | ❌      | ✅         | ✅        | ✅         |
| Easy Debug   | ❌      | ✅         | ❌        | ✅         |
| Easy IO      | ❌      | ❌         | ❌        | ✅         |
| Large Mesh   | ✅      | ✅         | ❌        | ✅         |
| GPU Support  | ✅      | ❌         | ✅        | ✅         |
| Efficiency   | ✅      | ❌         | ✅        | ✅         |
| Auto-diff    | ✅      | ❌         | ✅        | ✅         |
| DL Integrity | ❌      | ❌         | ✅        | ✅         |




The table above compares key features of TensorMesh with other popular FEM libraries:

- **Flexibility**: TensorMesh and scikit-fem provide flexible APIs for customizing weak form implementations. In contrast, FEniCS has strict prerequisites that limit cross-platform compatibility, while JAX-FEM only supports a fixed set of predefined problems.

- **Easy Debug**: Due to their straightforward execution flow and clear error messages, TensorMesh and scikit-fem enable effective debugging. FEniCS and JAX-FEM are more challenging to debug because they rely on JIT compilation.

- **Easy IO**: Thanks to [meshio](https://github.com/nschloe/meshio). TensorMesh provides simple and intuitive APIs for mesh import/export across multiple formats (GMSH, XDMF, etc). Other libraries often have more complex or limited IO capabilities.

- **GPU Support**: TensorMesh, FEniCS, and JAX-FEM leverage GPU acceleration for high-performance computing, while scikit-fem is limited to CPU execution.

- **Efficiency**: Through optimized implementations, TensorMesh achieves computational efficiency on par with mature libraries like FEniCS and JAX-FEM.

- **Auto-diff**: TensorMesh, FEniCS and JAX-FEM all support automatic differentiation, making them suitable for gradient-based optimization and machine learning applications.

- **DL Integration**: TensorMesh and JAX-FEM are designed with deep learning in mind, offering native PyTorch/JAX compatibility. Other libraries typically require additional wrapper code for deep learning workflows.

## Performance 

### Mac M2 Pro

Zero Boundary Poisson

![img](assets/comparison_3d_cpu_mac.png)




## Installation

To install TensorMesh via pip, follow these steps:

1. Install the package directly from GitHub using pip:
   ```bash
   pip install git+https://github.com/walkerchi/tensormesh.git@main
   ```

2. Or clone and install locally:
   ```bash
   git clone https://github.com/walkerchi/tensormesh.git
   cd tensormesh
   pip install -e .
   ```

Note: Make sure you have Python 3.8+ and pip installed on your system before proceeding with the installation.

### Optional Dependencies

For GPU support with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

For visualization with PyVista:
```bash
pip install pyvista
```

## Features

- **Mesh Generation**: Automatic mesh generation for common geometries (rectangles, circles, L-shapes, cubes, spheres, etc.)
- **Element Types**: Support for various element types including triangles, quadrilaterals, tetrahedra, hexahedra, pyramids, and prisms
- **High-Order Elements**: Support for polynomial orders up to 4 for most element types
- **Assemblers**: Flexible assembly system with `ElementAssembler`, `NodeAssembler`, and `FacetAssembler`
- **Sparse Solvers**: Multiple backend options (SciPy, PETSc, CuPy/cuSOLVER)
- **Visualization**: Integrated visualization with matplotlib and PyVista support
- **Automatic Differentiation**: Full PyTorch integration for gradient-based optimization
- **ODE Solvers**: Built-in time integration schemes for transient problems

## Quick Start

### Poisson Equation

```python
import tensormesh as tm

# Create mesh
mesh = tm.Mesh.gen_rectangle(chara_length=0.05)

# Define assemblers
class LaplaceAssembler(tm.ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv

class SourceAssembler(tm.NodeAssembler):
    def forward(self, v, f):
        return f * v

# Assemble system
K_asm = LaplaceAssembler.from_mesh(mesh)
f_asm = SourceAssembler.from_mesh(mesh)

K = K_asm()
f = f_asm(point_data={'f': source_term})

# Apply boundary conditions and solve
condenser = tm.Condenser(mesh.boundary_mask)
K_, f_ = condenser(K, f)
u = condenser.recover(K_.solve(f_))
```

### Quadrature Order

- line: `[1, ∞)`
- triangle: `[1, 20)`
- quad: `[1, ∞)`
- tetra: `[1, 10)`

### Shape Function Order

- line: `[1, 4]`
- triangle: `[1, 4]`
- quad: `[1, 4]`
- tetra: `[1, 4]`
- hexahedron: `[1, 4]`
- pyramid: `[1, 4]`
- prism: `[1, 4]`

## Examples

### Heat Equation

```bash
cd examples/heat
python heat.py
```

![img](assets/heat.gif)



### Wave Equation

```bash
cd examples/wave
python wave.py
```

![img](assets/wave.gif)

### More Examples

- **Poisson**: `examples/poisson/` - Various Poisson equation examples including optimization
- **Linear Elasticity**: `examples/linear_elasticity/` - Stress analysis and plate bending
- **Solid Mechanics**: `examples/solid mechanics/` - Hyperelasticity and plasticity
- **Inverse Problems**: `examples/inverse/` - Topology optimization and material design
- **Fluid Mechanics**: `examples/fluid/` - Navier-Stokes examples

## Documentation

Full documentation is available at the [TensorMesh Documentation](https://walkerchi.github.io/tensormesh/).

## Benchmark

See the `tensormesh-bench` repository for comprehensive benchmarks comparing TensorMesh with FEniCS, scikit-fem, and JAX-FEM.

## Contribution

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
git clone https://github.com/walkerchi/tensormesh.git
cd tensormesh
pip install -e ".[dev]"
pytest tests/
```

### Code Style

We follow PEP 8 guidelines. Please ensure your code passes linting before submitting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TensorMesh in your research, please cite:

```bibtex
@software{tensormesh,
  author = {Walker Chi},
  title = {TensorMesh: A PyTorch-based Finite Element Library},
  year = {2023},
  url = {https://github.com/walkerchi/tensormesh}
}
```

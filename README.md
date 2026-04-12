<p align="center">
  <img src="assets/logo.png" alt="TensorMesh Logo" width="400"/>
</p>

<h1 align="center">TensorMesh</h1>

<p align="center">
  A differentiable, GPU-accelerated Finite Element Method library built on PyTorch.
</p>

<p align="center">
  <a href="https://camlab-ethz.github.io/TensorMesh/">Documentation</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#examples">Examples</a>
</p>

---

## Highlights

- **Pure Python** — no JIT compilation, no DSL; debug with standard Python tools
- **GPU-accelerated** — assembly, solve, and differentiation all run on CUDA
- **Differentiable** — full PyTorch autograd support for inverse problems and optimization
- **Flexible weak forms** — define your PDE by writing a short `forward()` method
- **Mixed elements** — triangles, quads, tetrahedra, hexahedra, pyramids, prisms (up to order 4)
- **Batch solve** — solve thousands of PDE instances simultaneously via multi-RHS sparse LU

## Installation

**Requirements:** Python >= 3.10, PyTorch >= 2.0

```bash
pip install torch-sla>=0.1.4
pip install git+https://github.com/camlab-ethz/TensorMesh.git
```

Or install from source:

```bash
git clone https://github.com/camlab-ethz/TensorMesh.git
cd TensorMesh
pip install -e .
```

### Optional dependencies

```bash
pip install gmsh          # mesh generation
pip install pyvista       # 3D visualization
pip install petsc4py      # PETSc sparse solver backend
pip install cupy-cuda12x  # GPU sparse solver backend
```

## Quick Start

### Poisson Equation

Solve $-\Delta u = f$ on a unit square with homogeneous Dirichlet boundary conditions:

```python
from tensormesh import ElementAssembler, NodeAssembler, Mesh, Condenser

# Generate mesh
mesh = Mesh.gen_rectangle(chara_length=0.05)

# Define weak form: a(u,v) = integral of grad(u) . grad(v)
class LaplaceAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv

# Define load: l(v) = integral of f * v
class SourceAssembler(NodeAssembler):
    def forward(self, v, f):
        return f * v

# Assemble
K = LaplaceAssembler.from_mesh(mesh)()
f = SourceAssembler.from_mesh(mesh)(point_data={"f": source_term})

# Apply BCs and solve
condenser = Condenser(mesh.boundary_mask)
K_, f_ = condenser(K, f)
u = condenser.recover(K_.solve(f_))
```

### Heat Equation (Implicit Euler)

```python
from tensormesh import ElementAssembler, Mesh, Condenser

mesh = Mesh.gen_rectangle(chara_length=0.02)

class MassAssembler(ElementAssembler):
    def forward(self, u, v):
        return u * v

class StiffnessAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv

M = MassAssembler.from_mesh(mesh)()
A = StiffnessAssembler.from_mesh(mesh)()

dt = 5e-5
K = M + dt * A                        # SparseMatrix arithmetic
condenser = Condenser(mesh.boundary_mask)
K_ = condenser(K)[0]

for step in range(100):
    F_ = condenser.condense_rhs(M @ U)
    U  = condenser.recover(K_.solve(F_))
```

## Architecture

The core workflow: **Mesh → Assembler → SparseMatrix → Condenser → Solve**

| Module | Description |
|--------|-------------|
| `tensormesh.mesh` | Mesh data structure, generation (`gen_rectangle`, `gen_circle`, `gen_cube`, ...), I/O |
| `tensormesh.element` | Shape functions, quadrature rules, element transformations (order 1-4) |
| `tensormesh.assemble` | `ElementAssembler`, `NodeAssembler`, `FacetAssembler` for matrix/vector assembly |
| `tensormesh.sparse` | `SparseMatrix` with multiple solver backends (SciPy, PETSc, CuPy, cuDSS) |
| `tensormesh.operator` | `Condenser` for Dirichlet boundary conditions via static condensation |
| `tensormesh.ode` | Time integrators: explicit/implicit Euler, midpoint, Runge-Kutta |
| `tensormesh.dataset` | Parametric PDE dataset generation (Poisson, Heat, Wave, Elasticity) |
| `tensormesh.visualization` | Matplotlib and PyVista plotting backends |

## Examples

<!-- TODO: add example figures/animations -->

| Category | Examples | Description |
|----------|----------|-------------|
| **Basics** | `examples/basics/` | Mesh visualization, basis functions, element gallery |
| **Poisson** | `examples/poisson/` | 2D/3D Poisson, batch solver, h-adaptivity |
| **Diffusion** | `examples/diffusion/` | Heat equation, Allen-Cahn phase field |
| **Wave** | `examples/wave/` | Wave equation with central difference scheme |
| **Dataset** | `examples/dataset/` | Batch dataset generation for ML (heat, wave) |
| **Fluid** | `examples/fluid/` | Lid-driven cavity, cylinder flow, Rayleigh-Benard, Taylor-Green |
| **Solid** | `examples/solid/` | Cantilever beam, hyperelasticity, contact, plasticity |
| **Distributed** | `examples/distributed/` | Graph coloring, mesh partitioning, multi-GPU assembly |

## Supported Elements

| Element | Geometric Order | Quadrature Order |
|---------|:-:|:-:|
| Line | 1-4 | 1+ |
| Triangle | 1-4 | 1-19 |
| Quadrilateral | 1-4 | 1+ |
| Tetrahedron | 1-4 | 1-9 |
| Hexahedron | 1-4 | 1+ |
| Pyramid | 1-4 | 1+ |
| Prism | 1-4 | 1+ |

## Documentation

Full documentation: [camlab-ethz.github.io/TensorMesh](https://camlab-ethz.github.io/TensorMesh/)

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

## Citation

If you use TensorMesh in your research, please cite:

```bibtex
@article{wen2026tensorgalerkin,
  title={Learning, Solving and Optimizing PDEs with TensorGalerkin: 
         an Efficient High-Performance Galerkin Assembly Algorithm},
  author={Wen, Shizheng and Chi, Mingyuan and Yu, Tianwei and Moseley, Ben and Michelis, Mike Yan and Ren, Pu and Sun, Hao and Mishra, Siddhartha},
  journal={arXiv preprint arXiv:2602.05052},
  year={2026}
}
```

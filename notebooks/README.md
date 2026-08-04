# TensorMesh Colab Notebooks

Browser-runnable companions to [`examples/`](../examples) — open any notebook on Google Colab, run the first cell to install [`tensormesh-fem`](https://pypi.org/project/tensormesh-fem/), and experiment without installing anything locally.

### Start here

| Notebook | Topic | Open |
|---|---|---|
| [`basics.ipynb`](basics.ipynb) | Elements, Lagrange basis functions, mesh generation — the data structures everything else is built on | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/basics.ipynb) |
| [`poisson.ipynb`](poisson.ipynb) | The TensorMesh "hello world": Mesh → Assembler → Condenser → Solve | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/poisson.ipynb) |

### Scalar problems

| Notebook | Topic | Open |
|---|---|---|
| [`poisson_3d.ipynb`](poisson_3d.ipynb) | The same weak form on tetrahedra, rendered as a half-domain cutaway | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/poisson_3d.ipynb) |
| [`poisson_h_adaptivity.ipynb`](poisson_h_adaptivity.ipynb) | h-adaptive refinement on the L-shape: solve → estimate → mark → remesh | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/poisson_h_adaptivity.ipynb) |
| [`helmholtz.ipynb`](helmholtz.ipynb) | Complex-valued FEM end to end, verified against a plane wave | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/helmholtz.ipynb) |

### Time-dependent problems

| Notebook | Topic | Open |
|---|---|---|
| [`heat.ipynb`](heat.ipynb) | Implicit Euler; the stepping operator is factorised once and reused | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/heat.ipynb) |
| [`wave.ipynb`](wave.ipynb) | Central differences, with an energy-conservation read-out | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/wave.ipynb) |
| [`allen_cahn.ipynb`](allen_cahn.ipynb) | Nonlinear phase field: a Newton solve inside every time step | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/allen_cahn.ipynb) |

### Fluids — mixed assembly

| Notebook | Topic | Open |
|---|---|---|
| [`stokes_taylor_hood.ipynb`](stokes_taylor_hood.ipynb) | P2-P1 Taylor-Hood mixed elements + convergence study | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/stokes_taylor_hood.ipynb) |
| [`cavity.ipynb`](cavity.ipynb) | Lid-driven cavity — steady Navier-Stokes by Picard iteration | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/cavity.ipynb) |
| [`flow_obstacles.ipynb`](flow_obstacles.ipynb) | Flow past obstacles — CSG meshing on a Taylor-Hood discretization | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/flow_obstacles.ipynb) |

### Solid mechanics

| Notebook | Topic | Open |
|---|---|---|
| [`cantilever_beam.ipynb`](cantilever_beam.ipynb) | 3D linear elasticity, checked against beam theory | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/cantilever_beam.ipynb) |
| [`hyperelastic_beam.ipynb`](hyperelastic_beam.ipynb) | Large deformation: write the strain energy, let autograd + LBFGS do the rest | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/hyperelastic_beam.ipynb) |

### Differentiable FEM

| Notebook | Topic | Open |
|---|---|---|
| [`coefficient_identification.ipynb`](coefficient_identification.ipynb) | An inverse problem solved by autograd **through** the FEM solve — no adjoint code | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/coefficient_identification.ipynb) |
| [`poisson_galerkin.ipynb`](poisson_galerkin.ipynb) | Physics-informed learning: train a network on the assembled Galerkin residual | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/poisson_galerkin.ipynb) |

Maintenance notes:

- The install cell pins the `tensormesh-fem` release the notebook content was written against; bump the pin (and re-run every notebook) as part of each release.
- `examples/*.py` stays the source of truth — when an example changes, update its notebook counterpart. Each notebook's header links to the specific gallery page and source file it mirrors.
- Notebooks are committed **without executed outputs** to keep the repository small; running the first cell on Colab produces them.
- The 3D notebooks additionally install `pyvista` plus a virtual framebuffer, and the animated ones install `ffmpeg`; the extra packages are listed in each notebook's install cell.
- gmsh writes its meshing log below Python's stdout, so mesh generation is wrapped in a small `quiet()` helper defined in the install cell. Remove the wrapper to see the mesher's output.
- The generator that builds these lives outside the repo (see the maintainer runbook); notebooks are hand-curated rather than mechanically converted, because a plain script-to-notebook conversion collapses into one unreadable cell.

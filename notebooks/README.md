# TensorMesh Colab Notebooks

Browser-runnable companions to [`examples/`](../examples) — open any notebook on Google Colab, run the first cell to install [`tensormesh-fem`](https://pypi.org/project/tensormesh-fem/), and experiment without installing anything locally.

| Notebook | Topic | Open |
|---|---|---|
| [`poisson.ipynb`](poisson.ipynb) | Poisson equation — the TensorMesh "hello world" (Mesh → Assembler → Condenser → Solve) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/poisson.ipynb) |
| [`stokes_taylor_hood.ipynb`](stokes_taylor_hood.ipynb) | Stokes flow with P2-P1 Taylor-Hood mixed elements + convergence study | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/stokes_taylor_hood.ipynb) |
| [`flow_obstacles.ipynb`](flow_obstacles.ipynb) | Steady Navier-Stokes past obstacles — Picard iteration on a Taylor-Hood discretization | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camlab-ethz/TensorMesh/blob/main/notebooks/flow_obstacles.ipynb) |

Maintenance notes:

- The install cell pins the `tensormesh-fem` release the notebook content was written against; bump the pin (and re-run the notebook) as part of each release.
- `examples/*.py` stays the source of truth — when an example changes, update its notebook counterpart.
- Notebooks are committed **with executed outputs** so GitHub and Colab show the expected figures and convergence tables up front; re-running them (previous point) refreshes the outputs at the same time.

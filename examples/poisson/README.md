# Poisson Equation

Elliptic PDE examples covering basic solve, 3D extension, batch solving, and h-adaptivity.

## Scripts

| Script | Description |
|--------|-------------|
| `poisson.py` | 2D Poisson on a rectangle with multi-frequency source term |
| `poisson_3d.py` | 3D Poisson on a unit cube with tetrahedral mesh |
| `poisson_batch_solver.py` | Batch solve with `PoissonMultiFrequency` dataset (2D/3D), includes analytical comparison |
| `poisson_h_adaptivity.py` | H-adaptive refinement on an L-shaped domain with gradient singularity |

## Problem Setup

- **PDE:** $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$
- **Source Term:** Multi-frequency Fourier series via `PoissonMultiFrequency`
- **Domains:** Rectangle, cube, L-shaped

## Usage

```bash
python poisson.py                            # basic 2D example
python poisson_3d.py                         # 3D example
python poisson_batch_solver.py --mode 2d     # batch 2D
python poisson_batch_solver.py --mode 3d     # batch 3D
python poisson_h_adaptivity.py               # adaptive refinement (requires gmsh >= 4.8)
```

## Output

- `poisson.png`: 2D solution contour
- `poisson_3d.vtu`: 3D solution (open with ParaView)
- `poisson_batch_solver_2d.png`, `poisson_batch_solver_3d.vtu`: batch solver results
- `poisson_h_adaptivity.png`: convergence plot (adaptive vs uniform) and final mesh

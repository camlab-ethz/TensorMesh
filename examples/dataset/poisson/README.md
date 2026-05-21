# Poisson Equation Dataset Generation

Batch generation of Poisson equation solutions for machine learning workflows.

## Problem Setup

- **PDE:** $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$
- **Geometry:** Disk of radius $0.5$ centered at $(0.5, 0.5)$ (triangular mesh, $h = 0.008$)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Source Term:** 1000 random samples from `PoissonMultiFrequency` with $K = 16$ Fourier modes

## Features

- **Batch solve:** Builds the load for all 1000 samples at once via the consistent mass-matrix form $b = M f^\top$, then solves the condensed system with a multi-RHS direct solve (one factorization, many back-substitutions)
- **GPU acceleration:** Automatic GPU detection; torch-sla auto-dispatches the sparse solve (cuDSS on GPU, scipy on CPU)
- **Benchmarking:** Compares GPU vs CPU solve time and reports the max GPU–CPU discrepancy
- **Self-check:** Verifies the batched load $M f^\top$ matches the per-sample `NodeAssembler` load before solving

## Usage

```bash
python poisson_dataset.py
```

## Output

- `poisson_dataset.npz`: `solutions` array of shape `(batch_size, n_dofs)`, `sources`, mesh `points`, the Fourier coefficients `a`, and parameters (`chara_length`, `K_modes`, `batch_size`)
- `poisson_dataset.png`: visualization of 5 samples

# Heat Equation Dataset Generation

Batch generation of heat equation snapshots for machine learning workflows.

## Problem Setup

- **PDE:** $u_t = D^2 \Delta u$
- **Geometry:** L-shaped domain (triangular mesh, $h = 0.008$)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Initial Conditions:** 1000 random samples from `HeatMultiFrequency` with $d = 16$ Fourier modes
- **Time Integration:** Implicit Euler, $\Delta t = 5 \times 10^{-5}$, 100 steps

## Features

- **Batch solve:** Solves all 1000 samples simultaneously using multi-RHS sparse LU
- **GPU acceleration:** Automatic GPU detection with CuPy backend
- **Benchmarking:** Compares GPU vs CPU solve time

## Usage

```bash
python heat_dataset.py
```

## Output

- `heat_dataset.npz`: snapshots array of shape `(n_steps, n_dofs, batch_size)`, mesh points, and parameters
- `heat_dataset.mp4`: animation of the first sample (GPU vs CPU comparison)

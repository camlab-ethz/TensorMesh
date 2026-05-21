# Wave Equation Dataset Generation

Batch generation of wave equation snapshots for machine learning workflows.

## Problem Setup

- **PDE:** $u_{tt} = c^2 \Delta u$
- **Geometry:** Circular domain (center $(0.5, 0.5)$, radius $0.5$, $h = 0.015$)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Initial Conditions:** 1000 random samples from `WaveMultiFrequency` with $K = 16$ Fourier modes, zero initial velocity
- **Time Integration:** Central difference scheme, $c = 2.0$, $\Delta t = 0.001$, 100 steps

## Features

- **Batch solve:** Solves all 1000 samples simultaneously using multi-RHS sparse LU
- **GPU acceleration:** Automatic GPU detection with CuPy backend
- **Benchmarking:** Compares GPU vs CPU solve time

## Usage

```bash
python wave_dataset.py
```

## Output

- `wave_dataset.npz`: snapshots array of shape `(n_steps, n_dofs, batch_size)`, mesh points, Fourier coefficients, and parameters
- `wave_dataset.mp4`: animation of the five samples

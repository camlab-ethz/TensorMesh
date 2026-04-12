# Heat Equation

Transient heat diffusion with implicit Euler time stepping.

## Problem Setup

- **PDE:** $u_t = D^2 \Delta u$
- **Geometry:** Unit square $[0,1]^2$ (2nd-order triangular mesh)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Initial Condition:** Multi-frequency Fourier series via `HeatMultiFrequency`
- **Time Integration:** Implicit Euler, $\Delta t = 5 \times 10^{-5}$, 100 steps

## Usage

```bash
python heat.py
```

## Output

- `heat.mp4`: animation comparing FEM prediction with analytical ground truth

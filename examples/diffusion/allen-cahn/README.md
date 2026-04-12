# Allen-Cahn Equation

Nonlinear phase-field evolution solved with Newton's method at each time step.

## Problem Setup

- **PDE:** $u_t = \Delta u + \varepsilon^2 u(1 - u^2)$
- **Geometry:** Unit square $[0,1]^2$ (triangular mesh)
- **Boundary Conditions:** Natural (no-flux) boundary conditions
- **Initial Condition:** Multi-frequency Fourier series via `PoissonMultiFrequency`
- **Nonlinear Solver:** Newton iteration (max 50 iterations per step, tolerance $10^{-10}$)
- **Time Integration:** Implicit Euler, $\Delta t = 10^{-6}$, 200 steps

## Usage

```bash
python ac.py
```

## Output

- `Allen-Cahn.mp4`: animation of the phase-field evolution

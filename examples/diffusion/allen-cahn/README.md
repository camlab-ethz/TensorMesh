# Allen-Cahn Equation

Nonlinear phase-field evolution solved with Newton's method at each time
step. `ac.py` and `ac_torch_sla.py` solve the *same* fully-implicit
backward-Euler step and produce identical phase fields to round-off — they
differ only in how the per-step Newton solve is driven. `ac_convex_concave.py`
solves a *different* time discretisation — Eyre's convex-concave splitting —
which is unconditionally energy-stable and so stays well-behaved at far
larger time steps.

## Scripts

| Script | Description |
|--------|-------------|
| `ac.py` | The Newton loop written out by hand: assemble the tangent `K` and residual `R`, `K.solve(R)`, update, repeat until `‖R‖` is small. |
| `ac_torch_sla.py` | The same step handed to `torch_sla.nonlinear_solve` — one call per time step. Newton / Picard / Anderson with an Armijo line search and an adjoint backward pass, all packaged. |
| `ac_convex_concave.py` | Eyre's convex splitting: the convex `c^3` reaction stays implicit, the concave `-c` term is taken explicitly. The Newton tangent is SPD and the scheme is unconditionally energy-stable, so `--dt` can be raised far beyond the fully-implicit limit. |

`ac_torch_sla.py` passes the FEM consistent tangent explicitly as
`jacobian_fn` (rather than letting `nonlinear_solve` build it via
autograd). Note the sign: `KAssembler` assembles the *negative* tangent
`K = -∂R/∂c`, so `jacobian_fn` returns `-K.values` to hand back the true
Jacobian `J`, which `nonlinear_solve` then steps with `J du = -R`.

## Problem Setup

- **PDE:** $u_t = \Delta u + \varepsilon^2 u(1 - u^2)$
- **Geometry:** Unit square $[0,1]^2$ (triangular mesh)
- **Boundary Conditions:** Natural (no-flux) boundary conditions
- **Initial Condition:** Multi-frequency Fourier series via `PoissonMultiFrequency`
- **Nonlinear Solver:** Newton iteration (max 50 iterations per step, tolerance $10^{-10}$)
- **Time Integration:** Implicit Euler, $\Delta t = 10^{-6}$, 200 steps

## Usage

```bash
python ac.py                # hand-written Newton loop
python ac_torch_sla.py      # same step via torch_sla.nonlinear_solve
python ac_convex_concave.py # Eyre convex splitting (unconditionally stable)
```

## Output

- `Allen-Cahn.mp4`: animation of the phase-field evolution (`ac.py`)
- `Allen-Cahn-torch-sla.mp4`: the identical evolution driven by `nonlinear_solve` (`ac_torch_sla.py`)
- `Allen-Cahn-convex-concave-dt<dt>.mp4`: the convex-splitting evolution (`ac_convex_concave.py`); the file name encodes the `--dt` used

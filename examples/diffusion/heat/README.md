# Heat Equation

Transient heat diffusion with implicit Euler time stepping. Two scripts
solve the *same* problem and produce identical snapshots to machine
precision — they differ only in how the time loop is written.

## Scripts

| Script | Description |
|--------|-------------|
| `heat.py` | The backward-Euler step written out by hand. The clearest pattern when the scheme is fixed. |
| `heat_ode.py` | The integrator-class version: subclass `tensormesh.ode.ImplicitLinearEuler`, wire `Condenser` through its three boundary-condition hooks, and the loop collapses to a single `stepper.step(t, U, dt)` call. Pick this when you want to swap integrators (e.g. backward Euler ↔ midpoint) without rewriting the loop. |

In `heat_ode.py` the stage RHS uses `Condenser.restrict` rather than
`condense_rhs`: a stage *slope* must not carry the Dirichlet correction
$-K_{io}\,u_o$, since a Dirichlet DOF has zero time-derivative by
definition, regardless of its value. The recovered slope is then
prolonged back to full DOF with zeros in the boundary slots.

## Problem Setup

- **PDE:** $u_t = D^2 \Delta u$
- **Geometry:** Unit square $[0,1]^2$ (2nd-order triangular mesh)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Initial Condition:** Multi-frequency Fourier series via `HeatMultiFrequency`
- **Time Integration:** Implicit Euler, $\Delta t = 5 \times 10^{-5}$, 100 steps

## Usage

```bash
python heat.py        # hand-written backward-Euler loop
python heat_ode.py    # same problem via tensormesh.ode.ImplicitLinearEuler
```

## Output

- `heat.mp4`: animation comparing FEM prediction with analytical ground truth (`heat.py`)
- `heat_ode.mp4`: identical animation produced through the ODE integrator (`heat_ode.py`)

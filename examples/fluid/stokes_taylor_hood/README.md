# Stokes Taylor-Hood

Steady incompressible Stokes flow on the unit square using a mixed
Taylor-Hood discretization:

- Velocity: quadratic P2, two components
- Pressure: linear P1 on the corner nodes of the same P2 mesh
- Weak form: `mu grad(u):grad(v) - p div(v) - q div(u)`
- Boundary data: exact divergence-free trigonometric velocity
- Body force: manufactured from `-mu Delta u + grad p = f`
- Pressure gauge: one P1 pressure node is pinned

The exact velocity is generated from the stream function
`psi = sin(pi x)^2 sin(pi y)^2`, and the exact pressure is
`p = sin(pi x) cos(pi y)`. The script refines from `h = 0.1` through three
halvings and reports the velocity `H1` error and pressure `L2` error.

## Usage

```bash
python stokes_taylor_hood.py
```

The script writes `stokes_taylor_hood_convergence.png` with the error curves
and `stokes_taylor_hood.png` with speed, pressure, and velocity-error panels
on the finest mesh.

# J2 Plasticity with Newton's Method (Thick-Walled Cylinder)

A long tube under internal pressure, loaded until the whole wall flows
plastically: the classical elastic-plastic thick cylinder (Hill, 1950), in plane
strain with von Mises (J2) plasticity. Every load step is solved by Newton's
method with the consistent tangent of the radial return, so the iterations
converge quadratically, and the result is checked against the closed-form
elastic response and the limit pressure.

## Problem Setup

- **Geometry (2D, plane strain):** a quarter of the tube, inner radius
  a = 100 mm, outer radius b = 200 mm; u_y = 0 on the x axis and u_x = 0 on the
  y axis (symmetry).
- **Material:** E = 200 GPa, nu = 0.3, sigma_y = 250 MPa, perfect plasticity
  (`--hardening` adds linear isotropic hardening, sigma_y + H p).
- **Loading:**
  - default: the inner wall is pushed out to 6 times the first-yield
    displacement in 30 steps (displacement control), which reaches the
    collapse plateau;
  - `--control pressure`: the internal pressure rises to 0.98 times the limit
    pressure, acting along the facet normal on the inner wall.
- **Mesh:** six-node (quadratic) triangles on a polar grid, 12 radial x 16
  angular cells, radial spacing growing with r, built with NumPy (no gmsh):
  825 nodes. The mid-edge nodes of the arcs lie on the circles.
- **Solver:**
  - radial return at every quadrature point, with the closed-form consistent
    tangent; the plastic state stays at the quadrature points and reaches the
    assemblers as `quadrature_data` (stress for the internal forces, tangent
    moduli for the stiffness matrix);
  - Newton per load step: the first iteration carries the change of the
    prescribed displacement through the body, the next ones correct;
    convergence at a residual of 1e-10 relative to the applied or reaction
    forces.

## Analytical references

| quantity | formula | value |
|---|---|---|
| elastic stiffness p / u_r(a) (Lamé, plane strain) | E (b² − a²) / ((1 + ν) a ((1 − 2ν) a² + b²)) | 1048.95 MPa/mm |
| first-yield pressure p_e (von Mises at r = a) | from the Lamé stresses, with σ_zz = ν (σ_rr + σ_θθ) | 108.07 MPa |
| limit pressure (perfect plasticity) | (2 / √3) σ_y ln(b / a) | 200.10 MPa |

## Usage

```bash
python plasticity_cylinder.py                       # displacement control, perfect plasticity
python plasticity_cylinder.py --control pressure    # pressure control, up to 0.98 p_lim
python plasticity_cylinder.py --hardening 2000      # H = 2 GPa
python plasticity_cylinder.py --n_radial 24 --n_angular 32
```

The default run takes a few seconds on a CPU. `--device cuda` runs on a GPU;
the Newton systems then use cuDSS when torch-sla has it, and SciPy on the host
otherwise.

## Output

- `output/plasticity_cylinder.png`: the equivalent plastic strain at the end of
  loading, the pressure-displacement curve against the three references, and
  the Newton residuals of an elastic step, the first plastic step, a
  half-plastic step and the last one.
- Console, per load step: inner-wall displacement, pressure, plastic fraction
  of the wall, Newton residuals.

## Tests

`tests/assemble/test_plasticity_cylinder_example.py` checks:

- the radial return: plastic points end on the yield surface, elastic points
  keep the elastic stress, the plastic flow is deviatoric, and the consistent
  tangent equals a forward-mode autodiff derivative of the stress update;
- the assembled stiffness against central differences of the residual;
- the mesh: area, inner and outer circles, symmetry edges;
- the elastic response against Lamé, under pressure and displacement control;
- first yield at the inner wall;
- the limit pressure of perfect plasticity (within 1 %), reached from below,
  and Newton converging in at most six updates per step;
- CUDA against CPU, when a GPU is present.

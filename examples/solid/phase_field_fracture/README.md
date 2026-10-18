# Phase-Field Fracture (Single-Edge-Notched Tension)

Brittle fracture with the variational phase-field model. A damage field
`alpha` in [0, 1] regularizes the crack over a length `ell`, and the crack
path follows from minimizing one energy. Damage cannot heal, so every load step
is a bound-constrained minimization. It is solved by alternate minimization,
with `torch_sla.vi_solve` (a bound-constrained Newton solve) for both fields.

## Problem Setup

- **Geometry (2D, plane strain):** unit square [0, 1]^2 mm, with a slit along
  y = 0.5 from the left edge to the centre (the elements above it have their
  own copies of its nodes).
- **Material:** E = 210 kN/mm^2, nu = 0.3, Gc = 2.7e-3 kN/mm, ell = 0.015 mm.
- **Model:**
  - AT1 (default) or AT2 crack model (`--model`);
  - Amor volumetric/deviatoric split, so compression does not damage;
  - residual stiffness g0 = 1e-6.
- **Boundary conditions:** bottom clamped; top u_x = 0, u_y = delta, in increments of 1e-4 mm.
- **Mesh:** union-jack triangles built with NumPy (no gmsh), mirror-symmetric about
  the crack line: h = ell / 2 in a band around y = 0.5, about 6,400 nodes.
- **Solver:**
  - Alternate minimization per load step:
    - displacement solve, with the Dirichlet values as fixed bounds (lower = upper);
    - damage solve, with bounds [alpha_prev, 1].
  - Residuals are autograd gradients of the energy; the Jacobians are assembled bilinear forms.
  - Convergence: the free-dof displacement residual drops below 1e-4 of the reactions.

## Usage

Needs torch-sla with `vi_solve` (0.4.0 or later).

```bash
python phase_field_fracture.py                        # AT1, ell = 0.015 mm
python phase_field_fracture.py --model at2
python phase_field_fracture.py --ell 0.04 --h 0.02    # coarse and quick
python phase_field_fracture.py --relaxation 1.6       # over-relaxed damage update
```

The default run takes about a minute on one CPU node, most of it in the load
steps where damage grows.

`--device cuda` runs on a GPU. The Newton systems then use cuDSS when torch-sla
has it, and SciPy on the host otherwise.

## Output

- `output/phase_field_fracture.png`: damage around the slit tip at peak load, the
  broken specimen, and the load-displacement curve.
- `output/phase_field_fracture_crack.png`: the broken specimen, displacements magnified.
- Console, per load step: reaction force, fracture energy, maximum damage,
  number of staggered iterations.

## Tests

`tests/assemble/test_phase_field_fracture_example.py` checks:

- the assembled Hessians against differences of the autograd residuals;
- the AT1 homogeneous solution (elastic limit and damage level) against its
  closed form;
- that damage never decreases, including after unloading;
- that damage prescribed with `alpha_fixed` is held exactly;
- on a coarse SENT run, that the crack runs straight through the ligament and
  dissipates about Gc times its length.

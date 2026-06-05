# Drucker-Prager triaxial compression

Example-only Drucker-Prager plasticity driver for a pressure-dependent soil or weak-rock material.

This example is intentionally local to `examples/solid/geomechanics/drucker_prager_triaxial/`. It does not add a public `tensormesh.assemble` API. The goal is to settle TensorMesh conventions for geomechanics before promoting any material model into the library core.

## Problem setup

- Geometry: unit cube generated with `gen_cube`.
- Kinematics: small strain, displacement-controlled affine triaxial driver.
- Material: associated Drucker-Prager plasticity with linear isotropic hardening.
- TensorMesh internal sign convention: stress is tension-positive.
- Reported geomechanics quantities: axial stress and mean pressure are printed compression-positive.
- State variables:
  - plastic strain tensor `eps_p`,
  - scalar plastic multiplier/history variable `alpha`.

The local custom assembler follows the same lifecycle as TensorMesh J2 plasticity:

1. store per-quadrature state in `self.history[etype]`,
2. pass previous-step state to `energy(...)` through `element_data`,
3. call `update_state(u)` after each converged load step under `torch.no_grad()`.

## Yield convention

The internal Drucker-Prager yield function is written with TensorMesh's tension-positive stress convention:

```text
f = q + eta * I1 - (k + H * alpha) <= 0
```

where:

```text
I1 = tr(sigma)
q  = sqrt(3/2 s:s)
p  = -I1 / 3   # compression-positive mean pressure, reporting only
```

Because compression gives negative `I1`, higher confinement delays yield.

## Sanity checks

The script runs two confinement levels:

- `p0 = 0 kPa`,
- `p0 = 100 kPa`.

It checks that:

- the higher-confinement case reaches elastic trial yield later,
- the committed plastic history variable is monotonic.

## Usage

From this directory:

```bash
python drucker_prager_triaxial.py
```

For a fast numerical-only run:

```bash
python drucker_prager_triaxial.py --no-plot --steps 16
```

## Output

- Console sanity-check summary.
- `drucker_prager_triaxial.png` showing:
  - axial stress versus axial compression strain,
  - maximum committed plastic multiplier versus axial compression strain.

## Notes

This first version is deliberately modest. It is a constitutive/geomechanics convention example, not a full laboratory triaxial test with exact constant radial stress control. A future follow-up can add a boundary-value problem or a promoted `tensormesh/assemble/geomechanics.py` API after maintainers agree on the model surface.

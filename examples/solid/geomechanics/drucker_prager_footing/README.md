# Drucker-Prager strip footing

This example solves a nonlinear geomechanics boundary-value problem: a
rectangular soil block loaded by a centered strip footing, with a
pressure-dependent Drucker-Prager soil model and load stepping.

It is deliberately example-only. It does not add a public geomechanics API. The
constitutive code is copied locally (Torch only), so the file is self-contained.

## How it differs from `elastic_footing`

`elastic_footing` solves the same footing geometry with **linear elasticity** and
a single direct solve. This example reuses that geometry and the roller-style
boundary conditions, but replaces the constitutive model with an example-local
**Drucker-Prager plasticity** assembler and ramps the footing pressure in load
steps. The result is a nonlinear, path-dependent simulation: a settlement bowl
forms, a plastic zone develops beneath the footing, and the load-settlement
curve bends away from the initial elastic stiffness as plasticity accumulates.

## How it reuses the Drucker-Prager triaxial pattern

The local assembler follows the same per-quadrature history lifecycle as the
`drucker_prager_triaxial` example:

1. per-quadrature state is stored in `self.history[etype]` (`eps_p`, `alpha`);
2. previous-step state is passed to `energy(...)` through `element_data`
   (`eps_p_n`, `alpha_n`);
3. after each converged load step, `update_state(u)` commits the new state under
   `torch.no_grad()`.

At each load step the total potential energy `internal - external` is minimized
over the free displacement DOFs with L-BFGS.

TensorMesh keeps stress tension-positive internally. For geomechanics reporting,
settlement is shown positive downward: `settlement = -u_y`.

## Run

```bash
python drucker_prager_footing.py
```

For a fast numerical-only run:

```bash
python drucker_prager_footing.py --no-plot --steps 6 --chara-length 0.60
```

## Outputs

- A console summary: nodes, load steps, final footing pressure, final footing
  and maximum settlement, final maximum/mean plastic history, and the plastic
  centroid.
- `drucker_prager_footing.png`, a three-panel figure:
  - settlement field on the deformed mesh,
  - committed Drucker-Prager plastic history beneath the footing,
  - load-settlement curve with the initial elastic tangent for reference.

## Sanity checks

The associated test (`tests/assemble/test_drucker_prager_footing_example.py`)
checks, on CPU + float64, that:

- the footing settles downward and plasticity develops;
- the committed plastic history and settlement grow monotonically with load;
- the plastic zone localizes under the centered footing;
- a low-load / high-cohesion case stays essentially elastic and matches the
  `elastic_footing` settlement closely;
- a higher-cohesion soil develops less plasticity than a lower-cohesion soil.

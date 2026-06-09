# Elastic strip footing

This example solves a small linear-elastic geomechanics boundary-value problem:
a rectangular soil block is loaded by a centered strip footing on the top
surface.

The example is deliberately small and example-only.  It does not add a public
geomechanics API.  It reuses TensorMesh's existing solid-mechanics direct-solve
workflow:

```text
mesh -> LinearElasticityElementAssembler -> Condenser -> SparseMatrix.solve
```

## Model

- 2D plane-strain-style soil block with unit out-of-plane thickness.
- Linear isotropic elasticity.
- Bottom boundary: vertical roller support, `u_y = 0`.
- Side boundaries: horizontal rollers, `u_x = 0`.
- Top surface: free except for a lumped downward load over the footing patch.

The internal TensorMesh displacement convention is unchanged.  For
geomechanics reporting, settlement is shown as positive downward:

```text
settlement = -u_y
```

## Run

```bash
python elastic_footing.py
```

For a fast numerical-only run:

```bash
python elastic_footing.py --no-plot --chara-length 0.5
```

The default run writes:

```text
elastic_footing.png
```

## Sanity checks

The example reports:

- total applied vertical load,
- vertical reaction at fixed vertical DOFs,
- reaction/load relative error,
- footing settlement,
- maximum settlement.

The associated test checks load/reaction balance and confirms that doubling the
footing pressure approximately doubles the settlement.

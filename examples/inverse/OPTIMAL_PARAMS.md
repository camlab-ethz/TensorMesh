# Multi-Material Topology Optimization - Optimal Parameters

This example implements the JAX-FEM multi-material topology optimization problem.

## Reference

JAX-FEM Source: `applications/outdated/top_opt/multi_material.py`
JAX-FEM Tutorial: https://deepmodeling.github.io/jax-fem/learn/structure_material_co_design/example.html

## Material Model (Matching JAX-FEM Exactly)

```
E = E_min + θ₁^p × (θ₂^p × E1 + (1 - θ₂^p) × E2)
```

Where:
- θ₁ = structural density (1 = solid, 0 = void)
- θ₂ = material selection (1 = strong E1, 0 = weak E2)
- E1 = E_max = 70 kPa (strong material)
- E2 = 0.2 × E_max = 14 kPa (weak material)
- E_min = 70 Pa (void stability)
- p = 3 (SIMP penalization)

## Volume Constraint (JAX-FEM Style)

```
g = sum(θ₁ × (θ₂ × 1 + (1 - θ₂) × 0.4)) / n / vf - 1 ≤ 0
```

This weighted constraint:
- Strong material (θ₂=1) contributes full volume
- Weak material (θ₂=0) contributes only 40% volume

## Best Configuration

```bash
python structure_material_codesign.py \
    --epoch 400 \
    --chara_length 0.08 \
    --lr 0.03 \
    --lambda_vol 5000 \
    --vf 0.4 \
    --filter_radius 0.12 \
    --beta_init 1 \
    --beta_max 32 \
    --beta_interval 40
```

**Results:**
- Final compliance: 1.1287e-03
- θ₁ mean (structure): 0.401
- θ₂ mean (material): 0.998
- Weighted volume: 0.400 (target: 0.4)
- Constraint g: 0.0000 (satisfied)
- Max displacement: 0.0011

## Parameter Description

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--length` | 5.0 | Domain length (scaled from JAX-FEM's 50) |
| `--height` | 3.0 | Domain height (scaled from JAX-FEM's 30) |
| `--epoch` | 400 | Optimization iterations |
| `--chara_length` | 0.08 | Mesh element size |
| `--lr` | 0.03 | Adam learning rate |
| `--lambda_vol` | 5000 | Volume constraint penalty |
| `--vf` | 0.4 | Target volume fraction |
| `--filter_radius` | 0.12 | Density filter radius |
| `--beta_init` | 1 | Initial Heaviside sharpness |
| `--beta_max` | 32 | Maximum Heaviside sharpness |
| `--beta_interval` | 40 | Epochs between beta increases |

## Comparison with JAX-FEM

| Feature | JAX-FEM | TensorMesh |
|---------|---------|------------|
| Material model | θ₁^p(θ₂^pE1+(1-θ₂^p)E2) | ✅ Same |
| Volume constraint | Weighted (JAX-FEM style) | ✅ Same |
| Optimizer | MMA | Adam (similar results) |
| Domain | 50×30 | 5×3 (scaled) |
| Load | Right-center point | ✅ Same |
| BC | Left clamped | ✅ Same |

## Notes

1. **θ₂ converges to 1.0**: This is physically correct for linear elasticity - stronger material always reduces compliance.

2. **For true co-design (mixed materials)**: JAX-FEM uses J2 plasticity where yield stress distribution matters. Our linear elastic model doesn't capture this.

3. **Volume constraint satisfied**: The weighted volume constraint matches JAX-FEM's formulation exactly.

## Output Files

- `codesign_result.png/pdf` - θ₁, θ₂, combined design, and displacement
- `codesign_convergence.png/pdf` - Compliance and volume history

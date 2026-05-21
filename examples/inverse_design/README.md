# Inverse Design & Identification

Differentiable-FEM inverse problems and design. Every gradient here comes
from PyTorch `autograd` through the FEM solve (the adjoint backward of
`SparseMatrix.solve`) — no hand-coded sensitivity equations.

For the theory (adjoint cost, what is differentiable, wiring a network in)
see the [Differentiability user guide](../../docs/source/user_guide/differentiability.rst);
this folder is the runnable companion to the gallery's *Inverse Design &
Identification* chapter.

## Scripts

| Script | Description |
|--------|-------------|
| `coefficient_identification.py` | Recover a spatially varying diffusion coefficient $\kappa(x)$ from one observed solution (gradients through the matrix entries) |
| `thermal_topology.py` | Thermal-compliance topology optimization with the built-in `OCOptimizer` (Optimality Criteria) |
| `compliance_topology.py` | Structural cantilever compliance minimization (SIMP + MMA), set up to match a JAX-FEM reference |
| `mma_optimizer.py` | Helper: a small Method-of-Moving-Asymptotes optimizer used by `compliance_topology.py` |

## Problem Setup

- **Coefficient identification:** $-\nabla\cdot(\kappa(x)\,\nabla u) = f$ on $[0,1]^2$, $u = 0$ on $\partial\Omega$; recover $\kappa$ by Adam on $\kappa = 1 + \tanh\theta$, minimizing $\lVert u_\theta - u_\text{obs}\rVert^2$.
- **Thermal topology:** minimize $J = b^T u$ s.t. $K(\rho)u = b$ and a volume cap, SIMP conductivity $\kappa(\rho) = \kappa_\text{min} + (1-\kappa_\text{min})\rho^p$, per-element density $\rho$.
- **Structural topology:** minimize compliance $\mathbf{u}^T\mathbf{K}(\rho)\mathbf{u}$ on a $60\times30$ plane-stress cantilever, left edge fixed, bottom-right loaded; SIMP penalty $p=3$, volume fraction $0.5$.

## Usage

```bash
python coefficient_identification.py    # ~1 min CPU; --device cuda, --n-iter, --chara-length
python thermal_topology.py              # a few seconds; --device cuda, --n-iter, --vf
python compliance_topology.py           # cantilever; --device, --vf, --max-iters, --nx, --ny
```

`compliance_topology.py` additionally requires `meshio` and `tqdm`
(VTK export + progress bar); the other two need only the core install
plus `matplotlib`.

## Output

- `coefficient_id_loss.png`, `coefficient_id_fields.png`: optimisation history and true/recovered/error $\kappa$ fields
- `thermal_topology.png`: density-evolution snapshots and convergence
- `output/`: `compliance_topology.py` writes its convergence plot, animation, and VTK frames here

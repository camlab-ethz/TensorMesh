# Physics-Informed Learning

Train a neural network by minimizing the **assembled Galerkin residual**
instead of solving the FEM system. TensorMesh assembles the weak-form
operators once; because `SparseMatrix` is autograd-traced, the residual
`||K u - F||²` back-propagates straight into the network weights — no
linear solve, no hand-coded adjoint.

Runnable companion to the gallery's *Physics-Informed Learning* chapter;
see the [Differentiability user guide](../../docs/source/user_guide/differentiability.rst)
for how the assembled system is differentiable.

## Scripts

| Script | Description |
|--------|-------------|
| `poisson_galerkin.py` | Represent the Poisson solution by a tanh MLP $u_\theta(x,y)$ and minimize the discrete Galerkin residual |

## Problem Setup

- **PDE:** $-\Delta u = f$ in $[0,1]^2$, $u = 0$ on $\partial\Omega$, manufactured solution $u = \sin\pi x\,\sin\pi y$
- **Network:** fully-connected MLP, input $(x,y)$, output $u$, tanh activations
- **Objective:** $\min_\theta \lVert K_- u_\theta - F_-\rVert^2$ on the interior system ($K$ = Laplace stiffness, $F = M f$, boundary conditions via `Condenser`)
- **Training:** Adam warm-up, then an LBFGS refine (the squared residual is ill-conditioned)

## Usage

```bash
python poisson_galerkin.py        # ~10 s CPU
# flags: --device cuda, --chara-length, --adam-iters, --lbfgs-iters, --width, --depth
```

## Output

- `poisson_galerkin_loss.png`: relative Galerkin residual and $L^2$ error vs iteration
- `poisson_galerkin_fields.png`: exact / learned / error solution fields

# Inverse Problems and Topology Optimization Examples

This directory contains examples of inverse problems and design optimization using TensorMesh's differentiable FEM capabilities.

## Examples

### 1. SIMP Topology Optimization (`simp_topology_optimization.py`)

Classic minimum compliance topology optimization using the SIMP (Solid Isotropic Material with Penalization) method.

**Problem Description:**
- Objective: Minimize compliance (maximize stiffness)
- Design variable: Element density $\rho \in [0, 1]$
- Constraint: Volume fraction

**Usage:**
```bash
python simp_topology_optimization.py --epoch 100 --vf 0.5 --n_elem_x 80 --n_elem_y 40
```

### 2. Structure/Material Co-design (`structure_material_codesign.py`)

SIMP topology optimization inspired by JAX-FEM's co-design example. Originally designed for multi-material optimization, simplified to single-material SIMP for consistency with JAX-FEM results.

**Problem Description:**
- Objective: Minimize compliance
- Design variable: $\rho$ (density, 0=void, 1=solid)
- Constraint: Volume fraction
- Method: Optimality Criteria (OC) with density filter and Heaviside projection

**Reference:** 
[JAX-FEM Tutorial: Structure/Material Co-design](https://deepmodeling.github.io/jax-fem/learn/structure_material_co_design/example.html)

**Usage:**
```bash
python structure_material_codesign.py --epoch 100 --vf 0.5 --n_elem_x 80 --n_elem_y 40
```

---

## Mathematical Background

### SIMP Topology Optimization Problem

The SIMP (Solid Isotropic Material with Penalization) method is a density-based topology optimization approach. The goal is to find the optimal material distribution that minimizes structural compliance (i.e., maximizes stiffness) under a volume constraint.

#### Problem Formulation

$$
\begin{aligned}
\min_{\boldsymbol{\rho}} \quad & C(\boldsymbol{\rho}) = \mathbf{u}^T \mathbf{K}(\boldsymbol{\rho}) \mathbf{u} = \mathbf{F}^T \mathbf{u} \\[8pt]
\text{subject to:} \quad & \mathbf{K}(\boldsymbol{\rho}) \mathbf{u} = \mathbf{F} \\[4pt]
& \frac{V(\boldsymbol{\rho})}{V_0} = \frac{\sum_e \rho_e v_e}{\sum_e v_e} \leq \bar{v} \\[4pt]
& 0 < \rho_{\min} \leq \rho_e \leq 1, \quad \forall e
\end{aligned}
$$

where:
- $C$: Compliance (strain energy, work done by external forces)
- $\boldsymbol{\rho} = \{\rho_1, \rho_2, \ldots, \rho_{n_e}\}$: Element densities (design variables)
- $\mathbf{u}$: Global displacement vector
- $\mathbf{K}$: Global stiffness matrix
- $\mathbf{F}$: External force vector
- $\bar{v}$: Target volume fraction (e.g., 0.5)
- $\rho_{\min}$: Minimum density to avoid singularity (e.g., $10^{-3}$)

#### SIMP Interpolation

The element stiffness matrix is interpolated using the SIMP power law:

$$
\mathbf{K}_e(\rho_e) = \left[ E_{\min} + \rho_e^p (E_0 - E_{\min}) \right] \mathbf{K}_e^0
$$

or equivalently:

$$
E_e(\rho_e) = E_{\min} + \rho_e^p (E_0 - E_{\min})
$$

where:
- $E_0$: Young's modulus of solid material
- $E_{\min}$: Small stiffness for void (numerical stability, e.g., $10^{-9} E_0$)
- $p$: Penalization power (typically $p = 3$)
- $\mathbf{K}_e^0$: Element stiffness matrix with unit Young's modulus

The penalization power $p > 1$ discourages intermediate densities, driving the solution towards a clear 0-1 (void/solid) design.

---

### Sensitivity Analysis

The sensitivity (gradient) of compliance with respect to element density:

$$
\frac{\partial C}{\partial \rho_e} = -\mathbf{u}_e^T \frac{\partial \mathbf{K}_e}{\partial \rho_e} \mathbf{u}_e
$$

Using SIMP interpolation:

$$
\frac{\partial \mathbf{K}_e}{\partial \rho_e} = p \rho_e^{p-1} (E_0 - E_{\min}) \mathbf{K}_e^0
$$

Therefore:

$$
\boxed{\frac{\partial C}{\partial \rho_e} = -p \rho_e^{p-1} (E_0 - E_{\min}) \mathbf{u}_e^T \mathbf{K}_e^0 \mathbf{u}_e}
$$

**Key observation:** $\frac{\partial C}{\partial \rho_e} < 0$ because:
- $p, \rho_e, E_0 - E_{\min} > 0$
- $\mathbf{u}_e^T \mathbf{K}_e^0 \mathbf{u}_e > 0$ (positive definite)

This means increasing density always reduces compliance (increases stiffness).

---

### Optimality Criteria (OC) Method

The OC method is a heuristic update scheme derived from KKT optimality conditions. It is efficient, stable, and widely used in topology optimization.

#### Lagrangian

$$
\mathcal{L} = C + \lambda \left( \sum_e \rho_e v_e - V^* \right)
$$

#### KKT Conditions

At optimum, the KKT stationarity condition gives:

$$
\frac{\partial C}{\partial \rho_e} + \lambda \frac{\partial V}{\partial \rho_e} = 0
$$

where $\frac{\partial V}{\partial \rho_e} = v_e$ (element volume).

#### OC Update Rule

Based on the optimality condition, the update rule is:

$$
\boxed{\rho_e^{\text{new}} = \rho_e \cdot B_e^\eta}
$$

where:

$$
B_e = \frac{-\partial C / \partial \rho_e}{\lambda \cdot \partial V / \partial \rho_e} = \frac{-\partial C / \partial \rho_e}{\lambda \cdot v_e}
$$

and $\eta = 0.5$ is the damping exponent.

#### Move Limits

To ensure stability, move limits are applied:

$$
\rho_e^{\text{new}} = \max\left(\rho_{\min}, \max\left(\rho_e - \Delta, \min\left(1, \min\left(\rho_e + \Delta, \rho_e B_e^\eta\right)\right)\right)\right)
$$

where $\Delta$ is the move limit (typically 0.2).

#### Bisection for Lagrange Multiplier

The Lagrange multiplier $\lambda$ is found by bisection to satisfy the volume constraint:

```python
lambda_low, lambda_high = 1e-9, 1e9
while (lambda_high - lambda_low) / (lambda_low + lambda_high) > 1e-4:
    lambda_mid = 0.5 * (lambda_low + lambda_high)
    
    # OC update with current lambda
    Be = (-dc / (lambda_mid * dv)) ** 0.5
    rho_new = (rho * Be).clamp(rho - move, rho + move).clamp(rho_min, 1.0)
    
    # Adjust lambda based on volume
    if rho_new.mean() > vf:
        lambda_low = lambda_mid
    else:
        lambda_high = lambda_mid
```

---

### Density Filter

To avoid checkerboard patterns and ensure mesh-independence, a density filter is applied:

$$
\tilde{\rho}_i = \frac{\sum_{j \in N_i} H_{ij} \rho_j}{\sum_{j \in N_i} H_{ij}}
$$

where:

$$
H_{ij} = \max(0, r_{\min} - \|x_i - x_j\|)
$$

- $r_{\min}$: Filter radius (typically 1.5 × element size)
- $N_i$: Set of elements within filter radius of element $i$

#### Sensitivity Filter (Alternative)

Instead of filtering densities, we can filter sensitivities:

$$
\widetilde{\frac{\partial C}{\partial \rho_e}} = \frac{1}{\rho_e \sum_i H_{ei}} \sum_{i \in N_e} H_{ei} \rho_i \frac{\partial C}{\partial \rho_i}
$$

---

### Heaviside Projection

To obtain sharper 0-1 designs, a Heaviside projection is applied after filtering:

$$
\bar{\rho} = \frac{\tanh(\beta \eta) + \tanh(\beta(\tilde{\rho} - \eta))}{\tanh(\beta \eta) + \tanh(\beta(1 - \eta))}
$$

where:
- $\beta$: Sharpness parameter (increased during optimization, e.g., 1 → 2 → 4 → 8 → ...)
- $\eta$: Threshold (typically 0.5)

**Beta Continuation:** Start with small $\beta$ (smooth) and gradually increase to obtain sharp boundaries.

---

### Algorithm Summary

```
Initialize ρ = vf (uniform density)
Initialize β = 1 (Heaviside sharpness)

for iteration = 1 to max_iter:
    # 1. Apply density filter
    ρ_filtered = filter(ρ)
    
    # 2. Apply Heaviside projection
    ρ_phys = heaviside(ρ_filtered, β)
    
    # 3. Compute element stiffness with SIMP
    E_e = E_min + ρ_phys^p * (E_0 - E_min)
    
    # 4. Assemble and solve FEM
    K(ρ_phys) u = F
    
    # 5. Compute compliance and sensitivities
    C = F^T u
    dc/dρ = -p * ρ_phys^(p-1) * u_e^T K_e^0 u_e
    
    # 6. Filter sensitivities (chain rule)
    dc/dρ = filter_sensitivity(dc/dρ)
    
    # 7. OC update with bisection
    ρ_new = OC_update(ρ, dc/dρ, λ)
    
    # 8. Beta continuation (every N iterations)
    if iteration % N == 0:
        β = min(β * 2, β_max)
    
    # Check convergence
    if |C_new - C_old| / C_old < tol:
        break
```

---

## Output Files

| File | Description |
|------|-------------|
| `*_result.png/pdf` | Final design and displacement field |
| `*_convergence.png/pdf` | Optimization history (compliance, volume) |
| `*_animation.mp4` | Animation of optimization process with boundary conditions |

---

## TensorMesh Features Used

1. **Differentiable FEM**: Automatic differentiation through the linear solve
2. **ElementAssembler**: Custom element stiffness with design-dependent properties
3. **Condenser**: Dirichlet boundary condition handling
4. **SparseMatrix.solve()**: Linear system solution with gradient support

---

## Comparison with JAX-FEM

| Feature | JAX-FEM | TensorMesh |
|---------|---------|------------|
| Backend | JAX | PyTorch |
| Autodiff | JAX transforms | PyTorch autograd |
| Solver | UMFPACK | scipy/torch/petsc |
| Optimizer | MMA (built-in) | OC method / torch.optim |
| GPU Support | Yes | Yes |

Both frameworks support differentiable FEM for inverse problems and optimization.

---

## GE Bracket Benchmark (`bracket_benchmark.py`)

Reproduces the [torch-fem bracket topology optimization example](https://github.com/meyer-nils/torch-fem/blob/main/examples/optimization/solid/bracket.ipynb) and compares three gradient computation methods.

### Problem Description

- **Geometry**: GE Jet Engine Bracket (3D, 42,548 nodes, 214,524 tetrahedra)
- **Objective**: Minimize compliance (4 load cases)
- **Constraint**: Volume fraction = 15%
- **Method**: SIMP with Optimality Criteria (OC) optimizer

### Methods Compared

1. **torch-fem**: Reference implementation using torch-fem's analytic sensitivity
2. **TensorMesh-autograd**: PyTorch autodiff through sparse solve
3. **TensorMesh-analytic**: Hand-derived analytic sensitivity (same formula as torch-fem)

### Usage

```bash
# Quick test (5 epochs) - uses AMG solver by default
python bracket_benchmark.py --epochs 5

# Full benchmark (50 epochs)
python bracket_benchmark.py --epochs 50

# Plot from cached results
python bracket_benchmark.py --plot-only
```

### Benchmark Results (3 epochs, AMG solver)

| Method | Final Compliance | Avg Time/Iter | Speedup |
|--------|------------------|---------------|---------|
| torch-fem | 4.810e+05 | 205.0 s | 1.0x (baseline) |
| TensorMesh-autograd | 4.810e+05 | 290.7 s | 0.71x |
| **TensorMesh-analytic** | **4.810e+05** | **195.6 s** | **1.05x faster** |

**Key Findings:**

1. **Accuracy**: All methods converge to identical compliance values
2. **Speed**: TensorMesh-analytic is slightly faster than torch-fem when using AMG solver with rigid body modes
3. **AMG preconditioner**: Critical for 3D elasticity - provides 4x speedup over Jacobi
4. **Analytic vs Autograd**: Analytic gradient is ~1.5x faster than autograd (no autodiff overhead)

### Convergence History

All methods show identical convergence:
- Epoch 0: 988,647 → Epoch 4: 266,595 (73% reduction)

### Solver Configuration

For 3D elasticity problems, **AMG with rigid body modes** is critical:

```python
# Build rigid body modes (3 translation modes for 3D)
B_rigid = torch.zeros(n_dofs, 3)
for i in range(3):
    B_rigid[i::3, i] = 1.0

# Solve with AMG
u = K.solve(F, backend='amg', B=B_rigid, tol=1e-5)
```

| Solver | Time | Notes |
|--------|------|-------|
| AMG + B | **12.9 s** | Best for 3D elasticity |
| AMG (no B) | 50.5 s | 4x slower without rigid body modes |
| CG + Jacobi | 57.3 s | May not converge |
| scipy direct | 171.3 s | Slow for large problems |

### Output Files

| File | Description |
|------|-------------|
| `output/benchmark_time.png` | Time comparison bar chart |
| `output/benchmark_effect.png` | Convergence + density visualization |
| `output/benchmark_cache.csv` | Summary results |
| `output/benchmark_history.csv` | Per-iteration history |

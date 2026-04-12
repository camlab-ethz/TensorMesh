# Dataset Generation for "Imposing BCs on Neural Operators" (arXiv 2602.04923)

This directory contains a complete TensorMesh-based reimplementation of the 18 FEM dataset generation pipeline from:

> **"Imposing Boundary Conditions on Neural Operators via Learned Function Extensions"**
> Sepehr Mousavi, Siddhartha Mishra, Laura De Lorenzis (2026)
> arXiv: [2602.04923](https://arxiv.org/abs/2602.04923)

The original code used DOLFINx/FEniCSx. This reimplementation uses **TensorMesh** (PyTorch-based FEM library), enabling GPU acceleration and automatic differentiation.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The 18 Datasets](#the-18-datasets)
3. [Code Architecture](#code-architecture)
4. [File Reference](#file-reference)
5. [PDE Formulations](#pde-formulations)
6. [Geometry Definitions](#geometry-definitions)
7. [Boundary Condition System](#boundary-condition-system)
8. [Output Format (HDF5)](#output-format-hdf5)

---

## Quick Start

### Generate a single dataset

```bash
cd examples/solid/dataset_paper

# Poisson on circle with Dirichlet BC, 10 samples
python generate_dataset.py --problem poisson --shape circle --id bc1 --size 10 --save-hdf5

# Elasticity on hollow circle with Steel, 5 samples
python generate_dataset.py --problem elasticity --shape circlehollow --id m1 --size 5 --save-hdf5
```

### Batch generate all 18 datasets

```bash
bash batch_generate.sh 256          # 256 samples each
bash batch_generate.sh 8448 ./data  # Full dataset (8448 samples)
```

### Visualize one sample per dataset

```bash
python visualize_all.py --output-dir ./figures
```

---

## The 18 Datasets

### Overview

The datasets are organized along three axes:

| Axis | Values | Description |
|------|--------|-------------|
| **PDE** | Poisson (9) / Elasticity (9) | Scalar vs. vector problem |
| **Geometry** | 6 shapes | Simple or with holes |
| **Configuration** | bc1/bc4/bc5 or m1/m2/m3 | BC type or material |

### Poisson Datasets (9)

Scalar PDE: $-\Delta u = f$ in $\Omega$ with mixed BCs on $\partial\Omega$.

| # | Dataset Name | Geometry | BC Config | Source $f$ |
|---|---|---|---|---|
| 1 | `poisson-circle-bc1` | Unit circle | Pure Dirichlet | Fixed radial cosine |
| 2 | `poisson-square-bc1` | $[-1,1]^2$ square | Pure Dirichlet | Fixed radial cosine |
| 3 | `poisson-boomerang-bc1` | Boomerang | Pure Dirichlet | Fixed radial cosine |
| 4 | `poisson-circle-bc4` | Unit circle | Mixed (D+N+R) | Fixed radial cosine |
| 5 | `poisson-square-bc4` | $[-1,1]^2$ square | Mixed (D+N+R) | Fixed radial cosine |
| 6 | `poisson-boomerang-bc4` | Boomerang | Mixed (D+N+R) | Fixed radial cosine |
| 7 | `poisson-circle-bc5` | Unit circle | Mixed (D+N+R) | Random radial sines |
| 8 | `poisson-square-bc5` | $[-1,1]^2$ square | Mixed (D+N+R) | Random radial sines |
| 9 | `poisson-boomerang-bc5` | Boomerang | Mixed (D+N+R) | Random radial sines |

**BC configurations:**
- **bc1**: 1 segment covering the entire boundary, all Dirichlet, random sinusoidal (12 modes for circle/square, 6 for boomerang)
- **bc4**: 4 segments with random joints; segment 0 = Dirichlet, segments 1-3 = randomly chosen Dirichlet/Neumann/Robin (33% each). Non-Dirichlet boundary fraction constrained to [0.2, 0.51].
- **bc5**: Same boundary setup as bc4, but the source function $f$ is also randomized (radial sine modes) instead of being a fixed cosine function.

### Elasticity Datasets (9)

Vector PDE: Neo-Hookean hyperelasticity with incremental loading. Displacement field $\mathbf{u}: \Omega \to \mathbb{R}^2$.

| # | Dataset Name | Geometry | Material | Load Steps |
|---|---|---|---|---|
| 10 | `elasticity-circlehollow-m1` | Hollow circle | Steel | 2 |
| 11 | `elasticity-squarehollow-m1` | Hollow square | Steel | 2 |
| 12 | `elasticity-boomcircletri-m1` | Boomerang + holes | Steel | 2 |
| 13 | `elasticity-circlehollow-m2` | Hollow circle | Bone | 4 |
| 14 | `elasticity-squarehollow-m2` | Hollow square | Bone | 4 |
| 15 | `elasticity-boomcircletri-m2` | Boomerang + holes | Bone | 4 |
| 16 | `elasticity-circlehollow-m3` | Hollow circle | Rubber | 25 |
| 17 | `elasticity-squarehollow-m3` | Hollow square | Rubber | 25 |
| 18 | `elasticity-boomcircletri-m3` | Boomerang + holes | Rubber | 25 |

**Material parameters:**

| ID | Material | $E$ (MPa) | $\nu$ | $\mu$ (MPa) | $\lambda$ (MPa) | $\sigma_c$ (MPa) | Disp. BC range | Traction range |
|---|---|---|---|---|---|---|---|---|
| m1 | Steel | 200,000 | 0.3 | 76,923 | 115,385 | 100 | 0.1--0.5 mm | 4--40 MPa |
| m2 | Bone | 10,000 | 0.3 | 3,846 | 5,769 | 10 | 0.4--2 mm | 0.4--4 MPa |
| m3 | Rubber | -- | ~0.495 | 10 | 1,000 | 50 | 0.1--0.4 m | 0.5--3 MPa |

**BC structure for elasticity** (4 segments on exterior, per-component):
- Segment 0: Homogeneous Neumann (traction-free)
- Segment 1: Dirichlet (prescribed displacement)
- Segments 2--3: Neumann (applied traction)
- Hole "brown": homogeneous Dirichlet; other holes: homogeneous Neumann

---

## Code Architecture

### Data Flow

```
                    Configuration
                         |
                         v
    +-------------------------------------------+
    |          generate_dataset.py               |
    |  (main entry point, CLI, sample loop)      |
    +-------------------------------------------+
         |            |            |           |
         v            v            v           v
    mesh_gen.py   boundary.py   poisson_    elasticity_
    (Gmsh mesh)   (random BC    solver.py   solver.py
                   functions)   (-Delta u   (NeoHookean
                       |         = f)        + LBFGS)
                       v
                  bc_segments.py
                  (segment node
                   assignment)
         |            |            |           |
         v            v            v           v
    +-------------------------------------------+
    |     Post-processing & Storage              |
    |  sdf.py  harmonic_extension.py  output.py  |
    +-------------------------------------------+
                         |
                         v
                   data/*.nc (HDF5)
```

### Pipeline for One Sample

1. **Mesh** (once, shared): `create_mesh(shape, order)` generates the FEM mesh via Gmsh + TensorMesh
2. **SDF** (once, shared): `compute_sdf_at_nodes()` and `compute_sdf_on_grid()` pre-compute signed distance
3. **Random BCs**: `BCGenerator.draw()` produces random boundary segments with random BC types and random Fourier-mode functions
4. **Validation**: `draw_valid_bcs()` retries until segment lengths and non-Dirichlet fractions satisfy constraints
5. **Node assignment**: `build_bc_masks_and_values()` maps boundary nodes to segments by angular position, evaluates BC functions, builds mask/value tensors
6. **PDE solve**:
   - *Poisson*: Assemble stiffness (Laplace) + source RHS + Robin mass matrix + Neumann RHS, condense Dirichlet DOFs, solve
   - *Elasticity*: LBFGS energy minimization with NeoHookean strain energy, incremental loading, component-wise Dirichlet/Neumann
7. **Harmonic extensions**: Solve 3 Laplace problems per BC dimension to smoothly extend $(\alpha, \beta, g)$ into the domain
8. **Store**: Write coordinates, solution, variables, SDF, extensions, and boundary data to HDF5

---

## File Reference

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `generate_dataset.py` | ~400 | Main CLI entry point. Contains all 18 dataset configurations, sample generation functions, and the main loop. |
| `boundary.py` | ~350 | Random BC function generators. Defines `BCGenerator`, `SegmentBCs`, and all BC distribution classes (`Dirichlet`, `Neumann`, `Robin`, `RandomBCTypes`, etc.). Also contains source function generators (`RandomRadialSines`, `get_centered_radial_cosine`). |
| `bc_segments.py` | ~230 | Boundary segmentation. Computes angle-based parametric positions for boundary nodes, assigns nodes to segments, evaluates BC functions at nodes, and validates segment constraints. |
| `mesh_gen.py` | ~310 | Mesh generation for all 6 geometries. Uses TensorMesh built-ins for circle/square, and Gmsh geo API for boomerang and hollow geometries. Includes adaptive refinement near boundaries. |
| `geometry.py` | ~190 | Geometry curve definitions. Boomerang (transformed circle), circle/polygon boundaries, SmoothJoint curve approximation for holes, and hole configurations for all 3 hollow geometries. |
| `poisson_solver.py` | ~250 | Poisson solver with mixed BCs. Assembles stiffness via `LaplaceElementAssembler`, Robin boundary mass matrix via edge-based integration, Neumann RHS via consistent P1 load vectors. Uses `Condenser` for Dirichlet static condensation. |
| `elasticity_solver.py` | ~260 | Hyperelasticity solver. Custom `NeoHookeanModel` with $J$-clamping, LBFGS optimizer with incremental loading, component-wise Dirichlet/Neumann BCs, and Green-Lagrange strain + Cauchy stress post-processing via nodal averaging. |
| `harmonic_extension.py` | ~100 | Solves $-\Delta\phi = 0$ with Dirichlet BCs to extend boundary data ($\alpha$, $\beta$, $g$) smoothly into the interior. Used by neural operators to encode BC information. |
| `sdf.py` | ~175 | Signed distance function computation. Chunked min-distance to boundary curves, ray-casting inside/outside test, SDF on 256x256 grid with gradient via finite differences. |
| `output.py` | ~150 | HDF5 I/O matching original dataset format. Variable-length datasets for mesh-varying quantities. |

### Scripts

| File | Purpose |
|------|---------|
| `visualize_all.py` | Generates 1 sample per dataset and saves a PNG visualization. Poisson: scalar field plot. Elasticity: 6-panel plot ($u_x$, $u_y$, $\varepsilon_{11}$, $\varepsilon_{12}$, $\sigma_{11}$, $\sigma_{12}$). |
| `batch_generate.sh` | Shell script to generate all 18 datasets with configurable sample count and output directory. |

---

## PDE Formulations

### Poisson Equation

$$-\Delta u = f \quad \text{in } \Omega$$

with boundary conditions on segments of $\partial\Omega$:

| Type | Strong form | Weak form contribution |
|------|------------|----------------------|
| Dirichlet | $u = g$ | Static condensation (`Condenser`) |
| Neumann | $-\partial u / \partial n = g$ | $+ \int_{\Gamma_N} g \, v \, dS$ to RHS |
| Robin | $-\partial u / \partial n + \alpha u = g$ | $+ \int_{\Gamma_R} \alpha \, u \, v \, dS$ to stiffness, $+ \int_{\Gamma_R} g \, v \, dS$ to RHS |

**FEM discretization**: P1 triangles (order=1), `LaplaceElementAssembler` for stiffness, `NodeAssembler` for source RHS, edge-based integration for boundary terms.

### Hyperelasticity (Neo-Hookean)

Compressible Neo-Hookean strain energy density:

$$\Psi(\mathbf{F}) = \frac{\mu}{2}(I_1 - d) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2$$

where $\mathbf{F} = \mathbf{I} + \nabla\mathbf{u}$ is the deformation gradient, $J = \det\mathbf{F}$, $I_1 = \text{tr}(\mathbf{F}^T\mathbf{F})$, and $d$ is the spatial dimension.

The first Piola-Kirchhoff stress:

$$\mathbf{P} = \mu\mathbf{F} + (\lambda \ln J - \mu)\mathbf{F}^{-T}$$

**Solver**: LBFGS energy minimization with incremental loading. At each load step $k$, BCs are scaled by $m_k = k / (N_\text{steps} - 1)$.

**Post-processing** (nodal averaging from quadrature points):
- Green-Lagrange strain: $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$, stored as $(E_{11}, E_{12}, E_{21}, E_{22})$
- Cauchy stress: $\boldsymbol{\sigma} = \frac{1}{J}\mathbf{P}\mathbf{F}^T$, stored as $(\sigma_{11}, \sigma_{12}, \sigma_{21}, \sigma_{22})$

**FEM discretization**: P2 triangles (order=2, 6-node triangles) for all elasticity datasets.

---

## Geometry Definitions

### Simple Geometries (Poisson)

| Geometry | Definition | Mesh Size | Typical Nodes |
|----------|-----------|-----------|---------------|
| **Circle** | Unit disk $r=1$, center $(0,0)$ | $h = 0.025$ | ~6,000 |
| **Square** | $[-1,1]^2$ | $h = 2/70 \approx 0.029$ | ~5,800 |
| **Boomerang** | Transformed circle: $(x,y) \to (x, 2x^2+y) \to (x, 0.7y) \to$ normalize to $[-1,1]^2$ | $h = 0.025$ | ~14,800 |

### Hollow Geometries (Elasticity)

| Geometry | Outer | Holes | Typical Nodes (P2) |
|----------|-------|-------|-------------------|
| **CircleHollow** | Circle $r=1$ | "purple": SmoothJoint curve; "brown": rotated rectangle | ~47,000 |
| **SquareHollow** | Square $[-1,1]^2$ | "purple": SmoothJoint curve; "brown": rotated boomerang | ~68,000 |
| **BoomCircleTri** | Boomerang | "purple": circle $r=0.2$ at $(0,-0.6)$; "pink": circle $r=0.2$ at $(-0.4,0)$; "brown": triangle at $(0.2,-0.2)$ | ~68,000 |

All meshes use **adaptive refinement** near boundaries (element size $h/3$ within margin 0.3--0.4 of boundary, $h$ in interior) via Gmsh `Distance`+`Threshold` fields.

---

## Boundary Condition System

### Random BC Function Generation

BCs are generated as **angle-based Fourier mode functions**:

$$g(\mathbf{x}) = s \cdot \sin(\theta + \phi_0) \cdot \sum_{k=1}^{M} c_k \sin(k\theta + \phi_k)$$

where $\theta = \arctan2(y - C_y, x - C_x) / R$, and $s$, $c_k$, $\phi_k$ are randomly drawn per sample.

### Segment Structure

The boundary is divided into **segments** by random "joint" positions on the parametric curve $[0,1)$. Each segment is assigned a BC type (Dirichlet/Neumann/Robin) independently per spatial dimension. Boundary mesh nodes are assigned to segments based on their angular position from a center point.

### Validation Constraints

- All segments must have parametric length $\geq 0.1$
- For bc4/bc5: total non-Dirichlet boundary fraction must be in $[0.2, 0.51]$

---

## Output Format (HDF5)

Each dataset is stored as a single `.nc` (HDF5) file with the following structure:

```
file.nc
  count: int                              # Number of samples written
  coordinates/[i, 0, d]                   # [N, 1, 2] var-length: mesh node coordinates (transposed)
  bbox/
    grid/[i, 0, d, 256]                   # [N, 1, 2, 256]: x/y grid coordinates
    sdf/[i, 0, 0, 256, 256]              # [N, 1, 1, 256, 256]: SDF on regular grid
  interior/
    sdf/[i, 0, 0]                         # [N, 1, 1] var-length: SDF at mesh nodes
    sdf_grad/[i, 0, d]                    # [N, 1, 2] var-length: SDF gradient at nodes
    solution/[i, 0, d]                    # [N, 1, ndims] var-length: FEM solution
    source/[i, 0, 0]                      # [N, 1, 1] var-length (Poisson only)
    strain/[i, 0, c]                      # [N, 1, 4] var-length (Elasticity only)
    cauchystress/[i, 0, c]                # [N, 1, 4] var-length (Elasticity only)
    extensions/
      {dim}/                              # "0" for Poisson, "0" and "1" for Elasticity
        alpha/[i, 0, 0]                   # Harmonic extension of alpha coefficient
        beta/[i, 0, 0]                    # Harmonic extension of beta coefficient
        g/[i, 0, 0]                       # Harmonic extension of g values
  boundaries/
    {dim}/
      dirichlet/
        indices/[i, 0, 0]                 # Boundary node indices (var-length int32)
        g/[i, 0, 0]                       # BC values (var-length float64)
      neumann/
        indices/[i, 0, 0]
        g/[i, 0, 0]
      robin/
        indices/[i, 0, 0]
        g/[i, 0, 0]
        alpha/[i, 0, 0]                   # Robin coefficient
```

Variable-length arrays use `h5py.vlen_dtype` to handle meshes with different node counts.

---

## Dependencies

- **TensorMesh** (PyTorch FEM library) with `torch-sla` backend
- **Gmsh** (mesh generation, installed via `pip install gmsh`)
- **h5py** (HDF5 output)
- **matplotlib** (visualization)
- **scipy** (SDF grid interpolation)

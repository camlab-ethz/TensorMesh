# Helmholtz scattering (DtN boundary treatment)

This folder contains two self-contained TensorMesh gallery demos. They are scripts,
not a package: there is no shared helper module, command-line parser, or private
project dependency.

- `circular_dtn.py` runs a Dirichlet obstacle, a Neumann obstacle, and a penetrable
  circular interface.
- `periodic_dtn.py` runs a finite slab, a Dirichlet-backed layer, and a
  Neumann-backed layer.

Both solve

$$
\nabla\cdot(a\nabla u)+\omega^2bu=0,
\qquad
k_j=\omega\sqrt{b_j/a_j}.
$$

Keeping $a$ and $b$ separate is important: the local wavenumber $k_j$ controls
phase, while $a\partial_nu$ is the continuous material flux.

## Run the examples

From a TensorMesh checkout with the test dependencies installed, run:

```bash
python examples/wave/helmholtz_dtn/circular_dtn.py
python examples/wave/helmholtz_dtn/periodic_dtn.py
```

Each script runs three analytical-reference cases and writes its figures to
`docs/source/_static/wave/helmholtz_dtn/`.  Edit the constants near the top of
a script to change the material, geometry, incident field, mesh size, or modal
cutoff.

## Reading the circular demo

The script follows the same order as TensorMesh's Helmholtz example:

1. define the material parameters and `HelmholtzAssembler`;
2. generate the mesh;
3. assemble the weighted stiffness and mass terms with `ElementAssembler`;
4. integrate Fourier traces with `FacetAssembler`;
5. add the circular DtN block and solve;
6. evaluate the exact Bessel--Hankel field and render all fields with `Mesh.plot`.

The incident mode is

$$
u^{\rm inc}(r,\theta)=J_n(k_0r)e^{in\theta}.
$$

The physical radius $R_c$ and artificial radius $R_\Gamma$ are independent. The
Dirichlet and Neumann cases choose the outgoing coefficient to satisfy the obstacle
condition. The transmission case solves a two-by-two system enforcing continuity of
$u$ and $a\partial_ru$.

## Reading the periodic demo

The periodic script uses a plane wave and a layer $0<x_2<h$. For each Rayleigh order,

$$
\alpha_m=\alpha+\frac{2\pi m}{L},
\qquad
\beta_{j,m}=\sqrt{k_j^2-\alpha_m^2},
$$

with the outgoing square-root branch. The slab has radiation ports above and below.
The wall cases have only a top port, so they report no artificial transmission value.
The incident order is propagating, $|\alpha|<k_0$, and retained modes must avoid
Wood anomalies. A cutoff of `NUM_MODES = 0` is supported and keeps only the
incident Rayleigh order.

Lower-case $r_m,t_m$ are complex amplitudes. The power fractions are

$$
\mathcal R_m=
\frac{\operatorname{Re}(a_0\beta_{0,m})}
     {\operatorname{Re}(a_0\beta_{0,0})}
\left|\frac{r_m}{A}\right|^2,
\qquad
\mathcal T_m=
\frac{\operatorname{Re}(a_b\beta_{b,m})}
     {\operatorname{Re}(a_0\beta_{0,0})}
\left|\frac{t_m}{A}\right|^2.
$$

For a periodic Dirichlet wall, the code intentionally performs these operations in
order: full volume and DtN assembly, `BlochReducer(sign=+1)`, explicit $T^Hb$,
`Condenser` on the master degrees of freedom, solve, then reverse recovery. A Neumann
wall is natural and does not use `Condenser`.

## Complex-valued assembly

Geometry, shape functions, and quadrature stay `float64`. The element assembler,
matrix, right-hand side, Bloch phase, and solution use `complex128`. Because the
facet geometry path is real, a mode is integrated as cosine and negative-sine
channels and combined afterward.

For modal moments $\Psi$, both scripts form the nonlocal block as

```python
Psi.conj() @ torch.diag(symbols) @ Psi.T
```

The last factor is a plain transpose because `Psi.T @ u` computes the modal
coefficients. The system is a general complex Helmholtz system, not Hermitian or
positive definite. The demos therefore call TensorMesh `SparseMatrix.solve` with the
SciPy LU backend selected explicitly instead of requesting an SPD solver.

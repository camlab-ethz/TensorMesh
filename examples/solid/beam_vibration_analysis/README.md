# Beam vibration analysis — cantilever cylinder (modes + FRF)

A 3D **structural-vibration** example: a clamped-free circular steel rod
(length $L = 1\,\mathrm{m}$, radius $R = 25\,\mathrm{mm}$) analysed for its
natural **mode shapes** and its harmonic **displacement FRF** (frequency-response
function). It exercises TensorMesh linear elasticity for a generalized
eigenproblem plus a modal harmonic response. Example-only: no public API is added.

## Model

Linear elasticity on a tetrahedral mesh gives the stiffness $K$ and consistent
mass $M$; clamping the base face $z = 0$ removes those DOFs.

**Modes** — generalized eigenproblem (shift-invert Lanczos near $\sigma = 0$):

$$K\,\phi = \omega^2 M\,\phi, \qquad f_n = \omega_n / (2\pi).$$

**FRF** — by modal superposition with the mass-normalized modes
($\phi_r^{\mathsf T} M \phi_r = 1$) and an isotropic (hysteretic) loss factor
$\eta = 2\%$, i.e. a $1\%$ modal damping ratio ($\zeta = \eta/2$):

$$u(\omega) = \sum_r \frac{\phi_r\,(\phi_r\cdot F)}{\omega_r^2(1 + i\eta) - \omega^2}.$$

The tip response to a transverse ($x$) tip force is normalized by its static
($\omega\!\to\!0$) value, giving a **dimensionless dynamic amplification**
$|u_x|/u_\text{static}$ — 1 below the first resonance, peaking on each bending
mode and dropping into antiresonances between them.

$K$ comes from `LinearElasticityElementAssembler`; the consistent mass is the
scalar `MassElementAssembler` lifted to the three vector components,
$M_\text{vec} = \rho\,(M_\text{scalar} \otimes I_3)$. The eigen solve uses SciPy
sparse (post-processing), keeping the compute core numpy-free.

```text
mesh (gmsh cylinder, tets) -> assemble (K, M) -> eig (modes)
                                              -> modal FRF -> plot modes + FRF
```

## Run

```bash
python vibration_cylinder.py
```

`run_demo(...)` returns diagnostics; `main()` exposes `--n-modes`, `--mesh-h-mm`,
`--frf-points`, `--no-plot`, `--modes-output`, `--frf-output`.

## What it shows

* **Mode shapes** (`vibration_cylinder_modes.png`) — deformed 3D cylinders
  (colored by displacement magnitude, tip amplitude exaggerated, undeformed axis
  drawn as a dashed line), one mode of each kind. Because the section is circular
  each **bending** mode is doubled (two orthogonal bending planes), so modes come
  in near-equal pairs; above them sit the first **torsion** and **axial** modes.
* **Displacement FRF** (`vibration_cylinder_frf.png`) — tip transverse
  displacement amplitude $|u_x|$ (m) vs frequency: sharp peaks at the bending
  frequencies with antiresonances between them. The transverse tip force excites
  only bending, so the torsion/axial modes show no peak.

## Accuracy

The first bending frequency lands at 37.2 Hz vs the Euler-Bernoulli cantilever
value $f_1 = (\beta_1 L)^2/(2\pi)\sqrt{EI/(\rho A L^4)} = 36.2$ Hz. Linear
tetrahedra are a little stiff in bending, so the FEM frequencies sit ~2–3 %
above the slender-beam analytic and drop toward it as the mesh is refined
(`--mesh-h-mm 5`); the axial mode already matches to <0.1 %.

> Quadratic (`tetra10`) elements are **not** used here: gmsh's second-order
> tet edge-node ordering does not match TensorMesh's, which corrupts the curved
> Jacobian (the mass matrix under-integrates the volume by ~17 %). The example
> therefore stays on linear tets and relies on mesh refinement for accuracy.

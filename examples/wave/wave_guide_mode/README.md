# Waveguide mode analysis — rectangular dielectric waveguide

A 2D **photonic** example: the guided modes of a rectangular dielectric
waveguide, found from its cross-section — a high-index rectangular core
(width $w$ × height $h$, index $n_\text{core}$) in a lower-index cladding
($n_\text{clad}$). Example-only: no public API is added.

## Model

In the weakly-guiding (scalar) approximation the transverse field $E(x,y)$ and
propagation constant $\beta$ satisfy the transverse Helmholtz equation

$$\nabla_t^2 E + \big(k_0^2\,n(x,y)^2 - \beta^2\big)E = 0,\qquad k_0 = 2\pi/\lambda_0,$$

a generalized eigenproblem for $\beta^2$. With the builtin stiffness $K$
(`LaplaceElementAssembler`), mass $M$ (`MassElementAssembler`) and the
index-weighted mass $M_\varepsilon = \int n^2\,\phi_i\phi_j$
(`ScaledMassElementAssembler`):

$$\big(K - k_0^2 M_\varepsilon\big)\,E = -\beta^2 M\,E.$$

The field is clamped to zero on the outer box (guided fields decay in the
cladding), so the modes come from a shift-invert Lanczos solve near the core
light line $\beta^2 = (k_0 n_\text{core})^2$. A mode is **guided** when its
effective index $n_\text{eff} = \beta/k_0$ satisfies
$n_\text{clad} < n_\text{eff} < n_\text{core}$.

```text
mesh (gmsh, core in box) -> assemble (K, M, M_eps) -> eig (beta^2)
                                                   -> guided modes (n_eff, fields)
```

## Run

```bash
python waveguide_modes.py
```

`run_demo(...)` returns diagnostics; `main()` exposes `--core-w-um`,
`--core-h-um`, `--n-core`, `--n-clad`, `--lam0-nm`, `--n-modes`, `--mesh-h-um`,
`--no-plot`, `--output`.

The mode count is set by the two normalized frequencies
$V_x = k_0\,(w/2)\,\mathrm{NA}$ and $V_y = k_0\,(h/2)\,\mathrm{NA}$ with
$\mathrm{NA}=\sqrt{n_\text{core}^2-n_\text{clad}^2}$. The default
($w=18\,\mu\mathrm{m}$, $h=14\,\mu\mathrm{m}$, $n_\text{core}=1.450$,
$n_\text{clad}=1.444$, $\lambda_0=1.55\,\mu\mathrm{m}$) gives
$V_x\approx4.81$, $V_y\approx3.74$ and seven guided modes (the gallery shows the
first six).

## What it shows

`waveguide_modes.png` — the guided transverse mode fields $E(x,y)$ (red/blue =
field sign, core outlined). Modes are labelled $E_{pq}$ by their lobe counts
along $x$ and $y$: $E_{11}$ (single lobe), $E_{21}$ / $E_{12}$ (dipoles along
$x$ / $y$), $E_{22}$ (quadrupole), $E_{31}$ / $E_{13}$ (three lobes across the
width / height).

## Accuracy

Each computed effective index is compared to the separable **Marcatili**
approximation: solve the symmetric-slab dispersion $U\tan U = W$ (even) /
$-U\cot U = W$ (odd), $U^2+W^2=(a\,k_0\,\mathrm{NA})^2$, independently across the
width and height, then combine $\beta^2 = k_0^2 n_\text{core}^2 - k_x^2 - k_y^2$.
FEM and Marcatili $n_\text{eff}$ agree to ~$10^{-4}$ (Marcatili is itself
approximate — it neglects the field in the core corners), and both give the same
guided-mode set:

| mode | $n_\text{eff}$ (FEM) | $n_\text{eff}$ (Marcatili) |
|-----:|---------------------:|---------------------------:|
| $E_{11}$ | 1.44902 | 1.44891 |
| $E_{21}$ | 1.44784 | 1.44763 |
| $E_{12}$ | 1.44732 | 1.44703 |
| $E_{22}$ | 1.44614 | 1.44574 |
| $E_{31}$ | 1.44595 | 1.44560 |
| $E_{13}$ | 1.44478 | 1.44430 |

Refining the mesh (`--mesh-h-um`) or box (`box_factor`) tightens the FEM values.

# Helmholtz resonator

A 2D **acoustic Helmholtz resonator** — a duct with a necked side cavity, driven
by a plane-wave port — solved in the frequency domain on TensorMesh. It showcases
the library boundary operators [`tensormesh.robin_operator` /
`tensormesh.port_source`](../../../tensormesh/bc.py) (built on
`FacetBilinearAssembler`): the port line mass and the incident-wave load.

Example-only: no public API is added. It reuses the scalar-Helmholtz assembly
(`LaplaceElementAssembler`, `MassElementAssembler`), the boundary operators, and
the complex sparse solve.

## Model

Scalar pressure acoustics $\nabla^2 p + k^2 p = 0$ on the air domain (duct + neck
+ cavity, meshed as one union). All walls are sound-hard (natural Neumann); the
inlet $x=0$ is a first-order plane-wave port that both injects the incident wave
and absorbs the reflected one:

$$\frac{\partial p}{\partial n} + i k\, p = 2 i k\, p_0 \quad\text{on the inlet.}$$

In the weak form the port adds a boundary mass $B=\int_\Gamma \phi_i \phi_j\,\mathrm dS$
(scaled by $i k$) to the operator and a load $e=\int_\Gamma \phi_i\,\mathrm dS$
(scaled by $2 i k p_0$) to the right-hand side — both assembled by
`robin_operator` / `port_source` and folded onto the volume layout so the whole
operator stays one complex `SparseMatrix`:

```text
mesh -> Laplace/Mass assembler -> port operators -> (K - k^2 M + i k B) p = 2 i k p0 e
```

The input impedance $Z(f)=p_\text{avg}/u_\text{in}$ is read at the inlet over a
frequency sweep; its peaks are the resonances (the Helmholtz mode plus duct
harmonics).

## Run

```bash
python helmholtz_resonator.py
```

`run_demo(...)` returns diagnostics and `main()` exposes `--no-plot` /
`--output`. The demo reports the fundamental resonance near **357 Hz** with a
cavity pressure gain of ~12x over the incident amplitude.

## What it shows

- **Input impedance** $|Z|/Z_0$ over the sweep (log scale) with the cavity
  pressure response — resonance and anti-resonance peaks.
- **Acoustic field** at the resonance: total pressure `Re(p)` and sound-pressure
  level, with the pressure node line across the neck.
- **Boundary conditions**: a schematic of the port (red) and sound-hard walls
  (blue) over the duct + cavity geometry.

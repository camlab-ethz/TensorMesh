# Optical ring resonator (waveguide-coupled microdisk)

A 2D **photonic** example: a silicon bus waveguide evanescently couples light
into a silicon **microdisk** (both $n=3.48$) embedded in silica ($n=1.44$), with
a stretched-coordinate **PML** frame making the domain open. On resonance the
disk lights up in a **whispering-gallery mode** (WGM) — a bright ring of
azimuthal lobes on the rim.

It exercises the open-domain wave operators
[`tensormesh.cartesian_pml`](../../../tensormesh/pml.py) +
`AnisotropicLaplaceElementAssembler` + `ScaledMassElementAssembler`. Example-only:
no public API is added.

## Model

In TM polarization the out-of-plane field $E_z$ obeys a scalar Helmholtz
equation with the PML's complex coordinate stretch $s_x, s_y$:

$$\nabla\cdot(\mathbf\Lambda\,\nabla E_z) + k_0^2\,\varepsilon_r\, s_x s_y\, E_z = \text{source},
\qquad \mathbf\Lambda = \mathrm{diag}(s_y/s_x,\ s_x/s_y).$$

Outside the PML $s_x=s_y=1$ and this is the ordinary Helmholtz equation;
$\varepsilon_r$ is $n_\text{core}^2$ in the waveguide and disk, $n_\text{clad}^2$
elsewhere. A line-current source in the waveguide launches the guided mode.

```text
mesh (gmsh, conforming) -> assemble (K_PML - k0^2 M) -> complex solve -> E_z
```

The example uses an inline `ElementAssembler` that evaluates $\varepsilon_r$ and
the stretch **per quadrature point** (sharper at the Si/SiO2 interfaces than
nodal coefficients); the reusable library path is
`cartesian_pml` + `AnisotropicLaplaceElementAssembler` +
`ScaledMassElementAssembler`, which reproduces the same operator and matches the
2D Hankel Green function to correlation 0.994.

The PML validity was checked against a purely-absorbing reference: without it a
PEC/hard box traps the field into a standing wave (~8x the amplitude).

## Run

```bash
python optical_ring_resonator.py
```

`run_demo(...)` returns diagnostics; `main()` exposes `--lam0-nm`, `--order`
(1 or 2), `--mesh-h-nm`, `--no-plot`, `--output`.

**On resonance vs off resonance.** The disk WGM is a sharp, high-Q resonance, so
the drive wavelength must sit on it. With linear (P1) elements numerical
dispersion blue-shifts the physical ~1.55 µm resonance to ~1.512 µm (the default
`lam0`); driving there gives a clean rim mode, while off resonance the disk fills
with a messy radial mix. Refining the mesh or `--order 2` moves the numerical
resonance back toward 1.55 µm.

## What it shows

A two-panel figure of the on-resonance field: `Re(E_z)` (the guided mode feeding
a ring of alternating rim lobes) and `|E_z|` (the bright whispering-gallery ring,
the waveguide depleted as power couples into the disk), both decaying into the
PML frame.

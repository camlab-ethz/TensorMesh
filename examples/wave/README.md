# Wave Equation

Transient wave propagation using central difference time integration.

## Problem Setup

- **PDE:** $u_{tt} = c^2 \Delta u$
- **Geometry:** Unit square $[0,1]^2$ (triangular mesh)
- **Boundary Conditions:** Homogeneous Dirichlet ($u = 0$ on $\partial\Omega$)
- **Initial Condition:** Multi-frequency Fourier series via `WaveMultiFrequency`
- **Time Integration:** Central difference (explicit), $c = 2.0$, $\Delta t = 0.001$, 100 steps

## Usage

```bash
python wave.py
```

## Output

- `wave.mp4`: animation comparing FEM prediction with analytical ground truth
- `wave_energy.png`: kinetic / potential / total mechanical energy vs time — the total stays flat, confirming the central-difference scheme conserves energy under the CFL condition

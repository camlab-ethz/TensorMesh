# Fluid Dynamics Examples 🌊

This directory contains examples of fluid dynamics simulations using TensorMesh.

## Examples

### 1. Lid-Driven Cavity (`cavity.py`)
The classic benchmark for incompressible Navier-Stokes solvers. A square cavity with a moving lid creates a large central vortex and smaller corner vortices.
- **Physics**: Steady-state Incompressible Navier-Stokes.
- **Technique**: Picard iteration with PSPG/SUPG stabilization for P1/P1 elements.

### 2. Kármán Vortex Street (`vortex_street.py`)
Simulation of flow past a circular cylinder. At Re=100, vortices are shed periodically, forming a beautiful pattern behind the cylinder.
- **Physics**: Transient Incompressible Navier-Stokes.
- **Technique**: Backward Euler time stepping, stabilized finite elements.

### 3. Rayleigh-Bénard Convection (`rayleigh_benard.py`)
A fluid layer heated from below. When the Rayleigh number exceeds a critical value, buoyancy overcomes diffusion, and convection cells form.
- **Physics**: Boussinesq-coupled Navier-Stokes and Heat Equation.
- **Technique**: Monolithic Picard iteration for velocity, pressure, and temperature.

### 4. Flow Past Multiple Obstacles (`flow_logo.py`)
Demonstrates TensorMesh's ability to handle complex geometries by simulating flow through a channel with multiple circular obstacles.
- **Physics**: Steady-state Navier-Stokes.
- **Technique**: Non-linear solver on non-structured meshes generated via Gmsh.

## How to Run

Navigate to this directory and run any script with Python:

```bash
python cavity.py
python vortex_street.py
python rayleigh_benard.py
python flow_logo.py
```

Results (images or MP4 videos) will be saved in the same directory.


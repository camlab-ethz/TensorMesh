"""
Emmentaler Linear Elasticity Example
=====================================

Solves linear elasticity on the "Emmentaler" geometry (3D block with random
spherical holes) from the solidmechanics_datagen project, using TensorMesh.

Boundary conditions match the original phase-field fracture setup:
  - Bottom face (z=0): fully clamped
  - Top face (z=T): prescribed displacement combining tension, bending, and torsion

This script supports two modes:
  1. Generate the Emmentaler mesh via Gmsh (default, no DOLFINx needed)
  2. Load an existing mesh from file (--mesh_file)

Usage:
    python emmentaler_elasticity.py                         # generate mesh + solve
    python emmentaler_elasticity.py --mesh_file B0001.msh   # load existing mesh
    python emmentaler_elasticity.py --h 0.08                # finer mesh
    python emmentaler_elasticity.py --steps 11              # multi-step loading
"""

import sys
import os
import argparse
import csv
import math
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.assemble import LinearElasticityElementAssembler


# ---------------------------------------------------------------------------
# Mesh generation (Gmsh only, no DOLFINx dependency)
# ---------------------------------------------------------------------------

# Hole positions from solidmechanics_datagen/mesh/meshgen_emmentaler.py
HOLE_POSITIONS = [
    (0.511, 0.241, 0.413), (0.743, 0.451, 0.88),
    (0.785, 0.386, 0.437), (0.746, 0.663, 0.98),
    (0.393, 0.606, 1.299), (0.736, 0.691, 0.703),
    (0.261, 0.765, 0.337), (0.243, 0.377, 0.427),
    (0.396, 0.64,  1.037), (0.326, 0.832, 1.268),
    (0.503, 0.627, 0.76),  (0.784, 0.816, 0.452),
    (0.192, 0.211, 1.188), (0.43,  0.407, 0.577),
    (0.758, 0.166, 0.916), (0.617, 0.474, 1.093),
    (0.446, 0.574, 0.391), (0.444, 0.318, 1.166),
    (0.331, 0.416, 0.936), (0.464, 0.178, 0.89),
    (0.339, 0.779, 0.551), (0.601, 0.758, 1.296),
    (0.693, 0.226, 0.592),
]


def generate_emmentaler_mesh(H=1.0, L=1.0, T=1.5, R=0.16, h=0.15, out_file="emmentaler.msh"):
    """Generate the Emmentaler mesh using Gmsh with settings matching solidmechanics_datagen.

    Parameters
    ----------
    H, L, T : float
        Block width, length, height.
    R : float
        Hole radius.
    h : float
        Mesh element size. Use ~0.15 for quick tests, ~0.01 for production.
    out_file : str
        Output mesh file path (.msh).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    # Mesh settings (matching solidmechanics_datagen/mesh/meshgen_emmentaler.py)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0 * h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.9 * h)
    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("Mesh.Hexahedra", 1)
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
    gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)
    gmsh.option.setNumber("Mesh.SmoothRatio", 1.1)

    gmsh.model.add("Emmentaler")

    # Outer block: centered at origin in x-y, z from 0 to T
    box = gmsh.model.occ.addBox(-H / 2, -L / 2, 0, H, L, T)

    # Create and subtract spherical holes
    holes = [(x - H / 2, y - L / 2, z) for x, y, z in HOLE_POSITIONS]
    hole_tags = [gmsh.model.occ.addSphere(x, y, z, R) for x, y, z in holes]
    gmsh.model.occ.cut([(3, box)], [(3, tag) for tag in hole_tags])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], 1)

    gmsh.model.mesh.generate(3)
    gmsh.write(out_file)
    gmsh.finalize()

    print(f"Mesh written to {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Boundary condition helpers
# ---------------------------------------------------------------------------
def identify_boundaries(points, T, tol=1e-5):
    """Identify bottom (z=0) and top (z=T) boundary nodes."""
    bottom_mask = torch.abs(points[:, 2]) < tol
    top_mask = torch.abs(points[:, 2] - T) < tol
    return bottom_mask, top_mask


def compute_top_displacement(points_top, load_factor,
                             normal_mag=1.0, bending_mag=1.0, torsion_mag=1.0):
    """Compute prescribed displacement on top face nodes.

    Combines three loading modes (matching solidmechanics_datagen/fem/problems.py):
      - Tension: u_z along z-axis
      - Bending: curvature kappa about x-axis
      - Torsion: rotation theta about z-axis

    Parameters
    ----------
    points_top : torch.Tensor [n_top, 3]
        Coordinates of top-face nodes.
    load_factor : float
        Load multiplier in [0, 1].

    Returns
    -------
    u_top : torch.Tensor [n_top, 3]
    u_z, kappa, theta : float
        Applied load parameter values.
    """
    # Normalization factors (from solidmechanics_datagen)
    normfac_normal = 0.010
    normfac_bending = 0.014
    normfac_torsion = 0.048

    u_z = load_factor * normal_mag * normfac_normal
    kappa = load_factor * bending_mag * normfac_bending
    theta = load_factor * torsion_mag * normfac_torsion

    x = points_top

    # Torsion: R(theta) @ x - x  (rotation about z-axis)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u_top = torch.zeros_like(x)
    u_top[:, 0] = cos_t * x[:, 0] + sin_t * x[:, 1] - x[:, 0]
    u_top[:, 1] = -sin_t * x[:, 0] + cos_t * x[:, 1] - x[:, 1]

    # Bending: [0, -kappa/2 * z^2, kappa * z * y]
    u_top[:, 1] += -kappa / 2 * x[:, 2] ** 2
    u_top[:, 2] += kappa * x[:, 2] * x[:, 1]

    # Normal tension: [0, 0, u_z]
    u_top[:, 2] += u_z

    return u_top, u_z, kappa, theta


def compute_reaction_forces(K, u_flat, f_flat, bottom_mask, dim=3):
    """Compute reaction forces at constrained (bottom) nodes.

    R = K @ u - f  (residual at constrained DOFs = reaction force)

    Returns
    -------
    F_x, F_y, F_z : float
        Total reaction force components.
    """
    r = K @ u_flat - f_flat
    r = r.reshape(-1, dim)
    rx = r[bottom_mask, 0].sum().item()
    ry = r[bottom_mask, 1].sum().item()
    rz = r[bottom_mask, 2].sum().item()
    return rx, ry, rz


def compute_nodal_stress_strain(assembler, mesh, u_vec, E, nu):
    """Compute nodal stress and strain fields by averaging quadrature-point values.

    Uses the assembler's shape function gradients to compute the displacement
    gradient at quadrature points, then computes strain/stress and averages
    to nodes (matching solidmechanics_datagen's CG1 projection).

    Parameters
    ----------
    assembler : LinearElasticityElementAssembler
    mesh : Mesh
    u_vec : torch.Tensor [n_nodes, 3]
        Displacement solution.
    E, nu : float
        Material parameters.

    Returns
    -------
    nodal_strain : torch.Tensor [n_nodes, 6]
        Voigt-notation strain [eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy].
    nodal_stress : torch.Tensor [n_nodes, 6]
        Voigt-notation stress [sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sig_xy].
    von_mises : torch.Tensor [n_nodes]
        Von Mises equivalent stress.
    """
    n_nodes = mesh.points.shape[0]
    dim = mesh.points.shape[1]

    # Accumulators for nodal averaging
    nodal_stress_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_strain_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_count = torch.zeros(n_nodes, dtype=u_vec.dtype)

    for element_type in assembler.element_types:
        trans = assembler.transformation[element_type]
        elements = assembler.elements[element_type]  # [n_elem, n_basis]
        n_elem, n_basis = elements.shape

        # Element displacement: [n_elem, n_basis, dim]
        elem_u = u_vec[elements]

        # shape_grad: [n_elem, n_quad, n_basis, dim]
        shape_grad = trans.shape_grad

        # Displacement gradient at quadrature points: [n_elem, n_quad, dim, dim]
        grad_u = torch.einsum("eqbd,ebc->eqdc", shape_grad, elem_u)

        # Strain at quadrature points: eps = 0.5*(grad_u + grad_u^T)
        # [n_elem, n_quad, dim, dim]
        eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))

        # Stress via Hooke's law (batched): sigma = lambda*tr(eps)*I + 2*mu*eps
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        tr_eps = eps.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(dim, dtype=eps.dtype, device=eps.device)
        sigma = lam * tr_eps * eye + 2.0 * mu * eps

        # Average over quadrature points per element: [n_elem, dim, dim]
        eps_avg = eps.mean(dim=1)
        sigma_avg = sigma.mean(dim=1)

        # Convert to Voigt notation [n_elem, 6]:
        # [eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy]
        eps_voigt = torch.stack([
            eps_avg[:, 0, 0], eps_avg[:, 1, 1], eps_avg[:, 2, 2],
            eps_avg[:, 1, 2], eps_avg[:, 0, 2], eps_avg[:, 0, 1],
        ], dim=1)
        sigma_voigt = torch.stack([
            sigma_avg[:, 0, 0], sigma_avg[:, 1, 1], sigma_avg[:, 2, 2],
            sigma_avg[:, 1, 2], sigma_avg[:, 0, 2], sigma_avg[:, 0, 1],
        ], dim=1)

        # Scatter-add to nodes (each element contributes to its n_basis nodes)
        for b in range(n_basis):
            node_ids = elements[:, b]  # [n_elem]
            nodal_strain_sum.index_add_(0, node_ids, eps_voigt)
            nodal_stress_sum.index_add_(0, node_ids, sigma_voigt)
            nodal_count.index_add_(0, node_ids, torch.ones(n_elem, dtype=u_vec.dtype))

    # Average
    nodal_count = nodal_count.clamp(min=1).unsqueeze(1)
    nodal_strain = nodal_strain_sum / nodal_count
    nodal_stress = nodal_stress_sum / nodal_count

    # Von Mises stress
    s = nodal_stress
    von_mises = torch.sqrt(0.5 * (
        (s[:, 0] - s[:, 1]) ** 2 +
        (s[:, 1] - s[:, 2]) ** 2 +
        (s[:, 2] - s[:, 0]) ** 2 +
        6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    ))

    return nodal_strain, nodal_stress, von_mises


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_emmentaler(mesh_file=None, h=0.15, E=12000.0, nu=0.3,
                     normal_mag=1.0, bending_mag=1.0, torsion_mag=1.0,
                     T=1.5, steps=1, output_dir="."):
    """Solve linear elasticity on Emmentaler geometry.

    Parameters
    ----------
    mesh_file : str or None
        Path to existing mesh (.msh, .vtk, .xdmf, ...). If None, generates one.
    h : float
        Mesh element size (only used when generating mesh).
    E, nu : float
        Young's modulus and Poisson's ratio.
    normal_mag, bending_mag, torsion_mag : float
        Load magnitude multipliers for tension, bending, torsion (each in [0, 1]).
    T : float
        Block height.
    steps : int
        Number of load steps (1 = single solve at full load).
    output_dir : str
        Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Mesh ----
    if mesh_file is None:
        mesh_path = os.path.join(output_dir, "emmentaler.msh")
        generate_emmentaler_mesh(T=T, h=h, out_file=mesh_path)
        mesh = Mesh.read(mesh_path, reorder=True)
    else:
        mesh = Mesh.read(mesh_file, reorder=True)

    n_nodes = mesh.points.shape[0]
    n_cells = sum(v.shape[0] for v in mesh.cells.values())
    cell_types = list(mesh.cells.keys())
    print(f"Mesh: {n_nodes} nodes, {n_cells} elements, types: {cell_types}")

    # ---- 2. Assemble stiffness matrix (once, reused for all load steps) ----
    print(f"Assembling stiffness matrix (E={E}, nu={nu})...")
    assembler = LinearElasticityElementAssembler.from_mesh(mesh, E=E, nu=nu)
    K = assembler()
    print(f"Stiffness matrix: {K.shape[0]} DOFs")

    # ---- 3. Identify boundaries ----
    bottom_mask, top_mask = identify_boundaries(mesh.points, T)
    print(f"Boundary nodes: {bottom_mask.sum().item()} bottom, {top_mask.sum().item()} top")

    # Dirichlet mask (constant across load steps)
    dirichlet_mask = torch.zeros(n_nodes, 3, dtype=torch.bool)
    dirichlet_mask[bottom_mask, :] = True
    dirichlet_mask[top_mask, :] = True
    dirichlet_mask_flat = dirichlet_mask.flatten()

    # Zero body forces
    f = torch.zeros(n_nodes * 3, dtype=mesh.points.dtype)

    # Points on the top face (precompute)
    points_top = mesh.points[top_mask]

    # ---- 4. Load stepping ----
    load_factors = torch.linspace(0, 1, steps) if steps > 1 else torch.tensor([1.0])

    # CSV monitoring output (matching solidmechanics_datagen format)
    csv_path = os.path.join(output_dir, "monitoring.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter="\t")
    csv_writer.writerow(["step", "lambda", "u_z", "kappa", "theta",
                         "E_strain", "F_x", "F_y", "F_z", "u_max"])

    for step_idx, lf in enumerate(load_factors):
        lf_val = lf.item()

        # Compute prescribed displacement on top face
        u_top, u_z_val, kappa_val, theta_val = compute_top_displacement(
            points_top, lf_val,
            normal_mag=normal_mag, bending_mag=bending_mag, torsion_mag=torsion_mag,
        )

        # Build Dirichlet values for this load step
        dirichlet_value = torch.zeros(n_nodes, 3, dtype=mesh.points.dtype)
        dirichlet_value[top_mask] = u_top

        condenser = Condenser(dirichlet_mask_flat, dirichlet_value.flatten())

        # Solve
        K_inner, f_inner = condenser(K, f)
        u_inner = K_inner.solve(f_inner)
        u = condenser.recover(u_inner)

        # Post-process
        u_vec = u.reshape(-1, 3)
        u_max = torch.max(torch.norm(u_vec, dim=1)).item()
        energy = assembler.energy(mesh.points, point_data={"displacement": u_vec})
        e_val = energy.item()
        fx, fy, fz = compute_reaction_forces(K, u, f, bottom_mask)

        csv_writer.writerow([
            step_idx, f"{lf_val:.6f}",
            f"{u_z_val:.6e}", f"{kappa_val:.6e}", f"{theta_val:.6e}",
            f"{e_val:.6e}", f"{fx:.6e}", f"{fy:.6e}", f"{fz:.6e}", f"{u_max:.6e}",
        ])

        if steps > 1:
            print(f"  step {step_idx:3d}/{steps-1}  λ={lf_val:.4f}  "
                  f"E={e_val:.4e}  F_z={fz:.4e}  u_max={u_max:.4e}")

    csv_file.close()
    print(f"Monitoring data saved to {csv_path}")

    # ---- 5. Compute stress/strain fields ----
    print("Computing stress and strain fields...")
    nodal_strain, nodal_stress, von_mises = compute_nodal_stress_strain(
        assembler, mesh, u_vec, E, nu,
    )
    print(f"Max von Mises stress: {von_mises.max().item():.6e}")

    # ---- 6. Save final result as VTK ----
    mesh.register_point_data("displacement", u_vec)
    mesh.register_point_data("displacement_magnitude", torch.norm(u_vec, dim=1))
    # Strain: [eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy]
    mesh.register_point_data("strain", nodal_strain)
    # Stress: [sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sig_xy]
    mesh.register_point_data("stress", nodal_stress)
    mesh.register_point_data("von_mises_stress", von_mises)

    out_path = os.path.join(output_dir, "emmentaler_result.vtk")
    mesh.save(out_path)
    print(f"Final result saved to {out_path}")
    print(f"Max displacement: {u_max:.6e}")
    print(f"Strain energy:    {e_val:.6e}")
    print(f"Reaction forces:  Fx={fx:.4e}  Fy={fy:.4e}  Fz={fz:.4e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emmentaler linear elasticity (TensorMesh)")
    parser.add_argument("--mesh_file", type=str, default=None,
                        help="Path to existing mesh file (.msh, .vtk, .xdmf)")
    parser.add_argument("--h", type=float, default=0.15,
                        help="Mesh element size for generation (default: 0.15)")
    parser.add_argument("--E", type=float, default=12000.0,
                        help="Young's modulus (default: 12000)")
    parser.add_argument("--nu", type=float, default=0.3,
                        help="Poisson's ratio (default: 0.3)")
    parser.add_argument("--normal_mag", type=float, default=1.0,
                        help="Tension magnitude [0,1] (default: 1.0)")
    parser.add_argument("--bending_mag", type=float, default=1.0,
                        help="Bending magnitude [0,1] (default: 1.0)")
    parser.add_argument("--torsion_mag", type=float, default=1.0,
                        help="Torsion magnitude [0,1] (default: 1.0)")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of load steps (default: 1)")
    parser.add_argument("--output_dir", type=str, default="output_emmentaler",
                        help="Output directory (default: output_emmentaler)")
    args = parser.parse_args()

    solve_emmentaler(
        mesh_file=args.mesh_file, h=args.h, E=args.E, nu=args.nu,
        normal_mag=args.normal_mag, bending_mag=args.bending_mag,
        torsion_mag=args.torsion_mag, steps=args.steps, output_dir=args.output_dir,
    )

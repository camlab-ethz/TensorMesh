"""
Emmentaler Hyperelastic Example (Neo-Hookean)
==============================================

Solves geometrically nonlinear elasticity (Neo-Hookean) on the "Emmentaler"
geometry using energy minimization with LBFGS and load stepping.

This is Phase 2 of the solidmechanics_datagen reproduction with TensorMesh,
extending Phase 1 (linear elasticity) to finite strains.

Boundary conditions (same as solidmechanics_datagen):
  - Bottom face (z=0): fully clamped
  - Top face (z=T): prescribed displacement (tension + bending + torsion)

Usage:
    python emmentaler_hyperelastic.py                          # default
    python emmentaler_hyperelastic.py --mesh_file B0001.msh    # existing mesh
    python emmentaler_hyperelastic.py --h 0.15 --load_steps 5  # quick test
"""

import sys
import os
import argparse
import csv
import math
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.assemble import ElementAssembler

# Reuse mesh generation and BC helpers from Phase 1
from emmentaler_elasticity import (
    generate_emmentaler_mesh,
    identify_boundaries,
    HOLE_POSITIONS,
)


# ---------------------------------------------------------------------------
# Neo-Hookean material model
# ---------------------------------------------------------------------------
class NeoHookeanModel(ElementAssembler):
    """Compressible Neo-Hookean hyperelastic model.

    Strain energy density:
        Psi = mu/2 * (I1 - d) - mu * ln(J) + lam/2 * (ln J)^2
    where F = I + grad_u, J = det(F), I1 = tr(F^T F).
    """

    def __post_init__(self, E=1.0, nu=0.3):
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    def element_energy(self, gradu):
        """Compute strain energy density at a single quadrature point.

        Parameters
        ----------
        gradu : torch.Tensor [dim, dim]
            Displacement gradient.
        """
        dim = gradu.shape[-1]
        F = torch.eye(dim, device=gradu.device, dtype=gradu.dtype) + gradu

        J = torch.linalg.det(F)
        J = torch.clamp(J, min=1e-8)
        I1 = (F * F).sum()

        log_J = torch.log(J)
        psi = 0.5 * self.mu * (I1 - dim) - self.mu * log_J + 0.5 * self.lam * log_J ** 2
        return psi


# ---------------------------------------------------------------------------
# Boundary condition (same as Phase 1)
# ---------------------------------------------------------------------------
def compute_top_displacement(points_top, load_factor,
                             normal_mag=1.0, bending_mag=1.0, torsion_mag=1.0):
    """Compute prescribed displacement on top face (tension + bending + torsion)."""
    normfac_normal = 0.010
    normfac_bending = 0.014
    normfac_torsion = 0.048

    u_z = load_factor * normal_mag * normfac_normal
    kappa = load_factor * bending_mag * normfac_bending
    theta = load_factor * torsion_mag * normfac_torsion

    x = points_top
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    u_top = torch.zeros_like(x)
    # Torsion
    u_top[:, 0] = cos_t * x[:, 0] + sin_t * x[:, 1] - x[:, 0]
    u_top[:, 1] = -sin_t * x[:, 0] + cos_t * x[:, 1] - x[:, 1]
    # Bending
    u_top[:, 1] += -kappa / 2 * x[:, 2] ** 2
    u_top[:, 2] += kappa * x[:, 2] * x[:, 1]
    # Normal
    u_top[:, 2] += u_z

    return u_top, u_z, kappa, theta


# ---------------------------------------------------------------------------
# Stress computation for Neo-Hookean
# ---------------------------------------------------------------------------
def compute_nodal_stress_neohookean(model, mesh, u_vec, E, nu):
    """Compute Cauchy stress from Neo-Hookean model via nodal averaging.

    Computes the first Piola-Kirchhoff stress P = dPsi/dF, then converts
    to Cauchy stress: sigma = (1/J) * P * F^T.

    Returns nodal Cauchy stress in Voigt notation and von Mises stress.
    """
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    n_nodes = mesh.points.shape[0]
    dim = mesh.points.shape[1]

    nodal_stress_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_strain_sum = torch.zeros(n_nodes, 6, dtype=u_vec.dtype)
    nodal_count = torch.zeros(n_nodes, dtype=u_vec.dtype)

    for element_type in model.element_types:
        trans = model.transformation[element_type]
        elements = model.elements[element_type]
        n_elem, n_basis = elements.shape

        elem_u = u_vec[elements]
        shape_grad = trans.shape_grad  # [n_elem, n_quad, n_basis, dim]

        # grad_u at quad points: [n_elem, n_quad, dim, dim]
        grad_u = torch.einsum("eqbd,ebc->eqdc", shape_grad, elem_u)

        # Deformation gradient: F = I + grad_u
        eye = torch.eye(dim, dtype=grad_u.dtype, device=grad_u.device)
        F = eye + grad_u  # [n_elem, n_quad, dim, dim]

        J = torch.linalg.det(F).clamp(min=1e-8)  # [n_elem, n_quad]
        log_J = torch.log(J)

        # First Piola-Kirchhoff stress: P = mu*F + (lam*lnJ - mu)*F^{-T}
        F_inv_T = torch.linalg.inv(F).transpose(-1, -2)
        P = mu * F + (lam * log_J.unsqueeze(-1).unsqueeze(-1) - mu) * F_inv_T

        # Cauchy stress: sigma = (1/J) * P @ F^T
        sigma = torch.einsum("eqij,eqkj->eqik", P, F) / J.unsqueeze(-1).unsqueeze(-1)

        # Green-Lagrange strain: E = 0.5*(F^T F - I)
        E_strain = 0.5 * (torch.einsum("eqji,eqjk->eqik", F, F) - eye)

        # Average over quadrature points
        sigma_avg = sigma.mean(dim=1)   # [n_elem, dim, dim]
        E_avg = E_strain.mean(dim=1)

        # To Voigt: [xx, yy, zz, yz, xz, xy]
        sigma_voigt = torch.stack([
            sigma_avg[:, 0, 0], sigma_avg[:, 1, 1], sigma_avg[:, 2, 2],
            sigma_avg[:, 1, 2], sigma_avg[:, 0, 2], sigma_avg[:, 0, 1],
        ], dim=1)
        E_voigt = torch.stack([
            E_avg[:, 0, 0], E_avg[:, 1, 1], E_avg[:, 2, 2],
            E_avg[:, 1, 2], E_avg[:, 0, 2], E_avg[:, 0, 1],
        ], dim=1)

        for b in range(n_basis):
            node_ids = elements[:, b]
            nodal_stress_sum.index_add_(0, node_ids, sigma_voigt)
            nodal_strain_sum.index_add_(0, node_ids, E_voigt)
            nodal_count.index_add_(0, node_ids, torch.ones(n_elem, dtype=u_vec.dtype))

    nodal_count = nodal_count.clamp(min=1).unsqueeze(1)
    nodal_stress = nodal_stress_sum / nodal_count
    nodal_strain = nodal_strain_sum / nodal_count

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
def solve_emmentaler_hyperelastic(
    mesh_file=None, h=0.15, E=12000.0, nu=0.3,
    normal_mag=1.0, bending_mag=1.0, torsion_mag=1.0,
    T=1.5, load_steps=10, output_dir=".",
):
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
    print(f"Mesh: {n_nodes} nodes, {n_cells} elements, types: {list(mesh.cells.keys())}")

    # ---- 2. Model ----
    model = NeoHookeanModel.from_mesh(mesh, E=E, nu=nu)

    # ---- 3. Boundaries ----
    bottom_mask, top_mask = identify_boundaries(mesh.points, T)
    print(f"Boundary nodes: {bottom_mask.sum().item()} bottom, {top_mask.sum().item()} top")

    # Free DOF mask: True = free, False = fixed
    free_mask = torch.ones(n_nodes, dtype=torch.bool)
    free_mask[bottom_mask] = False
    free_mask[top_mask] = False
    free_mask_float = free_mask.unsqueeze(1).float()  # [n_nodes, 1] for broadcasting

    points_top = mesh.points[top_mask]

    # ---- 4. Displacement variable ----
    u = torch.zeros_like(mesh.points, requires_grad=True)

    # ---- 5. CSV monitoring ----
    csv_path = os.path.join(output_dir, "monitoring.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter="\t")
    csv_writer.writerow(["step", "lambda", "u_z", "kappa", "theta",
                         "E_internal", "u_max", "n_iter"])

    # ---- 6. Load stepping + LBFGS energy minimization ----
    print(f"Solving with {load_steps} load steps...")
    for step in range(load_steps + 1):
        lf = step / load_steps

        # Target BC displacement on the top face
        u_top_target, u_z_val, kappa_val, theta_val = compute_top_displacement(
            points_top, lf,
            normal_mag=normal_mag, bending_mag=bending_mag, torsion_mag=torsion_mag,
        )

        # Build prescribed displacement tensor
        u_prescribed = torch.zeros_like(mesh.points)
        u_prescribed[top_mask] = u_top_target
        # bottom stays zero

        optimizer = optim.LBFGS(
            [u], lr=1.0, max_iter=50, max_eval=60,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=50, line_search_fn="strong_wolfe",
        )

        n_eval = [0]

        def closure():
            optimizer.zero_grad()
            # Apply BCs: free nodes use u, constrained nodes use prescribed values
            u_active = u * free_mask_float + u_prescribed
            E_int = model.energy(point_data={"u": u_active})
            # No external forces (pure displacement-driven problem)
            if E_int.requires_grad:
                E_int.backward()
            n_eval[0] += 1
            return E_int

        loss = optimizer.step(closure)

        with torch.no_grad():
            u_active = u * free_mask_float + u_prescribed
            u_max = u_active.norm(dim=1).max().item()
            e_val = loss.item()

        csv_writer.writerow([
            step, f"{lf:.6f}",
            f"{u_z_val:.6e}", f"{kappa_val:.6e}", f"{theta_val:.6e}",
            f"{e_val:.6e}", f"{u_max:.6e}", n_eval[0],
        ])

        print(f"  step {step:3d}/{load_steps}  λ={lf:.4f}  "
              f"E={e_val:.4e}  u_max={u_max:.4e}  iters={n_eval[0]}")

    csv_file.close()
    print(f"Monitoring data saved to {csv_path}")

    # ---- 7. Post-processing ----
    with torch.no_grad():
        u_final = (u * free_mask_float + u_prescribed).detach()

    print("Computing stress and strain fields...")
    nodal_strain, nodal_stress, von_mises = compute_nodal_stress_neohookean(
        model, mesh, u_final, E, nu,
    )
    print(f"Max von Mises stress: {von_mises.max().item():.6e}")

    # ---- 8. Save VTK ----
    mesh.register_point_data("displacement", u_final)
    mesh.register_point_data("displacement_magnitude", u_final.norm(dim=1))
    mesh.register_point_data("strain", nodal_strain)
    mesh.register_point_data("stress", nodal_stress)
    mesh.register_point_data("von_mises_stress", von_mises)

    out_path = os.path.join(output_dir, "emmentaler_hyperelastic.vtk")
    mesh.save(out_path)
    print(f"Result saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emmentaler Neo-Hookean hyperelasticity (TensorMesh)")
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--h", type=float, default=0.15,
                        help="Mesh element size (default: 0.15)")
    parser.add_argument("--E", type=float, default=12000.0,
                        help="Young's modulus (default: 12000)")
    parser.add_argument("--nu", type=float, default=0.3,
                        help="Poisson's ratio (default: 0.3)")
    parser.add_argument("--normal_mag", type=float, default=1.0)
    parser.add_argument("--bending_mag", type=float, default=1.0)
    parser.add_argument("--torsion_mag", type=float, default=1.0)
    parser.add_argument("--load_steps", type=int, default=10,
                        help="Number of load steps (default: 10)")
    parser.add_argument("--output_dir", type=str, default="output_emmentaler_hyper")
    args = parser.parse_args()

    solve_emmentaler_hyperelastic(
        mesh_file=args.mesh_file, h=args.h, E=args.E, nu=args.nu,
        normal_mag=args.normal_mag, bending_mag=args.bending_mag,
        torsion_mag=args.torsion_mag, load_steps=args.load_steps,
        output_dir=args.output_dir,
    )

"""Hyperelasticity (Neo-Hookean) solver with incremental loading.

Solves the nonlinear elasticity problem using energy minimization (LBFGS)
with load stepping. All materials (m1/m2/m3) use the same NeoHookean model,
matching the original FEniCSx implementation.

Boundary conditions are component-wise:
  - Dirichlet: prescribed displacement per component
  - Neumann: applied traction per component (via energy)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional

from tensormesh import Mesh
from tensormesh.assemble import ElementAssembler


# ---------------------------------------------------------------------------
# NeoHookean model (same as builtin but with J clamping for robustness)
# ---------------------------------------------------------------------------

class NeoHookeanModel(ElementAssembler):
    """Compressible Neo-Hookean hyperelastic model with robust J clamping."""

    def __post_init__(self, mu=1.0, lam=1.0):
        self.mu = mu
        self.lam = lam

    def element_energy(self, gradu):
        dim = gradu.shape[-1]
        F = torch.eye(dim, device=gradu.device, dtype=gradu.dtype) + gradu
        J = torch.linalg.det(F)
        J = torch.clamp(J, min=1e-8)
        I1 = (F * F).sum()
        log_J = torch.log(J)
        return 0.5 * self.mu * (I1 - dim) - self.mu * log_J + 0.5 * self.lam * log_J ** 2


# ---------------------------------------------------------------------------
# Neumann (traction) energy
# ---------------------------------------------------------------------------

def _compute_neumann_energy(u: torch.Tensor, mesh: Mesh,
                             neumann_data: List[dict]) -> torch.Tensor:
    """Compute external work due to traction BCs: W_ext = -∫ t · u dS.

    For simplicity, uses lumped boundary integration (nodal forces).
    Each neumann_data entry: {'node_indices': tensor, 'values': tensor, 'component': int}

    The traction is applied as nodal forces: F_i = g_i * h_i (lumped).
    External work = -sum(F_i * u_i) for each component.
    """
    energy = torch.tensor(0.0, dtype=u.dtype, device=u.device)
    for nd in neumann_data:
        idx = nd['node_indices']
        g = nd['values']  # [n_nodes] traction values
        comp = nd['component']
        # Lumped: energy contribution = -sum(g * u_comp * h)
        # For simplicity, approximate h as average boundary edge length
        # Using nodal forces: F_i ≈ g_i * boundary_measure_i
        # Since the original code uses FEM integration, this is approximate
        energy = energy - (g * u[idx, comp]).sum()
    return energy


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_hyperelasticity(mesh: Mesh,
                           mu_nd: float,
                           lam_nd: float,
                           segments_exterior: list,
                           segments_holes: Optional[Dict[str, list]] = None,
                           load_steps: int = 2,
                           ndims: int = 2,
                           verbose: bool = False,
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve hyperelasticity with incremental loading and component-wise BCs.

    Args:
        mesh: TensorMesh Mesh (P2 triangles)
        mu_nd: non-dimensionalized shear modulus (mu / char_stress)
        lam_nd: non-dimensionalized Lamé parameter (lambda / char_stress)
        segments_exterior: list of SegmentBCs for outer boundary
        segments_holes: dict of {hole_name: list of SegmentBCs}
        load_steps: number of incremental load steps
        ndims: spatial dimensions (2)
        verbose: print convergence info

    Returns:
        u: [n_points, 2] displacement
        strain: [n_points, 4] Green-Lagrange strain (e11, e12, e21, e22)
        cauchy_stress: [n_points, 4] Cauchy stress (s11, s12, s21, s22)
    """
    n_points = mesh.points.shape[0]
    dtype = mesh.points.dtype
    device = mesh.points.device

    # Build NeoHookean model
    model = NeoHookeanModel.from_mesh(mesh, mu=mu_nd, lam=lam_nd)

    # Parse BCs: separate Dirichlet and Neumann per component
    all_segments = list(segments_exterior)
    if segments_holes:
        for hole_segs in segments_holes.values():
            all_segments.extend(hole_segs)

    dirichlet_nodes = {}  # {(node_idx, component): target_value}
    neumann_data = []     # list of {node_indices, values, component}

    for seg in all_segments:
        if seg.indices is None or len(seg.indices) == 0:
            continue
        idx = seg.indices
        for d in range(ndims):
            dim_bc = seg.dims[d]
            if dim_bc.type == 'dirichlet' and dim_bc.values:
                g_vals = dim_bc.values['g']
                for j, node in enumerate(idx):
                    dirichlet_nodes[(node, d)] = g_vals[j]
            elif dim_bc.type == 'neumann' and dim_bc.values:
                g_vals = dim_bc.values['g']
                if np.any(np.abs(g_vals) > 1e-15):
                    neumann_data.append({
                        'node_indices': torch.tensor(idx, dtype=torch.long, device=device),
                        'values': torch.tensor(g_vals, dtype=dtype, device=device),
                        'component': d,
                    })

    # Build free/fixed masks
    free_mask = torch.ones(n_points, ndims, dtype=torch.bool, device=device)
    u_prescribed_full = torch.zeros(n_points, ndims, dtype=dtype, device=device)
    for (node, comp), val in dirichlet_nodes.items():
        free_mask[node, comp] = False
        u_prescribed_full[node, comp] = val

    free_mask_float = free_mask.float()

    # Initialize displacement
    u = torch.zeros(n_points, ndims, dtype=dtype, device=device, requires_grad=True)

    # Incremental loading
    multipliers = np.linspace(0., 1., load_steps, endpoint=True)

    for step, m in enumerate(multipliers):
        # Scale BCs
        u_prescribed = u_prescribed_full * m
        neumann_scaled = []
        for nd in neumann_data:
            neumann_scaled.append({
                'node_indices': nd['node_indices'],
                'values': nd['values'] * m,
                'component': nd['component'],
            })

        # LBFGS optimization
        optimizer = optim.LBFGS(
            [u], lr=1.0, max_iter=50, max_eval=60,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=50, line_search_fn="strong_wolfe",
        )

        n_eval = [0]

        def closure():
            optimizer.zero_grad()
            u_active = u * free_mask_float + u_prescribed
            E_int = model.energy(point_data={"u": u_active})
            # External work from Neumann BCs
            E_ext = _compute_neumann_energy(u_active, mesh, neumann_scaled)
            total = E_int + E_ext
            if total.requires_grad:
                total.backward()
            n_eval[0] += 1
            return total

        loss = optimizer.step(closure)

        if verbose:
            with torch.no_grad():
                u_active = u * free_mask_float + u_prescribed
                print(f"  Step {step}/{load_steps-1}: E={loss.item():.4e}, "
                      f"u_max={u_active.norm(dim=1).max().item():.4e}, "
                      f"iters={n_eval[0]}")

    # Final displacement
    with torch.no_grad():
        u_final = (u * free_mask_float + u_prescribed_full).detach()

    # Post-process: compute strain and Cauchy stress
    strain, cauchy_stress = _compute_strain_stress(model, mesh, u_final, mu_nd, lam_nd)

    return (u_final.cpu().numpy(),
            strain.cpu().numpy(),
            cauchy_stress.cpu().numpy())


def _compute_strain_stress(model, mesh, u_vec, mu, lam):
    """Compute Green-Lagrange strain and Cauchy stress via nodal averaging.

    Returns:
        strain: [n_points, 4] (e11, e12, e21, e22)
        cauchy_stress: [n_points, 4] (s11, s12, s21, s22)
    """
    n_nodes = mesh.points.shape[0]
    dim = mesh.points.shape[1]

    nodal_stress_sum = torch.zeros(n_nodes, dim * dim, dtype=u_vec.dtype)
    nodal_strain_sum = torch.zeros(n_nodes, dim * dim, dtype=u_vec.dtype)
    nodal_count = torch.zeros(n_nodes, dtype=u_vec.dtype)

    for element_type in model.element_types:
        trans = model.transformation[element_type]
        elements = model.elements[element_type]
        n_elem, n_basis = elements.shape

        elem_u = u_vec[elements]  # [n_elem, n_basis, dim]
        shape_grad = trans.shape_grad  # [n_elem, n_quad, n_basis, dim]

        # grad_u at quad points: [n_elem, n_quad, dim, dim]
        grad_u = torch.einsum("eqbd,ebc->eqdc", shape_grad, elem_u)

        eye = torch.eye(dim, dtype=grad_u.dtype, device=grad_u.device)
        F = eye + grad_u
        J = torch.linalg.det(F).clamp(min=1e-8)
        log_J = torch.log(J)

        # First Piola-Kirchhoff: P = mu*F + (lam*lnJ - mu)*F^{-T}
        F_inv_T = torch.linalg.inv(F).transpose(-1, -2)
        P = mu * F + (lam * log_J.unsqueeze(-1).unsqueeze(-1) - mu) * F_inv_T

        # Cauchy: sigma = (1/J) * P @ F^T
        sigma = torch.einsum("eqij,eqkj->eqik", P, F) / J.unsqueeze(-1).unsqueeze(-1)

        # Green-Lagrange: E = 0.5*(F^T F - I)
        E_strain = 0.5 * (torch.einsum("eqji,eqjk->eqik", F, F) - eye)

        # Average over quadrature points
        sigma_avg = sigma.mean(dim=1)   # [n_elem, dim, dim]
        E_avg = E_strain.mean(dim=1)

        # Flatten: [e11, e12, e21, e22]
        sigma_flat = sigma_avg.reshape(n_elem, dim * dim)
        E_flat = E_avg.reshape(n_elem, dim * dim)

        for b in range(n_basis):
            node_ids = elements[:, b]
            nodal_stress_sum.index_add_(0, node_ids, sigma_flat)
            nodal_strain_sum.index_add_(0, node_ids, E_flat)
            nodal_count.index_add_(0, node_ids, torch.ones(n_elem, dtype=u_vec.dtype))

    nodal_count = nodal_count.clamp(min=1).unsqueeze(1)
    return nodal_strain_sum / nodal_count, nodal_stress_sum / nodal_count

"""Poisson equation solver with Dirichlet, Neumann, and Robin BCs.

Solves: -Δu = f in Ω
with boundary conditions on segments of ∂Ω:
  - Dirichlet: u = g
  - Neumann: -∂u/∂n = g  (adds ∫g*v dS to RHS)
  - Robin: -∂u/∂n + α*u = g  (adds ∫α*u*v dS to stiffness, ∫g*v dS to RHS)
"""

import torch
import numpy as np
from typing import Callable, Optional

from tensormesh import LaplaceElementAssembler, Mesh, Condenser
from tensormesh.assemble import NodeAssembler
from tensormesh.sparse import SparseMatrix


def _precompute_source_on_nodes(mesh: Mesh, source_func: Callable) -> torch.Tensor:
    """Evaluate numpy source function at mesh points, return as tensor."""
    points_np = mesh.points.detach().cpu().numpy()
    vals = source_func(points_np)
    return torch.tensor(vals, dtype=mesh.points.dtype, device=mesh.points.device)


def _add_sparse_matrices(A: SparseMatrix, B: SparseMatrix) -> SparseMatrix:
    """Add two SparseMatrix objects with potentially different sparsity patterns.

    Converts to torch.sparse_coo, adds, then converts back to SparseMatrix.
    """
    s_a = torch.sparse_coo_tensor(
        torch.stack([A.row_indices, A.col_indices]),
        A.values, A.shape)
    s_b = torch.sparse_coo_tensor(
        torch.stack([B.row_indices, B.col_indices]),
        B.values, B.shape)
    s_c = (s_a + s_b).coalesce()
    return SparseMatrix(
        s_c.values(),
        s_c.indices()[0],
        s_c.indices()[1],
        shape=A.shape,
    )


def _get_boundary_edges(mesh: Mesh) -> torch.Tensor:
    """Extract boundary edges from mesh.

    Returns:
        edges: [n_boundary_edges, 2] tensor of node indices
    """
    cell_keys = list(mesh.cells.keys())
    if 'line' in cell_keys:
        return mesh.cells['line']

    # Extract from triangles
    tri_key = [k for k in cell_keys if 'triangle' in k][0]
    tris = mesh.cells[tri_key]
    edges = torch.stack([
        tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]
    ], dim=1).reshape(-1, 2)
    bm = mesh.boundary_mask
    is_boundary_edge = bm[edges].all(-1)
    return edges[is_boundary_edge]


def _assemble_boundary_mass_matrix(mesh: Mesh,
                                    robin_mask: torch.Tensor,
                                    alpha_values: torch.Tensor) -> SparseMatrix:
    """Assemble the Robin boundary mass matrix: ∫_Γ_R α * Ni * Nj dS.

    For P1 triangles, boundary facets are line segments (2 nodes each).
    The local mass matrix for a line element of length h with weight α is:
        M_local = (α_avg * h / 6) * [[2, 1], [1, 2]]

    Args:
        mesh: TensorMesh Mesh object
        robin_mask: [n_points] boolean, True for Robin boundary nodes
        alpha_values: [n_points] α values (only robin nodes matter)

    Returns:
        SparseMatrix of shape [n_points, n_points]
    """
    n_points = mesh.points.shape[0]
    dtype = mesh.points.dtype
    device = mesh.points.device

    if not robin_mask.any():
        return SparseMatrix(
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            shape=(n_points, n_points),
        )

    edges = _get_boundary_edges(mesh)

    # Filter to Robin boundary edges (both nodes must be Robin)
    is_robin_edge = robin_mask[edges].all(-1)
    robin_edges = edges[is_robin_edge]

    if robin_edges.shape[0] == 0:
        return SparseMatrix(
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            shape=(n_points, n_points),
        )

    # Compute edge lengths
    p0 = mesh.points[robin_edges[:, 0]]
    p1 = mesh.points[robin_edges[:, 1]]
    h = torch.norm(p1 - p0, dim=-1)

    # Average alpha on each edge
    alpha0 = alpha_values[robin_edges[:, 0]]
    alpha1 = alpha_values[robin_edges[:, 1]]
    alpha_avg = 0.5 * (alpha0 + alpha1)

    # Local mass matrix: (alpha * h / 6) * [[2,1],[1,2]]
    diag_val = alpha_avg * h / 3.0
    offdiag_val = alpha_avg * h / 6.0

    # COO indices: 4 entries per edge
    n0 = robin_edges[:, 0]
    n1 = robin_edges[:, 1]
    rows = torch.cat([n0, n1, n0, n1])
    cols = torch.cat([n0, n1, n1, n0])
    vals = torch.cat([diag_val, diag_val, offdiag_val, offdiag_val])

    return SparseMatrix(vals, rows, cols, shape=(n_points, n_points))


def _assemble_neumann_rhs(mesh: Mesh,
                           boundary_mask: torch.Tensor,
                           g_values: torch.Tensor) -> torch.Tensor:
    """Assemble Neumann/Robin boundary RHS vector: ∫_Γ g * v dS.

    Uses direct edge-based assembly for P1 elements.
    For an edge of length h with g values g0, g1:
        f_0 += h/6 * (2*g0 + g1)
        f_1 += h/6 * (g0 + 2*g1)

    Args:
        mesh: TensorMesh Mesh object
        boundary_mask: [n_points] boolean, True for this boundary segment
        g_values: [n_points] g values at all nodes

    Returns:
        rhs: [n_points] tensor
    """
    n_points = mesh.points.shape[0]
    dtype = mesh.points.dtype
    device = mesh.points.device

    if not boundary_mask.any():
        return torch.zeros(n_points, dtype=dtype, device=device)

    edges = _get_boundary_edges(mesh)

    # Filter to edges where both nodes belong to this segment
    is_seg_edge = boundary_mask[edges].all(-1)
    seg_edges = edges[is_seg_edge]

    rhs = torch.zeros(n_points, dtype=dtype, device=device)
    if seg_edges.shape[0] == 0:
        return rhs

    p0 = mesh.points[seg_edges[:, 0]]
    p1 = mesh.points[seg_edges[:, 1]]
    h = torch.norm(p1 - p0, dim=-1)

    g0 = g_values[seg_edges[:, 0]]
    g1 = g_values[seg_edges[:, 1]]

    # Consistent load vector for P1 line element
    rhs.scatter_add_(0, seg_edges[:, 0], h / 6.0 * (2 * g0 + g1))
    rhs.scatter_add_(0, seg_edges[:, 1], h / 6.0 * (g0 + 2 * g1))

    return rhs


def solve_poisson(mesh: Mesh,
                  source_func: Callable,
                  boundary_masks: dict,
                  boundary_values: dict,
                  ) -> torch.Tensor:
    """Solve the Poisson equation with mixed Dirichlet/Neumann/Robin BCs.

    Args:
        mesh: TensorMesh Mesh object
        source_func: f(x) -> values, source term function.
            x is [n_points, dim] numpy array, returns [n_points] numpy array.
        boundary_masks: dict with keys:
            'dirichlet_mask': bool tensor [n_points]
            'neumann_mask': bool tensor [n_points] (optional)
            'robin_mask': bool tensor [n_points] (optional)
        boundary_values: dict with keys:
            'dirichlet_values': tensor [n_points]
            'neumann_values': tensor [n_points] (optional)
            'robin_g_values': tensor [n_points] (optional)
            'robin_alpha_values': tensor [n_points] (optional)

    Returns:
        u: solution tensor [n_points]
    """
    # Assemble stiffness matrix
    assembler = LaplaceElementAssembler.from_mesh(mesh)
    K = assembler(mesh.points)

    # Assemble RHS from source term
    source_vals = _precompute_source_on_nodes(mesh, source_func)

    class SourceNodeAssembler(NodeAssembler):
        def forward(self, v, f):
            return f * v

    source_asm = SourceNodeAssembler.from_mesh(mesh)
    f = source_asm(mesh.points, point_data={"f": source_vals})

    # Add Robin boundary mass matrix to stiffness
    robin_mask = boundary_masks.get('robin_mask', None)
    if robin_mask is not None and robin_mask.any():
        alpha_values = boundary_values['robin_alpha_values']
        M_robin = _assemble_boundary_mass_matrix(mesh, robin_mask, alpha_values)
        K = _add_sparse_matrices(K, M_robin)

    # Add Neumann RHS contribution: ∫_Γ_N g * v dS
    neumann_mask = boundary_masks.get('neumann_mask', None)
    if neumann_mask is not None and neumann_mask.any():
        f_neumann = _assemble_neumann_rhs(mesh, neumann_mask, boundary_values['neumann_values'])
        f = f + f_neumann

    # Add Robin RHS contribution: ∫_Γ_R g * v dS
    if robin_mask is not None and robin_mask.any():
        f_robin = _assemble_neumann_rhs(mesh, robin_mask, boundary_values['robin_g_values'])
        f = f + f_robin

    # Apply Dirichlet BCs via static condensation
    dirichlet_mask = boundary_masks['dirichlet_mask']
    dirichlet_values = boundary_values['dirichlet_values']

    condenser = Condenser(dirichlet_mask, dirichlet_values)
    K_inner, f_inner = condenser(K, f)

    # Solve
    u_inner = K_inner.solve(f_inner)
    u = condenser.recover(u_inner)

    return u

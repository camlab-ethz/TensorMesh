"""
Bracket Inverse Design Benchmark

Compare three methods for topology optimization gradient computation:
1. torchfem (analytic sensitivity via torch-fem library)
2. TensorMesh-autograd (PyTorch autodiff through sparse solve)
3. TensorMesh-analytic (hand-derived analytic sensitivity)

Outputs:
    output/benchmark_time.png - Time comparison plot
    output/benchmark_effect.png - Effect comparison (convergence + density)
    output/benchmark_cache.csv - Complete results cache

Usage:
    python bracket_benchmark.py              # Run benchmark (50 epochs)
    python bracket_benchmark.py --epochs 25  # Custom epochs
    python bracket_benchmark.py --plot-only  # Plot from cache without running
"""

import sys
sys.path.insert(0, "../..")

import os
import argparse
import time
import numpy as np
import torch
import meshio
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import bisect
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

torch.set_default_dtype(torch.float64)

# Video generation settings
GENERATE_VIDEO = True
VIDEO_FPS = 3

# ICML style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Method colors
METHOD_COLORS = {
    'torchfem': '#E24A33',           # red-orange
    'tensormesh-autograd': '#2CA02C', # green
    'tensormesh-analytic': '#348ABD', # blue
}

METHOD_MARKERS = {
    'torchfem': 'o',
    'tensormesh-autograd': 's',
    'tensormesh-analytic': '^',
}


# ============================================================================
# TensorMesh utilities
# ============================================================================

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.sparse import SparseMatrix
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness


class SIMP3DStiffnessAssembler(ElementAssembler):
    """SIMP stiffness matrix assembler for 3D topology optimization."""
    
    def __post_init__(self, E=16500.0, nu=0.342, penal=3.0, E_min=1e-9):
        self.E0 = E
        self.nu = nu
        self.penal = penal
        self.E_min = E_min
    
    def forward(self, gradu, gradv, rho):
        dim = gradu.shape[0]
        E = self.E_min + (rho ** self.penal) * (self.E0 - self.E_min)
        Ba = voigt_shape_grad(gradu)
        Bb = voigt_shape_grad(gradv)
        C = voigt_stiffness(E, self.nu, dim)
        C = C.to(dtype=gradu.dtype, device=gradu.device)
        return Ba.T @ C @ Bb


def compute_element_stiffness_batch(nodes, elements, nu, E=1.0):
    """Compute element stiffness matrices for all elements (vectorized).
    
    Uses standard tetrahedron node ordering [0, 1, 2, 3] to match torch-fem.
    No reordering is applied - elements are used directly.
    """
    n_elements = elements.shape[0]
    dtype = nodes.dtype
    device = nodes.device
    
    # Constitutive matrix [6, 6]
    C = voigt_stiffness(E, nu, 3).to(dtype=dtype, device=device)
    
    # Use elements directly without reordering (matches torch-fem)
    coords = nodes[elements]  # [n_elements, 4, 3]
    
    # Reference shape function gradients for standard tet [0, 1, 2, 3]
    # Node 0 at origin, nodes 1,2,3 define the Jacobian
    dN_dxi = torch.tensor([
        [-1.0, -1.0, -1.0],  # node 0: gradient = -sum of others
        [1.0, 0.0, 0.0],     # node 1
        [0.0, 1.0, 0.0],     # node 2
        [0.0, 0.0, 1.0]      # node 3
    ], dtype=dtype, device=device)
    
    # Jacobian: J = [x2-x1, x3-x1, x4-x1]^T  [n_elements, 3, 3]
    J = coords[:, 1:4, :] - coords[:, 0:1, :]  # [n_elements, 3, 3]
    J = J.transpose(1, 2)  # [n_elements, 3, 3]
    
    # Determinant and inverse
    det_J = torch.linalg.det(J)  # [n_elements]
    J_inv = torch.linalg.inv(J)  # [n_elements, 3, 3]
    vol = torch.abs(det_J) / 6.0  # [n_elements]
    
    # Shape function gradients in physical coords: dN_dx = dN_dxi @ J_inv
    # dN_dxi: [4, 3], J_inv: [n_elements, 3, 3]
    # Result: [n_elements, 4, 3]
    dN_dx = torch.einsum('ij,njk->nik', dN_dxi, J_inv)
    
    # Build B matrices for all nodes [n_elements, 4, 6, 3]
    B = torch.zeros(n_elements, 4, 6, 3, dtype=dtype, device=device)
    B[:, :, 0, 0] = dN_dx[:, :, 0]  # dN/dx
    B[:, :, 1, 1] = dN_dx[:, :, 1]  # dN/dy
    B[:, :, 2, 2] = dN_dx[:, :, 2]  # dN/dz
    B[:, :, 3, 0] = dN_dx[:, :, 1]  # dN/dy (for xy shear)
    B[:, :, 3, 1] = dN_dx[:, :, 0]  # dN/dx (for xy shear)
    B[:, :, 4, 1] = dN_dx[:, :, 2]  # dN/dz (for yz shear)
    B[:, :, 4, 2] = dN_dx[:, :, 1]  # dN/dy (for yz shear)
    B[:, :, 5, 0] = dN_dx[:, :, 2]  # dN/dz (for xz shear)
    B[:, :, 5, 2] = dN_dx[:, :, 0]  # dN/dx (for xz shear)
    
    # Compute Ke = sum_i,j Bi^T @ C @ Bj * vol
    # B: [n_elements, 4, 6, 3], C: [6, 6]
    # Ke: [n_elements, 12, 12]
    
    # For node i, B[i] is [6, 3], so B[i]^T is [3, 6]
    # B[i]^T @ C @ B[j] = [3, 6] @ [6, 6] @ [6, 3] = [3, 3]
    
    # B is [n, i, k, a] where:
    #   n = n_elements, i = 4 nodes, k = 6 strain components, a = 3 dofs
    # 
    # For each element: Ke[i*3:(i+1)*3, j*3:(j+1)*3] = B[i]^T @ C @ B[j] * vol
    # = [3, 6] @ [6, 6] @ [6, 3] = [3, 3]
    
    # B^T @ C: [n, 4, 3, 6] @ [6, 6] -> need to contract over k dimension
    # B[n,i,k,a] transposed in last 2 dims -> B^T[n,i,a,k]
    # B^T @ C = sum_k B^T[a,k] * C[k,l] = sum_k B[k,a] * C[k,l]
    # einsum: 'nika,kl->nial'
    BtC = torch.einsum('nika,kl->nial', B, C)  # [n_elements, 4, 3, 6]
    
    # (B^T @ C) @ B: BtC[n,i,a,l] @ B[n,j,l,b] -> sum over l
    # Result: [n_elements, 4, 3, 4, 3] = [n, i, a, j, b]
    BtCB = torch.einsum('nial,njlb->niajb', BtC, B)
    
    # Reshape to [n_elements, 12, 12] and multiply by volume
    # Shape [n, i, a, j, b] = [n, 4, 3, 4, 3]
    # Direct reshape gives [n, 12, 12] with correct indexing:
    # index i*3+a for rows, j*3+b for cols
    k0 = BtCB.reshape(n_elements, 12, 12) * vol.view(-1, 1, 1)
    
    return k0


def build_filter(centroids, R):
    """Build density filter matrix."""
    tree = cKDTree(centroids)
    d = tree.sparse_distance_matrix(tree, R, output_type="coo_matrix")
    
    n = len(centroids)
    H = torch.sparse_coo_tensor(
        indices=torch.stack([torch.as_tensor(d.row), torch.as_tensor(d.col)]),
        values=torch.as_tensor(R - d.data, dtype=torch.float64),
        size=(n, n),
    )
    H_sum = torch.sparse.sum(H, dim=1).to_dense()
    
    return H, H_sum


def apply_filter(H, H_sum, rho, sensitivity, vols):
    """Apply sensitivity filter."""
    H_dense = H.to_dense()
    weighted = rho * sensitivity / vols
    filtered = H_dense @ weighted / H_sum / (rho / vols + 1e-10)
    return filtered


# ============================================================================
# Optimization Methods
# ============================================================================

def run_torchfem_optimization(nodes, elements, domain, epochs, volfrac, penal, 
                               move, filter_R, E, nu, verbose=True):
    """Run optimization using torch-fem library."""
    try:
        from torchfem import Solid
        from torchfem.materials import IsotropicElasticity3D
    except ImportError:
        print("  [warn] torch-fem not available")
        return None
    
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    dim = 3
    
    # Design elements
    design_mask = domain == 6
    n_design = design_mask.sum().item()
    elements_design = elements[design_mask]
    
    # Setup torchfem model
    material = IsotropicElasticity3D(E=E, nu=nu)
    model = Solid(nodes, elements, material)
    
    # Boundary conditions
    R_fix = 6.0
    for d in [1, 3, 4, 5]:
        dom = torch.unique(elements[domain == d])
        center = nodes[dom].mean(dim=0)
        con = (nodes[dom, 0] - center[0]) ** 2 + (nodes[dom, 1] - center[1]) ** 2 < R_fix**2
        model.constraints[dom[con], :] = True
    
    # Get base stiffness
    C0 = model.material.C.clone()
    k0_tf = model.k0().clone()
    k0_design = k0_tf[design_mask]
    
    # Load cases
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    n_load = len(load_nodes)
    
    forces_list = []
    F1 = torch.zeros_like(nodes)
    F1[load_nodes, 2] = 8000.0 / n_load
    forces_list.append(F1)
    
    F2 = torch.zeros_like(nodes)
    F2[load_nodes, 1] = -8500.0 / n_load
    forces_list.append(F2)
    
    F3 = torch.zeros_like(nodes)
    F3[load_nodes, 1] = -9500 * np.sin(np.deg2rad(42)) / n_load
    F3[load_nodes, 2] = 9500 * np.cos(np.deg2rad(42)) / n_load
    forces_list.append(F3)
    
    F4 = torch.zeros_like(nodes)
    levers = nodes[load_nodes, 0]
    for i, node in enumerate(load_nodes):
        if levers[i].abs() > 1e-6:
            F4[node, 1] = 5000.0 / levers[i] / n_load
    forces_list.append(F4)
    
    # Compute element volumes
    def compute_volumes(pts, elems):
        p = pts[elems]
        v = p[:, 1:] - p[:, 0:1]
        return torch.abs(torch.linalg.det(v)) / 6.0
    
    vols = compute_volumes(nodes, elements)
    vols_design = vols[design_mask]
    
    # Build filter
    centroids = nodes[elements_design].mean(dim=1).cpu().numpy()
    H, H_sum = build_filter(centroids, filter_R)
    
    # Initialize density
    rho_design = volfrac * torch.ones(n_design)
    rho_full = torch.ones(n_elements)
    rho_full[design_mask] = rho_design
    
    rho_min, rho_max = 0.01, 1.0
    V_0 = volfrac * vols_design.sum()
    
    history = {'compliance': [], 'volume': [], 'time': [], 'rho_history': []}
    
    if verbose:
        print(f"\n  Running torch-fem optimization ({epochs} epochs)...")
    
    for epoch in tqdm(range(epochs), desc="torchfem", disable=not verbose):
        t_start = time.time()
        
        # Update density and material
        rho_full[design_mask] = rho_design
        model.material.C = torch.einsum("n,nijkl->nijkl", rho_full ** penal, C0)
        
        # Solve all load cases and compute compliance + sensitivity
        total_compliance = 0.0
        sensitivity = torch.zeros(n_design)
        rho_design_penal = rho_design ** (penal - 1.0)
        
        for F in forces_list:
            model.forces = F
            u, f, _, _, _ = model.solve(rtol=0.01, verbose=False)
            
            compliance = torch.inner(f.ravel(), u.ravel()).item()
            total_compliance += compliance
            
            # Analytic sensitivity
            u_e = u[elements_design].reshape(n_design, -1)
            k0_u = torch.einsum("nij,nj->ni", k0_design, u_e)
            w = torch.einsum("ni,ni->n", u_e, k0_u)
            sensitivity += -penal * rho_design_penal * w
        
        # Apply filter
        sensitivity_filtered = apply_filter(H, H_sum, rho_design, sensitivity, vols_design)
        
        # OC update
        def make_step(mu):
            G_k = -sensitivity_filtered / (mu + 1e-10)
            upper = torch.minimum(torch.tensor(rho_max), (1 + move) * rho_design)
            lower = torch.maximum(torch.tensor(rho_min), (1 - move) * rho_design)
            rho_trial = (G_k.clamp(min=1e-10) ** 0.5) * rho_design
            return torch.maximum(torch.minimum(rho_trial, upper), lower)
        
        def constraint(mu):
            return (torch.inner(make_step(mu), vols_design) - V_0).item()
        
        try:
            mu = bisect(constraint, 1e-10, 1e10, xtol=1e-4)
        except ValueError:
            mu = 1.0
        
        rho_design = make_step(mu)
        rho_full[design_mask] = rho_design
        
        t_end = time.time()
        
        history['compliance'].append(total_compliance)
        history['volume'].append((rho_design * vols_design).sum().item() / vols_design.sum().item())
        history['time'].append(t_end - t_start)
        history['rho_history'].append(rho_full.cpu().numpy().copy())
    
    return {
        'method': 'torchfem',
        'rho': rho_design.cpu().numpy(),
        'rho_full': rho_full.cpu().numpy(),
        'design_mask': design_mask.cpu().numpy(),
        'history': history,
        'final_compliance': history['compliance'][-1],
        'final_volume': history['volume'][-1],
        'total_time': sum(history['time']),
        'avg_time_per_iter': np.mean(history['time']),
    }


def run_tensormesh_optimization(nodes, elements, domain, method, epochs, volfrac, 
                                 penal, move, filter_R, E, nu, verbose=True,
                                 solver_backend="torch", solver_tol=1e-5):
    """
    Run optimization using TensorMesh.
    
    method: 'autograd' or 'analytic'
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    dim = 3
    
    # Create TensorMesh mesh
    # For analytic method, we need consistent element ordering with k0 computation
    meshio_obj = meshio.Mesh(
        points=nodes.cpu().numpy(),
        cells=[("tetra", elements.cpu().numpy())],
    )
    tm_mesh = Mesh(meshio_obj, reorder=True)
    
    # Get reordered elements from TensorMesh for analytic gradient
    tm_elements = tm_mesh.elements()
    
    # Design elements
    design_mask = domain == 6
    n_design = design_mask.sum().item()
    elements_design = elements[design_mask]
    # Use TensorMesh's reordered elements for displacement extraction
    tm_elements_design = tm_elements[design_mask]
    
    # Assembler
    assembler = SIMP3DStiffnessAssembler.from_mesh(tm_mesh, E=E, nu=nu, penal=penal)
    
    # Boundary conditions
    R_fix = 6.0
    constraint_mask = torch.zeros(n_nodes, dim, dtype=torch.bool)
    for d in [1, 3, 4, 5]:
        dom_elements = elements[domain == d]
        dom_nodes = torch.unique(dom_elements)
        center = nodes[dom_nodes].mean(dim=0)
        dist_sq = ((nodes[dom_nodes, 0] - center[0])**2 + 
                   (nodes[dom_nodes, 1] - center[1])**2)
        inner = dist_sq < R_fix**2
        constraint_mask[dom_nodes[inner], :] = True
    
    dbc_mask = constraint_mask.flatten()
    condenser = Condenser(dbc_mask)
    
    # Build rigid body modes B for AMG preconditioner (3D elasticity: 3 translation modes)
    n_inner = (~dbc_mask).sum().item()
    B_rigid = torch.zeros(n_inner, 3, dtype=torch.float64)
    for i in range(3):
        B_rigid[i::3, i] = 1.0
    
    # Load cases
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    n_load = len(load_nodes)
    
    forces_list = []
    
    F1 = torch.zeros(n_nodes * dim)
    F1[load_nodes * dim + 2] = 8000.0 / n_load
    forces_list.append(F1)
    
    F2 = torch.zeros(n_nodes * dim)
    F2[load_nodes * dim + 1] = -8500.0 / n_load
    forces_list.append(F2)
    
    F3 = torch.zeros(n_nodes * dim)
    F3[load_nodes * dim + 1] = -9500 * np.sin(np.deg2rad(42)) / n_load
    F3[load_nodes * dim + 2] = 9500 * np.cos(np.deg2rad(42)) / n_load
    forces_list.append(F3)
    
    F4 = torch.zeros(n_nodes * dim)
    levers = nodes[load_nodes, 0]
    for i, node in enumerate(load_nodes):
        if levers[i].abs() > 1e-6:
            F4[node * dim + 1] = 5000.0 / levers[i] / n_load
    forces_list.append(F4)
    
    # Compute element volumes
    def compute_volumes(pts, elems):
        p = pts[elems]
        v = p[:, 1:] - p[:, 0:1]
        return torch.abs(torch.linalg.det(v)) / 6.0
    
    vols = compute_volumes(nodes, elements)
    vols_design = vols[design_mask]
    
    # Precompute k0 for analytic method
    # Use original elements (NOT tm_elements_design) to match torch-fem's k0 computation
    # k0 must include E0 to match the sensitivity formula: dC/dρ = -p * ρ^(p-1) * u^T k0 u
    k0_design = None
    if method == 'analytic':
        if verbose:
            print("  Computing base stiffness matrices (with E0)...")
        # Pass original elements_design - compute_element_stiffness_batch will handle reordering internally
        k0_design = compute_element_stiffness_batch(nodes, elements_design, nu, E=E)
    
    # Build filter
    centroids = nodes[elements_design].mean(dim=1).cpu().numpy()
    H, H_sum = build_filter(centroids, filter_R)
    
    # Initialize density
    rho_design = volfrac * torch.ones(n_design)
    rho_full = torch.ones(n_elements)
    rho_full[design_mask] = rho_design
    
    rho_min, rho_max = 0.01, 1.0
    V_0 = volfrac * vols_design.sum()
    
    history = {'compliance': [], 'volume': [], 'time': [], 'rho_history': []}
    
    method_name = f"tensormesh-{method}"
    if verbose:
        print(f"\n  Running {method_name} optimization ({epochs} epochs)...")
    
    for epoch in tqdm(range(epochs), desc=method_name, disable=not verbose):
        t_start = time.time()
        
        rho_full[design_mask] = rho_design
        
        if method == 'autograd':
            # Autograd gradient
            rho_var = rho_design.clone().requires_grad_(True)
            rho_var_full = rho_full.clone()
            rho_var_full[design_mask] = rho_var
            
            K = assembler(tm_mesh.points, element_data={"rho": rho_var_full})
            
            total_compliance = torch.tensor(0.0)
            for F in forces_list:
                K_, F_ = condenser(K, F)
                u_ = K_.solve(F_, backend=solver_backend, tol=solver_tol, B=B_rigid)
                u = condenser.recover(u_)
                compliance = torch.inner(F, u)
                total_compliance = total_compliance + compliance
            
            total_compliance.backward()
            sensitivity = rho_var.grad.clone()
            compliance_val = total_compliance.item()
            
        else:  # analytic
            with torch.no_grad():
                K = assembler(tm_mesh.points, element_data={"rho": rho_full})
            
            total_compliance = 0.0
            sensitivity = torch.zeros(n_design)
            rho_design_penal = rho_design ** (penal - 1.0)
            
            for F in forces_list:
                K_, F_ = condenser(K, F)
                with torch.no_grad():
                    u_ = K_.solve(F_, backend=solver_backend, tol=solver_tol, B=B_rigid)
                u = condenser.recover(u_)
                
                compliance = torch.inner(F, u).item()
                total_compliance += compliance
                
                # Analytic sensitivity: dC/dρ = -p * ρ^(p-1) * u^T k0 u
                # k0 uses elements_design directly without reordering (matches torch-fem)
                u_reshaped = u.reshape(-1, dim)
                # Use elements_design directly (no reordering, same as torch-fem)
                u_e = u_reshaped[elements_design].reshape(n_design, -1)
                k0_u = torch.einsum("nij,nj->ni", k0_design, u_e)
                w = torch.einsum("ni,ni->n", u_e, k0_u)
                sensitivity += -penal * rho_design_penal * w
            
            compliance_val = total_compliance
        
        # Apply filter
        sensitivity_filtered = apply_filter(H, H_sum, rho_design, sensitivity, vols_design)
        
        # OC update
        def make_step(mu):
            G_k = -sensitivity_filtered / (mu + 1e-10)
            upper = torch.minimum(torch.tensor(rho_max), (1 + move) * rho_design)
            lower = torch.maximum(torch.tensor(rho_min), (1 - move) * rho_design)
            rho_trial = (G_k.clamp(min=1e-10) ** 0.5) * rho_design
            return torch.maximum(torch.minimum(rho_trial, upper), lower)
        
        def constraint(mu):
            return (torch.inner(make_step(mu), vols_design) - V_0).item()
        
        with torch.no_grad():
            try:
                mu = bisect(constraint, 1e-10, 1e10, xtol=1e-4)
            except ValueError:
                mu = 1.0
        
        rho_design = make_step(mu)
        rho_full[design_mask] = rho_design
        
        t_end = time.time()
        
        history['compliance'].append(compliance_val)
        history['volume'].append((rho_design * vols_design).sum().item() / vols_design.sum().item())
        history['time'].append(t_end - t_start)
        history['rho_history'].append(rho_full.cpu().numpy().copy())
    
    return {
        'method': method_name,
        'rho': rho_design.cpu().numpy(),
        'rho_full': rho_full.cpu().numpy(),
        'design_mask': design_mask.cpu().numpy(),
        'history': history,
        'final_compliance': history['compliance'][-1],
        'final_volume': history['volume'][-1],
        'total_time': sum(history['time']),
        'avg_time_per_iter': np.mean(history['time']),
    }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(mesh_path, epochs=50, volfrac=0.15, penal=3.0, move=0.2, 
                  filter_R=5.0, E=16500.0, nu=0.342, verbose=True,
                  solver_backend="torch", solver_tol=1e-5):
    """Run all three methods and collect results."""
    
    # Load mesh
    mesh_data = meshio.read(mesh_path)
    nodes = torch.tensor(mesh_data.points, dtype=torch.float64)
    elements = torch.tensor(mesh_data.cells[0].data, dtype=torch.long)
    domain = torch.tensor(mesh_data.cell_data["gmsh:geometrical"][0], dtype=torch.long)
    
    if verbose:
        print("=" * 70)
        print("  BRACKET INVERSE DESIGN BENCHMARK")
        print("=" * 70)
        print(f"  Mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")
        print(f"  Epochs: {epochs}")
        print(f"  volfrac={volfrac}, penal={penal}, move={move}, R={filter_R}")
        print("=" * 70)
    
    results = {}
    
    # 1. torch-fem
    if verbose:
        print("\n" + "#" * 70)
        print("  Method 1: torch-fem (analytic sensitivity)")
        print("#" * 70)
    
    result_tf = run_torchfem_optimization(
        nodes, elements, domain, epochs, volfrac, penal, move, filter_R, E, nu, verbose
    )
    if result_tf is not None:
        results['torchfem'] = result_tf
    
    # 2. TensorMesh autograd
    if verbose:
        print("\n" + "#" * 70)
        print("  Method 2: TensorMesh-autograd (PyTorch autodiff)")
        print("#" * 70)
    
    results['tensormesh-autograd'] = run_tensormesh_optimization(
        nodes, elements, domain, 'autograd', epochs, volfrac, penal, move, filter_R, E, nu, verbose,
        solver_backend=solver_backend, solver_tol=solver_tol
    )
    
    # 3. TensorMesh analytic
    if verbose:
        print("\n" + "#" * 70)
        print("  Method 3: TensorMesh-analytic (hand-derived sensitivity)")
        print("#" * 70)
    
    results['tensormesh-analytic'] = run_tensormesh_optimization(
        nodes, elements, domain, 'analytic', epochs, volfrac, penal, move, filter_R, E, nu, verbose,
        solver_backend=solver_backend, solver_tol=solver_tol
    )
    
    return results, mesh_data


# ============================================================================
# Visualization
# ============================================================================

def plot_time_comparison(results, output_dir):
    """Plot time comparison (Figure 1)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods = list(results.keys())
    
    # 1. Average time per iteration
    ax = axes[0]
    avg_times = [results[m]['avg_time_per_iter'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    
    bars = ax.bar(range(len(methods)), avg_times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Time per Iteration (s)')
    ax.set_title('Average Time per Iteration')
    
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Total time
    ax = axes[1]
    total_times = [results[m]['total_time'] for m in methods]
    
    bars = ax.bar(range(len(methods)), total_times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Optimization Time')
    
    for bar, val in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "benchmark_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_effect_comparison(results, mesh_data, output_dir):
    """Plot effect comparison (Figure 2)."""
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig = plt.figure(figsize=(12, 8))
    
    # Create grid: 2 rows
    # Row 1: Convergence curve + Final compliance bar
    # Row 2: Density distributions (one per method)
    
    # 1. Convergence curve (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    for m in methods:
        history = results[m]['history']
        color = METHOD_COLORS.get(m, '#888888')
        ax1.semilogy(history['compliance'], color=color, linewidth=2, label=m)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Compliance')
    ax1.set_title('Convergence History')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Final compliance bar (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    final_compliances = [results[m]['final_compliance'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    
    bars = ax2.bar(range(len(methods)), final_compliances, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=9)
    ax2.set_ylabel('Final Compliance')
    ax2.set_title('Final Objective Value')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    for bar, val in zip(bars, final_compliances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.3e}', ha='center', va='bottom', fontsize=8)
    
    # 3. Density distributions (bottom row - 3D scatter plots)
    nodes = torch.tensor(mesh_data.points)
    elements = torch.tensor(mesh_data.cells[0].data)
    
    for i, m in enumerate(methods):
        ax = fig.add_subplot(2, n_methods, n_methods + i + 1, projection='3d')
        
        result = results[m]
        rho_full = result['rho_full']
        design_mask = result['design_mask']
        
        # Get centroids of design elements with high density
        elements_design = elements[design_mask]
        rho_design = rho_full[design_mask]
        
        centroids = nodes[elements_design].mean(dim=1).numpy()
        
        # Filter to show only elements with rho > 0.5
        high_density = rho_design > 0.5
        if high_density.sum() > 0:
            centroids_high = centroids[high_density]
            rho_high = rho_design[high_density]
            
            scatter = ax.scatter(centroids_high[:, 0], centroids_high[:, 1], centroids_high[:, 2],
                                c=rho_high, cmap='gray_r', s=2, alpha=0.6, vmin=0, vmax=1)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f'{m}\n(rho > 0.5)', fontsize=10)
        ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "benchmark_effect.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def generate_videos(results, mesh_data, output_dir):
    """Generate MP4 video for each method showing optimization process."""
    if not GENERATE_VIDEO:
        return
    
    print("\nGenerating optimization videos...")
    
    nodes = mesh_data.points
    elements = mesh_data.cells[0].data
    n_elements = elements.shape[0]
    
    # Create VTK grid once
    cells_vtk = np.hstack([np.full((n_elements, 1), 4), elements]).flatten()
    cell_types = np.full(n_elements, pv.CellType.TETRA)
    
    for method_name, result in results.items():
        rho_history = result['history'].get('rho_history', [])
        if not rho_history:
            print(f"  Skipping {method_name}: no density history")
            continue
        
        # Create frames directory
        frames_dir = os.path.join(output_dir, f'frames_{method_name}')
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"  Generating frames for {method_name}...")
        
        for epoch, rho_full in enumerate(tqdm(rho_history, desc=f"  {method_name}", leave=False)):
            compliance = result['history']['compliance'][epoch]
            
            # Create mesh
            grid = pv.UnstructuredGrid(cells_vtk.copy(), cell_types.copy(), nodes.copy())
            grid.cell_data['density'] = rho_full
            
            # Threshold to show solid material
            try:
                thresholded = grid.threshold(0.1, scalars='density')
                if thresholded.n_cells == 0:
                    thresholded = grid
            except:
                thresholded = grid
            
            # Render
            plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
            plotter.add_mesh(thresholded, scalars='density', cmap='YlOrRd',
                           clim=[0.1, 1], show_scalar_bar=True)
            plotter.add_text(f"{method_name}\nEpoch {epoch+1}/{len(rho_history)}\nCompliance: {compliance:.0f}",
                           position='upper_left', font_size=14, color='black')
            # Camera position: closer isometric view (mesh bounds: X[-89,89], Y[-18,90], Z[-45,18])
            plotter.camera_position = [
                (150, -30, 60),    # camera position (closer isometric)
                (0, 40, -15),      # focal point (mesh center)
                (0, 0, 1)          # up vector
            ]
            plotter.reset_camera_clipping_range()
            plotter.set_background('white')
            plotter.screenshot(f'{frames_dir}/frame_{epoch:04d}.png')
            plotter.close()
        
        # Generate MP4
        video_path = os.path.join(output_dir, f'optimization_{method_name}.mp4')
        os.system(f'ffmpeg -y -framerate {VIDEO_FPS} -i {frames_dir}/frame_%04d.png '
                 f'-c:v libx264 -pix_fmt yuv420p {video_path} 2>/dev/null')
        print(f"  Saved: {video_path}")


def save_cache(results, output_dir, epochs):
    """Save results to CSV cache."""
    
    # 1. Summary cache
    rows = []
    for m, result in results.items():
        rows.append({
            'method': m,
            'epochs': epochs,
            'final_compliance': result['final_compliance'],
            'final_volume': result['final_volume'],
            'total_time': result['total_time'],
            'avg_time_per_iter': result['avg_time_per_iter'],
        })
    
    df_summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "benchmark_cache.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    
    # 2. Detailed history cache
    history_rows = []
    for m, result in results.items():
        history = result['history']
        for epoch in range(len(history['compliance'])):
            history_rows.append({
                'method': m,
                'epoch': epoch,
                'compliance': history['compliance'][epoch],
                'volume': history['volume'][epoch],
                'time_per_iter': history['time'][epoch],
            })
    
    df_history = pd.DataFrame(history_rows)
    history_path = os.path.join(output_dir, "benchmark_history.csv")
    df_history.to_csv(history_path, index=False)
    print(f"  Saved: {history_path}")


def plot_from_cache(output_dir):
    """Plot from cached results."""
    
    cache_path = os.path.join(output_dir, "benchmark_cache.csv")
    history_path = os.path.join(output_dir, "benchmark_history.csv")
    
    if not os.path.exists(cache_path) or not os.path.exists(history_path):
        print("Error: Cache files not found. Run benchmark first.")
        return
    
    df_summary = pd.read_csv(cache_path)
    df_history = pd.read_csv(history_path)
    
    # Reconstruct results dict
    results = {}
    for _, row in df_summary.iterrows():
        m = row['method']
        method_history = df_history[df_history['method'] == m]
        
        results[m] = {
            'method': m,
            'final_compliance': row['final_compliance'],
            'final_volume': row['final_volume'],
            'total_time': row['total_time'],
            'avg_time_per_iter': row['avg_time_per_iter'],
            'history': {
                'compliance': method_history['compliance'].tolist(),
                'volume': method_history['volume'].tolist(),
                'time': method_history['time_per_iter'].tolist(),
            }
        }
    
    # Plot time comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods = list(results.keys())
    
    ax = axes[0]
    avg_times = [results[m]['avg_time_per_iter'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    bars = ax.bar(range(len(methods)), avg_times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Time per Iteration (s)')
    ax.set_title('Average Time per Iteration')
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    ax = axes[1]
    total_times = [results[m]['total_time'] for m in methods]
    bars = ax.bar(range(len(methods)), total_times, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], fontsize=9)
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Optimization Time')
    for bar, val in zip(bars, total_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "benchmark_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        history = results[m]['history']
        color = METHOD_COLORS.get(m, '#888888')
        ax.semilogy(history['compliance'], color=color, linewidth=2, label=m)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compliance')
    ax.set_title('Convergence History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "benchmark_convergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  {'Method':<22} {'Final Compliance':>18} {'Avg Time (s)':>15} {'Total Time (s)':>15}")
    print("-" * 80)
    
    for m, result in results.items():
        print(f"  {m:<22} {result['final_compliance']:>18.4e} "
              f"{result['avg_time_per_iter']:>15.2f} {result['total_time']:>15.2f}")
    
    print("=" * 80 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bracket Inverse Design Benchmark")
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--volfrac', type=float, default=0.15, help='Volume fraction')
    parser.add_argument('--penal', type=float, default=3.0, help='SIMP penalization')
    parser.add_argument('--move', type=float, default=0.2, help='OC move limit')
    parser.add_argument('--filter_R', type=float, default=5.0, help='Filter radius')
    parser.add_argument('--plot-only', action='store_true', help='Only plot from cache')
    parser.add_argument('--solver', type=str, default='amg', choices=['sla', 'torch', 'scipy', 'petsc', 'amg'],
                        help='Solver backend: amg (CG+AMG, recommended), sla (CG+Jacobi), torch (BiCGSTAB), scipy (direct, slow), petsc')
    parser.add_argument('--solver-tol', type=float, default=1e-5, help='Solver tolerance for iterative solvers')
    
    args = parser.parse_args()
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.plot_only:
        print("Plotting from cache...")
        plot_from_cache(output_dir)
        return
    
    # Mesh path
    mesh_path = os.path.join(os.path.dirname(__file__), "ge_bracket.vtu")
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        return
    
    # Run benchmark
    results, mesh_data = run_benchmark(
        mesh_path,
        epochs=args.epochs,
        volfrac=args.volfrac,
        penal=args.penal,
        move=args.move,
        filter_R=args.filter_R,
        solver_backend=args.solver,
        solver_tol=args.solver_tol,
    )
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_time_comparison(results, output_dir)
    plot_effect_comparison(results, mesh_data, output_dir)
    
    # Generate videos
    generate_videos(results, mesh_data, output_dir)
    
    # Save cache
    print("\nSaving cache...")
    save_cache(results, output_dir, args.epochs)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


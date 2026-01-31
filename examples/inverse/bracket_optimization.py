"""
GE Jet Engine Bracket Topology Optimization

Reproduces the torch-fem bracket example using TensorMesh.
Compares three gradient computation methods:
1. Analytic sensitivity (as in torch-fem)
2. Automatic differentiation (PyTorch autograd through solve)
3. Explicit adjoint method

Reference:
    https://github.com/meyer-nils/torch-fem/blob/main/examples/optimization/solid/bracket.ipynb
    https://grabcad.com/challenges/ge-jet-engine-bracket-challenge

Usage:
    python bracket_optimization.py --method autograd --epochs 25
    python bracket_optimization.py --method analytic --epochs 25
    python bracket_optimization.py --method adjoint --epochs 25
    python bracket_optimization.py --benchmark  # Compare all methods
"""

import sys
sys.path.insert(0, "../..")

import os
import argparse
import time
import numpy as np
import torch
import meshio
from scipy.spatial import cKDTree
from scipy.optimize import bisect
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.sparse import SparseMatrix
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness

torch.set_default_dtype(torch.float64)


# ============================================================================
# SIMP 3D Stiffness Assembler
# ============================================================================

class SIMP3DStiffnessAssembler(ElementAssembler):
    """
    SIMP stiffness matrix assembler for 3D topology optimization.
    
    Material model:
        E(ρ) = E_min + ρ^p * (E_0 - E_min)
    """
    
    def __post_init__(self, E=16500.0, nu=0.342, penal=3.0, E_min=1e-9):
        self.E0 = E
        self.nu = nu
        self.penal = penal
        self.E_min = E_min
    
    def forward(self, gradu, gradv, rho):
        """Compute stiffness matrix contribution with SIMP interpolation."""
        dim = gradu.shape[0]
        E = self.E_min + (rho ** self.penal) * (self.E0 - self.E_min)
        Ba = voigt_shape_grad(gradu)
        Bb = voigt_shape_grad(gradv)
        C = voigt_stiffness(E, self.nu, dim)
        C = C.to(dtype=gradu.dtype, device=gradu.device)
        return Ba.T @ C @ Bb


# ============================================================================
# Gradient Computation Methods
# ============================================================================

def compute_autograd_gradient(mesh, assembler, condenser, forces_list, 
                               rho_full, design_mask, penal):
    """
    Automatic differentiation gradient through sparse solve.
    Uses PyTorch autograd to compute dC/dρ.
    """
    n_design = design_mask.sum().item()
    rho_design = rho_full[design_mask].clone().requires_grad_(True)
    
    rho_var = rho_full.clone()
    rho_var[design_mask] = rho_design
    
    # Assemble
    K = assembler(mesh.points, element_data={"rho": rho_var})
    
    total_compliance = torch.tensor(0.0, dtype=rho_full.dtype, device=rho_full.device)
    
    for F in forces_list:
        K_, F_ = condenser(K, F)
        u_ = K_.solve(F_, backend="scipy")
        u = condenser.recover(u_)
        compliance = torch.inner(F.flatten(), u.flatten())
        total_compliance = total_compliance + compliance
    
    total_compliance.backward()
    sensitivity = rho_design.grad.clone()
    
    return total_compliance.item(), sensitivity


def compute_analytic_gradient(mesh, assembler, condenser, forces_list,
                               rho_full, design_mask, penal, k0_design, elements_design):
    """
    Analytic gradient computation (as in torch-fem).
    dC/dρ = -p * ρ^(p-1) * u_e^T @ k0_e @ u_e
    """
    n_design = design_mask.sum().item()
    dim = 3
    
    # Assemble global stiffness
    with torch.no_grad():
        K = assembler(mesh.points, element_data={"rho": rho_full})
    
    total_compliance = 0.0
    sensitivity = torch.zeros(n_design, dtype=rho_full.dtype, device=rho_full.device)
    rho_design = rho_full[design_mask]
    
    for F in forces_list:
        K_, F_ = condenser(K, F)
        with torch.no_grad():
            u_ = K_.solve(F_, backend="scipy")
        u = condenser.recover(u_)
        
        compliance = torch.inner(F.flatten(), u.flatten()).item()
        total_compliance += compliance
        
        # Extract element displacements
        u_reshaped = u.reshape(-1, dim)
        u_e = u_reshaped[elements_design].reshape(n_design, -1)
        
        # Analytic sensitivity
        k0_u = torch.einsum("nij,nj->ni", k0_design, u_e)
        w = torch.einsum("ni,ni->n", u_e, k0_u)
        sensitivity += -penal * (rho_design ** (penal - 1.0)) * w
    
    return total_compliance, sensitivity


def compute_adjoint_gradient(mesh, assembler, condenser, forces_list,
                              rho_full, design_mask, penal, k0_design, elements_design):
    """
    Explicit adjoint method.
    For compliance, adjoint = u (self-adjoint), so this is equivalent to analytic.
    """
    # For compliance minimization, the adjoint method gives the same result
    # as the analytic method because dC/du = 2*K*u = 2*F, and adjoint = u
    return compute_analytic_gradient(mesh, assembler, condenser, forces_list,
                                     rho_full, design_mask, penal, k0_design, elements_design)


# ============================================================================
# Precompute Element Stiffness Matrices
# ============================================================================

def compute_element_stiffness_batch(nodes, elements, nu):
    """
    Compute element stiffness matrices (E=1) for all elements.
    Returns [n_elements, 12, 12] tensor.
    """
    n_elements = elements.shape[0]
    dtype = nodes.dtype
    device = nodes.device
    
    C = voigt_stiffness(1.0, nu, 3).to(dtype=dtype, device=device)
    
    k0 = torch.zeros(n_elements, 12, 12, dtype=dtype, device=device)
    
    for idx in tqdm(range(n_elements), desc="Computing k0", leave=False):
        elem_nodes = elements[idx]
        coords = nodes[elem_nodes]
        
        # Shape function gradients
        dN_dxi = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=dtype, device=device)
        
        J = coords[1:4] - coords[0:1]
        J_inv = torch.linalg.inv(J.T)
        det_J = torch.linalg.det(J)
        vol = abs(det_J) / 6.0
        
        dN_dx = dN_dxi @ J_inv
        
        def make_B(dN):
            B = torch.zeros(6, 3, dtype=dtype, device=device)
            B[0, 0] = dN[0]
            B[1, 1] = dN[1]
            B[2, 2] = dN[2]
            B[3, 0] = dN[1]
            B[3, 1] = dN[0]
            B[4, 1] = dN[2]
            B[4, 2] = dN[1]
            B[5, 0] = dN[2]
            B[5, 2] = dN[0]
            return B
        
        Ke = torch.zeros(12, 12, dtype=dtype, device=device)
        for i in range(4):
            Bi = make_B(dN_dx[i])
            for j in range(4):
                Bj = make_B(dN_dx[j])
                Ke[i*3:(i+1)*3, j*3:(j+1)*3] = Bi.T @ C @ Bj * vol
        
        k0[idx] = Ke
    
    return k0


# ============================================================================
# Filter
# ============================================================================

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
# Main Optimization
# ============================================================================

def run_optimization(mesh_path, method_name, epochs=25, volfrac=0.15, penal=3.0,
                     move=0.2, filter_R=5.0, E=16500.0, nu=0.342, verbose=True):
    """Run topology optimization."""
    
    device = torch.device("cpu")
    dtype = torch.float64
    
    # Load mesh
    mesh_data = meshio.read(mesh_path)
    nodes = torch.tensor(mesh_data.points, dtype=dtype, device=device)
    elements = torch.tensor(mesh_data.cells[0].data, dtype=torch.long, device=device)
    domain = torch.tensor(mesh_data.cell_data["gmsh:geometrical"][0], dtype=torch.long, device=device)
    
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    dim = 3
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  GE BRACKET TOPOLOGY OPTIMIZATION - {method_name.upper()}")
        print(f"{'='*70}")
        print(f"  Nodes: {n_nodes}, Elements: {n_elements}")
        print(f"  volfrac={volfrac}, penal={penal}, move={move}, R={filter_R}")
        print(f"{'='*70}\n")
    
    # Create TensorMesh mesh (with reorder for correct node ordering)
    meshio_obj = meshio.Mesh(
        points=mesh_data.points,
        cells=[("tetra", mesh_data.cells[0].data)],
    )
    tm_mesh = Mesh(meshio_obj, reorder=True)
    
    # Design elements (domain 6)
    design_mask = (domain == 6)
    n_design = design_mask.sum().item()
    elements_design = elements[design_mask]
    
    if verbose:
        print(f"  Design elements: {n_design}")
    
    # Compute element volumes
    def compute_volumes(nodes, elems):
        p = nodes[elems]
        v = p[:, 1:] - p[:, 0:1]
        return torch.abs(torch.linalg.det(v)) / 6.0
    
    vols = compute_volumes(nodes, elements)
    vols_design = vols[design_mask]
    
    # Assembler
    assembler = SIMP3DStiffnessAssembler.from_mesh(tm_mesh, E=E, nu=nu, penal=penal)
    
    # Boundary conditions
    R_fix = 6.0
    constraint_mask = torch.zeros(n_nodes, dim, dtype=torch.bool, device=device)
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
    
    if verbose:
        print(f"  Constrained DOFs: {dbc_mask.sum().item()}")
    
    # Load cases
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    n_load = len(load_nodes)
    
    forces_list = []
    
    F1 = torch.zeros(n_nodes * dim, dtype=dtype, device=device)
    F1[load_nodes * dim + 2] = 8000.0 / n_load
    forces_list.append(F1)
    
    F2 = torch.zeros(n_nodes * dim, dtype=dtype, device=device)
    F2[load_nodes * dim + 1] = -8500.0 / n_load
    forces_list.append(F2)
    
    F3 = torch.zeros(n_nodes * dim, dtype=dtype, device=device)
    F3[load_nodes * dim + 1] = -9500 * np.sin(np.deg2rad(42)) / n_load
    F3[load_nodes * dim + 2] = 9500 * np.cos(np.deg2rad(42)) / n_load
    forces_list.append(F3)
    
    F4 = torch.zeros(n_nodes * dim, dtype=dtype, device=device)
    levers = nodes[load_nodes, 0]
    for i, node in enumerate(load_nodes):
        if levers[i].abs() > 1e-6:
            F4[node * dim + 1] = 5000.0 / levers[i] / n_load
    forces_list.append(F4)
    
    if verbose:
        print(f"  Load cases: {len(forces_list)}")
    
    # Precompute k0 for analytic/adjoint methods
    k0_design = None
    if method_name in ['analytic', 'adjoint']:
        if verbose:
            print("  Computing base stiffness matrices...")
        k0_design = compute_element_stiffness_batch(nodes, elements_design, nu)
    
    # Build filter
    centroids = nodes[elements_design].mean(dim=1).cpu().numpy()
    H, H_sum = build_filter(centroids, filter_R)
    H = H.to(dtype=dtype, device=device)
    H_sum = H_sum.to(dtype=dtype, device=device)
    
    # Initialize density
    rho_design = volfrac * torch.ones(n_design, dtype=dtype, device=device)
    rho_full = torch.ones(n_elements, dtype=dtype, device=device)
    rho_full[design_mask] = rho_design
    
    rho_min, rho_max = 0.01, 1.0
    V_0 = volfrac * vols_design.sum()
    
    # History
    history = {'compliance': [], 'volume': [], 'time': []}
    
    if verbose:
        print(f"\nStarting optimization with {method_name} method...")
    
    # Optimization loop
    for epoch in tqdm(range(epochs), desc="Optimizing", disable=not verbose):
        t_start = time.time()
        
        # Update full density
        rho_full[design_mask] = rho_design
        
        # Compute gradient
        if method_name == 'autograd':
            compliance, sensitivity = compute_autograd_gradient(
                tm_mesh, assembler, condenser, forces_list,
                rho_full, design_mask, penal
            )
        elif method_name == 'analytic':
            compliance, sensitivity = compute_analytic_gradient(
                tm_mesh, assembler, condenser, forces_list,
                rho_full, design_mask, penal, k0_design, elements_design
            )
        else:  # adjoint
            compliance, sensitivity = compute_adjoint_gradient(
                tm_mesh, assembler, condenser, forces_list,
                rho_full, design_mask, penal, k0_design, elements_design
            )
        
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
        
        t_end = time.time()
        
        history['compliance'].append(compliance)
        history['volume'].append((rho_design * vols_design).sum().item() / vols_design.sum().item())
        history['time'].append(t_end - t_start)
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}: C={compliance:.4e}, V={history['volume'][-1]:.4f}, t={history['time'][-1]:.2f}s")
    
    # Final results
    results = {
        'rho': rho_design.cpu().numpy(),
        'rho_full': rho_full.cpu().numpy(),
        'design_mask': design_mask.cpu().numpy(),
        'history': history,
        'method': method_name,
        'final_compliance': history['compliance'][-1],
        'final_volume': history['volume'][-1],
        'total_time': sum(history['time']),
        'avg_time_per_iter': np.mean(history['time']),
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Final compliance: {results['final_compliance']:.4e}")
        print(f"  Final volume fraction: {results['final_volume']:.4f}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Average time per iteration: {results['avg_time_per_iter']:.2f}s")
        print(f"{'='*70}\n")
    
    return results, mesh_data


# ============================================================================
# Benchmarking and Visualization
# ============================================================================

def run_benchmark(mesh_path, epochs=10, **kwargs):
    """Run all methods and compare."""
    results = {}
    mesh_data = None
    
    for method in ['analytic', 'autograd', 'adjoint']:
        print(f"\n{'#'*70}")
        print(f"  Running {method.upper()} method")
        print(f"{'#'*70}")
        
        results[method], mesh_data = run_optimization(
            mesh_path, method, epochs=epochs, **kwargs
        )
    
    return results, mesh_data


def plot_comparison(results, save_path="bracket_comparison.png"):
    """Plot comparison of methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {'analytic': 'blue', 'autograd': 'green', 'adjoint': 'red'}
    
    # Compliance history
    ax = axes[0]
    for method, result in results.items():
        ax.semilogy(result['history']['compliance'], color=colors.get(method, 'gray'),
                    linewidth=2, label=method)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compliance')
    ax.set_title('Convergence History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time per iteration
    ax = axes[1]
    methods = list(results.keys())
    times = [results[m]['avg_time_per_iter'] for m in methods]
    ax.bar(methods, times, color=[colors.get(m, 'gray') for m in methods])
    ax.set_ylabel('Time per Iteration (s)')
    ax.set_title('Computational Cost')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Final compliance
    ax = axes[2]
    compliances = [results[m]['final_compliance'] for m in methods]
    ax.bar(methods, compliances, color=[colors.get(m, 'gray') for m in methods])
    ax.set_ylabel('Final Compliance')
    ax.set_title('Final Objective')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def export_result(mesh_data, results, method='autograd', save_path="bracket_result.vtu"):
    """Export result to VTU."""
    result = results[method]
    rho_full = result['rho_full']
    
    out_mesh = meshio.Mesh(
        points=mesh_data.points,
        cells=mesh_data.cells,
        cell_data={"rho": [rho_full]}
    )
    out_mesh.write(save_path)
    print(f"Saved: {save_path}")


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "="*70)
    print("  COMPARISON TABLE")
    print("="*70)
    print(f"  {'Method':<12} {'Final Compliance':>18} {'Avg Time (s)':>15} {'Total Time (s)':>15}")
    print("-"*70)
    
    for method, result in results.items():
        print(f"  {method:<12} {result['final_compliance']:>18.4e} "
              f"{result['avg_time_per_iter']:>15.2f} {result['total_time']:>15.2f}")
    
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GE Bracket Topology Optimization")
    
    parser.add_argument('--method', type=str, default='autograd',
                        choices=['analytic', 'autograd', 'adjoint'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--volfrac', type=float, default=0.15)
    parser.add_argument('--penal', type=float, default=3.0)
    parser.add_argument('--move', type=float, default=0.2)
    parser.add_argument('--filter_R', type=float, default=5.0)
    parser.add_argument('--benchmark', action='store_true')
    
    args = parser.parse_args()
    
    mesh_path = os.path.join(os.path.dirname(__file__), "ge_bracket.vtu")
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        return
    
    if args.benchmark:
        results, mesh_data = run_benchmark(
            mesh_path,
            epochs=args.epochs,
            volfrac=args.volfrac,
            penal=args.penal,
            move=args.move,
            filter_R=args.filter_R
        )
        print_comparison_table(results)
        plot_comparison(results)
        export_result(mesh_data, results)
    else:
        result, mesh_data = run_optimization(
            mesh_path,
            method_name=args.method,
            epochs=args.epochs,
            volfrac=args.volfrac,
            penal=args.penal,
            move=args.move,
            filter_R=args.filter_R
        )
        results = {args.method: result}
        export_result(mesh_data, results, method=args.method)


if __name__ == "__main__":
    main()

"""
Alignment check between torchfem and TensorMesh for bracket inverse design.

This script verifies that compliance and gradients match between the two implementations,
generates comparison visualizations, and saves results to CSV cache.

Outputs:
    output/alignment_diff.png - Visualization of differences
    output/alignment_cache.csv - Detailed comparison data
"""

import sys
sys.path.insert(0, "../..")

import os
import time
import numpy as np
import torch
import meshio
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

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

# Colors
TORCHFEM_COLOR = '#E24A33'    # red-orange
TENSORMESH_COLOR = '#2CA02C'  # green

def main():
    print("=" * 70)
    print("  BRACKET ALIGNMENT CHECK: torchfem vs TensorMesh")
    print("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # Load mesh
    # ========================================================================
    mesh_path = os.path.join(os.path.dirname(__file__), "ge_bracket.vtu")
    mesh = meshio.read(mesh_path)
    
    nodes = torch.tensor(mesh.points, dtype=torch.float64)
    elements = torch.tensor(mesh.cells[0].data)
    domain = torch.tensor(mesh.cell_data["gmsh:geometrical"][0])
    
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    dim = 3
    
    print(f"\nMesh: {n_nodes} nodes, {n_elements} elements")
    
    # Optimization parameters
    volfrac = 0.15
    penal = 3.0
    R_fix = 6.0
    E = 16500.0
    nu = 0.342
    
    # Design elements
    design_mask = domain == 6
    n_design = design_mask.sum().item()
    print(f"Design elements: {n_design}")
    
    # Load nodes
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    
    # ========================================================================
    # torch-fem reference
    # ========================================================================
    print("\n--- torch-fem reference ---")
    
    torchfem_available = False
    compliance_tf = None
    sensitivity_tf = None
    time_tf = None
    u_tf = None
    
    try:
        from torchfem import Solid
        from torchfem.materials import IsotropicElasticity3D
        
        material = IsotropicElasticity3D(E=E, nu=nu)
        model = Solid(nodes, elements, material)
        
        # Boundary conditions
        for d in [1, 3, 4, 5]:
            dom = torch.unique(elements[domain == d])
            center = nodes[dom].mean(dim=0)
            con = (nodes[dom, 0] - center[0]) ** 2 + (nodes[dom, 1] - center[1]) ** 2 < R_fix**2
            model.constraints[dom[con], :] = True
        
        # Get base stiffness before applying SIMP
        C0 = model.material.C.clone()
        k0_tf = model.k0().clone()
        
        # Apply SIMP density
        rho_tf = torch.ones(n_elements)
        rho_tf[design_mask] = volfrac
        model.material.C = torch.einsum("n,nijkl->nijkl", rho_tf ** penal, C0)
        
        # Load case 1
        load_case_1 = torch.zeros_like(nodes)
        load_case_1[load_nodes, 2] = 8000 / len(load_nodes)
        model.forces = load_case_1
        
        print("Solving with torch-fem...")
        t0 = time.time()
        u_tf, f_tf, _, _, _ = model.solve(rtol=0.01, verbose=False)
        t1 = time.time()
        time_tf = t1 - t0
        
        compliance_tf = torch.inner(f_tf.ravel(), u_tf.ravel()).item()
        print(f"  Solve time: {time_tf:.2f}s")
        print(f"  Compliance: {compliance_tf:.6e}")
        print(f"  Max displacement: {u_tf.abs().max().item():.6e}")
        
        # Compute analytic sensitivity for design elements
        design_elements = elements[design_mask]
        rho_design_tf = volfrac * torch.ones(n_design)
        k0_design = k0_tf[design_mask]
        
        # dC/dρ = -p * ρ^(p-1) * u_e^T @ k0_e @ u_e
        u_j = u_tf[design_elements].reshape(n_design, -1)
        w_k = torch.einsum("...i, ...ij, ...j", u_j, k0_design, u_j)
        sensitivity_tf = -penal * rho_design_tf ** (penal - 1.0) * w_k
        
        print(f"  Sensitivity mean: {sensitivity_tf.mean().item():.6e}")
        print(f"  Sensitivity range: [{sensitivity_tf.min().item():.6e}, {sensitivity_tf.max().item():.6e}]")
        
        torchfem_available = True
        
    except Exception as e:
        print(f"  torch-fem error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # TensorMesh implementation
    # ========================================================================
    print("\n--- TensorMesh implementation ---")
    
    from tensormesh import Mesh, Condenser, ElementAssembler
    from tensormesh.sparse import SparseMatrix
    from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness
    
    # Create mesh (with reorder for tetrahedron)
    meshio_obj = meshio.Mesh(
        points=mesh.points,
        cells=[("tetra", mesh.cells[0].data)],
        cell_data={"domain": [mesh.cell_data["gmsh:geometrical"][0]]}
    )
    tm_mesh = Mesh(meshio_obj, reorder=True)
    
    # SIMP assembler
    class SIMP3DAssembler(ElementAssembler):
        def __post_init__(self, E=16500.0, nu=0.342, penal=3.0, E_min=1e-9):
            self.E0 = E
            self.nu = nu
            self.penal = penal
            self.E_min = E_min
        
        def forward(self, gradu, gradv, rho):
            dim = gradu.shape[0]
            E_eff = self.E_min + (rho ** self.penal) * (self.E0 - self.E_min)
            Ba = voigt_shape_grad(gradu)
            Bb = voigt_shape_grad(gradv)
            C = voigt_stiffness(E_eff, self.nu, dim)
            C = C.to(dtype=gradu.dtype, device=gradu.device)
            return Ba.T @ C @ Bb
    
    # Full rho vector
    rho_full = torch.ones(n_elements)
    rho_full[design_mask] = volfrac
    
    # Assembler
    assembler = SIMP3DAssembler.from_mesh(tm_mesh, E=E, nu=nu, penal=penal)
    
    # Boundary conditions
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
    
    # Load case 1
    F = torch.zeros(n_nodes, dim)
    F[load_nodes, 2] = 8000.0 / len(load_nodes)
    F = F.flatten()
    
    print("Assembling and solving with TensorMesh...")
    t0 = time.time()
    K = assembler(tm_mesh.points, element_data={"rho": rho_full})
    K_, F_ = condenser(K, F)
    u_ = K_.solve(F_, backend="scipy")
    u_tm = condenser.recover(u_)
    t1 = time.time()
    time_tm = t1 - t0
    
    compliance_tm = torch.inner(F, u_tm).item()
    print(f"  Total time: {time_tm:.2f}s")
    print(f"  Compliance: {compliance_tm:.6e}")
    print(f"  Max displacement: {u_tm.abs().max().item():.6e}")
    
    # ========================================================================
    # TensorMesh autograd gradient
    # ========================================================================
    print("\n--- TensorMesh autograd gradient ---")
    
    rho_design = volfrac * torch.ones(n_design)
    rho_var = rho_design.clone().requires_grad_(True)
    
    rho_full_var = torch.ones(n_elements)
    rho_full_var = rho_full_var.clone()
    rho_full_var[design_mask] = rho_var
    
    t0 = time.time()
    K_grad = assembler(tm_mesh.points, element_data={"rho": rho_full_var})
    K_g, F_g = condenser(K_grad, F)
    u_g = K_g.solve(F_g, backend="scipy")
    u_full = condenser.recover(u_g)
    compliance_grad = torch.inner(F, u_full)
    compliance_grad.backward()
    sensitivity_tm = rho_var.grad.clone()
    t1 = time.time()
    time_tm_grad = t1 - t0
    
    print(f"  Autograd time: {time_tm_grad:.2f}s")
    print(f"  Sensitivity mean: {sensitivity_tm.mean().item():.6e}")
    print(f"  Sensitivity range: [{sensitivity_tm.min().item():.6e}, {sensitivity_tm.max().item():.6e}]")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    
    results = {
        'compliance_tensormesh': compliance_tm,
        'time_tensormesh': time_tm,
        'time_tensormesh_grad': time_tm_grad,
        'sensitivity_tm_mean': sensitivity_tm.mean().item(),
        'sensitivity_tm_min': sensitivity_tm.min().item(),
        'sensitivity_tm_max': sensitivity_tm.max().item(),
    }
    
    if torchfem_available:
        comp_diff = abs(compliance_tm - compliance_tf) / abs(compliance_tf) * 100
        print(f"\n  Compliance:")
        print(f"    torch-fem:   {compliance_tf:.6e}")
        print(f"    TensorMesh:  {compliance_tm:.6e}")
        print(f"    Difference:  {comp_diff:.6f}%")
        
        # Compare sensitivities
        sens_corr = torch.corrcoef(torch.stack([sensitivity_tf, sensitivity_tm]))[0, 1].item()
        sens_diff = (sensitivity_tm - sensitivity_tf).abs()
        sens_mae = sens_diff.mean().item()
        sens_max_diff = sens_diff.max().item()
        sens_rel = sens_mae / sensitivity_tf.abs().mean().item() * 100
        
        print(f"\n  Sensitivity:")
        print(f"    torch-fem mean:   {sensitivity_tf.mean().item():.6e}")
        print(f"    TensorMesh mean:  {sensitivity_tm.mean().item():.6e}")
        print(f"    Correlation:      {sens_corr:.6f}")
        print(f"    Mean abs diff:    {sens_mae:.6e} ({sens_rel:.2f}%)")
        print(f"    Max abs diff:     {sens_max_diff:.6e}")
        
        # Update results
        results.update({
            'compliance_torchfem': compliance_tf,
            'compliance_diff_percent': comp_diff,
            'time_torchfem': time_tf,
            'sensitivity_tf_mean': sensitivity_tf.mean().item(),
            'sensitivity_tf_min': sensitivity_tf.min().item(),
            'sensitivity_tf_max': sensitivity_tf.max().item(),
            'sensitivity_corr': sens_corr,
            'sensitivity_mae': sens_mae,
            'sensitivity_max_diff': sens_max_diff,
            'sensitivity_rel_diff_percent': sens_rel,
        })
        
        if comp_diff < 0.01 and sens_corr > 0.999:
            results['status'] = 'ALIGNED'
            print("\n  *** RESULTS ALIGNED! ***")
        elif comp_diff < 1.0:
            results['status'] = 'CLOSE'
            print("\n  *** COMPLIANCE ALIGNED, SENSITIVITY CLOSE ***")
        else:
            results['status'] = 'DIFFER'
            print("\n  *** RESULTS MAY DIFFER - CHECK IMPLEMENTATION ***")
        
        # ====================================================================
        # Generate comparison figure
        # ====================================================================
        print("\nGenerating comparison figure...")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. Compliance comparison (bar chart)
        ax = axes[0, 0]
        methods = ['torch-fem', 'TensorMesh']
        compliances = [compliance_tf, compliance_tm]
        colors = [TORCHFEM_COLOR, TENSORMESH_COLOR]
        bars = ax.bar(methods, compliances, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Compliance')
        ax.set_title(f'Compliance Comparison (diff: {comp_diff:.4f}%)')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        for bar, val in zip(bars, compliances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.4e}', ha='center', va='bottom', fontsize=8)
        
        # 2. Sensitivity scatter plot
        ax = axes[0, 1]
        sens_tf_np = sensitivity_tf.cpu().numpy()
        sens_tm_np = sensitivity_tm.cpu().numpy()
        ax.scatter(sens_tf_np, sens_tm_np, alpha=0.3, s=5, c='#348ABD')
        
        # Identity line
        lims = [min(sens_tf_np.min(), sens_tm_np.min()),
                max(sens_tf_np.max(), sens_tm_np.max())]
        ax.plot(lims, lims, 'k--', linewidth=1, label='y=x')
        ax.set_xlabel('torch-fem sensitivity')
        ax.set_ylabel('TensorMesh sensitivity')
        ax.set_title(f'Sensitivity Correlation (r = {sens_corr:.6f})')
        ax.legend(loc='upper left')
        ax.set_aspect('equal', adjustable='box')
        
        # 3. Sensitivity difference histogram
        ax = axes[1, 0]
        diff_np = (sensitivity_tm - sensitivity_tf).cpu().numpy()
        ax.hist(diff_np, bins=50, color='#988ED5', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Sensitivity Difference (TensorMesh - torch-fem)')
        ax.set_ylabel('Count')
        ax.set_title(f'Difference Distribution (MAE: {sens_mae:.4e})')
        
        # 4. Relative difference per element
        ax = axes[1, 1]
        rel_diff = sens_diff / (sensitivity_tf.abs() + 1e-10)
        rel_diff_np = rel_diff.cpu().numpy() * 100
        ax.hist(rel_diff_np, bins=50, color='#FF7F0E', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Relative Difference (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'Relative Difference Distribution (mean: {rel_diff_np.mean():.2f}%)')
        ax.set_xlim(0, min(100, rel_diff_np.max() * 1.1))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, "alignment_diff.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
        
    else:
        print("\n  torch-fem not available for comparison")
        print(f"  TensorMesh compliance: {compliance_tm:.6e}")
        results['status'] = 'TORCHFEM_UNAVAILABLE'
    
    # ========================================================================
    # Save CSV cache
    # ========================================================================
    csv_path = os.path.join(output_dir, "alignment_cache.csv")
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    print("\n" + "=" * 70)
    print("  ALIGNMENT CHECK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()









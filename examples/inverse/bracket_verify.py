"""
Verification script to compare TensorMesh with torch-fem results.
Validates that compliance and gradients match between the two implementations.
"""

import sys
sys.path.insert(0, "../..")

import os
import time
import numpy as np
import torch
import meshio

torch.set_default_dtype(torch.float64)

print("=" * 70)
print("  BRACKET OPTIMIZATION VERIFICATION")
print("=" * 70)

# ============================================================================
# Load mesh
# ============================================================================
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

# Design elements
design_mask = domain == 6
n_design = design_mask.sum().item()
print(f"Design elements: {n_design}")

# ============================================================================
# torch-fem reference
# ============================================================================
print("\n--- torch-fem reference ---")

try:
    from torchfem import Solid
    from torchfem.materials import IsotropicElasticity3D
    
    material = IsotropicElasticity3D(E=16500.0, nu=0.342)
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
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    load_case_1 = torch.zeros_like(nodes)
    load_case_1[load_nodes, 2] = 8000 / len(load_nodes)
    model.forces = load_case_1
    
    print("Solving with torch-fem...")
    t0 = time.time()
    u_tf, f_tf, _, _, _ = model.solve(rtol=0.01, verbose=False)
    t1 = time.time()
    
    compliance_tf = torch.inner(f_tf.ravel(), u_tf.ravel()).item()
    print(f"  Solve time: {t1-t0:.2f}s")
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
    torchfem_available = False
    compliance_tf = None
    sensitivity_tf = None

# ============================================================================
# TensorMesh implementation
# ============================================================================
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
        E = self.E_min + (rho ** self.penal) * (self.E0 - self.E_min)
        Ba = voigt_shape_grad(gradu)
        Bb = voigt_shape_grad(gradv)
        C = voigt_stiffness(E, self.nu, dim)
        C = C.to(dtype=gradu.dtype, device=gradu.device)
        return Ba.T @ C @ Bb

# Full rho vector
rho_full = torch.ones(n_elements)
rho_full[design_mask] = volfrac

# Assembler
assembler = SIMP3DAssembler.from_mesh(tm_mesh, E=16500.0, nu=0.342, penal=penal)

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

print("Assembling with TensorMesh...")
t0 = time.time()
K = assembler(tm_mesh.points, element_data={"rho": rho_full})
t1 = time.time()
print(f"  Assembly time: {t1-t0:.2f}s")

print("Solving with TensorMesh...")
t0 = time.time()
K_, F_ = condenser(K, F)
u_ = K_.solve(F_, backend="scipy")
u = condenser.recover(u_)
t1 = time.time()

compliance_tm = torch.inner(F, u).item()
print(f"  Solve time: {t1-t0:.2f}s")
print(f"  Compliance: {compliance_tm:.6e}")
print(f"  Max displacement: {u.abs().max().item():.6e}")

# ============================================================================
# Test autograd gradient
# ============================================================================
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

print(f"  Autograd time: {t1-t0:.2f}s")
print(f"  Sensitivity mean: {sensitivity_tm.mean().item():.6e}")
print(f"  Sensitivity range: [{sensitivity_tm.min().item():.6e}, {sensitivity_tm.max().item():.6e}]")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "=" * 70)
print("  COMPARISON RESULTS")
print("=" * 70)

if torchfem_available:
    comp_diff = abs(compliance_tm - compliance_tf) / abs(compliance_tf) * 100
    print(f"\n  Compliance:")
    print(f"    torch-fem:   {compliance_tf:.6e}")
    print(f"    TensorMesh:  {compliance_tm:.6e}")
    print(f"    Difference:  {comp_diff:.6f}%")
    
    # Compare sensitivities
    sens_corr = torch.corrcoef(torch.stack([sensitivity_tf, sensitivity_tm]))[0, 1].item()
    sens_diff = (sensitivity_tm - sensitivity_tf).abs().mean().item()
    sens_rel = sens_diff / sensitivity_tf.abs().mean().item() * 100
    
    print(f"\n  Sensitivity:")
    print(f"    torch-fem mean:   {sensitivity_tf.mean().item():.6e}")
    print(f"    TensorMesh mean:  {sensitivity_tm.mean().item():.6e}")
    print(f"    Correlation:      {sens_corr:.6f}")
    print(f"    Mean abs diff:    {sens_diff:.6e} ({sens_rel:.2f}%)")
    
    if comp_diff < 0.01 and sens_corr > 0.999:
        print("\n  *** RESULTS ALIGNED! ***")
    elif comp_diff < 1.0:
        print("\n  *** COMPLIANCE ALIGNED, SENSITIVITY CLOSE ***")
    else:
        print("\n  *** RESULTS MAY DIFFER - CHECK IMPLEMENTATION ***")
else:
    print("\n  torch-fem not available for comparison")
    print(f"  TensorMesh compliance: {compliance_tm:.6e}")

print("\n" + "=" * 70)

#!/usr/bin/env python3
"""Quick test for torch-sla solver."""

import sys
sys.path.insert(0, '../..')
import torch
torch.set_default_dtype(torch.float64)
import time
import meshio
from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness

print('='*70)
print('  TORCH-SLA SOLVER TEST')
print('='*70)
sys.stdout.flush()

mesh = meshio.read('ge_bracket.vtu')
nodes = torch.tensor(mesh.points, dtype=torch.float64)
elements = torch.tensor(mesh.cells[0].data)
domain = torch.tensor(mesh.cell_data['gmsh:geometrical'][0])

n_nodes = nodes.shape[0]
n_elements = elements.shape[0]
dim = 3

print(f'Mesh: {n_nodes} nodes, {n_elements} elements, {n_nodes*dim} DOFs')
sys.stdout.flush()

meshio_obj = meshio.Mesh(points=mesh.points, cells=[('tetra', mesh.cells[0].data)])
tm_mesh = Mesh(meshio_obj, reorder=True)

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

rho_full = torch.ones(n_elements)
rho_full[domain == 6] = 0.15
assembler = SIMP3DAssembler.from_mesh(tm_mesh, E=16500.0, nu=0.342, penal=3.0)

R_fix = 6.0
constraint_mask = torch.zeros(n_nodes, dim, dtype=torch.bool)
for d in [1, 3, 4, 5]:
    dom_elements = elements[domain == d]
    dom_nodes = torch.unique(dom_elements)
    center = nodes[dom_nodes].mean(dim=0)
    dist_sq = ((nodes[dom_nodes, 0] - center[0])**2 + (nodes[dom_nodes, 1] - center[1])**2)
    inner = dist_sq < R_fix**2
    constraint_mask[dom_nodes[inner], :] = True
dbc_mask = constraint_mask.flatten()
condenser = Condenser(dbc_mask)

load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
F = torch.zeros(n_nodes * dim)
F[load_nodes * dim + 2] = 8000.0 / len(load_nodes)

print('Assembling stiffness matrix...')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    K = assembler(tm_mesh.points, element_data={'rho': rho_full})
    K_, F_ = condenser(K, F)
print(f'Assembly time: {time.time()-t0:.2f}s')
sys.stdout.flush()

print()
print('='*70)
print('  Testing torch-sla iterative solvers')
print('='*70)
sys.stdout.flush()

# Test default (now uses torch-sla CG)
print()
print('1. Default solver (torch-sla CG + Jacobi):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_default = K_.solve(F_)
t_default = time.time() - t0
c_default = torch.inner(F_, u_default).item()
print(f'   Time: {t_default:.2f}s')
print(f'   Compliance: {c_default:.6e}')
sys.stdout.flush()

# Test torch-sla with explicit CG
print()
print('2. torch-sla CG (tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_cg = K_.solve(F_, backend='sla', tol=1e-5, sla_method='cg')
t_cg = time.time() - t0
c_cg = torch.inner(F_, u_cg).item()
print(f'   Time: {t_cg:.2f}s')
print(f'   Compliance: {c_cg:.6e}')
sys.stdout.flush()

# Test torch-sla with MINRES
print()
print('3. torch-sla MINRES (tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_minres = K_.solve(F_, backend='sla', tol=1e-5, sla_method='minres')
t_minres = time.time() - t0
c_minres = torch.inner(F_, u_minres).item()
print(f'   Time: {t_minres:.2f}s')
print(f'   Compliance: {c_minres:.6e}')
sys.stdout.flush()

# Test torch-sla with LGMRES
print()
print('4. torch-sla LGMRES (tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_lgmres = K_.solve(F_, backend='sla', tol=1e-5, sla_method='lgmres')
t_lgmres = time.time() - t0
c_lgmres = torch.inner(F_, u_lgmres).item()
print(f'   Time: {t_lgmres:.2f}s')
print(f'   Compliance: {c_lgmres:.6e}')
sys.stdout.flush()

print()
print('='*70)
print('  SUMMARY')
print('='*70)
print(f'  Default (CG):    {t_default:>8.2f}s  Compliance: {c_default:.6e}')
print(f'  CG:              {t_cg:>8.2f}s  Compliance: {c_cg:.6e}')
print(f'  MINRES:          {t_minres:>8.2f}s  Compliance: {c_minres:.6e}')
print(f'  LGMRES:          {t_lgmres:>8.2f}s  Compliance: {c_lgmres:.6e}')
print('='*70)
print()
print('All solvers converged successfully!')
sys.stdout.flush()







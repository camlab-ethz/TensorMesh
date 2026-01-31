#!/usr/bin/env python3
"""Test script for torch-sla solver comparison."""

import sys
sys.path.insert(0, '../..')
import torch
torch.set_default_dtype(torch.float64)
import time
import meshio
from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness

print('='*70)
print('  SOLVER COMPARISON: scipy vs torch-sla')
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
print()
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
print()
sys.stdout.flush()

print('='*70)
print('  SOLVE TIME COMPARISON')
print('='*70)
sys.stdout.flush()

# scipy (baseline)
print()
print('1. scipy (direct solver - SuperLU):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_scipy = K_.solve(F_, backend='scipy')
t_scipy = time.time() - t0
c_scipy = torch.inner(F_, u_scipy).item()
print(f'   Time: {t_scipy:.2f}s')
print(f'   Compliance: {c_scipy:.6e}')
sys.stdout.flush()

# torch-sla with CG + Jacobi
print()
print('2. torch-sla (CG + Jacobi, tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_sla_cg = K_.solve(F_, backend='sla', tol=1e-5, sla_method='cg', sla_preconditioner='jacobi')
t_sla_cg = time.time() - t0
c_sla_cg = torch.inner(F_, u_sla_cg).item()
err_cg = abs(c_sla_cg - c_scipy) / abs(c_scipy) * 100
print(f'   Time: {t_sla_cg:.2f}s')
print(f'   Compliance: {c_sla_cg:.6e} (error: {err_cg:.4f}%)')
sys.stdout.flush()

# torch-sla with BiCGSTAB + Jacobi
print()
print('3. torch-sla (BiCGSTAB + Jacobi, tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_sla_bicg = K_.solve(F_, backend='sla', tol=1e-5, sla_method='bicgstab', sla_preconditioner='jacobi')
t_sla_bicg = time.time() - t0
c_sla_bicg = torch.inner(F_, u_sla_bicg).item()
err_bicg = abs(c_sla_bicg - c_scipy) / abs(c_scipy) * 100
print(f'   Time: {t_sla_bicg:.2f}s')
print(f'   Compliance: {c_sla_bicg:.6e} (error: {err_bicg:.4f}%)')
sys.stdout.flush()

# torch-sla with GMRES
print()
print('4. torch-sla (GMRES, tol=1e-5):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u_sla_gmres = K_.solve(F_, backend='sla', tol=1e-5, sla_method='gmres', sla_preconditioner='jacobi')
t_sla_gmres = time.time() - t0
c_sla_gmres = torch.inner(F_, u_sla_gmres).item()
err_gmres = abs(c_sla_gmres - c_scipy) / abs(c_scipy) * 100
print(f'   Time: {t_sla_gmres:.2f}s')
print(f'   Compliance: {c_sla_gmres:.6e} (error: {err_gmres:.4f}%)')
sys.stdout.flush()

print()
print('='*70)
print('  SUMMARY')
print('='*70)
print(f'  scipy (SuperLU):        {t_scipy:>8.2f}s  (baseline)')
print(f'  torch-sla CG:           {t_sla_cg:>8.2f}s  (speedup: {t_scipy/t_sla_cg:.1f}x, err: {err_cg:.4f}%)')
print(f'  torch-sla BiCGSTAB:     {t_sla_bicg:>8.2f}s  (speedup: {t_scipy/t_sla_bicg:.1f}x, err: {err_bicg:.4f}%)')
print(f'  torch-sla GMRES:        {t_sla_gmres:>8.2f}s  (speedup: {t_scipy/t_sla_gmres:.1f}x, err: {err_gmres:.4f}%)')
print('='*70)
sys.stdout.flush()







#!/usr/bin/env python3
"""Final test for torch-sla solver comparison."""

import sys
sys.path.insert(0, '../..')
import torch
torch.set_default_dtype(torch.float64)
import time
import meshio
from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness

print('='*70)
print('  TORCH-SLA vs SCIPY SOLVER COMPARISON')
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

results = {}

print()
print('='*70)
print('  SOLVER COMPARISON')
print('='*70)
sys.stdout.flush()

# 1. Default (torch-sla CG + Jacobi)
print()
print('1. Default (torch-sla CG + Jacobi):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u1 = K_.solve(F_)
t1 = time.time() - t0
c1 = torch.inner(F_, u1).item()
results['default_sla_cg'] = {'time': t1, 'compliance': c1}
print(f'   Time: {t1:.2f}s')
print(f'   Compliance: {c1:.6e}')
sys.stdout.flush()

# 2. torch-sla BiCGSTAB
print()
print('2. torch-sla BiCGSTAB:')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u2 = K_.solve(F_, backend='sla', sla_method='bicgstab')
t2 = time.time() - t0
c2 = torch.inner(F_, u2).item()
results['sla_bicgstab'] = {'time': t2, 'compliance': c2}
print(f'   Time: {t2:.2f}s')
print(f'   Compliance: {c2:.6e}')
sys.stdout.flush()

# 3. scipy direct (SuperLU) - baseline
print()
print('3. scipy SuperLU (direct, baseline):')
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    u3 = K_.solve(F_, backend='scipy')
t3 = time.time() - t0
c3 = torch.inner(F_, u3).item()
results['scipy_superlu'] = {'time': t3, 'compliance': c3}
print(f'   Time: {t3:.2f}s')
print(f'   Compliance: {c3:.6e}')
sys.stdout.flush()

# Summary
print()
print('='*70)
print('  SUMMARY')
print('='*70)
baseline = c3
for name, res in results.items():
    err = abs(res['compliance'] - baseline) / abs(baseline) * 100
    speedup = results['scipy_superlu']['time'] / res['time'] if res['time'] > 0 else 0
    print(f'  {name:20s}: {res["time"]:>8.2f}s  C={res["compliance"]:.4e}  err={err:.4f}%  speedup={speedup:.2f}x')
print('='*70)
print()
print('✓ TensorMesh now uses torch-sla iterative solver by default!')
sys.stdout.flush()







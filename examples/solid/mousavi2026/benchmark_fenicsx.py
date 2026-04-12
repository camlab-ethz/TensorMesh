"""FEniCSx benchmark: Poisson-boomerang-bc4 + linear-elasticity-squarehollow.

Usage (in fenics conda env):
    conda activate fenics
    python benchmark_fenicsx.py --input-dir ./benchmark_results --problem poisson
    python benchmark_fenicsx.py --input-dir ./benchmark_results --problem elasticity
"""

import argparse
import gc
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl


def load_mesh(msh_path):
    from dolfinx.io.gmsh import read_from_msh
    return read_from_msh(msh_path, MPI.COMM_SELF, gdim=2).mesh


def build_scalar_node_map(mesh_dx, points_orig):
    V = fem.functionspace(mesh_dx, ("Lagrange", 1))
    coords = V.tabulate_dof_coordinates()[:, :2]
    tree = cKDTree(points_orig)
    dists, dof2orig = tree.query(coords)
    assert dists.max() < 1e-10, f"Node mismatch: max dist = {dists.max():.2e}"
    n_orig = points_orig.shape[0]
    orig2dof = np.full(n_orig, -1, dtype=int)
    orig2dof[dof2orig] = np.arange(len(dof2orig))
    return V, dof2orig, orig2dof


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

def run_poisson(input_dir, n_runs):
    print("=" * 60)
    print("Example 1: Poisson-boomerang-bc4 (FEniCSx)")
    print("=" * 60)
    pdir = os.path.join(input_dir, 'poisson_boomerang')
    bc = dict(np.load(os.path.join(pdir, 'bc_arrays.npz'), allow_pickle=True))
    pts = bc['points']
    mesh = load_mesh(os.path.join(pdir, 'mesh.msh'))
    V, dof2orig, orig2dof = build_scalar_node_map(mesh, pts)
    print(f"  Mesh: {mesh.geometry.x.shape[0]} nodes")

    def solve_once():
        u_t = ufl.TrialFunction(V)
        v_t = ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
        f = fem.Function(V)
        f.x.array[:] = bc['source_values'][dof2orig]
        L = f * v_t * ufl.dx

        dir_dofs = np.where(bc['dirichlet_mask'].astype(bool)[dof2orig])[0]
        u_D = fem.Function(V)
        u_D.x.array[dir_dofs] = bc['dirichlet_values'][dof2orig[dir_dofs]]
        bcs = [fem.dirichletbc(u_D, dir_dofs)]

        if bc['neumann_mask'].astype(bool).any():
            g_n = fem.Function(V)
            g_n.x.array[:] = bc['neumann_values'][dof2orig]
            L += g_n * v_t * ufl.ds

        if bc['robin_mask'].astype(bool).any():
            af = fem.Function(V)
            af.x.array[:] = bc['robin_alpha_values'][dof2orig]
            a += af * u_t * v_t * ufl.ds
            gf = fem.Function(V)
            gf.x.array[:] = bc['robin_g_values'][dof2orig]
            L += gf * v_t * ufl.ds

        prob = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="p_",
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = prob.solve()
        return uh.x.array[orig2dof].copy()

    # Warmup
    solve_once()
    gc.collect(); PETSc.garbage_cleanup()

    times = []
    for i in range(n_runs):
        gc.collect(); PETSc.garbage_cleanup()
        t0 = time.perf_counter()
        u_out = solve_once()
        times.append(time.perf_counter() - t0)
        gc.collect(); PETSc.garbage_cleanup()

    print(f"  Times: {[f'{t*1000:.1f}ms' for t in times]}")
    np.savez(os.path.join(pdir, 'solution_fenicsx.npz'), u=u_out)
    with open(os.path.join(pdir, 'timing_fenicsx.json'), 'w') as f_out:
        json.dump({'cpu_times_ms': [t * 1000 for t in times],
                    'cpu_mean_ms': float(np.mean(times) * 1000),
                    'cpu_std_ms': float(np.std(times) * 1000)}, f_out, indent=2)


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------

def run_elasticity(input_dir, n_runs):
    """Elasticity benchmark using low-level PETSc API to avoid dolfinx 0.10 LinearProblem crash."""
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

    print("=" * 60)
    print("Example 2: Linear-elasticity-squarehollow (FEniCSx)")
    print("=" * 60)
    edir = os.path.join(input_dir, 'elasticity_squarehollow')
    bc = dict(np.load(os.path.join(edir, 'bc_arrays.npz'), allow_pickle=True))
    pts = bc['points']
    E, nu = float(bc['E']), float(bc['nu'])
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 - 2 * nu) * (1 + nu))
    n_nodes = pts.shape[0]

    mesh = load_mesh(os.path.join(edir, 'mesh.msh'))
    print(f"  Mesh: {mesh.geometry.x.shape[0]} nodes")

    # Scalar node map
    V_s = fem.functionspace(mesh, ("Lagrange", 1))
    coords_s = V_s.tabulate_dof_coordinates()[:, :2]
    tree = cKDTree(pts)
    _, sdof2orig = tree.query(coords_s)
    n_sdofs = V_s.dofmap.index_map.size_local
    orig2sdof = {}
    for i in range(n_sdofs):
        orig2sdof[sdof2orig[i]] = i

    # Vector function space and forms (build once)
    V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    bs = V.dofmap.bs
    u_t = ufl.TrialFunction(V)
    v_t = ufl.TestFunction(V)
    eps = lambda u: 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    sig = lambda u: lam * ufl.tr(eps(u)) * ufl.Identity(2) + 2 * mu * eps(u)
    a_form = ufl.inner(sig(u_t), eps(v_t)) * ufl.dx
    L_form = ufl.inner(fem.Constant(mesh, np.array([0., 0.])), v_t) * ufl.dx

    # Parse segments and build BCs
    n_segs = 0
    while f'seg_{n_segs}_indices' in bc:
        n_segs += 1

    bcs_list = []
    for seg_i in range(n_segs):
        idx = bc.get(f'seg_{seg_i}_indices', np.array([])).astype(int)
        if len(idx) == 0:
            continue
        for d in range(2):
            tk = f'seg_{seg_i}_dim_{d}_type'
            gk = f'seg_{seg_i}_dim_{d}_g'
            if tk not in bc or gk not in bc:
                continue
            btype = str(bc[tk])
            gv = bc[gk]
            if btype == 'dirichlet':
                dofs, vals = [], []
                for j, nd in enumerate(idx):
                    if nd in orig2sdof:
                        dofs.append(orig2sdof[nd] * bs + d)
                        vals.append(gv[j])
                if dofs:
                    da = np.array(dofs, dtype=np.int32)
                    uf = fem.Function(V)
                    uf.x.array[da] = np.array(vals)
                    bcs_list.append(fem.dirichletbc(uf, da))
            elif btype == 'neumann' and np.any(np.abs(gv) > 1e-15):
                tf = fem.Function(V)
                for j, nd in enumerate(idx):
                    if nd in orig2sdof:
                        tf.x.array[orig2sdof[nd] * bs + d] = gv[j]
                L_form += ufl.inner(tf, v_t) * ufl.ds

    # Compile forms once
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Low-level assemble + solve
    def solve_once():
        A = assemble_matrix(a_compiled, bcs=bcs_list)
        A.assemble()
        b = assemble_vector(L_compiled)
        apply_lifting(b, [a_compiled], [bcs_list])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs_list)
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        x = A.createVecRight()
        ksp.solve(b, x)
        u_arr = x.getArray().copy()
        ksp.destroy(); A.destroy(); b.destroy(); x.destroy()
        return u_arr

    # Warmup
    solve_once()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        u_arr = solve_once()
        times.append(time.perf_counter() - t0)

    u_res = np.zeros((n_nodes, 2))
    for i in range(n_sdofs):
        for dc in range(2):
            u_res[sdof2orig[i], dc] = u_arr[i * bs + dc]

    print(f"  Times: {[f'{t*1000:.1f}ms' for t in times]}")
    np.savez(os.path.join(edir, 'solution_fenicsx.npz'), u=u_res)
    with open(os.path.join(edir, 'timing_fenicsx.json'), 'w') as f_out:
        json.dump({'cpu_times_ms': [t * 1000 for t in times],
                    'cpu_mean_ms': float(np.mean(times) * 1000),
                    'cpu_std_ms': float(np.std(times) * 1000)}, f_out, indent=2)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='./benchmark_results')
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--problem', default='all', choices=['poisson', 'elasticity', 'all'])
    args = parser.parse_args()

    if args.problem in ['poisson', 'all']:
        run_poisson(args.input_dir, args.n_runs)
    if args.problem in ['elasticity', 'all']:
        if args.problem == 'all':
            print()
        run_elasticity(args.input_dir, args.n_runs)

    print("\nFEniCSx benchmark done.")


if __name__ == '__main__':
    main()

"""TensorMesh benchmark: Poisson-boomerang-bc4 + linear-elasticity-squarehollow.

Solves both problems on CPU and GPU, exports mesh/BC/solution data for FEniCSx comparison.

Usage:
    source ~/venvs/tensorgalerkin/bin/activate
    python benchmark_tensormesh.py --output-dir ./benchmark_results
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from mesh_gen import create_mesh, _generate_mesh_from_boundary, CACHE_DIR
from geometry import get_boomerang_boundary, get_squarehollow_holes, polygon_boundary
from boundary import (
    BCGenerator, Dirichlet, Neumann, RandomBCTypes, NeumannHomogenous,
    get_centered_radial_cosine, get_random_scalar_function,
)
from bc_segments import draw_valid_bcs, build_bc_masks_and_values
from poisson_solver import solve_poisson


# ---------------------------------------------------------------------------
# Poisson benchmark
# ---------------------------------------------------------------------------

def setup_poisson_boomerang_bc4(seed=42):
    """Create mesh + fixed BCs for Poisson-boomerang-bc4."""
    np.random.seed(seed)
    mesh = create_mesh('boomerang', order=1).double()

    C, R = (0., -0.375), 0.625
    bc_gen = BCGenerator(ndims=1, dists=[
        Dirichlet(rng=(1.0, 4.0), modes=6, C=C, R=R),
        RandomBCTypes(
            ranges={'dirichlet': (1.0, 4.0), 'neumann': (2.0, 10.0),
                    'robin': ((2.0, 10.0), (0.2, 0.6))},
            modes={'dirichlet': 6, 'neumann': 4, 'robin': (4, 3)}, C=C, R=R),
        RandomBCTypes(
            ranges={'dirichlet': (1.0, 4.0), 'neumann': (2.0, 10.0),
                    'robin': ((2.0, 10.0), (0.2, 0.6))},
            modes={'dirichlet': 6, 'neumann': 4, 'robin': (4, 3)}, C=C, R=R),
        RandomBCTypes(
            ranges={'dirichlet': (1.0, 4.0), 'neumann': (2.0, 10.0),
                    'robin': ((2.0, 10.0), (0.2, 0.6))},
            modes={'dirichlet': 6, 'neumann': 4, 'robin': (4, 3)}, C=C, R=R),
    ])

    shape_bcs = draw_valid_bcs(bc_gen, min_non_dirichlet_length=0.2,
                                max_non_dirichlet_length=0.51, ndims=1)
    bc_data = build_bc_masks_and_values(mesh.points, mesh.boundary_mask,
                                         shape_bcs, ndims=1, center=C)
    source_func = get_centered_radial_cosine(ord=2, scale=20.0)

    return mesh, bc_data, source_func


def solve_poisson_timed(mesh, bc_data, source_func, n_runs=5, warmup=1, device='cpu'):
    """Solve Poisson and return solution + timing."""
    if device != 'cpu':
        mesh = mesh.to(device)
        for k in ['dirichlet_mask', 'neumann_mask', 'robin_mask',
                   'dirichlet_values', 'neumann_values', 'robin_g_values', 'robin_alpha_values']:
            if k in bc_data and bc_data[k] is not None:
                bc_data[k] = bc_data[k].to(device)

    masks = {k: bc_data[k] for k in ['dirichlet_mask', 'neumann_mask', 'robin_mask']}
    vals = {k: bc_data[k] for k in ['dirichlet_values', 'neumann_values',
                                      'robin_g_values', 'robin_alpha_values']}

    # Warmup
    for _ in range(warmup):
        u = solve_poisson(mesh, source_func, masks, vals)
        if device != 'cpu':
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = solve_poisson(mesh, source_func, masks, vals)
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return u.detach().cpu().numpy(), times


# ---------------------------------------------------------------------------
# Linear elasticity benchmark
# ---------------------------------------------------------------------------

def setup_elasticity_squarehollow(seed=2):
    """Create mesh + fixed BCs for linear-elasticity-squarehollow."""
    np.random.seed(seed)
    mesh = create_mesh('squarehollow', order=1).double()

    # Steel: E=200GPa, nu=0.3
    E, nu = 200.0e3, 0.3
    char_stress = 100.0
    disp_rng = (0.1e-3, 0.5e-3)
    trac_rng_nd = (4.0 / char_stress, 40.0 / char_stress)

    bc_gen = BCGenerator(ndims=2, dists=[
        NeumannHomogenous(),
        Dirichlet(rng=disp_rng, modes=2),
        Neumann(rng=trac_rng_nd, modes=2),
        Neumann(rng=trac_rng_nd, modes=2),
    ])
    shape_bcs = draw_valid_bcs(bc_gen, min_non_dirichlet_length=0.0,
                                max_non_dirichlet_length=1.0, ndims=2)
    bc_data = build_bc_masks_and_values(mesh.points, mesh.boundary_mask,
                                         shape_bcs, ndims=2, center=(0., 0.))
    return mesh, bc_data, E, nu


def solve_elasticity_timed(mesh, bc_data, E, nu, n_runs=5, warmup=1, device='cpu'):
    """Solve linear elasticity and return solution + timing."""
    from tensormesh import LinearElasticityElementAssembler, Condenser
    from poisson_solver import _assemble_neumann_rhs, _get_boundary_edges

    if device != 'cpu':
        mesh_d = mesh.to(device)
        bc_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in bc_data.items()}
    else:
        mesh_d = mesh
        bc_d = bc_data

    def _solve_once():
        # Assemble stiffness
        asm = LinearElasticityElementAssembler.from_mesh(mesh_d, E=E, nu=nu)
        K = asm(mesh_d.points)

        n_pts = mesh_d.points.shape[0]
        dtype = mesh_d.points.dtype
        dev = mesh_d.points.device

        # Build component-wise Dirichlet mask and values
        # For 2D: dirichlet_mask [2*n_pts], dirichlet_values [2*n_pts]
        dir_mask_2d = torch.zeros(2 * n_pts, dtype=torch.bool, device=dev)
        dir_vals_2d = torch.zeros(2 * n_pts, dtype=dtype, device=dev)

        # RHS = 0 (no body force)
        f = torch.zeros(2 * n_pts, dtype=dtype, device=dev)

        segments = bc_d['segments']
        for seg in segments:
            if seg.indices is None or len(seg.indices) == 0:
                continue
            idx = torch.tensor(seg.indices, dtype=torch.long, device=dev)
            for d in range(2):
                dim_bc = seg.dims[d]
                if dim_bc.type == 'dirichlet' and dim_bc.values and 'g' in dim_bc.values:
                    g = torch.tensor(dim_bc.values['g'], dtype=dtype, device=dev)
                    dir_mask_2d[idx * 2 + d] = True
                    dir_vals_2d[idx * 2 + d] = g
                elif dim_bc.type == 'neumann' and dim_bc.values and 'g' in dim_bc.values:
                    g_vals = dim_bc.values['g']
                    if np.any(np.abs(g_vals) > 1e-15):
                        # Lumped Neumann: f[node*2+d] += g * boundary_measure
                        g_t = torch.tensor(g_vals, dtype=dtype, device=dev)
                        f.scatter_add_(0, idx * 2 + d, g_t)

        condenser = Condenser(dir_mask_2d, dir_vals_2d)
        K_, f_ = condenser(K, f)
        u_inner = K_.solve(f_)
        u = condenser.recover(u_inner)
        return u.reshape(n_pts, 2)

    # Warmup
    for _ in range(warmup):
        u = _solve_once()
        if device != 'cpu':
            torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(n_runs):
        if device != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        u = _solve_once()
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return u.detach().cpu().numpy(), times


# ---------------------------------------------------------------------------
# Export BC data for FEniCSx
# ---------------------------------------------------------------------------

def export_bc_data(bc_data, filepath):
    """Export BC segment data as .npz for FEniCSx to read."""
    segments = bc_data['segments']
    export = {}
    export['n_segments'] = len(segments)
    for i, seg in enumerate(segments):
        export[f'seg_{i}_center'] = seg.center
        export[f'seg_{i}_radius'] = seg.radius
        export[f'seg_{i}_indices'] = seg.indices if seg.indices is not None else np.array([])
        export[f'seg_{i}_ndims'] = len(seg.dims)
        for d, dim_bc in enumerate(seg.dims):
            export[f'seg_{i}_dim_{d}_type'] = dim_bc.type
            if dim_bc.values:
                for k, v in dim_bc.values.items():
                    export[f'seg_{i}_dim_{d}_{k}'] = v
    np.savez(filepath, **export)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./benchmark_results')
    parser.add_argument('--n-runs', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Example 1: Poisson-boomerang-bc4 ----
    print("=" * 60)
    print("Example 1: Poisson-boomerang-bc4")
    print("=" * 60)
    poisson_dir = os.path.join(args.output_dir, 'poisson_boomerang')
    os.makedirs(poisson_dir, exist_ok=True)

    mesh_p, bc_p, source_p = setup_poisson_boomerang_bc4(seed=42)
    print(f"Mesh: {mesh_p.points.shape[0]} nodes, {mesh_p.boundary_mask.sum().item()} boundary")

    # Export mesh
    mesh_path = os.path.join(poisson_dir, 'mesh.msh')
    if not os.path.exists(mesh_path):
        import shutil
        src = os.path.join(CACHE_DIR, [f for f in os.listdir(CACHE_DIR) if 'boomerang' in f][0])
        shutil.copy(src, mesh_path)

    # Export BC data
    export_bc_data(bc_p, os.path.join(poisson_dir, 'bc_data.npz'))

    # Also save source function params and raw BC arrays for FEniCSx
    pts_np = mesh_p.points.detach().cpu().numpy()
    np.savez(os.path.join(poisson_dir, 'bc_arrays.npz'),
             points=pts_np,
             boundary_mask=mesh_p.boundary_mask.cpu().numpy(),
             dirichlet_mask=bc_p['dirichlet_mask'].cpu().numpy(),
             dirichlet_values=bc_p['dirichlet_values'].cpu().numpy(),
             neumann_mask=bc_p['neumann_mask'].cpu().numpy(),
             neumann_values=bc_p['neumann_values'].cpu().numpy(),
             robin_mask=bc_p['robin_mask'].cpu().numpy(),
             robin_g_values=bc_p['robin_g_values'].cpu().numpy(),
             robin_alpha_values=bc_p['robin_alpha_values'].cpu().numpy(),
             source_values=source_p(pts_np))

    # Solve CPU
    print("  Solving (CPU)...")
    u_cpu, times_cpu = solve_poisson_timed(mesh_p, bc_p, source_p,
                                            n_runs=args.n_runs, device='cpu')
    print(f"  CPU times: {[f'{t*1000:.1f}ms' for t in times_cpu]}")
    np.savez(os.path.join(poisson_dir, 'solution_tensormesh_cpu.npz'), u=u_cpu)

    # Solve GPU
    gpu_times = None
    if torch.cuda.is_available():
        print("  Solving (GPU)...")
        # Re-setup to get fresh tensors (bc_data was modified by CPU solve's .to())
        mesh_p2, bc_p2, source_p2 = setup_poisson_boomerang_bc4(seed=42)
        u_gpu, times_gpu = solve_poisson_timed(mesh_p2, bc_p2, source_p2,
                                                n_runs=args.n_runs, device='cuda')
        print(f"  GPU times: {[f'{t*1000:.1f}ms' for t in times_gpu]}")
        np.savez(os.path.join(poisson_dir, 'solution_tensormesh_gpu.npz'), u=u_gpu)
        gpu_times = times_gpu

    timing = {
        'cpu_times_ms': [t * 1000 for t in times_cpu],
        'cpu_mean_ms': np.mean(times_cpu) * 1000,
        'cpu_std_ms': np.std(times_cpu) * 1000,
    }
    if gpu_times:
        timing['gpu_times_ms'] = [t * 1000 for t in gpu_times]
        timing['gpu_mean_ms'] = np.mean(gpu_times) * 1000
        timing['gpu_std_ms'] = np.std(gpu_times) * 1000
    with open(os.path.join(poisson_dir, 'timing_tensormesh.json'), 'w') as f:
        json.dump(timing, f, indent=2)

    # ---- Example 2: Linear-elasticity-squarehollow ----
    print()
    print("=" * 60)
    print("Example 2: Linear-elasticity-squarehollow")
    print("=" * 60)
    elast_dir = os.path.join(args.output_dir, 'elasticity_squarehollow')
    os.makedirs(elast_dir, exist_ok=True)

    mesh_e, bc_e, E, nu = setup_elasticity_squarehollow(seed=2)
    print(f"Mesh: {mesh_e.points.shape[0]} nodes, {mesh_e.boundary_mask.sum().item()} boundary")

    # Export mesh
    mesh_path_e = os.path.join(elast_dir, 'mesh.msh')
    if not os.path.exists(mesh_path_e):
        import shutil
        src = os.path.join(CACHE_DIR, [f for f in os.listdir(CACHE_DIR) if 'squarehollow' in f][0])
        shutil.copy(src, mesh_path_e)

    # Export BC data
    export_bc_data(bc_e, os.path.join(elast_dir, 'bc_data.npz'))

    pts_e = mesh_e.points.detach().cpu().numpy()
    bc_arrays_e = {
        'points': pts_e,
        'boundary_mask': mesh_e.boundary_mask.cpu().numpy(),
        'E': E, 'nu': nu,
    }
    segments = bc_e['segments']
    for i, seg in enumerate(segments):
        if seg.indices is not None:
            bc_arrays_e[f'seg_{i}_indices'] = seg.indices
            for d in range(2):
                bc_arrays_e[f'seg_{i}_dim_{d}_type'] = seg.dims[d].type
                if seg.dims[d].values and 'g' in seg.dims[d].values:
                    bc_arrays_e[f'seg_{i}_dim_{d}_g'] = seg.dims[d].values['g']
    np.savez(os.path.join(elast_dir, 'bc_arrays.npz'), **bc_arrays_e)

    # Solve CPU
    print("  Solving (CPU)...")
    u_cpu_e, times_cpu_e = solve_elasticity_timed(mesh_e, bc_e, E, nu,
                                                    n_runs=args.n_runs, device='cpu')
    print(f"  CPU times: {[f'{t*1000:.1f}ms' for t in times_cpu_e]}")
    np.savez(os.path.join(elast_dir, 'solution_tensormesh_cpu.npz'), u=u_cpu_e)

    # Solve GPU
    gpu_times_e = None
    if torch.cuda.is_available():
        print("  Solving (GPU)...")
        mesh_e2, bc_e2, E2, nu2 = setup_elasticity_squarehollow(seed=2)
        u_gpu_e, times_gpu_e = solve_elasticity_timed(mesh_e2, bc_e2, E2, nu2,
                                                        n_runs=args.n_runs, device='cuda')
        print(f"  GPU times: {[f'{t*1000:.1f}ms' for t in times_gpu_e]}")
        np.savez(os.path.join(elast_dir, 'solution_tensormesh_gpu.npz'), u=u_gpu_e)
        gpu_times_e = times_gpu_e

    timing_e = {
        'cpu_times_ms': [t * 1000 for t in times_cpu_e],
        'cpu_mean_ms': np.mean(times_cpu_e) * 1000,
        'cpu_std_ms': np.std(times_cpu_e) * 1000,
    }
    if gpu_times_e:
        timing_e['gpu_times_ms'] = [t * 1000 for t in gpu_times_e]
        timing_e['gpu_mean_ms'] = np.mean(gpu_times_e) * 1000
        timing_e['gpu_std_ms'] = np.std(gpu_times_e) * 1000
    with open(os.path.join(elast_dir, 'timing_tensormesh.json'), 'w') as f:
        json.dump(timing_e, f, indent=2)

    print("\nTensorMesh benchmark done. Results in:", args.output_dir)


if __name__ == '__main__':
    main()

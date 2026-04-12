"""Compare TensorMesh vs FEniCSx benchmark results.

Usage:
    python benchmark_compare.py --results-dir ./benchmark_results
"""

import argparse
import json
import os

import numpy as np


def compare_solutions(dir_path, name, is_vector=False):
    """Load and compare solutions from TensorMesh and FEniCSx."""
    u_fx = np.load(os.path.join(dir_path, 'solution_fenicsx.npz'))['u']
    u_tm_cpu = np.load(os.path.join(dir_path, 'solution_tensormesh_cpu.npz'))['u']

    gpu_path = os.path.join(dir_path, 'solution_tensormesh_gpu.npz')
    u_tm_gpu = np.load(gpu_path)['u'] if os.path.exists(gpu_path) else None

    # Match array shapes
    n = min(len(u_fx), len(u_tm_cpu))
    u_fx = u_fx[:n]
    u_tm_cpu = u_tm_cpu[:n]

    # Error metrics
    if is_vector:
        diff_cpu = u_tm_cpu - u_fx
        l2_cpu = np.sqrt(np.sum(diff_cpu ** 2)) / np.sqrt(np.sum(u_fx ** 2) + 1e-30)
        linf_cpu = np.max(np.abs(diff_cpu))
        ref_max = np.max(np.abs(u_fx))
    else:
        diff_cpu = u_tm_cpu - u_fx
        l2_cpu = np.sqrt(np.sum(diff_cpu ** 2)) / np.sqrt(np.sum(u_fx ** 2) + 1e-30)
        linf_cpu = np.max(np.abs(diff_cpu))
        ref_max = np.max(np.abs(u_fx))

    result = {
        'name': name,
        'n_nodes': n,
        'cpu_l2_rel': l2_cpu,
        'cpu_linf': linf_cpu,
        'ref_max': ref_max,
    }

    if u_tm_gpu is not None:
        u_tm_gpu = u_tm_gpu[:n]
        diff_gpu = u_tm_gpu - u_fx
        result['gpu_l2_rel'] = np.sqrt(np.sum(diff_gpu ** 2)) / np.sqrt(np.sum(u_fx ** 2) + 1e-30)
        result['gpu_linf'] = np.max(np.abs(diff_gpu))

    return result


def load_timing(dir_path):
    """Load timing data from JSON files."""
    tm_path = os.path.join(dir_path, 'timing_tensormesh.json')
    fx_path = os.path.join(dir_path, 'timing_fenicsx.json')

    tm = json.load(open(tm_path)) if os.path.exists(tm_path) else {}
    fx = json.load(open(fx_path)) if os.path.exists(fx_path) else {}
    return tm, fx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='./benchmark_results')
    args = parser.parse_args()

    examples = [
        ('poisson_boomerang', 'Poisson (boomerang, bc4)', False),
        ('elasticity_squarehollow', 'Elasticity (squarehollow)', True),
    ]

    print()
    print("=" * 80)
    print("  Benchmark Results: TensorMesh vs FEniCSx")
    print("=" * 80)

    for subdir, label, is_vector in examples:
        dir_path = os.path.join(args.results_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"\n  {label}: SKIPPED (no data)")
            continue

        # Solution comparison
        err = compare_solutions(dir_path, label, is_vector)
        tm_t, fx_t = load_timing(dir_path)

        print(f"\n  {label}")
        print(f"  {'─' * 70}")
        print(f"  Nodes: {err['n_nodes']}")
        print(f"  Reference solution max: {err['ref_max']:.4e}")
        print()

        # Error table
        print(f"  {'Solver':<25} {'L2 Rel Error':<15} {'Linf Error':<15}")
        print(f"  {'─' * 55}")
        print(f"  {'FEniCSx (CPU)':<25} {'(reference)':<15} {'(reference)':<15}")
        print(f"  {'TensorMesh (CPU)':<25} {err['cpu_l2_rel']:<15.2e} {err['cpu_linf']:<15.2e}")
        if 'gpu_l2_rel' in err:
            print(f"  {'TensorMesh (GPU)':<25} {err['gpu_l2_rel']:<15.2e} {err['gpu_linf']:<15.2e}")

        # Timing table
        print()
        print(f"  {'Solver':<25} {'Mean (ms)':<15} {'Std (ms)':<15}")
        print(f"  {'─' * 55}")
        if fx_t:
            print(f"  {'FEniCSx (CPU)':<25} {fx_t.get('cpu_mean_ms', 0):<15.1f} {fx_t.get('cpu_std_ms', 0):<15.1f}")
        if tm_t:
            print(f"  {'TensorMesh (CPU)':<25} {tm_t.get('cpu_mean_ms', 0):<15.1f} {tm_t.get('cpu_std_ms', 0):<15.1f}")
            if 'gpu_mean_ms' in tm_t:
                print(f"  {'TensorMesh (GPU)':<25} {tm_t.get('gpu_mean_ms', 0):<15.1f} {tm_t.get('gpu_std_ms', 0):<15.1f}")

        # Speedup
        if fx_t and tm_t:
            fx_mean = fx_t.get('cpu_mean_ms', 1)
            tm_cpu_mean = tm_t.get('cpu_mean_ms', 1)
            print()
            print(f"  Speedup (CPU): {fx_mean / tm_cpu_mean:.1f}x")
            if 'gpu_mean_ms' in tm_t:
                tm_gpu_mean = tm_t['gpu_mean_ms']
                print(f"  Speedup (GPU): {fx_mean / tm_gpu_mean:.1f}x")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()

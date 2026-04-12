"""
Assembly Benchmark: Single-GPU vs Multi-GPU (multiprocessing)
==============================================================
Multi-GPU assembly using torch.multiprocessing.

Usage:
    python benchmark_assembly_mp.py                          # auto-detect GPUs
    python benchmark_assembly_mp.py --start 50 --step 50 --max 500
"""

import argparse
import time
import gc
import numpy as np
import torch
import torch.multiprocessing as mp
import meshio
import tensormesh as tm
from tensormesh.mesh import Mesh
from tensormesh.distributed import DistributedMesh


# ─── Structured mesh generation ─────────────────────────────────────

def gen_structured_cube(n: int) -> Mesh:
    """Structured tet mesh on [0,1]^3.  Points=(n+1)^3, tets=5*n^3."""
    lin = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    n1 = n + 1
    def idx(i, j, k):
        return i * n1 * n1 + j * n1 + k

    ii, jj, kk = np.mgrid[0:n, 0:n, 0:n]
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
    v0 = idx(ii, jj, kk);     v1 = idx(ii+1, jj, kk)
    v2 = idx(ii+1, jj+1, kk); v3 = idx(ii, jj+1, kk)
    v4 = idx(ii, jj, kk+1);   v5 = idx(ii+1, jj, kk+1)
    v6 = idx(ii+1, jj+1, kk+1); v7 = idx(ii, jj+1, kk+1)

    tets = np.concatenate([
        np.stack([v0, v1, v3, v4], axis=1),
        np.stack([v1, v2, v3, v6], axis=1),
        np.stack([v1, v4, v5, v6], axis=1),
        np.stack([v3, v4, v6, v7], axis=1),
        np.stack([v1, v3, v4, v6], axis=1),
    ], axis=0)

    m_io = meshio.Mesh(points=points, cells=[("tetra", tets)])
    mesh = Mesh(m_io)
    pts = mesh.points
    is_boundary = (
        (pts[:, 0] == 0) | (pts[:, 0] == 1) |
        (pts[:, 1] == 0) | (pts[:, 1] == 1) |
        (pts[:, 2] == 0) | (pts[:, 2] == 1)
    )
    mesh.register_point_data("is_boundary", is_boundary)
    return mesh


# ─── GPU helpers ────────────────────────────────────────────────────

def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def peak_mb(device_idx=0):
    return torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)


def fmt_mb(mb):
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


# ─── Worker function for multiprocessing ────────────────────────────

def _mp_assemble_worker(rank, submesh, device_id, quadrature_order, return_dict):
    """Run assembly in a separate process.

    Each process gets its own Python interpreter → no GIL contention.
    """
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)

        orig_nid = submesh.point_data['orig_nid'].clone()
        submesh.to(device)

        asm = tm.LaplaceElementAssembler.from_mesh(
            submesh, quadrature_order=quadrature_order
        )

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device_id)

        t0 = time.perf_counter()
        K_local = asm()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        mem = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)

        values = K_local.edata.cpu()
        global_row = orig_nid[K_local.row.cpu()]
        global_col = orig_nid[K_local.col.cpu()]

        return_dict[rank] = {
            'values': values,
            'row': global_row,
            'col': global_col,
            'time': elapsed,
            'mem_mb': mem,
        }
    except Exception as e:
        return_dict[rank] = {'error': str(e)}


def bench_multi_gpu_mp(mesh, num_partitions, quadrature_order=2):
    """Multi-GPU assembly using torch.multiprocessing.

    Returns (wall_time, per_gpu_asm_time_list, per_gpu_peak_mb_list) or None on OOM.
    """
    reset_gpu()

    dmesh = DistributedMesh(mesh, num_partitions=num_partitions)

    # Use mp.Manager for inter-process communication
    manager = mp.Manager()
    return_dict = manager.dict()

    t_wall_start = time.perf_counter()

    processes = []
    for i in range(num_partitions):
        submesh = dmesh.submeshes[i]
        if submesh is None:
            return_dict[i] = {
                'values': torch.tensor([], dtype=torch.float64),
                'row': torch.tensor([], dtype=torch.long),
                'col': torch.tensor([], dtype=torch.long),
                'time': 0.0, 'mem_mb': 0.0,
            }
            continue

        device_id = dmesh.devices[i].index if dmesh.devices[i].index is not None else 0
        p = mp.Process(
            target=_mp_assemble_worker,
            args=(i, submesh, device_id, quadrature_order, return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    t_wall = time.perf_counter() - t_wall_start

    # Check for errors / OOM
    for i in range(num_partitions):
        if i not in return_dict:
            return None
        if 'error' in return_dict[i]:
            err = return_dict[i]['error']
            if 'out of memory' in err.lower():
                return None
            raise RuntimeError(f"Partition {i}: {err}")

    times = [return_dict[i]['time'] for i in range(num_partitions)]
    mems = [return_dict[i]['mem_mb'] for i in range(num_partitions)]

    del dmesh
    reset_gpu()
    return t_wall, times, mems


def bench_single_gpu(mesh):
    """Single-GPU assembly.  Returns (time_s, peak_mb) or None on OOM."""
    reset_gpu()
    try:
        mesh_gpu = mesh.clone().cuda()
        torch.cuda.synchronize()
        reset_gpu()

        t0 = time.perf_counter()
        K = tm.LaplaceElementAssembler.from_mesh(mesh_gpu)()
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        mem = peak_mb(0)

        del K, mesh_gpu
        reset_gpu()
        return t, mem
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
            reset_gpu()
            return None
        raise


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Assembly benchmark: 1-GPU vs N-GPU (multiprocessing)")
    parser.add_argument("--partitions", type=int, default=None,
                        help="Number of GPU partitions (default: all GPUs)")
    parser.add_argument("--start", type=int, default=30,
                        help="Starting n (divisions per axis)")
    parser.add_argument("--step", type=int, default=20,
                        help="Step size for n")
    parser.add_argument("--max", type=int, default=1000,
                        help="Max n to try")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    num_partitions = args.partitions or num_gpus

    print(f"GPUs: {num_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"Partitions: {num_partitions}")
    print(f"Method: torch.multiprocessing (true parallel, no GIL)")
    print()

    header = (
        f"{'n':>5} | {'Points':>12} | {'Tets':>12} | "
        f"{'1-GPU time':>10} {'1-GPU mem':>10} | "
        f"{'N-GPU wall':>10} {'N-GPU max_t':>11} {'N-GPU mem/card':>14} | "
        f"{'Speedup':>7} {'Mem save':>9}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    single_oom = False
    multi_oom = False

    n = args.start
    while n <= args.max:
        mesh = gen_structured_cube(n)
        n_pts = mesh.n_points
        n_tets = sum(mesh.cells[k].shape[0] for k in mesh.cells.keys())

        # Single GPU
        if not single_oom:
            res_single = bench_single_gpu(mesh)
            if res_single is None:
                single_oom = True
                s_time_str = "OOM"
                s_mem_str = "OOM"
            else:
                s_time, s_mem = res_single
                s_time_str = f"{s_time:.2f}s"
                s_mem_str = fmt_mb(s_mem)
        else:
            s_time_str = "OOM"
            s_mem_str = "OOM"
            res_single = None

        # Multi GPU (multiprocessing)
        if not multi_oom:
            res_multi = bench_multi_gpu_mp(mesh, num_partitions)
            if res_multi is None:
                multi_oom = True
                m_wall_str = "OOM"
                m_maxt_str = "OOM"
                m_mem_str = "OOM"
            else:
                m_wall, m_times, m_mems = res_multi
                m_wall_str = f"{m_wall:.2f}s"
                m_maxt_str = f"{max(m_times):.2f}s"
                m_max_mem = max(m_mems)
                m_mem_str = fmt_mb(m_max_mem)
        else:
            m_wall_str = "OOM"
            m_maxt_str = "OOM"
            m_mem_str = "OOM"
            res_multi = None

        # Speedup & memory saving
        if res_single and res_multi:
            speedup = f"{s_time / m_wall:.2f}x"
            mem_save = f"{s_mem / max(m_mems):.1f}x"
        elif res_single is None and res_multi:
            speedup = "inf"
            mem_save = "inf"
        else:
            speedup = "-"
            mem_save = "-"

        print(
            f"{n:>5} | {n_pts:>12,} | {n_tets:>12,} | "
            f"{s_time_str:>10} {s_mem_str:>10} | "
            f"{m_wall_str:>10} {m_maxt_str:>11} {m_mem_str:>14} | "
            f"{speedup:>7} {mem_save:>9}",
            flush=True
        )

        del mesh
        reset_gpu()

        if multi_oom:
            print("\nMulti-GPU hit OOM. Stopping.")
            break

        n += args.step

    print(sep)
    print("Done!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

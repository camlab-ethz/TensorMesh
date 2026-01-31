#!/usr/bin/env python3
"""
Generate comprehensive comparison plots for batch Poisson FEM solver.

Shows:
1. CPU vs CUDA performance at different batch_sizes (fixed mesh, LU cached)
2. CUDA multi-mesh methods: Sequential vs Block Diagonal
"""

import os
import sys
import time
import gc
import argparse
import csv
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Import tensormesh components
# ============================================================================

from tensormesh.mesh import Mesh
from tensormesh.dataset.equation.poisson import PoissonMultiFrequency, PoissonMultiFrequency3D
from tensormesh.dataset.equation.poisson_solver import BatchPoissonSolver


def _load_components():
    """Return tensormesh components."""
    return {
        "Mesh": Mesh,
        "PoissonMultiFrequency": PoissonMultiFrequency,
        "PoissonMultiFrequency3D": PoissonMultiFrequency3D,
        "BatchPoissonFEMSolver": BatchPoissonSolver,  # Use actual class name
    }


def _now():
    return time.perf_counter()


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# ============================================================================
# Benchmark: Fixed mesh (LU cached)
# ============================================================================

def bench_fixed_mesh(
    devices: List[str],
    dim: int,
    chara_length: float,
    batch_sizes: List[int],
    K: int,
    repeats: int,
):
    """Benchmark with fixed mesh (LU decomposition cached)."""
    comps = _load_components()
    Mesh = comps["Mesh"]
    BatchPoissonFEMSolver = comps["BatchPoissonFEMSolver"]
    PoissonMultiFrequency = comps["PoissonMultiFrequency"]
    PoissonMultiFrequency3D = comps["PoissonMultiFrequency3D"]
    
    if dim == 2:
        mesh = Mesh.gen_rectangle(chara_length=chara_length)
        domain = "rectangle"
    else:
        mesh = Mesh.gen_cube(chara_length=chara_length)
        domain = "cube"
    
    dof = mesh.n_points
    print(f"[bench_fixed_mesh] dim={dim}, dof={dof}")
    
    results = {"dof": dof, "devices": {}}
    
    for device_str in devices:
        dev = torch.device(device_str)
        print(f"\n  Device: {dev}")
        
        solver = BatchPoissonFEMSolver(mesh, device=dev)
        points = solver.mesh.points
        
        dev_results = {"batch_sizes": [], "times": [], "errors": []}
        
        for bs in batch_sizes:
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            
            times = []
            errors = []
            
            try:
                for rep in range(repeats):
                    torch.manual_seed(rep)
                    
                    if dim == 2:
                        a = torch.rand((bs, K, K), device=dev) * 2 - 1
                        eq = PoissonMultiFrequency(a=a)
                    else:
                        a = torch.rand((bs, K, K, K), device=dev) * 2 - 1
                        eq = PoissonMultiFrequency3D(a=a)
                    
                    _sync(dev)
                    t0 = _now()
                    f = eq.source_term(points, domain=domain)
                    u_fem = solver.solve(f)
                    _sync(dev)
                    t = _now() - t0
                    
                    u_ana = eq.solution(points)
                    err = (u_fem - u_ana).abs()
                    if err.dim() == 1:
                        l2 = err.pow(2).mean().sqrt().item()
                    else:
                        l2 = err.pow(2).mean(dim=1).sqrt().mean().item()
                    
                    times.append(t)
                    errors.append(l2)
                
                dev_results["batch_sizes"].append(bs)
                dev_results["times"].append(np.mean(times))
                dev_results["errors"].append(np.mean(errors))
                print(f"    bs={bs:>6}: t={np.mean(times):.4f}s, L2={np.mean(errors):.2e}")
                
            except torch.cuda.OutOfMemoryError:
                print(f"    bs={bs:>6}: OOM")
                break
        
        results["devices"][device_str] = dev_results
    
    return results


# ============================================================================
# Benchmark: Multi-mesh (Sequential vs Block Diagonal)
# ============================================================================

def solve_multi_mesh_sequential(meshes, f_list, dev):
    """Solve each mesh independently (sequential)."""
    comps = _load_components()
    BatchPoissonFEMSolver = comps["BatchPoissonFEMSolver"]
    
    solutions = []
    for mesh, f in zip(meshes, f_list):
        solver = BatchPoissonFEMSolver(mesh, device=dev)
        u = solver.solve(f)
        solutions.append(u)
    return solutions


def solve_multi_mesh_block_diagonal(meshes, f_list, dev):
    """Assemble block-diagonal matrix and solve together."""
    # For simplicity, we just call sequential here
    # A true block-diagonal implementation would need sparse block assembly
    # which is more complex. This is a placeholder showing the concept.
    return solve_multi_mesh_sequential(meshes, f_list, dev)


def bench_multi_mesh(
    device: str,
    dim: int,
    chara_length: float,
    n_meshes_list: List[int],
    K: int,
    repeats: int,
):
    """Benchmark multi-mesh solving methods."""
    comps = _load_components()
    Mesh = comps["Mesh"]
    PoissonMultiFrequency = comps["PoissonMultiFrequency"]
    PoissonMultiFrequency3D = comps["PoissonMultiFrequency3D"]
    
    dev = torch.device(device)
    domain = "rectangle" if dim == 2 else "cube"
    
    results = {"n_meshes": [], "sequential": [], "per_mesh_dof": None}
    
    for n_meshes in n_meshes_list:
        print(f"\n  n_meshes={n_meshes}...")
        
        # Create meshes
        meshes = []
        for i in range(n_meshes):
            if dim == 2:
                mesh = Mesh.gen_rectangle(chara_length=chara_length)
            else:
                mesh = Mesh.gen_cube(chara_length=chara_length)
            meshes.append(mesh)
        
        if results["per_mesh_dof"] is None:
            results["per_mesh_dof"] = meshes[0].n_points
        
        # Generate source terms
        f_list = []
        for i, mesh in enumerate(meshes):
            torch.manual_seed(i)
            mesh_dev = mesh.to(dev)
            points = mesh_dev.points
            if dim == 2:
                a = torch.rand((1, K, K), device=dev) * 2 - 1
                eq = PoissonMultiFrequency(a=a)
            else:
                a = torch.rand((1, K, K, K), device=dev) * 2 - 1
                eq = PoissonMultiFrequency3D(a=a)
            f = eq.source_term(points, domain=domain).squeeze(0)
            f_list.append(f)
        
        # Time sequential
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        
        t_seq_list = []
        for rep in range(repeats):
            _sync(dev)
            t0 = _now()
            _ = solve_multi_mesh_sequential(meshes, f_list, dev)
            _sync(dev)
            t_seq_list.append(_now() - t0)
        
        results["n_meshes"].append(n_meshes)
        results["sequential"].append(np.mean(t_seq_list))
        
        print(f"    sequential: {np.mean(t_seq_list):.3f}s")
    
    return results


# ============================================================================
# Plotting
# ============================================================================

def _apply_style():
    """Apply ICML-style matplotlib settings."""
    plt.rcParams.update({
        # ICML font style (Times New Roman / serif)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Font sizes
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Lines and markers
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        # Grid
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.axisbelow": True,
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def _draw_slope_triangle(ax, x_center, y_center, slope, color, size=0.3, label_offset=(0.1, 0.1)):
    """Draw a slope triangle annotation in log-log space.
    
    Args:
        ax: matplotlib axis
        x_center: center x position (log scale)
        y_center: center y position (log scale)
        slope: the slope to annotate
        color: triangle color
        size: triangle size in log units
        label_offset: (dx, dy) offset for slope label
    """
    # In log-log space, slope = dy/dx where dx and dy are in log10 units
    # We draw a right triangle: horizontal leg = size, vertical leg = size * slope
    
    log_x = np.log10(x_center)
    log_y = np.log10(y_center)
    
    # Triangle vertices (in log space)
    # Bottom-left, bottom-right, top-right
    dx = size
    dy = size * slope
    
    x0, y0 = 10**(log_x - dx/2), 10**(log_y - dy/2)  # bottom-left
    x1, y1 = 10**(log_x + dx/2), 10**(log_y - dy/2)  # bottom-right
    x2, y2 = 10**(log_x + dx/2), 10**(log_y + dy/2)  # top-right
    
    triangle = plt.Polygon([(x0, y0), (x1, y1), (x2, y2)], 
                           facecolor=color, edgecolor='black', 
                           alpha=0.7, linewidth=1.0, zorder=10)
    ax.add_patch(triangle)
    
    # Add slope label
    label_x = 10**(log_x + dx/2 + label_offset[0])
    label_y = 10**(log_y + label_offset[1])
    ax.text(label_x, label_y, f'{slope:.2f}', fontsize=10, fontweight='bold',
            color=color, ha='left', va='center')


def _compute_slope(x_arr, y_arr, x_threshold):
    """Compute slope in log-log space for x > threshold."""
    mask = x_arr > x_threshold
    if mask.sum() < 2:
        return None, None, None
    
    log_x = np.log10(x_arr[mask])
    log_y = np.log10(y_arr[mask])
    
    # Linear regression in log space
    slope, intercept = np.polyfit(log_x, log_y, 1)
    
    # Return slope and a good position for annotation (right side of data)
    x_pos = x_arr[mask][-2] if len(x_arr[mask]) > 2 else x_arr[mask][-1]
    y_pos = y_arr[mask][-2] if len(y_arr[mask]) > 2 else y_arr[mask][-1]
    
    return slope, x_pos, y_pos


def plot_comprehensive(
    fixed_mesh_results: Dict,
    multi_mesh_results: Dict,
    out_path: str,
    save_pdf: bool = False,
    no_title: bool = False,
):
    """Generate comparison plot - CPU vs CUDA with LU cached."""
    _apply_style()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    dof = fixed_mesh_results["dof"]
    
    # Style definitions - bright, vibrant colors
    styles = {
        "cpu": {"color": "#3498DB", "marker": "o", "linestyle": "-", "label": "CPU"},      # Bright blue
        "cuda": {"color": "#E74C3C", "marker": "s", "linestyle": "-", "label": "CUDA"},    # Coral red
    }
    
    # Slope thresholds for each device
    slope_thresholds = {
        "cpu": 1e2,   # batch_size > 10^2
        "cuda": 1e4,  # batch_size > 10^4
    }
    
    # Plot fixed mesh results: x = batch_size
    for dev_name, dev_data in fixed_mesh_results["devices"].items():
        bs_arr = np.array(dev_data["batch_sizes"])
        t_arr = np.array(dev_data["times"])
        
        style = styles.get(dev_name, {"color": "#457B9D", "marker": "o", "linestyle": "-", "label": dev_name})
        
        ax.plot(bs_arr, t_arr, 
                marker=style["marker"], color=style["color"], 
                linestyle=style["linestyle"], label=style["label"],
                linewidth=2.5, markersize=9)
        
        # Compute and draw slope triangle
        threshold = slope_thresholds.get(dev_name, 1e2)
        slope, x_pos, y_pos = _compute_slope(bs_arr, t_arr, threshold)
        
        if slope is not None:
            # Position triangle to avoid overlap between CPU and CUDA
            if dev_name == "cpu":
                # CPU: place triangle to the left of the line
                tri_x = x_pos * 0.08
                tri_y = y_pos * 2.5
                label_offset = (0.18, 0.12)
            else:
                # CUDA: place triangle to the lower right
                tri_x = bs_arr[-1] * 0.5  # Move more left
                tri_y = t_arr[-1] * 0.15  # Move further down
                label_offset = (0.18, 0.12)
            
            _draw_slope_triangle(ax, tri_x, tri_y, slope, style["color"], 
                                 size=0.32, label_offset=label_offset)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('batch size', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    if not no_title:
        ax.set_title(f'Batch Poisson 3D Solver (dof={dof})', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    
    if save_pdf:
        pdf_path = out_path.replace(".png", ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    
    plt.close(fig)


# ============================================================================
# Cache I/O
# ============================================================================

CSV_FIELDS = ["device", "batch_size", "time", "error", "dof", "dim", "K"]


def save_cache(results: Dict, cache_path: str):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    rows = []
    dof = results["dof"]
    
    for dev_name, dev_data in results["devices"].items():
        for bs, t, err in zip(dev_data["batch_sizes"], dev_data["times"], dev_data["errors"]):
            rows.append({
                "device": dev_name,
                "batch_size": bs,
                "time": t,
                "error": err,
                "dof": dof,
                "dim": dev_data.get("dim", 3),
                "K": dev_data.get("K", 8),
            })
    
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[cache] Saved: {cache_path}")


def load_cache(cache_path: str) -> Dict:
    """Load benchmark results from CSV."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    rows = []
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    if not rows:
        raise ValueError(f"Empty cache: {cache_path}")
    
    dof = int(rows[0]["dof"])
    devices = {}
    
    for r in rows:
        dev = r["device"]
        if dev not in devices:
            devices[dev] = {"batch_sizes": [], "times": [], "errors": []}
        devices[dev]["batch_sizes"].append(int(r["batch_size"]))
        devices[dev]["times"].append(float(r["time"]))
        devices[dev]["errors"].append(float(r["error"]))
    
    print(f"[cache] Loaded: {cache_path}")
    return {"dof": dof, "devices": devices}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Poisson FEM Benchmark")
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # bench subcommand
    p_bench = subparsers.add_parser("bench", help="Run benchmark and save to cache")
    p_bench.add_argument("--dim", type=int, default=3, choices=[2, 3])
    p_bench.add_argument("--chara_length", type=float, default=0.05)
    p_bench.add_argument("--K", type=int, default=8)
    p_bench.add_argument("--batch_sizes_cpu", default="1,16,256,4096,16384")
    p_bench.add_argument("--batch_sizes_cuda", default="1,16,256,4096,16384,65536,131072")
    p_bench.add_argument("--repeats", type=int, default=2)
    p_bench.add_argument("--cache", default="cache/poisson_bench.csv", help="Cache file path")
    
    # plot subcommand
    p_plot = subparsers.add_parser("plot", help="Plot from cache")
    p_plot.add_argument("--cache", default="cache/poisson_bench.csv", help="Cache file path")
    p_plot.add_argument("--output", default="poisson_comprehensive.png")
    p_plot.add_argument("--pdf", action="store_true")
    p_plot.add_argument("--no-title", action="store_true", dest="no_title")
    
    args = parser.parse_args()
    
    here = os.path.dirname(os.path.abspath(__file__))
    
    if args.cmd == "bench":
        batch_sizes_cpu = [int(x) for x in args.batch_sizes_cpu.split(",")]
        batch_sizes_cuda = [int(x) for x in args.batch_sizes_cuda.split(",")]
        
        print("=" * 60)
        print("Benchmark: CPU (LU cached)")
        print("=" * 60)
        
        results_cpu = bench_fixed_mesh(
            devices=["cpu"],
            dim=args.dim,
            chara_length=args.chara_length,
            batch_sizes=batch_sizes_cpu,
            K=args.K,
            repeats=args.repeats,
        )
        # Add metadata
        results_cpu["devices"]["cpu"]["dim"] = args.dim
        results_cpu["devices"]["cpu"]["K"] = args.K
        
        print("\n" + "=" * 60)
        print("Benchmark: CUDA (LU cached)")
        print("=" * 60)
        
        results_cuda = bench_fixed_mesh(
            devices=["cuda"],
            dim=args.dim,
            chara_length=args.chara_length,
            batch_sizes=batch_sizes_cuda,
            K=args.K,
            repeats=args.repeats,
        )
        results_cuda["devices"]["cuda"]["dim"] = args.dim
        results_cuda["devices"]["cuda"]["K"] = args.K
        
        # Merge and save
        results = {
            "dof": results_cpu["dof"],
            "devices": {
                **results_cpu["devices"],
                **results_cuda["devices"],
            }
        }
        
        cache_path = os.path.join(here, args.cache)
        save_cache(results, cache_path)
        
    elif args.cmd == "plot":
        cache_path = os.path.join(here, args.cache)
        results = load_cache(cache_path)
        
        out_path = os.path.join(here, args.output)
        plot_comprehensive(
            results,
            {},  # multi_mesh_results not used
            out_path,
            save_pdf=args.pdf,
            no_title=args.no_title,
        )


if __name__ == "__main__":
    main()


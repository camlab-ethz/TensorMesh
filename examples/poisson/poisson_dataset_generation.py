"""
Benchmark Batch Poisson Dataset Generation + FEM Solve.

What it does
------------
- Fixed mesh, sweeps over batch_size to measure generation + FEM solve performance.
- Validates correctness via L2 error against analytical solution.
- Supports both 2D and 3D Poisson equations.
- Compares CPU vs CUDA performance.
- Compares sequential vs block-diagonal multi-mesh solving.

Usage
-----
Run benchmark:
  python poisson_dataset_generation.py bench --devices cpu,cuda

Compare multi-mesh methods:
  python poisson_dataset_generation.py multi_mesh --n_meshes 10

Plot results:
  python poisson_dataset_generation.py plot --cache cache/poisson3d.csv
"""

import os
import sys
import time
import gc
import argparse
import csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append("../..")

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib

import importlib.util


def _load_poisson_classes():
    """Load PoissonMultiFrequency (2D) and PoissonMultiFrequency3D (3D) from source."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "../.."))
    mod_path = os.path.join(root, "tensormesh", "dataset", "equation", "poisson.py")
    spec = importlib.util.spec_from_file_location("tensormesh_dataset_equation_poisson", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "PoissonMultiFrequency"), getattr(mod, "PoissonMultiFrequency3D")


PoissonMultiFrequency, PoissonMultiFrequency3D = _load_poisson_classes()


# ============================================================================
# TensorMesh FEM Solver Components
# ============================================================================

_TENSORMESH_LOADED = False
_TM_COMPONENTS = {}


def _load_tensormesh_components():
    """Load TensorMesh components for FEM solving."""
    global _TENSORMESH_LOADED, _TM_COMPONENTS
    if _TENSORMESH_LOADED:
        return _TM_COMPONENTS
    
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "../.."))
    if root not in sys.path:
        sys.path.insert(0, root)
    
    from tensormesh.mesh import Mesh
    from tensormesh.assemble import LaplaceElementAssembler, NodeAssembler
    from tensormesh.operator import Condenser
    
    _TM_COMPONENTS["Mesh"] = Mesh
    _TM_COMPONENTS["LaplaceElementAssembler"] = LaplaceElementAssembler
    _TM_COMPONENTS["NodeAssembler"] = NodeAssembler
    _TM_COMPONENTS["Condenser"] = Condenser
    
    _TENSORMESH_LOADED = True
    return _TM_COMPONENTS


class BatchPoissonFEMSolver:
    """
    Batch Poisson FEM Solver with cached LU factorization.
    Solves: -Δu = f on Ω with u=0 on ∂Ω
    
    "Batch" means: same mesh, different right-hand sides (source terms f).
    
    LU factorization is computed ONCE in __init__ and reused for all solves.
    Each solve() call only does triangular solves (L^-1 and U^-1), which is O(n^2).
    """
    
    def __init__(self, mesh, device=None):
        comps = _load_tensormesh_components()
        LaplaceElementAssembler = comps["LaplaceElementAssembler"]
        NodeAssembler = comps["NodeAssembler"]
        Condenser = comps["Condenser"]
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.mesh = mesh.to(device)
        
        # Build stiffness matrix
        assembler = LaplaceElementAssembler.from_mesh(self.mesh)
        K_full = assembler(self.mesh.points)
        
        # Setup Dirichlet BC
        n_boundary = self.mesh.boundary_mask.sum().item()
        dirichlet_value = torch.zeros(n_boundary, device=device)
        self.condenser = Condenser(self.mesh.boundary_mask, dirichlet_value)
        
        # Condense stiffness matrix
        self.K, _ = self.condenser(K_full, torch.zeros(self.mesh.n_points, device=device))
        
        # Source term assembler
        class _FAssembler(NodeAssembler):
            def forward(self, v, fs):
                return v * fs
        
        self._f_assembler = _FAssembler.from_mesh(self.mesh)
        self.n_points = self.mesh.n_points
        
        # Pre-compute and cache LU factorization
        self._lu_cached = None
        self._backend = "torch"
        self._cache_lu()
    
    def _cache_lu(self):
        """Cache LU factorization for efficient batch solve."""
        K = self.K
        edata = K.edata.double()
        row = K.row
        col = K.col
        shape = K.shape
        
        if self.device.type == "cuda":
            try:
                import cupy as cp
                import cupyx.scipy.sparse
                from tensormesh.sparse.utils import tensor2cupy
                
                cp.cuda.Device(self.device.index or 0).use()
                A_cupy = cupyx.scipy.sparse.coo_matrix(
                    (tensor2cupy(edata), (tensor2cupy(row), tensor2cupy(col))),
                    shape=shape
                ).tocsc()
                self._lu_cached = cupyx.scipy.sparse.linalg.splu(A_cupy)
                self._backend = "cupy"
            except ImportError:
                self._backend = "torch"
        else:
            try:
                import scipy.sparse as sp
                from scipy.sparse.linalg import splu
                
                A_scipy = sp.coo_matrix(
                    (edata.numpy(), (row.numpy(), col.numpy())),
                    shape=shape
                ).tocsc()
                self._lu_cached = splu(A_scipy)
                self._backend = "scipy"
            except ImportError:
                self._backend = "torch"
    
    def solve(self, f: torch.Tensor) -> torch.Tensor:
        """
        Solve -Δu = f using cached LU factorization.
        
        Parameters
        ----------
        f : torch.Tensor
            Source term. Shape: [batch, n_nodes] or [n_nodes]
        
        Returns
        -------
        torch.Tensor
            Solution. Shape: [batch, n_nodes] or [n_nodes]
        """
        squeeze = f.dim() == 1
        if squeeze:
            f = f.unsqueeze(0)
        
        f_T = f.T
        F = self._f_assembler(self.mesh.points, point_data={"fs": f_T})
        F = F.reshape(self.n_points, -1)
        F_cond = self.condenser.condense_rhs(F)
        
        u_inner = self._lu_solve(F_cond)
        
        u_full = self.condenser.recover(u_inner)
        u = u_full.T
        
        if squeeze:
            u = u.squeeze(0)
        return u
    
    def _lu_solve(self, b: torch.Tensor) -> torch.Tensor:
        """Solve Kx = b using cached LU factorization."""
        orig_dtype = b.dtype
        b_f64 = b.double()
        
        if self._backend == "cupy" and self._lu_cached is not None:
            import cupy as cp
            from tensormesh.sparse.utils import tensor2cupy, cupy2tensor
            
            cp.cuda.Device(self.device.index or 0).use()
            b_cupy = tensor2cupy(b_f64)
            u_cupy = self._lu_cached.solve(b_cupy)
            u = cupy2tensor(u_cupy)
        elif self._backend == "scipy" and self._lu_cached is not None:
            b_np = b_f64.numpy()
            u_np = self._lu_cached.solve(b_np)
            u = torch.from_numpy(u_np)
        else:
            u = self.K.solve(b_f64)
        
        return u.to(orig_dtype)


# ============================================================================
# Multi-Mesh Solver: Block Diagonal vs Sequential
# ============================================================================

def solve_multi_mesh_sequential(meshes, f_list, device):
    """Solve multiple meshes sequentially (baseline)."""
    results = []
    for mesh, f in zip(meshes, f_list):
        solver = BatchPoissonFEMSolver(mesh, device=device)
        u = solver.solve(f)
        results.append(u)
    return results


def solve_multi_mesh_block_diagonal(meshes, f_list, device):
    """Solve multiple meshes by assembling a block diagonal matrix."""
    comps = _load_tensormesh_components()
    LaplaceElementAssembler = comps["LaplaceElementAssembler"]
    NodeAssembler = comps["NodeAssembler"]
    Condenser = comps["Condenser"]
    
    meshes_dev = [m.to(device) for m in meshes]
    
    K_list = []
    condensers = []
    n_inner_list = []
    
    for mesh in meshes_dev:
        assembler = LaplaceElementAssembler.from_mesh(mesh)
        K_full = assembler(mesh.points)
        
        n_boundary = mesh.boundary_mask.sum().item()
        dirichlet_value = torch.zeros(n_boundary, device=device)
        condenser = Condenser(mesh.boundary_mask, dirichlet_value)
        
        K_cond, _ = condenser(K_full, torch.zeros(mesh.n_points, device=device))
        
        K_list.append(K_cond)
        condensers.append(condenser)
        n_inner = (~mesh.boundary_mask).sum().item()
        n_inner_list.append(n_inner)
    
    # Build block diagonal sparse matrix
    total_n = sum(n_inner_list)
    all_edata = []
    all_row = []
    all_col = []
    offset = 0
    
    for K, n_inner in zip(K_list, n_inner_list):
        all_edata.append(K.edata)
        all_row.append(K.row + offset)
        all_col.append(K.col + offset)
        offset += n_inner
    
    big_edata = torch.cat(all_edata)
    big_row = torch.cat(all_row)
    big_col = torch.cat(all_col)
    
    # Assemble RHS for each mesh
    class _FAssembler(NodeAssembler):
        def forward(self, v, fs):
            return v * fs
    
    rhs_list = []
    for mesh, condenser, f in zip(meshes_dev, condensers, f_list):
        f_assembler = _FAssembler.from_mesh(mesh)
        f_dev = f.to(device)
        if f_dev.dim() == 1:
            f_dev = f_dev.unsqueeze(0)
        f_T = f_dev.T
        F = f_assembler(mesh.points, point_data={"fs": f_T})
        F = F.reshape(mesh.n_points, -1)
        F_cond = condenser.condense_rhs(F)
        rhs_list.append(F_cond)
    
    big_rhs = torch.cat(rhs_list, dim=0)
    
    # Solve block diagonal system
    big_edata_f64 = big_edata.double()
    big_rhs_f64 = big_rhs.double()
    
    if device.type == "cuda":
        import cupy as cp
        import cupyx.scipy.sparse
        from tensormesh.sparse.utils import tensor2cupy, cupy2tensor
        
        cp.cuda.Device(device.index or 0).use()
        A_cupy = cupyx.scipy.sparse.coo_matrix(
            (tensor2cupy(big_edata_f64), (tensor2cupy(big_row), tensor2cupy(big_col))),
            shape=(total_n, total_n)
        ).tocsc()
        lu = cupyx.scipy.sparse.linalg.splu(A_cupy)
        u_cupy = lu.solve(tensor2cupy(big_rhs_f64))
        big_u = cupy2tensor(u_cupy).float()
    else:
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        
        A_scipy = sp.coo_matrix(
            (big_edata_f64.numpy(), (big_row.numpy(), big_col.numpy())),
            shape=(total_n, total_n)
        ).tocsc()
        lu = splu(A_scipy)
        u_np = lu.solve(big_rhs_f64.numpy())
        big_u = torch.from_numpy(u_np).float()
    
    # Split solution back
    results = []
    offset = 0
    for mesh, condenser, n_inner in zip(meshes_dev, condensers, n_inner_list):
        u_inner = big_u[offset:offset+n_inner]
        u_full = condenser.recover(u_inner)
        results.append(u_full.T.squeeze(0) if u_full.dim() > 1 else u_full)
        offset += n_inner
    
    return results


def run_multi_mesh_bench(
    *,
    device: str,
    dim: int,
    n_meshes: int,
    chara_length: float,
    K: int,
    repeats: int,
):
    """Benchmark: compare sequential vs block-diagonal for multiple meshes."""
    comps = _load_tensormesh_components()
    Mesh = comps["Mesh"]
    
    dev = torch.device(device)
    print(f"[multi_mesh_bench] Device: {dev}, dim: {dim}, n_meshes: {n_meshes}")
    
    meshes = []
    for i in range(n_meshes):
        if dim == 2:
            mesh = Mesh.gen_rectangle(chara_length=chara_length)
            domain = "rectangle"
        else:
            mesh = Mesh.gen_cube(chara_length=chara_length)
            domain = "cube"
        meshes.append(mesh)
    
    mesh_n_points = meshes[0].n_points
    print(f"[multi_mesh_bench] Each mesh: {mesh_n_points} nodes")
    
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
    
    # Benchmark sequential
    t_seq_list = []
    for rep in range(repeats):
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        
        _sync(dev)
        t0 = _now()
        results_seq = solve_multi_mesh_sequential(meshes, f_list, dev)
        _sync(dev)
        t_seq = _now() - t0
        t_seq_list.append(t_seq)
    
    t_seq_mean = np.mean(t_seq_list)
    t_seq_std = np.std(t_seq_list, ddof=1) if len(t_seq_list) >= 2 else 0.0
    
    # Benchmark block diagonal
    t_block_list = []
    for rep in range(repeats):
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        
        _sync(dev)
        t0 = _now()
        results_block = solve_multi_mesh_block_diagonal(meshes, f_list, dev)
        _sync(dev)
        t_block = _now() - t0
        t_block_list.append(t_block)
    
    t_block_mean = np.mean(t_block_list)
    t_block_std = np.std(t_block_list, ddof=1) if len(t_block_list) >= 2 else 0.0
    
    # Check correctness
    max_diff = 0.0
    for u_seq, u_block in zip(results_seq, results_block):
        diff = (u_seq - u_block).abs().max().item()
        max_diff = max(max_diff, diff)
    
    print()
    print(f"{'Method':<20} {'Time (s)':<15} {'Std (s)':<15}")
    print("-" * 50)
    print(f"{'Sequential':<20} {t_seq_mean:<15.4f} {t_seq_std:<15.4f}")
    print(f"{'Block Diagonal':<20} {t_block_mean:<15.4f} {t_block_std:<15.4f}")
    print()
    print(f"Speedup (seq/block): {t_seq_mean / t_block_mean:.2f}x")
    print(f"Max diff (correctness check): {max_diff:.2e}")


# ============================================================================
# Benchmark
# ============================================================================

def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class BenchResult:
    """Result of a batch Poisson benchmark run."""
    device: str
    dim: int
    mesh_n_points: int
    chara_length: float
    batch_size: int
    K: int
    repeats: int
    
    t_total_mean: float
    t_total_std: float
    
    error_l2_mean: float
    peak_mem_bytes: Optional[float] = None


CSV_FIELDS: Tuple[str, ...] = (
    "key",
    "device",
    "dim",
    "mesh_n_points",
    "chara_length",
    "batch_size",
    "K",
    "repeats",
    "t_total_mean",
    "t_total_std",
    "error_l2_mean",
    "peak_mem_bytes",
)


def run_bench(
    *,
    devices: Sequence[str],
    dim: int,
    chara_length: float,
    batch_sizes: Sequence[int],
    K: int,
    repeats: int,
    cache_path: str,
    fixed_mesh: bool = True,
):
    """Run batch Poisson benchmark: generate source terms + FEM solve."""
    comps = _load_tensormesh_components()
    Mesh = comps["Mesh"]
    
    print(f"[bench] dim: {dim}, chara_length: {chara_length}, K: {K}")
    print(f"[bench] devices: {devices}")
    print(f"[bench] fixed_mesh: {fixed_mesh} {'(LU cached)' if fixed_mesh else '(rebuild each time)'}")
    
    if dim == 2:
        mesh = Mesh.gen_rectangle(chara_length=chara_length)
        domain = "rectangle"
    else:
        mesh = Mesh.gen_cube(chara_length=chara_length)
        domain = "cube"
    
    mesh_n_points = mesh.n_points
    print(f"[bench] Mesh: {mesh_n_points} nodes")
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    results = []
    
    for device_str in devices:
        dev = torch.device(device_str)
        print(f"\n[bench] Testing device: {dev}")
        
        if fixed_mesh:
            solver = BatchPoissonFEMSolver(mesh, device=dev)
            points = solver.mesh.points
        
        print(f"{'batch_size':>10} {'t_total(s)':>12} {'L2 error':>12}")
        print("-" * 36)
        
        for batch_size in batch_sizes:
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            
            t_total_list, errors_l2 = [], []
            peak_mem = 0.0
            
            try:
                for rep in range(repeats):
                    torch.manual_seed(rep)
                    
                    if not fixed_mesh:
                        solver = BatchPoissonFEMSolver(mesh, device=dev)
                        points = solver.mesh.points
                    
                    if dim == 2:
                        a = torch.rand((batch_size, K, K), device=dev) * 2 - 1
                        eq = PoissonMultiFrequency(a=a)
                    else:
                        a = torch.rand((batch_size, K, K, K), device=dev) * 2 - 1
                        eq = PoissonMultiFrequency3D(a=a)
                    
                    _sync(dev)
                    t0 = _now()
                    if not fixed_mesh:
                        solver = BatchPoissonFEMSolver(mesh, device=dev)
                        points = solver.mesh.points
                    f = eq.source_term(points, domain=domain)
                    u_fem = solver.solve(f)
                    _sync(dev)
                    t_total = _now() - t0
                    
                    u_analytical = eq.solution(points)
                    
                    error = (u_fem - u_analytical).abs()
                    if error.dim() == 1:
                        l2_err = error.pow(2).mean().sqrt().item()
                    else:
                        l2_err = error.pow(2).mean(dim=1).sqrt().mean().item()
                    
                    t_total_list.append(t_total)
                    errors_l2.append(l2_err)
                    
                    if dev.type == "cuda":
                        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(dev))
                
                t_total_mean = float(np.mean(t_total_list))
                t_total_std = float(np.std(t_total_list, ddof=1)) if len(t_total_list) >= 2 else 0.0
                l2_mean = float(np.mean(errors_l2))
                
                print(f"{batch_size:>10} {t_total_mean:>12.4f} {l2_mean:>12.2e}")
                
                key = f"{dev}|dim{dim}|bs{batch_size}|np{mesh_n_points}|K{K}"
                res = BenchResult(
                    device=str(dev),
                    dim=dim,
                    mesh_n_points=mesh_n_points,
                    chara_length=chara_length,
                    batch_size=batch_size,
                    K=K,
                    repeats=repeats,
                    t_total_mean=t_total_mean,
                    t_total_std=t_total_std,
                    error_l2_mean=l2_mean,
                    peak_mem_bytes=peak_mem if peak_mem > 0 else None,
                )
                row = asdict(res)
                row["key"] = key
                results.append(row)
                
            except torch.OutOfMemoryError:
                print(f"{batch_size:>10} {'OOM':>12} {'---':>12}")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                break
    
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})
    
    print()
    print(f"[bench] wrote: {cache_path}")


# ============================================================================
# Plot
# ============================================================================

def _apply_style():
    """Apply clean matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 170,
        "savefig.dpi": 170,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_cache(cache_path: str, out_dir: str, no_title: bool = False, save_pdf: bool = False):
    """Plot benchmark results."""
    _apply_style()
    
    if not os.path.exists(cache_path):
        raise RuntimeError(f"CSV cache not found: {cache_path}")

    rows: List[Dict[str, Any]] = []
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No cached rows found at {cache_path}")

    os.makedirs(out_dir, exist_ok=True)

    def to_i(v): return int(float(v))
    def to_f(v): return float(v)
    
    dim = to_i(rows[0]["dim"])
    mesh_n_points = to_i(rows[0]["mesh_n_points"])
    K = to_i(rows[0]["K"])
    
    devices = sorted(set(r["device"] for r in rows))
    
    device_styles = {
        "cpu": {"color": "#2E86AB", "marker": "o", "label": "CPU"},
        "cuda": {"color": "#A23B72", "marker": "s", "label": "CUDA"},
        "cuda:0": {"color": "#A23B72", "marker": "s", "label": "CUDA"},
        "cuda:1": {"color": "#E76F51", "marker": "^", "label": "CUDA:1"},
    }
    
    def get_style(dev):
        if dev in device_styles:
            return device_styles[dev]
        if "cuda" in dev:
            return {"color": "#A23B72", "marker": "s", "label": dev.upper()}
        return {"color": "#457B9D", "marker": "o", "label": dev.upper()}
    
    # Figure 1: Total time vs batch_size
    fig, ax = plt.subplots(figsize=(7, 5))
    
        for dev in devices:
        dev_rows = [r for r in rows if r["device"] == dev]
        batch_sizes = [to_i(r["batch_size"]) for r in dev_rows]
        t_total = [to_f(r["t_total_mean"]) for r in dev_rows]
        t_std = [to_f(r["t_total_std"]) for r in dev_rows]
        
        style = get_style(dev)
        ax.errorbar(batch_sizes, t_total, yerr=t_std,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, capthick=1.5)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('batch_size')
    ax.set_ylabel('time (s)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    if not no_title:
        ax.set_title(f'Poisson{dim}D Total Time (mesh={mesh_n_points}, K={K})')
    
        fig.tight_layout()
    out_path = os.path.join(out_dir, f"poisson{dim}d_bench.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"[plot] wrote: {out_path}")
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"[plot] wrote: {out_path.replace('.png', '.pdf')}")
        plt.close(fig)

    # Figure 2: Error validation
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    
        for dev in devices:
        dev_rows = [r for r in rows if r["device"] == dev]
        batch_sizes = [to_i(r["batch_size"]) for r in dev_rows]
        errors = [to_f(r["error_l2_mean"]) for r in dev_rows]
        
        style = get_style(dev)
        ax2.plot(batch_sizes, errors, color=style["color"], marker=style["marker"], label=style["label"])
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('batch_size')
    ax2.set_ylabel('L2 error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    if not no_title:
        ax2.set_title(f'Poisson{dim}D L2 Error (stable across batch_size)')
    
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, f"poisson{dim}d_error.png")
    fig2.savefig(out_path2, dpi=170, bbox_inches="tight")
    print(f"[plot] wrote: {out_path2}")
    if save_pdf:
        fig2.savefig(out_path2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig2)


def plot_comprehensive(
    cache_path: str,
    out_dir: str,
    dim: int = 3,
    chara_length: float = 0.1,
    K: int = 8,
    n_meshes_list: Sequence[int] = (5, 10, 20, 50),
    repeats: int = 3,
    save_pdf: bool = False,
):
    """Generate comprehensive comparison plot."""
    _apply_style()
    
    # Read bench data
    rows: List[Dict[str, Any]] = []
    if os.path.exists(cache_path):
        with open(cache_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    
    def to_i(v): return int(float(v))
    def to_f(v): return float(v)
    
    # Get mesh info
    if rows:
        mesh_dof = to_i(rows[0]["mesh_n_points"])
                    else:
        mesh_dof = 7315
    
    # Collect multi-mesh benchmark data
    comps = _load_tensormesh_components()
    Mesh = comps["Mesh"]
    
    print("[plot_comprehensive] Running multi-mesh benchmarks...")
    
    multi_mesh_data = {"n_meshes": [], "sequential": [], "block_diagonal": []}
    
    for n_meshes in n_meshes_list:
        print(f"  n_meshes={n_meshes}...", end=" ", flush=True)
        
        # Create meshes
        meshes = []
        for i in range(n_meshes):
            if dim == 2:
                mesh = Mesh.gen_rectangle(chara_length=chara_length)
                domain = "rectangle"
            else:
                mesh = Mesh.gen_cube(chara_length=chara_length)
                domain = "cube"
            meshes.append(mesh)
        
        # Generate source terms
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        # Time block diagonal
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        t_blk_list = []
        for rep in range(repeats):
            _sync(dev)
            t0 = _now()
            _ = solve_multi_mesh_block_diagonal(meshes, f_list, dev)
            _sync(dev)
            t_blk_list.append(_now() - t0)
        
        multi_mesh_data["n_meshes"].append(n_meshes)
        multi_mesh_data["sequential"].append(np.mean(t_seq_list))
        multi_mesh_data["block_diagonal"].append(np.mean(t_blk_list))
        
        print(f"seq={np.mean(t_seq_list):.3f}s, blk={np.mean(t_blk_list):.3f}s")
    
    # Create comprehensive figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: CPU vs CUDA (batch_size) with dof annotation
    ax1 = axes[0]
    
    if rows:
        devices = sorted(set(r["device"] for r in rows))
        device_styles = {
            "cpu": {"color": "#2E86AB", "marker": "o", "label": "CPU"},
            "cuda": {"color": "#A23B72", "marker": "s", "label": "CUDA"},
        }
        
        for dev in devices:
            dev_rows = [r for r in rows if r["device"] == dev]
            batch_sizes = [to_i(r["batch_size"]) for r in dev_rows]
            t_total = [to_f(r["t_total_mean"]) for r in dev_rows]
            
            style = device_styles.get(dev, {"color": "#457B9D", "marker": "^", "label": dev})
            ax1.plot(batch_sizes, t_total, color=style["color"], marker=style["marker"],
                    label=style["label"], linewidth=2, markersize=8)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('batch_size')
        ax1.set_ylabel('time (s)')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'CPU vs CUDA (dof={mesh_dof})', fontsize=13, fontweight='bold')
    
    # Right: CUDA methods comparison (n_meshes)
    ax2 = axes[1]
    
    n_meshes_arr = np.array(multi_mesh_data["n_meshes"])
    t_seq_arr = np.array(multi_mesh_data["sequential"])
    t_blk_arr = np.array(multi_mesh_data["block_diagonal"])
    
    ax2.plot(n_meshes_arr, t_seq_arr, 'o-', color='#E76F51', label='Sequential', linewidth=2, markersize=8)
    ax2.plot(n_meshes_arr, t_blk_arr, 's-', color='#2A9D8F', label='Block Diagonal', linewidth=2, markersize=8)
    
    ax2.set_xlabel('n_meshes')
    ax2.set_ylabel('time (s)')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Multi-Mesh Methods (CUDA, dof={meshes[0].n_points if meshes else "?"})', fontsize=13, fontweight='bold')
    
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"poisson{dim}d_comprehensive.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"[plot] wrote: {out_path}")
    if save_pdf:
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"[plot] wrote: {out_path.replace('.png', '.pdf')}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def _parse_int_list(s: str) -> List[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(out_dir, "cache")

    ap = argparse.ArgumentParser(description="Batch Poisson Dataset Generation Benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # bench
    ap_bench = sub.add_parser("bench", help="Run benchmark")
    ap_bench.add_argument("--devices", default="cpu,cuda", help="comma-separated devices")
    ap_bench.add_argument("--dim", type=int, default=3, choices=[2, 3], help="2D or 3D")
    ap_bench.add_argument("--chara_length", type=float, default=0.05, help="mesh characteristic length")
    ap_bench.add_argument("--batch_sizes", default="1,16,256,4096,16384", help="batch sizes to test")
    ap_bench.add_argument("--K", type=int, default=8, help="frequency parameter")
    ap_bench.add_argument("--repeats", type=int, default=3, help="repeats per config")
    ap_bench.add_argument("--cache", default=None, help="output CSV path")
    ap_bench.add_argument("--fixed_mesh", action="store_true", default=True)
    ap_bench.add_argument("--no-fixed_mesh", dest="fixed_mesh", action="store_false")
    
    # multi_mesh: compare sequential vs block diagonal
    ap_multi = sub.add_parser("multi_mesh", help="Compare sequential vs block-diagonal for multiple meshes")
    ap_multi.add_argument("--device", default="cuda", help="device to use")
    ap_multi.add_argument("--dim", type=int, default=3, choices=[2, 3], help="2D or 3D")
    ap_multi.add_argument("--n_meshes", type=int, default=10, help="number of meshes")
    ap_multi.add_argument("--chara_length", type=float, default=0.1, help="mesh characteristic length")
    ap_multi.add_argument("--K", type=int, default=8, help="frequency parameter")
    ap_multi.add_argument("--repeats", type=int, default=3, help="repeats")
    
    # plot
    ap_plot = sub.add_parser("plot", help="Plot benchmark results")
    ap_plot.add_argument("--cache", required=True, help="input CSV path")
    ap_plot.add_argument("--out_dir", default=out_dir)
    ap_plot.add_argument("--no-title", action="store_true")
    ap_plot.add_argument("--pdf", action="store_true")
    
    # plot_all: comprehensive comparison plot
    ap_plot_all = sub.add_parser("plot_all", help="Generate comprehensive comparison plot")
    ap_plot_all.add_argument("--cache", default=None, help="bench CSV path")
    ap_plot_all.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap_plot_all.add_argument("--chara_length", type=float, default=0.1, help="mesh chara_length for multi-mesh")
    ap_plot_all.add_argument("--K", type=int, default=8)
    ap_plot_all.add_argument("--n_meshes", default="5,10,20,50", help="n_meshes list for multi-mesh")
    ap_plot_all.add_argument("--repeats", type=int, default=2)
    ap_plot_all.add_argument("--out_dir", default=out_dir)
    ap_plot_all.add_argument("--pdf", action="store_true")

    args = ap.parse_args()

    if args.cmd == "bench":
        cache_path = args.cache or os.path.join(cache_dir, f"poisson{args.dim}d.csv")
        run_bench(
            devices=_parse_str_list(args.devices),
            dim=args.dim,
            chara_length=args.chara_length,
            batch_sizes=_parse_int_list(args.batch_sizes),
            K=args.K,
            repeats=args.repeats,
            cache_path=cache_path,
            fixed_mesh=args.fixed_mesh,
        )
    elif args.cmd == "multi_mesh":
        run_multi_mesh_bench(
            device=args.device,
            dim=args.dim,
            n_meshes=args.n_meshes,
            chara_length=args.chara_length,
            K=args.K,
            repeats=args.repeats,
        )
    elif args.cmd == "plot":
        plot_cache(
            args.cache,
            args.out_dir,
            no_title=getattr(args, "no_title", False),
            save_pdf=getattr(args, "pdf", False),
        )
    elif args.cmd == "plot_all":
        cache_path = args.cache or os.path.join(cache_dir, f"poisson{args.dim}d.csv")
        plot_comprehensive(
            cache_path=cache_path,
            out_dir=args.out_dir,
            dim=args.dim,
            chara_length=args.chara_length,
            K=args.K,
            n_meshes_list=_parse_int_list(args.n_meshes),
            repeats=args.repeats,
            save_pdf=args.pdf,
        )


if __name__ == "__main__":
    main()

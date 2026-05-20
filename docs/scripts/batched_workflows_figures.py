"""Generate the static figures used in docs/source/user_guide/batched_workflows.rst.

Run from the repo root, after activating the tensorgalerkin venv:

    python docs/scripts/batched_workflows_figures.py

Three figures, all built around the canonical pattern in the chapter
(same mesh, same stiffness matrix, many source terms):

* ``batched_rhs_scaling.png`` -- per-problem solve cost as a function of
  ``n_batch``, comparing a Python loop of single-RHS solves against a
  single 2D-RHS call to :meth:`SparseMatrix.solve`. CPU vs CUDA.
* ``throughput.png`` -- problems-per-second of the same batched solve as
  ``n_batch`` grows, on CPU and CUDA.
* ``memory_chunking.png`` -- assembly time and peak GPU memory as a
  function of the ``batch_size`` chunk knob on a fine mesh, plus the
  bit-equality of the resulting stiffness matrix.
"""

import gc
import math
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch

from tensormesh import Mesh, LaplaceElementAssembler, MassElementAssembler, Condenser


warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="float64 recommended")

OUT = os.path.join(os.path.dirname(__file__), "..", "source",
                   "_static", "user_guide", "batched_workflows")
os.makedirs(OUT, exist_ok=True)

torch.set_default_dtype(torch.float64)

HAS_CUDA = torch.cuda.is_available()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _build_poisson(device, h):
    """Build (mesh, K_condensed, M, condenser) for -Delta u = f on [0,1]^2."""
    mesh = Mesh.gen_rectangle(chara_length=h).to(device)
    K = LaplaceElementAssembler.from_mesh(mesh)().double()
    M = MassElementAssembler.from_mesh(mesh)().double()
    cond = Condenser(mesh.boundary_mask)
    zero = torch.zeros(mesh.n_points, dtype=torch.float64, device=device)
    K_, _ = cond(K, zero)
    return mesh, K_, M, cond


def _time(fn, n_warmup, n_runs, device):
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return min(times)  # best of n; smoother than mean for GPU.


# ---------------------------------------------------------------------------
# Figure 1: batched-RHS vs Python loop, per-problem cost.
# ---------------------------------------------------------------------------

def fig_batched_rhs_scaling():
    print("[fig 1] batched-RHS scaling...")
    n_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    series = {}
    for device in ["cpu"] + (["cuda"] if HAS_CUDA else []):
        print(f"  device={device}")
        mesh, K_, M, cond = _build_poisson(device, h=0.02)
        torch.manual_seed(0)
        n_dof = mesh.n_points
        B_full = torch.randn(n_dof, max(n_batches), dtype=torch.float64,
                             device=device)
        B_full = cond.condense_rhs(B_full)  # [n_inner, max_nb]

        loop_per, batch_per = [], []
        for nb in n_batches:
            B = B_full[:, :nb].contiguous()

            def run_loop():
                for i in range(nb):
                    _ = K_.solve(B[:, i])

            def run_batched():
                _ = K_.solve(B)

            t_loop = _time(run_loop, n_warmup=1, n_runs=3, device=device)
            t_batch = _time(run_batched, n_warmup=1, n_runs=3, device=device)
            loop_per.append(t_loop / nb * 1e3)     # ms/problem
            batch_per.append(t_batch / nb * 1e3)
            print(f"    nb={nb:4d}  loop={t_loop*1e3:8.2f}ms  "
                  f"batched={t_batch*1e3:8.2f}ms  "
                  f"speedup={t_loop/t_batch:5.1f}x")

        series[device] = (np.array(loop_per), np.array(batch_per))
        del mesh, K_, M, cond, B_full
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    palette = {
        "cpu":  ("#7f8c8d", "#2c3e50"),
        "cuda": ("#e67e22", "#c0392b"),
    }
    label_map = {"cpu": "CPU", "cuda": "GPU"}
    for device, (loop, batch) in series.items():
        c_loop, c_batch = palette[device]
        ax.loglog(n_batches, loop, "o--", color=c_loop, alpha=0.7,
                  linewidth=1.5, markersize=6,
                  label=f"{label_map[device]} Python loop")
        ax.loglog(n_batches, batch, "s-", color=c_batch,
                  linewidth=2.0, markersize=7,
                  label=f"{label_map[device]} 2D RHS  (K.solve(B))")

    ax.set_xlabel(r"batch size $n_{\mathrm{batch}}$")
    ax.set_ylabel("cost per problem  [ms]")
    ax.set_title("Same mesh, many RHS:  Python loop vs single 2D-RHS solve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT, "batched_rhs_scaling.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Figure 2: throughput (problems/sec).
# ---------------------------------------------------------------------------

def fig_throughput():
    print("[fig 2] throughput...")
    n_batches = [1, 4, 16, 64, 256, 1024]
    series = {}

    for device in ["cpu"] + (["cuda"] if HAS_CUDA else []):
        print(f"  device={device}")
        mesh, K_, M, cond = _build_poisson(device, h=0.02)
        torch.manual_seed(1)
        B_full = torch.randn(mesh.n_points, max(n_batches),
                             dtype=torch.float64, device=device)
        B_full = cond.condense_rhs(B_full)

        thr = []
        for nb in n_batches:
            B = B_full[:, :nb].contiguous()

            def run():
                _ = K_.solve(B)

            t = _time(run, n_warmup=1, n_runs=3, device=device)
            thr.append(nb / t)
            print(f"    nb={nb:4d}  total={t*1e3:8.2f}ms  "
                  f"throughput={nb / t:8.1f} problems/s")
        series[device] = np.array(thr)
        del mesh, K_, M, cond, B_full
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    style = {"cpu":  ("#2c3e50", "o-", "CPU"),
             "cuda": ("#c0392b", "s-", "GPU")}
    for device, thr in series.items():
        color, marker, lbl = style[device]
        ax.loglog(n_batches, thr, marker, color=color,
                  linewidth=2.0, markersize=7, label=lbl)
    ax.set_xlabel(r"batch size $n_{\mathrm{batch}}$")
    ax.set_ylabel("throughput  [problems / s]")
    ax.set_title("Throughput of a single 2D-RHS solve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT, "throughput.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: memory chunking on the assembly side.
# ---------------------------------------------------------------------------

def fig_memory_chunking():
    if not HAS_CUDA:
        print("[fig 3] skipped (no CUDA)")
        return

    print("[fig 3] memory chunking...")
    device = "cuda"
    # ``batch_size`` chunks the QUADRATURE dimension n_q, so the knob only
    # bites while batch_size < n_q -- and at the default quadrature order n_q
    # is tiny (3 for an order-2 triangle), small enough that the per-element
    # matrix and the COO index buffers dominate the peak. Use a higher
    # quadrature order so the chunkable integrand
    # [n_elem, n_q, n_basis, n_basis] is the dominant allocation and the
    # memory win is visible.
    mesh = Mesh.gen_rectangle(chara_length=0.005, order=2).to(device)
    quad_order = 5

    def assemble(bs):
        return LaplaceElementAssembler.from_mesh(
            mesh, quadrature_order=quad_order)(batch_size=bs).double()

    asm0 = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=quad_order)
    n_q = asm0.transformation[asm0.element_types[0]].n_quadrature

    # All batch_size values strictly below n_q (batch_size >= n_q is identical
    # to no chunking), ordered so the memory bars descend left -> right.
    bs_values = [-1, 4, 2, 1]

    # Reference matrix (un-chunked). Keep only its edata, on the CPU, so a full
    # second GPU matrix is not resident while we measure the peaks.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    K_ref = assemble(-1)
    ref_edata = K_ref.edata.detach().cpu()
    del K_ref
    gc.collect()
    torch.cuda.empty_cache()

    times, peaks, diffs = [], [], []
    for bs in bs_values:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        K = assemble(bs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e6  # MB

        diff = (K.edata.detach().cpu() - ref_edata).abs().max().item()
        diffs.append(diff)

        times.append(dt * 1e3)
        peaks.append(peak)
        n_chunk = 1 if bs == -1 else math.ceil(n_q / bs)
        print(f"    batch_size={str(bs):>3}  ({n_chunk} chunk(s))  "
              f"time={dt*1e3:7.2f}ms   peak={peak:7.1f}MB   "
              f"max|K - K_ref|={diff:.2e}")
        del K
        gc.collect()
        torch.cuda.empty_cache()

    # The first entry is the un-chunked reference; the rest are decreasing
    # chunk sizes, so peak memory drops left -> right while time rises.
    labels = ["full"] + [str(b) for b in bs_values[1:]]
    xs = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    color_time = "#2980b9"
    color_mem = "#c0392b"

    ax1.bar(xs - 0.18, times, width=0.36, color=color_time, alpha=0.85,
            label="assemble time")
    ax1.set_ylabel("assemble time  [ms]", color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(r"assembler ``batch_size``  (-1 = no chunking)")

    ax2 = ax1.twinx()
    ax2.bar(xs + 0.18, peaks, width=0.36, color=color_mem, alpha=0.85,
            label="peak GPU mem")
    ax2.set_ylabel("peak GPU memory  [MB]", color=color_mem)
    ax2.tick_params(axis="y", labelcolor=color_mem)

    ax1.set_title(f"Memory chunking on a {mesh.n_points}-node quadratic mesh "
                  f"($n_q={n_q}$ quadrature pts)\n"
                  f"max$|K - K_{{full}}| <$ {max(diffs):.0e}",
                  fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT, "memory_chunking.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")


if __name__ == "__main__":
    fig_batched_rhs_scaling()
    fig_throughput()
    fig_memory_chunking()
    print("done.")

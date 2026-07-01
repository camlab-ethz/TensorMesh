"""Acoustic transmission through a 2D phononic-crystal slab.

A normally-incident plane wave in water hits a finite slab of rigid (sound-hard)
cylinders arranged as a square lattice; we compute the power transmission
``T(f) = <|p|^2>_out / |p0|^2`` over a frequency sweep and recover the phononic
band gap.

It is deliberately example-only: no public API is added. It reuses TensorMesh's
scalar Helmholtz assembly and complex sparse solve:

    mesh -> Laplace/Mass assembler -> (K - k^2 M - i k B) p = -2 i k p0 e_in

The radiation boundary term ``B`` and the incident-field load ``e_in`` are
hand-rolled boundary line integrals (TensorMesh ships no facet assembler).
COMSOL's "Plane Wave Radiation" is a first-order absorbing condition, which this
matches exactly, so no PML is needed; at normal incidence the lateral periodic
walls are mirror-symmetry planes, equivalent to natural Neumann.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh import LaplaceElementAssembler, MassElementAssembler  # noqa: E402
from tensormesh.sparse import SparseMatrix, spsolve  # noqa: E402  -- complex solve

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class TransmissionSlab:
    """Parameters for the transmission example.

    Parameters
    ----------
    a:
        Square-lattice constant and strip height, in metres.
    r_over_a:
        Cylinder radius as a fraction of ``a`` (rigid, sound-hard).
    n_cyl:
        Number of cylinders along the propagation (x) direction.
    buffer_cells:
        Water buffer on each side of the slab, in units of ``a``.
    c:
        Speed of sound in water, in m/s.
    p0:
        Incident plane-wave pressure amplitude, in Pa.
    f_min, f_max, f_step:
        Frequency sweep bounds and step, in Hz.
    chara_per_a:
        Mesh elements per lattice constant (mesh density).
    """

    a: float = 0.01
    r_over_a: float = 0.45
    n_cyl: int = 10
    buffer_cells: float = 3.0
    c: float = 1481.0
    p0: float = 1.0
    f_min: float = 30.0e3
    f_max: float = 120.0e3
    f_gap_lo: float = 40.0e3       # below this: dense (pass band + lower gap edge)
    f_gap_hi: float = 103.0e3      # above this: dense (upper gap edge + pass band)
    f_step: float = 2.0e3          # coarse step inside the flat band gap
    f_step_dense: float = 0.5e3    # fine step in the pass bands / gap edges
    chara_per_a: float = 36.0

    @property
    def r(self) -> float:
        return self.r_over_a * self.a

    @property
    def lx(self) -> float:
        return (self.n_cyl + 2.0 * self.buffer_cells) * self.a

    @property
    def ly(self) -> float:
        return self.a

    @property
    def h(self) -> float:
        return self.a / self.chara_per_a

    @property
    def cyl_x(self) -> List[float]:
        x0 = self.buffer_cells * self.a
        return [x0 + (i + 0.5) * self.a for i in range(self.n_cyl)]

    @property
    def freqs(self) -> np.ndarray:
        """Non-uniform sweep: dense in the pass bands / gap edges, coarse in the
        flat band gap where the spectrum barely changes."""
        lo = np.arange(self.f_min, self.f_gap_lo, self.f_step_dense)
        mid = np.arange(self.f_gap_lo, self.f_gap_hi, self.f_step)
        hi = np.arange(self.f_gap_hi, self.f_max + 1e-9, self.f_step_dense)
        return np.unique(np.concatenate([lo, mid, hi]))


# --------------------------------------------------------------------------- #
# Mesh: strip with N circular holes (rigid cylinders), natural Neumann walls
# --------------------------------------------------------------------------- #
def build_mesh(problem: TransmissionSlab):
    """Mesh the fluid strip with the cylinders cut out (sound-hard holes)."""
    import gmsh

    msh = os.path.join(os.path.dirname(__file__), "_slab.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("transmission_slab")
        occ = gmsh.model.occ
        rect = occ.addRectangle(0.0, 0.0, 0.0, problem.lx, problem.ly)
        disks = [(2, occ.addDisk(cx, problem.ly / 2.0, 0.0, problem.r, problem.r))
                 for cx in problem.cyl_x]
        occ.cut([(2, rect)], disks)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", problem.h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", problem.h)
        gmsh.model.mesh.generate(2)
        gmsh.write(msh)
    finally:
        gmsh.finalize()

    from tensormesh.mesh import Mesh
    mesh = Mesh.from_file(msh, reorder=False)
    os.remove(msh)
    return mesh


# --------------------------------------------------------------------------- #
# Volume operators K (grad.grad) and M (mass) as TensorMesh sparse matrices
# --------------------------------------------------------------------------- #
def assemble_volume(mesh) -> Tuple[SparseMatrix, SparseMatrix]:
    K_asm = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)
    M_asm = MassElementAssembler.from_assembler(K_asm)
    return K_asm(mesh.points), M_asm(mesh.points)


# --------------------------------------------------------------------------- #
# Boundary line operators on x = 0 (input) and x = Lx (output)  [hand-rolled]
# --------------------------------------------------------------------------- #
def boundary_operators(mesh, problem: TransmissionSlab, tol: float = 1e-7):
    """Return (b_row, b_col, b_val, e_in, out_idx) as torch tensors.

    ``B = int_{Gin u Gout} phi_i phi_j ds`` (COO triplets) is the first-order
    radiation term; ``e_in = int_{Gin} phi_i ds`` is the incident-field load on
    the input edge; ``out_idx`` are the output-edge node indices.
    """
    pts = mesh.points.cpu().numpy()
    n = pts.shape[0]
    rows, cols, vals = [], [], []
    e_in = np.zeros(n)

    def add_edge_line(xb: float, is_input: bool):
        idx = np.where(np.abs(pts[:, 0] - xb) < tol)[0]
        order = idx[np.argsort(pts[idx, 1])]              # ordered along y
        for i in range(len(order) - 1):
            ia, ib = int(order[i]), int(order[i + 1])
            le = abs(pts[ib, 1] - pts[ia, 1])
            for r, c, w in ((ia, ia, 2), (ib, ib, 2), (ia, ib, 1), (ib, ia, 1)):
                rows.append(r); cols.append(c); vals.append(w * le / 6.0)
            if is_input:
                e_in[ia] += le / 2.0
                e_in[ib] += le / 2.0
        return order

    add_edge_line(0.0, is_input=True)
    out_idx = add_edge_line(problem.lx, is_input=False)
    return (torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(vals, dtype=torch.float64),
            torch.as_tensor(e_in, dtype=torch.float64),
            torch.as_tensor(out_idx, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Frequency sweep -> transmission spectrum
# --------------------------------------------------------------------------- #
def transmission_spectrum(problem: TransmissionSlab, verbose: bool = True) -> Dict[str, Any]:
    mesh = build_mesh(problem)
    K, M = assemble_volume(mesh)
    b_row, b_col, b_val, e_in, out_idx = boundary_operators(mesh, problem)
    n = int(mesh.points.shape[0])
    freqs = problem.freqs
    if verbose:
        print(f"mesh: {n} nodes, {mesh.elements().shape[0]} elements; "
              f"{len(out_idx)} output-edge nodes", flush=True)

    # Assemble A = K - k^2 M - i k B once per frequency as COO triplets and
    # hand them to the TensorMesh complex sparse solver (scipy-backed direct LU
    # via torch-sla); duplicate (row, col) entries are summed on coalesce.
    # time convention e^{-i w t}, outgoing dp/dn = +ik p:
    #   (K - k^2 M - i k B) p = -2 i k p0 e_in
    row = torch.cat([K.row, M.row, b_row])
    col = torch.cat([K.col, M.col, b_col])
    Kv = K.edata.to(torch.complex128)
    Mv = M.edata.to(torch.complex128)
    Bv = b_val.to(torch.complex128)
    e_in_c = e_in.to(torch.complex128)

    T = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        k = 2.0 * math.pi * f / problem.c
        val = torch.cat([Kv, -(k * k) * Mv, (-1j * k) * Bv])
        b = (-2j * k * problem.p0) * e_in_c
        p = spsolve(val, row, col, (n, n), b,
                    backend="scipy", method="lu", is_spd=False)
        T[i] = float(torch.mean(torch.abs(p[out_idx]) ** 2)) / (problem.p0 ** 2)
        if verbose and i % max(1, len(freqs) // 6) == 0:
            print(f"  f {i+1}/{len(freqs)}  ({f/1e3:.0f} kHz)  T={T[i]:.3f}", flush=True)

    return dict(freqs=freqs, T=T, n_nodes=n)


# --------------------------------------------------------------------------- #
# COMSOL reference (committed, offline): transmission spectrum T(f)
# --------------------------------------------------------------------------- #
def load_comsol_reference(path=None):
    """Load the committed COMSOL transmission spectrum
    (``comsol_reference_transmission.npz``: ``freqs_hz``, ``T``, ``c_water``).
    Returns ``None`` if the file is absent so the example still runs offline."""
    if path is None:
        path = Path(__file__).with_name("comsol_reference_transmission.npz")
    if not Path(path).exists():
        return None
    d = np.load(path)
    return dict(freqs=d["freqs_hz"], T=d["T"], c=float(d["c_water"]))


def compare_to_comsol(result: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, float]:
    """Mean / max absolute transmission difference on the COMSOL frequency
    grid (TensorMesh spectrum linearly interpolated onto COMSOL's frequencies,
    restricted to the overlapping band)."""
    f = ref["freqs"]
    inside = (f >= result["freqs"].min()) & (f <= result["freqs"].max())
    Tt = np.interp(f[inside], result["freqs"], result["T"])
    dT = np.abs(Tt - ref["T"][inside])
    return dict(mean=float(dT.mean()), max=float(dT.max()), n=int(dT.size))


# --------------------------------------------------------------------------- #
# Plot: slab geometry | transmission spectrum
# --------------------------------------------------------------------------- #
def plot_transmission(result: Dict[str, Any], problem: TransmissionSlab,
                      save_path, ref: Dict[str, Any] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    f = result["freqs"] / 1e3
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2.2], wspace=0.25)

    # -- Left panel: slab geometry -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    mm = 1e3
    ax.add_patch(Rectangle((0, 0), problem.lx * mm, problem.ly * mm,
                           fc="#cfe8ff", ec="k", lw=1))
    for cx in problem.cyl_x:
        ax.add_patch(Circle((cx * mm, problem.ly / 2 * mm), problem.r * mm,
                            fc="0.4", ec="k", lw=0.6))
    ax.annotate("incident\nplane wave", (0.5, problem.ly * mm * 1.6),
                color="tab:red", fontsize=8)
    ax.arrow(0.3, problem.ly * mm / 2, 2.2, 0, head_width=0.8, color="tab:red")
    ax.set_xlim(-3, problem.lx * mm + 3); ax.set_ylim(-6, 12)
    ax.set_aspect("equal")
    ax.set_title(f"Slab: {problem.n_cyl} rigid cylinders in water")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

    # -- Right panel: transmission spectrum (+ COMSOL reference if available) --
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(f, result["T"], "-", color="tab:blue", lw=1.6, label="TensorMesh")
    if ref is not None:
        ax.plot(ref["freqs"] / 1e3, ref["T"], "o", ms=4.5, mfc="none",
                mec="#D55E00", mew=1.0, label="COMSOL")
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel(r"transmission  $T = \langle|p|^2\rangle / |p_0|^2$")
    title = "Transmission through the phononic crystal slab"
    if "stats" in result:
        title += f"  (mean |dT| {result['stats']['mean']:.3f} vs COMSOL)"
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.05); ax.grid(True, alpha=0.3)
    if ref is not None:
        ax.legend(loc="upper right", fontsize=8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Demo runner
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, output_path=None,
             chara_per_a: float = 36.0, verbose: bool = True) -> Dict[str, Any]:
    """Run the transmission example and return diagnostics."""
    ref = load_comsol_reference()
    # Match COMSOL's sound speed when validating so the comparison is geometry,
    # not a 0.03% speed-of-sound offset.
    problem = TransmissionSlab(chara_per_a=chara_per_a,
                               c=ref["c"] if ref is not None else 1481.0)
    result = transmission_spectrum(problem, verbose=verbose)
    if ref is not None:
        result["stats"] = compare_to_comsol(result, ref)

    if make_plot:
        if output_path is None:
            output_path = Path(__file__).with_name("transmission_slab.png")
        plot_transmission(result, problem, output_path, ref=ref)

    T, freqs = result["T"], result["freqs"]
    gap = freqs[T < 0.05] / 1e3
    print("Transmission through a phononic-crystal slab")
    print(f"  nodes: {result['n_nodes']}")
    print(f"  frequencies: {len(freqs)} in [{freqs.min()/1e3:.0f}, {freqs.max()/1e3:.0f}] kHz")
    if gap.size:
        print(f"  band gap (T < 0.05): [{gap.min():.0f}, {gap.max():.0f}] kHz")
    print(f"  T range: [{T.min():.3f}, {T.max():.3f}]")
    if "stats" in result:
        s = result["stats"]
        print(f"  vs COMSOL ({s['n']} freqs): mean |dT| {s['mean']:.3f}  "
              f"max |dT| {s['max']:.3f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip PNG generation")
    parser.add_argument("--chara-per-a", type=float, default=36.0,
                        help="mesh elements per lattice constant")
    parser.add_argument("--output", type=str, default=None, help="output PNG path")
    args = parser.parse_args()
    run_demo(make_plot=not args.no_plot, output_path=args.output,
             chara_per_a=args.chara_per_a)


if __name__ == "__main__":
    main()

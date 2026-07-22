"""2D Helmholtz-resonator side-branch on a duct — port-driven pressure acoustics.

A straight air duct carries a plane wave injected by a **port** at the inlet;
a Helmholtz resonator (a narrow neck opening into a larger cavity) hangs off the
duct's far wall.  Near the Helmholtz resonance the cavity pressure is strongly
amplified and the port reactance sweeps through zero.  Everything is solved on
TensorMesh (torch); no external reference is needed.

Geometry (meters) — three axis-aligned rectangles, unioned::

    duct   : [0, D] x [0, Dh]                      D  = Dh = 0.1
    neck   : [D, D+Ln] x [0, Nh]                   Ln = 0.01, Nh = 0.002
    cavity : [D+Ln, D+Ln+Lc] x [0, Ch]             Lc = Ch = 0.05

with ``D = duct_length``, ``Dh = duct_height``, ``Ln = neck_length``,
``Nh = neck_height``, ``Lc = cavity_length``, ``Ch = cavity_height``.  The neck
opening (x = D, 0 <= y <= Nh) is interior after the union; the rest of the far
wall (x = D, Nh <= y <= Dh) is rigid.

Boundary conditions::

    inlet x = 0 : plane-wave port, incident amplitude p0 = 1  (absorbing + source)
    all others  : sound-hard,  dp/dn = 0                       (natural Neumann)

Helmholtz, time convention ``e^{+iwt}`` (outgoing ~ e^{-ikr}, dp/dn = -ik p):

    (K - k^2 M + i k B_in) p = 2 i k p0 e_in ,   k = w / c0 = 2 pi f / c0

  K_ij  = int grad phi_i . grad phi_j          (LaplaceElementAssembler)
  M_ij  = int phi_i phi_j                       (MassElementAssembler)
  B_in  = int_{x=0} phi_i phi_j ds              (hand-rolled inlet line mass)
  e_in  = int_{x=0} phi_i ds                    (incident plane-wave load)

The port line mass ``B_in`` and load ``e_in`` come from the library boundary
operators :func:`tensormesh.robin_operator` and :func:`tensormesh.port_source`
(themselves built on ``FacetBilinearAssembler``); ``B_in`` is folded onto the
volume sparsity pattern so the whole operator stays a single complex
``SparseMatrix`` (``SparseMatrix`` add needs matching layouts).  No public API is
added — this example only *uses* the library.

Workflow:  mesh -> assemble (K, M) -> port line operators -> per-f complex solve
           -> input impedance Z(f) + pressure field p(x, y; f0).

Run (env with torch + tensormesh + gmsh):
    python examples/wave/helmholtz_resonator/helmholtz_resonator.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from tensormesh import (LaplaceElementAssembler, MassElementAssembler, Mesh,
                        robin_operator, port_source)
from tensormesh.sparse.matrix import SparseMatrix

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Resonator:
    r"""All parameters of the duct + Helmholtz-resonator problem.

    Fields
    ------
    duct_length, duct_height : duct dimensions (m).
    neck_length, neck_height : neck length (along the duct axis) and aperture (m).
    cavity_length, cavity_height : cavity dimensions (m).
    rho0, c0 : air density (kg/m^3) and sound speed (m/s) at 20 C, 1 atm.
    p0 : incident plane-wave amplitude at the port (Pa).
    f_min, f_max, n_freq : frequency sweep (Hz).
    mesh_size : target element edge length (m).
    f_field : frequency at which the pressure field snapshot is drawn (Hz).
    """

    duct_length: float = 0.1
    duct_height: float = 0.1
    neck_length: float = 0.01
    neck_height: float = 0.002
    cavity_length: float = 0.05
    cavity_height: float = 0.05

    rho0: float = 1.2041
    c0: float = 343.2
    p0: float = 1.0

    f_min: float = 20.0
    f_max: float = 500.0
    n_freq: int = 500

    mesh_size: float = 0.0015
    f_field: float = 357.0

    @property
    def freqs(self) -> torch.Tensor:
        """Sweep frequencies (Hz), shape ``[n_freq]``."""
        return torch.linspace(self.f_min, self.f_max, self.n_freq, dtype=torch.float64)

    @property
    def total_length(self) -> float:
        """Overall x-extent of the geometry (m)."""
        return self.duct_length + self.neck_length + self.cavity_length

    @property
    def cavity_probe(self) -> Tuple[float, float]:
        """Point at the cavity center (m) — a convenient response probe."""
        return (self.duct_length + self.neck_length + 0.5 * self.cavity_length,
                0.5 * self.cavity_height)

    @property
    def Z0(self) -> float:
        """Characteristic (specific) acoustic impedance of air, ``rho0 c0``."""
        return self.rho0 * self.c0


# --------------------------------------------------------------------------- #
# 1. Mesh: the three rectangles, unioned
# --------------------------------------------------------------------------- #
def build_mesh(problem: Resonator, msh_path: Optional[str] = None) -> Mesh:
    """Mesh the duct + neck + cavity union with gmsh, return a TensorMesh ``Mesh``."""
    import gmsh

    if msh_path is None:
        msh_path = str(HERE / "_resonator.msh")
    h = problem.mesh_size

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("helmholtz_resonator")
        occ = gmsh.model.occ
        duct = occ.addRectangle(0.0, 0.0, 0.0, problem.duct_length, problem.duct_height)
        neck = occ.addRectangle(problem.duct_length, 0.0, 0.0,
                                problem.neck_length, problem.neck_height)
        cavity = occ.addRectangle(problem.duct_length + problem.neck_length, 0.0, 0.0,
                                  problem.cavity_length, problem.cavity_height)
        occ.fuse([(2, duct)], [(2, neck), (2, cavity)])
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()

    return Mesh.from_file(msh_path, reorder=False)


# --------------------------------------------------------------------------- #
# 2. Volume operators (real) — K = grad.grad, M = mass, sharing one layout
# --------------------------------------------------------------------------- #
def assemble_volume(mesh: Mesh) -> Tuple[SparseMatrix, SparseMatrix]:
    """Assemble the stiffness ``K`` and mass ``M`` matrices (real, shared layout)."""
    K_asm = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)
    M_asm = MassElementAssembler.from_assembler(K_asm)
    K = K_asm(mesh.points)
    M = M_asm(mesh.points)
    return K, M


# --------------------------------------------------------------------------- #
# 3. Port line operators on the inlet x = 0  [hand-rolled, folded onto K layout]
# --------------------------------------------------------------------------- #
def _fold_onto_layout(K: SparseMatrix, rows: torch.Tensor, cols: torch.Tensor,
                      vals: torch.Tensor) -> torch.Tensor:
    """Scatter boundary entries ``(rows, cols, vals)`` onto ``K``'s sparsity pattern.

    Returns a values vector aligned with ``K.row``/``K.col`` so that
    ``K.values + <this>`` is a legal same-layout combination.  Every boundary
    entry must already exist in ``K`` (boundary nodes sharing an edge are
    volume-adjacent), otherwise the assertion trips.
    """
    n = K.shape[0]
    key_t = K.row.long() * n + K.col.long()
    order = torch.argsort(key_t)
    sorted_keys = key_t[order]
    key_b = rows.long() * n + cols.long()
    pos = torch.searchsorted(sorted_keys, key_b)
    hit = order[pos]
    assert torch.equal(key_t[hit], key_b), "boundary entry missing from volume layout"
    out = torch.zeros_like(K.values, dtype=torch.float64)
    out.index_add_(0, hit, vals.to(torch.float64))
    return out


def port_operators(mesh: Mesh, K: SparseMatrix, tol: float = 1e-7
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inlet line mass ``B_in`` (as K-aligned values), load ``e_in``, inlet indices.

    The inlet is the straight edge ``x = 0``.  The boundary bilinear form
    ``B_in = int_inlet phi_i phi_j ds`` and the incident-wave load
    ``e_in = int_inlet phi_i ds`` are assembled by the library
    (:func:`tensormesh.robin_operator` / :func:`tensormesh.port_source`); ``B_in``
    is then folded onto ``K``'s sparsity pattern so the whole operator stays a
    single complex ``SparseMatrix`` (``SparseMatrix`` add needs matching layouts).
    """
    pts = mesh.points.to(torch.float64)
    inlet_mask = pts[:, 0].abs() < tol
    inlet = torch.where(inlet_mask)[0]

    B = robin_operator(mesh, inlet_mask, 1.0, points=pts)          # int_inlet phi_i phi_j ds
    e_in = port_source(mesh, inlet_mask, 1.0, points=pts).to(torch.float64)  # int_inlet phi_i ds
    B_vals = _fold_onto_layout(K, B.row, B.col, B.values.to(torch.float64))
    return B_vals, e_in, inlet


# --------------------------------------------------------------------------- #
# 4. Frequency sweep -> pressure fields, input impedance, cavity spectrum
# --------------------------------------------------------------------------- #
def solve_spectrum(problem: Resonator, verbose: bool = True) -> Dict:
    """Solve the port-driven Helmholtz problem over the sweep; return diagnostics.

    Returns a dict with ``freqs``, ``mesh``, ``points``, per-frequency input
    impedance ``Z`` (normalized by ``rho0 c0``), the cavity-probe magnitude
    spectrum ``probe``, and the complex pressure field ``p_field`` at
    ``problem.f_field``.
    """
    mesh = build_mesh(problem)
    K, M = assemble_volume(mesh)
    B_vals, e_in, inlet = port_operators(mesh, K)
    pts = mesh.points.to(torch.float64)

    Kv = K.values.to(torch.complex128)
    Mv = M.values.to(torch.complex128)
    Bv = B_vals.to(torch.complex128)
    e_in_c = e_in.to(torch.complex128)
    inlet_len = float(e_in.sum())              # = duct_height (int over the inlet)

    freqs = problem.freqs
    probe_pt = torch.tensor(problem.cavity_probe, dtype=torch.float64)
    jprobe = int(((pts - probe_pt) ** 2).sum(1).argmin())

    if verbose:
        print(f"mesh: {mesh.n_points} nodes, {mesh.n_elements} triangles; "
              f"{len(inlet)} inlet nodes", flush=True)

    z_norm = torch.zeros(len(freqs), dtype=torch.complex128)
    probe = torch.zeros(len(freqs), dtype=torch.float64)
    p_field: Optional[torch.Tensor] = None
    jf_field = int((freqs - problem.f_field).abs().argmin())

    for i, f in enumerate(freqs.tolist()):
        k = 2.0 * torch.pi * f / problem.c0
        A_vals = Kv - (k * k) * Mv + (1j * k) * Bv
        A = SparseMatrix(A_vals, K.row, K.col, K.shape)
        rhs = (2j * k * problem.p0) * e_in_c
        p = A.solve(rhs)

        # input impedance at the port from the reflection coefficient:
        # total p = p0 (1 + R) at the inlet -> R = <p>_inlet / p0 - 1,
        # normalized specific impedance z = (1 + R) / (1 - R).
        p_avg = (e_in_c @ p) / inlet_len
        R = p_avg / problem.p0 - 1.0
        z_norm[i] = (1.0 + R) / (1.0 - R)
        probe[i] = float(p[jprobe].abs())
        if i == jf_field:
            p_field = p.clone()

    return dict(freqs=freqs, mesh=mesh, points=pts, Z=z_norm, probe=probe,
                p_field=p_field, f_field=float(freqs[jf_field]),
                Z0=problem.Z0, jprobe=jprobe)


# --------------------------------------------------------------------------- #
# 5. Plots — (a) input impedance Z(f), (b) pressure field at f0
# --------------------------------------------------------------------------- #
def plot_impedance(res: Dict, save_path: str) -> None:
    """Port impedance magnitude ``|Z|/Z0`` (log scale) + cavity response.

    The system is lossless (``|R| = 1``), so ``Z`` is purely reactive: ``|Z|``
    peaks at the **Helmholtz resonance** and dips to zero at the
    **anti-resonance**.  Both extrema show cleanly on a logarithmic axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    f = res["freqs"].cpu().numpy()
    Z = res["Z"].cpu().numpy()
    Zmag = 1.0 / np.maximum(np.abs(Z), 1e-30)      # peaks at resonance
    probe = res["probe"].cpu().numpy()

    BLUE, GREEN = "#0072B2", "#009E73"
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True)

    ax0.semilogy(f, Zmag, "-", color=BLUE, lw=1.5)
    ax0.set_xlim(f.min(), f.max())
    ax0.set_ylabel(r"$|Z| / \rho_0 c_0$")
    ax0.set_title("Helmholtz-resonator port impedance (log scale)")
    ax0.grid(True, which="both", alpha=0.25)

    ax1.plot(f, probe, "-", color=GREEN, lw=1.8)
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel(r"$|p|$ at cavity center (Pa)")
    ax1.set_title("Cavity pressure response")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


def _refine(triang, values, subdiv: int):
    """Cubic-interpolated uniform refinement -> smooth fields (e.g. node lines)."""
    import matplotlib.tri as mtri

    refiner = mtri.UniformTriRefiner(triang)
    interp = mtri.CubicTriInterpolator(triang, values)
    return refiner.refine_field(values, interp, subdiv=subdiv)


def plot_field(res: Dict, problem: Resonator, save_path: str,
               p_ref: float = 20e-6, subdiv: int = 3) -> None:
    """Acoustic field at ``f_field``: total acoustic pressure ``Re(p)`` (Pa) and
    total sound pressure level (dB).

    Both panels are drawn on a cubically-refined triangulation so the pressure
    node (|p| -> 0, where the SPL plunges) renders as a smooth line instead of a
    jagged one.  SPL uses the RMS pressure against ``p_ref`` (20 uPa in air):
    ``Lp = 20 log10(|p| / (sqrt(2) p_ref))``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    pts = res["points"].cpu().numpy()
    tris = res["mesh"].cells["triangle"].cpu().numpy()
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tris)
    p = res["p_field"].cpu().numpy()
    f0 = res["f_field"]

    # refine Re and Im (smooth) on a shared finer mesh, then recombine
    fine, p_re = _refine(triang, p.real, subdiv)
    _, p_im = _refine(triang, p.imag, subdiv)
    amp = np.hypot(p_re, p_im)
    spl = 20.0 * np.log10(np.maximum(amp / np.sqrt(2.0), 1e-30) / p_ref)
    lim = float(np.abs(p_re).max())

    xmax = problem.total_length
    panels = [
        ("Total acoustic pressure (Pa)", p_re, "RdBu_r", "Pa",
         dict(vmin=-lim, vmax=lim)),
        ("Total sound pressure level (dB)", spl, "jet", "dB",
         dict(vmin=float(spl.min()), vmax=float(spl.max()))),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8))
    for ax, (title, data, cmap, unit, norm) in zip(axes, panels):
        tpc = ax.tripcolor(fine, data, shading="gouraud", cmap=cmap,
                           rasterized=True, **norm)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.005, xmax + 0.025)
        ax.set_ylim(-0.005, problem.duct_height + 0.005)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        cb = plt.colorbar(tpc, ax=ax, shrink=0.9, pad=0.02)
        cb.ax.set_title(unit, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # center the suptitle over the plot axes (not the figure, which the right
    # color-bars would otherwise skew left)
    pos = axes[0].get_position()
    fig.suptitle(f"Helmholtz resonator @ {f0:.0f} Hz",
                 x=pos.x0 + 0.5 * pos.width, y=0.99)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 6. Boundary-condition schematic
# --------------------------------------------------------------------------- #
def plot_boundary_conditions(problem: Resonator, save_path: str) -> None:
    """Schematic of the geometry with the boundary conditions labeled.

    Inlet (x = 0): plane-wave port (absorbing + incident source).  Every other
    outer wall: sound-hard (rigid, ``dp/dn = 0``).  The neck opening is an
    interior interface (continuity), not a boundary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    D, Dh = problem.duct_length, problem.duct_height
    Ln, Nh = problem.neck_length, problem.neck_height
    Lc, Ch = problem.cavity_length, problem.cavity_height
    PORT, HARD = "#D62728", "#1F5FBF"          # red = port, blue = sound-hard
    xmax = D + Ln + Lc

    fig, ax = plt.subplots(figsize=(10, 6.4))
    for x, y, w, h in ((0, 0, D, Dh), (D, 0, Ln, Nh), (D + Ln, 0, Lc, Ch)):
        ax.add_patch(Rectangle((x, y), w, h, fc="#e5eefb", ec="none", zorder=1))

    segs = [
        ((0, 0), (0, Dh), "port"),                      # inlet
        ((0, Dh), (D, Dh), "hard"),                     # duct top
        ((D, Dh), (D, Nh), "hard"),                     # duct right, above neck
        ((D, Nh), (D + Ln, Nh), "hard"),                # neck top
        ((D + Ln, Nh), (D + Ln, Ch), "hard"),           # cavity left, above neck
        ((D + Ln, Ch), (D + Ln + Lc, Ch), "hard"),      # cavity top
        ((D + Ln + Lc, Ch), (D + Ln + Lc, 0), "hard"),  # cavity right
        ((D + Ln + Lc, 0), (0, 0), "hard"),             # bottom
    ]
    for (x0, y0), (x1, y1), kind in segs:
        ax.plot([x0, x1], [y0, y1], "-",
                color=PORT if kind == "port" else HARD, lw=4, zorder=5,
                solid_capstyle="round")

    ax.text(0.5 * D, 0.5 * Dh, "duct\n(air)", ha="center", va="center",
            fontsize=12, color="#1e293b")
    ax.text(D + Ln + 0.5 * Lc, 0.5 * Ch, "cavity", ha="center", va="center",
            fontsize=12, color="#1e293b")

    ax.legend(handles=[
        Line2D([0], [0], color=PORT, lw=4, label="plane-wave port  (inlet $x=0$)"),
        Line2D([0], [0], color=HARD, lw=4, label=r"sound-hard walls  ($\partial p/\partial n = 0$)"),
    ], loc="upper right", fontsize=9, framealpha=0.96)

    ax.set_aspect("equal")
    ax.set_xlim(-0.04 * xmax, xmax * 1.06)
    ax.set_ylim(-0.06 * Dh, Dh * 1.12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Helmholtz resonator — port-driven duct with side cavity", fontsize=13)
    ax.grid(True, color="#f1f5f9", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 7. run_demo / main
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True,
             impedance_path: Optional[str] = None,
             field_path: Optional[str] = None,
             bc_path: Optional[str] = None,
             problem: Optional[Resonator] = None) -> Dict:
    """Run the case, save the two figures, print a summary, return diagnostics."""
    problem = problem or Resonator()
    res = solve_spectrum(problem)

    import numpy as np
    f = res["freqs"].cpu().numpy()
    probe = res["probe"].cpu().numpy()
    ipk = int(np.argmax(probe))
    print(f"resonance: f = {f[ipk]:.0f} Hz, "
          f"|p|_cavity = {probe[ipk]:.2f} Pa (incident p0 = {problem.p0:g})",
          flush=True)

    if make_plot:
        plot_boundary_conditions(problem, bc_path or str(HERE / "boundary_conditions.png"))
        plot_impedance(res, impedance_path or str(HERE / "impedance.png"))
        plot_field(res, problem, field_path or str(HERE / "acoustic_field.png"))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--mesh-size", type=float, default=Resonator.mesh_size)
    parser.add_argument("--f-field", type=float, default=Resonator.f_field,
                        help="frequency (Hz) for the field snapshot")
    parser.add_argument("--impedance-output", type=str, default=None)
    parser.add_argument("--field-output", type=str, default=None)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    problem = Resonator(mesh_size=args.mesh_size, f_field=args.f_field)
    run_demo(make_plot=not args.no_plot,
             impedance_path=args.impedance_output,
             field_path=args.field_output,
             problem=problem)


if __name__ == "__main__":
    main()

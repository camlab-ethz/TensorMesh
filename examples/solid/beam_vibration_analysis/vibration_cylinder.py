"""Cantilever cylinder eigenmodes + displacement FRF on TensorMesh — 3D elasticity.

A clamped-free circular steel rod (length ``L`` = 1 m, radius ``R`` = 25 mm) is
analysed for its structural vibration: the natural mode shapes and the harmonic
drive-point frequency-response function (FRF).  Linear elasticity on a tetrahedral
mesh gives the stiffness ``K`` and consistent mass ``M``; clamping the base face
removes those degrees of freedom.

The **modes** solve the generalized eigenproblem

    K phi = omega^2 M phi ,        f_n = omega_n / (2 pi) ,

for the lowest ``n_modes`` pairs (shift-invert Lanczos near ``sigma = 0``).  The
**FRF** is built by modal superposition with the mass-normalized modes and an
isotropic (hysteretic) loss factor ``eta`` = 2 % (a 1 % modal damping ratio,
``zeta = eta / 2``):

    u(omega) = sum_r  phi_r (phi_r . F) / ( omega_r^2 (1 + i eta) - omega^2 ) .

The tip-average ``|u_x|`` is normalized by its static (``omega -> 0``) value, so
the FRF is a **dimensionless** dynamic amplification: 1 below the first resonance,
peaking on each bending mode and dropping into antiresonances between them.

``K`` comes from :class:`~tensormesh.assemble.LinearElasticityElementAssembler`
and the consistent mass is the scalar
:class:`~tensormesh.assemble.MassElementAssembler` lifted to the 3 vector
components (``M_vec = rho * (M_scalar (x) I_3)``).  The eigen solve uses SciPy
sparse (post-processing), keeping the TensorMesh compute core numpy-free.
No public API is added here.

Workflow:  mesh (gmsh cylinder, tets) -> assemble (K, M) -> eig (modes)
           -> modal FRF -> plot mode shapes + FRF.

Run (env with torch + tensormesh + gmsh + scipy):
    python "examples/solid/beam_vibration_analysis/vibration_cylinder.py"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from tensormesh import Mesh
from tensormesh.assemble import (LinearElasticityElementAssembler,
                                 MassElementAssembler)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cylinder:
    r"""All parameters of the clamped-free steel cylinder (SI units).

    Fields
    ------
    length, radius : cylinder length ``L`` and section radius ``R`` (m).
    E, nu, rho : Young's modulus (Pa), Poisson ratio, density (kg/m^3).
    n_modes : number of eigenmodes to extract.
    eta : isotropic (hysteretic) loss factor; modal damping ratio is ``eta/2``.
    f0 : total transverse (x) tip force for the FRF (N).
    frf_f0, frf_f1, frf_n : FRF sweep start / stop (Hz) and number of points.
    mesh_h : target tetra edge length (m); default ``radius / 4``.
    """

    length: float = 1.0
    radius: float = 0.025
    E: float = 210.0e9
    nu: float = 0.3
    rho: float = 7850.0
    n_modes: int = 12
    eta: float = 0.02
    f0: float = 1.0
    frf_f0: float = 5.0
    frf_f1: float = 1500.0
    frf_n: int = 400
    mesh_h: Optional[float] = None

    @property
    def h(self) -> float:
        """Resolved tetra edge length (m); default ``radius / 4``."""
        return self.mesh_h or self.radius / 4.0

    @property
    def area(self) -> float:
        """Cross-section area ``pi R^2`` (m^2)."""
        return float(np.pi * self.radius ** 2)

    @property
    def inertia(self) -> float:
        """Second moment of area ``pi R^4 / 4`` (m^4)."""
        return float(np.pi * self.radius ** 4 / 4.0)

    def analytic_bending(self, n: int = 4) -> np.ndarray:
        r"""Euler-Bernoulli cantilever bending frequencies (Hz), first ``n``."""
        beta_l = np.array([1.875104, 4.694091, 7.854757, 10.995541])[:n]
        coeff = np.sqrt(self.E * self.inertia / (self.rho * self.area * self.length ** 4))
        return (beta_l ** 2) / (2.0 * np.pi) * coeff


# --------------------------------------------------------------------------- #
# 1. Mesh: tetrahedral cylinder, axis = z, base at z = 0
# --------------------------------------------------------------------------- #
def build_mesh(problem: Cylinder, msh_path: Optional[str] = None) -> Mesh:
    """gmsh tetrahedral mesh of the cylinder; returns a tetra-only Mesh."""
    import gmsh
    import meshio

    if msh_path is None:
        msh_path = str(HERE / "_cylinder.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        occ.addCylinder(0, 0, 0, 0, 0, problem.length, problem.radius)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", problem.h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", problem.h)
        gmsh.model.mesh.generate(3)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()

    # keep only the tetrahedra — lower-dim facets confuse a 3D volume assembler.
    m = meshio.read(msh_path)
    tets = np.vstack([c.data for c in m.cells if c.type == "tetra"])
    clean = meshio.Mesh(points=m.points, cells=[("tetra", tets)])
    return Mesh(clean, reorder=False)


# --------------------------------------------------------------------------- #
# 2. Assemble stiffness K and consistent (vector) mass M as SciPy sparse
# --------------------------------------------------------------------------- #
def assemble(problem: Cylinder, mesh: Mesh):
    """Return ``(K, M)`` as CSC (3N x 3N) with node-major, component-minor DOFs."""
    pts = mesh.points.to(torch.float64)

    Kasm = LinearElasticityElementAssembler.from_mesh(mesh, E=problem.E, nu=problem.nu)
    K = Kasm().to_scipy_coo().tocsc()

    # scalar consistent mass  M_ij = int phi_i phi_j ; lift to the 3 components:
    # for isotropic density the vector mass is rho * (M_scalar kron I_3), which
    # matches the assembler's node-major / component-minor DOF ordering.
    Masm = MassElementAssembler.from_mesh(mesh)
    Ms = Masm(pts).to_scipy_coo().tocsc()
    M = (problem.rho * sp.kron(Ms, sp.eye(3, format="csc"), format="csc")).tocsc()
    return K, M


# --------------------------------------------------------------------------- #
# 3. Boundary DOFs: clamp the base face z = 0
# --------------------------------------------------------------------------- #
def dof_masks(problem: Cylinder, mesh: Mesh):
    """Return ``(free_dofs, tip_nodes)``: free DOF indices + free-tip node ids."""
    pts = mesh.points.cpu().numpy()
    tol = problem.radius * 1e-4
    fixed_node = pts[:, 2] < tol
    tip_node = np.where(pts[:, 2] > problem.length - tol)[0]
    free = np.where(np.repeat(~fixed_node, 3))[0]
    return free, tip_node


# --------------------------------------------------------------------------- #
# 4. Eigenmodes:  K phi = omega^2 M phi
# --------------------------------------------------------------------------- #
def solve_modes(problem: Cylinder, K, M, free: np.ndarray) -> Dict:
    """Lowest ``n_modes`` natural frequencies (Hz) + full-length mode shapes."""
    Kff = K[free][:, free]
    Mff = M[free][:, free]
    # shift-invert near 0 pulls out the lowest eigenpairs of the GEVP.
    lam, vec = spla.eigsh(Kff, k=problem.n_modes, M=Mff, sigma=0.0, which="LM")
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    freqs = np.sqrt(np.maximum(lam, 0.0)) / (2.0 * np.pi)

    n = K.shape[0]
    shapes = np.zeros((problem.n_modes, n))
    shapes[:, free] = vec.T
    return dict(freqs=freqs, shapes=shapes)     # shapes[m] is a length-3N vector


# --------------------------------------------------------------------------- #
# 5. FRF:  drive-point tip receptance by modal superposition
# --------------------------------------------------------------------------- #
def solve_frf(problem: Cylinder, modes: Dict, tip: np.ndarray) -> Dict:
    r"""Tip transverse displacement amplitude ``|u_x|`` (m) vs frequency.

    Modal superposition with the mass-normalized modes ``phi_r`` (from
    :func:`solve_modes`, where ``phi^T M phi = I``) and hysteretic damping, driven
    by the transverse tip force ``F`` = ``f0`` (N):

        u(omega) = sum_r  phi_r (phi_r . F) / ( omega_r^2 (1 + i eta) - omega^2 ) .

    Returns the steady-state tip-face-average ``|u_x|`` in meters — the physical
    vibration amplitude of the free end at each drive frequency.
    """
    shapes, wr = modes["shapes"], 2.0 * np.pi * modes["freqs"]   # phi_r, omega_r
    tip_x = 3 * tip + 0                                          # x-DOF per tip node

    # modal force  q_r = phi_r . F  (total x-force f0 spread over the tip face)
    qr = shapes[:, tip_x].sum(axis=1) * (problem.f0 / len(tip))
    phi_tip = shapes[:, tip_x].mean(axis=1)                     # tip-average phi_r

    freqs = np.linspace(problem.frf_f0, problem.frf_f1, problem.frf_n)
    w2 = (2.0 * np.pi * freqs) ** 2
    denom = (wr[:, None] ** 2) * (1.0 + 1j * problem.eta) - w2[None, :]
    u_tip = np.abs(((phi_tip * qr)[:, None] / denom).sum(axis=0))   # |u_x| (m)
    return dict(frf_hz=freqs, frf_ux=u_tip)


# --------------------------------------------------------------------------- #
# 6. Plot: deformed mode shapes + FRF
# --------------------------------------------------------------------------- #
def _mode_kind(u: np.ndarray, pts: np.ndarray, problem: Cylinder) -> str:
    """Label a mode bending / torsion / axial from its cross-section motion.

    Grouping nodes into thin ``z``-slices, each slice's rigid motion splits into a
    transverse **translation** (bending), a net **twist** about the axis
    (torsion) and an axial **stretch** (axial).  A slice's *signed* twist cancels
    for a bending mode (odd across the section) but accumulates for torsion, so
    comparing the slice-summed magnitudes classifies the mode cleanly.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r2 = x * x + y * y
    zbin = np.round(z / problem.length * 200).astype(np.int64)

    def _slice_sum(v):                                      # sum |per-slice mean|
        order = np.argsort(zbin)
        _, first = np.unique(zbin[order], return_index=True)
        sums = np.add.reduceat(v[order], first)
        cnts = np.add.reduceat(np.ones_like(v[order]), first)
        return np.abs(sums / cnts).sum()

    bend = _slice_sum(u[:, 0]) + _slice_sum(u[:, 1])        # transverse translation
    axial = _slice_sum(u[:, 2])                             # z-stretch
    twist = _slice_sum((x * u[:, 1] - y * u[:, 0]) / (r2.mean() ** 0.5 + 1e-30))
    scores = {"bending": bend, "torsion": twist, "axial": axial}
    return max(scores, key=scores.get)


def plot_modes(res: Dict, problem: Cylinder, mesh: Mesh, save_path: str,
               show: Optional[list] = None) -> None:
    """Selected mode shapes as deformed 3D cylinders (colored by |u|).

    The 40:1-slender rod would vanish in a true-aspect 3D box, so each panel
    exaggerates the modal deflection (tip amplitude ~ 0.35 L) and stretches the
    transverse axes to match, with the undeformed skin drawn faintly for
    reference.  ``show`` is a list of 0-based mode indices; the default picks one
    of each kind (bending / torsion / axial) so the whole family is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = mesh.points.cpu().numpy()
    freqs, shapes = res["freqs"], res["shapes"]
    # cylinder skin nodes only -> a clean, light 3D point cloud.
    rr = np.hypot(pts[:, 0], pts[:, 1])
    skin = np.where((rr > problem.radius * 0.8)
                    | (pts[:, 2] < problem.radius * 0.15)
                    | (pts[:, 2] > problem.length - problem.radius * 0.15))[0]
    sp = pts[skin]

    if show is None:
        # one representative per kind, ascending, padded with the lowest bendings.
        kinds = [_mode_kind(shapes[i].reshape(-1, 3), pts, problem)
                 for i in range(len(freqs))]
        picks, seen = [], set()
        for i, k in enumerate(kinds):
            if k not in seen:
                picks.append(i); seen.add(k)
        for i in range(len(freqs)):
            if len(picks) >= 6:
                break
            if i not in picks:
                picks.append(i)
        show = sorted(picks[:6])

    amp = 0.30 * problem.length                            # target tip deflection
    R = problem.radius
    ncol = 3
    nrow = int(np.ceil(len(show) / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 3.4 * nrow))
    for panel, m in enumerate(show):
        u = shapes[m].reshape(-1, 3)
        umag = np.linalg.norm(u, axis=1)
        kind = _mode_kind(u, pts, problem)

        # Put the mode's dominant transverse component on the vertical plot axis
        # so every bending mode (whichever physical plane it lives in) reads as a
        # clean vertical deflection under one fixed camera; the other transverse
        # component becomes the (shallow) depth axis.
        vax = 0 if np.abs(u[:, 0]).sum() >= np.abs(u[:, 1]).sum() else 1
        dax = 1 - vax
        # torsion moves the rim, not the axis; scaling the rim to `amp` would make
        # it explode into a fan, so cap it to a few radii -> a visible twisted rod.
        target = 3.0 * R if kind == "torsion" else amp
        scale = target / (umag.max() + 1e-30)
        horiz, depth, vert = sp[:, 2], sp[:, dax], sp[:, vax]
        d_h = horiz + scale * u[skin, 2]
        d_d = depth + scale * u[skin, dax]
        d_v = vert + scale * u[skin, vax]

        # symmetric limits about each transverse range (base axis at 0 included)
        # so the deformed rod is centered in its panel; a generous floor keeps a
        # no-swing mode (axial) a legible rod rather than an edge-on sliver.
        floor = 0.10 * problem.length

        def _center(vals):
            lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
            mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), 0.5 * floor)
            return mid, half * 1.15

        cd, hd = _center(d_d)
        cv, hv = _center(d_v)

        ax = fig.add_subplot(nrow, ncol, panel + 1, projection="3d")
        # undeformed reference = the neutral axis (thin line, so it never occludes
        # the colored rod the way a solid grey tube does in matplotlib's 3D, which
        # has no reliable depth sorting between artists).
        ax.plot([0.0, problem.length], [0.0, 0.0], [0.0, 0.0],
                color="#999999", lw=1.6, ls="--", zorder=1)
        ax.scatter(d_h, d_d, d_v,
                   c=umag[skin] / umag.max(), cmap="turbo", s=11,
                   linewidths=0, zorder=2)
        ax.set_title(f"mode {m + 1}: {freqs[m]:.1f} Hz  ({kind})", fontsize=10)
        ax.set_xlabel("z (m)", fontsize=8)
        ax.set_xlim(0.0, problem.length)
        ax.set_ylim(cd - hd, cd + hd)
        ax.set_zlim(cv - hv, cv + hv)
        ax.set_box_aspect((problem.length, 2 * hd, 2 * hv))
        ax.set_xticks([0, problem.length])
        ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=18, azim=-70)
    fig.suptitle("Cantilever cylinder — natural mode shapes", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


def plot_frf(res: Dict, problem: Cylinder, save_path: str) -> None:
    """Tip transverse displacement amplitude ``|u_x|`` (m) vs frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.semilogy(res["frf_hz"], res["frf_ux"], color="#0072B2", lw=1.8, zorder=3)
    ax.set_xlim(problem.frf_f0, problem.frf_f1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"tip displacement amplitude  $|u_x|$  (m)")
    ax.set_title("Cantilever cylinder — drive-point displacement FRF")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 7. run_demo / main
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, modes_path: Optional[str] = None,
             frf_path: Optional[str] = None,
             problem: Optional[Cylinder] = None) -> Dict:
    """Solve modes + FRF, save figures, print a summary, return diagnostics."""
    torch.set_default_dtype(torch.float64)
    problem = problem or Cylinder()

    mesh = build_mesh(problem)
    K, M = assemble(problem, mesh)
    free, tip = dof_masks(problem, mesh)
    print(f"mesh: {mesh.points.shape[0]} nodes, "
          f"{mesh.cells['tetra'].shape[0]} tets; {len(tip)} tip nodes; "
          f"{K.shape[0] - len(free)} clamped DOFs", flush=True)

    modes = solve_modes(problem, K, M, free)
    frf = solve_frf(problem, modes, tip)
    res = dict(mesh=mesh, **modes, **frf)

    print("\nnatural frequencies (Hz):", flush=True)
    print(f"{'mode':>4} {'TensorMesh':>12}", flush=True)
    for i in range(problem.n_modes):
        print(f"{i + 1:>4} {modes['freqs'][i]:12.2f}", flush=True)
    # Euler-Bernoulli bending frequencies (each appears twice — two bend planes).
    print("Euler-Bernoulli bending (Hz): "
          + ", ".join(f"{f:.1f}" for f in problem.analytic_bending()), flush=True)

    if make_plot:
        plot_modes(res, problem, mesh,
                   modes_path or str(HERE / "vibration_cylinder_modes.png"))
        plot_frf(res, problem, frf_path or str(HERE / "vibration_cylinder_frf.png"))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-modes", type=int, default=12)
    parser.add_argument("--mesh-h-mm", type=float, default=None, help="tetra edge (mm)")
    parser.add_argument("--frf-points", type=int, default=400)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--modes-output", type=str, default=None)
    parser.add_argument("--frf-output", type=str, default=None)
    args = parser.parse_args()

    problem = Cylinder(n_modes=args.n_modes, frf_n=args.frf_points,
                       mesh_h=(args.mesh_h_mm * 1e-3) if args.mesh_h_mm else None)
    run_demo(make_plot=not args.no_plot, modes_path=args.modes_output,
             frf_path=args.frf_output, problem=problem)


if __name__ == "__main__":
    main()

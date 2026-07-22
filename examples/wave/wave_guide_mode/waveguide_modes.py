"""Rectangular dielectric waveguide mode analysis on TensorMesh — scalar Helmholtz.

The guided modes of a rectangular dielectric waveguide are found from its 2D
cross-section: a high-index rectangular core (width ``w`` x height ``h``, index
``n_core``) embedded in a lower-index cladding (``n_clad``).  In the
weakly-guiding (scalar) approximation the modal field ``E(x, y)`` and propagation
constant ``beta`` obey the transverse Helmholtz equation

    lap_t E + ( k0^2 n(x,y)^2 - beta^2 ) E = 0 ,      k0 = 2 pi / lam0 ,

a generalized eigenproblem for ``beta^2``.  Assembled with the builtin
:class:`~tensormesh.assemble.LaplaceElementAssembler` (stiffness ``K``),
:class:`~tensormesh.assemble.MassElementAssembler` (mass ``M``) and
:class:`~tensormesh.assemble.ScaledMassElementAssembler` (index-weighted mass
``M_eps = int n^2 phi_i phi_j``) it reads

    ( K - k0^2 M_eps ) E = -beta^2 M E .

The field is clamped to zero on the outer box (guided modes decay in the
cladding), so the eigenpairs come from a shift-invert Lanczos solve near the core
light line ``beta^2 = (k0 n_core)^2``.  A mode is **guided** when its effective
index ``n_eff = beta / k0`` lies between the cladding and core indices,
``n_clad < n_eff < n_core``.  Each is compared to the separable (Marcatili) slab
approximation.  No public API is added here.

Workflow:  mesh (gmsh, rectangular core in box) -> assemble (K, M, M_eps)
           -> generalized eig (beta^2) -> guided modes (n_eff + fields).

Run (env with torch + tensormesh + gmsh + scipy):
    python examples/wave/wave_guide_mode/waveguide_modes.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from tensormesh import Mesh
from tensormesh.assemble import (LaplaceElementAssembler, MassElementAssembler,
                                 ScaledMassElementAssembler)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Waveguide:
    r"""Rectangular step-index waveguide parameters (SI, meters).

    Fields
    ------
    core_w, core_h : core width and height (m).
    n_core, n_clad : core / cladding refractive indices.
    lam0 : vacuum wavelength (m).
    box_factor : outer (square) box half-width in units of ``max(w, h)``.
    n_modes : number of eigenpairs to request from the solver.
    mesh_h : target element edge length (m); default ``min(w, h) / 12``.
    """

    core_w: float = 18.0e-6
    core_h: float = 14.0e-6
    n_core: float = 1.450
    n_clad: float = 1.444
    lam0: float = 1.55e-6
    box_factor: float = 2.0
    n_modes: int = 12
    mesh_h: Optional[float] = None

    @property
    def k0(self) -> float:
        """Vacuum wavenumber ``2 pi / lam0`` (rad/m)."""
        return 2.0 * np.pi / self.lam0

    @property
    def NA(self) -> float:
        """Numerical aperture ``sqrt(n_core^2 - n_clad^2)``."""
        return float(np.sqrt(self.n_core ** 2 - self.n_clad ** 2))

    @property
    def box(self) -> float:
        """Outer box half-width (m)."""
        return self.box_factor * max(self.core_w, self.core_h)

    @property
    def Vx(self) -> float:
        """Normalized frequency across the width ``k0 (w/2) NA``."""
        return self.k0 * (self.core_w / 2.0) * self.NA

    @property
    def Vy(self) -> float:
        """Normalized frequency across the height ``k0 (h/2) NA``."""
        return self.k0 * (self.core_h / 2.0) * self.NA

    @property
    def h(self) -> float:
        """Resolved mesh edge length (m)."""
        return self.mesh_h or min(self.core_w, self.core_h) / 10.0


# --------------------------------------------------------------------------- #
# 1. Mesh: rectangular core centered in a square cladding box
# --------------------------------------------------------------------------- #
def build_mesh(problem: Waveguide, msh_path: Optional[str] = None) -> Mesh:
    """gmsh mesh of the square box with the core rectangle as an interior interface."""
    import gmsh

    if msh_path is None:
        msh_path = str(HERE / "_waveguide.msh")
    b, w, ht, hh = problem.box, problem.core_w, problem.core_h, problem.h
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        box = occ.addRectangle(-b, -b, 0, 2 * b, 2 * b)
        core = occ.addRectangle(-w / 2, -ht / 2, 0, w, ht)
        occ.fragment([(2, box)], [(2, core)])
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", hh)
        gmsh.option.setNumber("Mesh.MeshSizeMax", hh)
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()
    return Mesh.from_file(msh_path, reorder=False)


# --------------------------------------------------------------------------- #
# 2. Assemble K, M and the index-weighted mass M_eps = int n^2 phi phi
# --------------------------------------------------------------------------- #
def assemble(problem: Waveguide, mesh: Mesh):
    """Return CSC ``(A, B)`` with ``A = K - k0^2 M_eps``, ``B = M``, + free mask."""
    pts = mesh.points.to(torch.float64)
    xy = pts.cpu().numpy()

    K = LaplaceElementAssembler.from_mesh(mesh)(pts).to_scipy_coo().tocsc()
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=4)(pts).to_scipy_coo().tocsc()

    # nodal n^2:  n_core^2 inside the core rectangle, n_clad^2 in the cladding.
    in_core = (np.abs(xy[:, 0]) <= problem.core_w / 2) & \
              (np.abs(xy[:, 1]) <= problem.core_h / 2)
    n2 = np.where(in_core, problem.n_core ** 2, problem.n_clad ** 2)
    c = torch.as_tensor(n2, dtype=torch.float64)
    Meps = ScaledMassElementAssembler.from_mesh(mesh, quadrature_order=4)(
        pts, point_data={"c": c}).to_scipy_coo().tocsc()

    A = (K - problem.k0 ** 2 * Meps).tocsc()

    # clamp the outer box boundary (guided fields decay to zero there).
    tol = problem.h * 0.5
    on_box = (np.abs(np.abs(xy[:, 0]) - problem.box) < tol) | \
             (np.abs(np.abs(xy[:, 1]) - problem.box) < tol)
    free = np.where(~on_box)[0]
    return A, M, free


# --------------------------------------------------------------------------- #
# 3. Solve the generalized eigenproblem for the guided modes
# --------------------------------------------------------------------------- #
def solve_modes(problem: Waveguide, A, B, free: np.ndarray) -> Dict:
    """Guided modes: effective indices ``n_eff`` (desc) + full-length fields."""
    Aff = A[free][:, free]
    Bff = B[free][:, free]
    # (K - k0^2 M_eps) E = -beta^2 M E ; the most-guided modes sit just below the
    # core light line, so shift-invert near lambda = -(k0 n_core)^2.
    sigma = -(problem.k0 * problem.n_core) ** 2
    lam, vec = spla.eigsh(Aff, k=problem.n_modes, M=Bff, sigma=sigma, which="LM")

    beta2 = -lam
    n_eff = np.sqrt(np.clip(beta2, 0.0, None)) / problem.k0
    guided = (n_eff > problem.n_clad) & (n_eff < problem.n_core)

    order = np.argsort(n_eff[guided])[::-1]              # most-confined first
    idx = np.where(guided)[0][order]
    n = A.shape[0]
    fields = np.zeros((len(idx), n))
    for j, i in enumerate(idx):
        f = np.zeros(n)
        f[free] = vec[:, i]
        f /= np.abs(f).max()                             # normalize sign/amplitude
        fields[j] = f
    return dict(n_eff=n_eff[idx], fields=fields)


# --------------------------------------------------------------------------- #
# 4. Analytic reference: separable (Marcatili) symmetric-slab approximation
# --------------------------------------------------------------------------- #
def _slab_kx(half_width: float, k0: float, NA: float) -> List[float]:
    r"""Transverse wavenumbers ``kx`` of a symmetric slab of full width ``2 a``.

    Even / odd guided modes satisfy ``U tan U = W`` / ``-U cot U = W`` with
    ``U = kx a``, ``W = gamma a`` and ``U^2 + W^2 = (a k0 NA)^2``.
    """
    Vd = half_width * k0 * NA
    if Vd <= 0:
        return []
    grid = np.linspace(1e-6, Vd - 1e-9, 6000)

    def g(u):                                            # even (+) and odd (-) merged
        w = np.sqrt(max(Vd * Vd - u * u, 0.0))
        return u * np.tan(u) - w, u / np.tan(u) + w      # (even, odd) residuals

    kxs: List[float] = []
    for sel in (0, 1):                                   # even then odd branch
        vals = np.array([g(u)[sel] for u in grid])
        for i in range(len(grid) - 1):
            a0, a1 = vals[i], vals[i + 1]
            if np.isfinite(a0) and np.isfinite(a1) and a0 * a1 < 0 \
               and abs(a0) < 10 and abs(a1) < 10:        # skip tan/cot poles
                try:
                    u = brentq(lambda u: g(u)[sel], grid[i], grid[i + 1], maxiter=200)
                    kxs.append(u / half_width)
                except ValueError:
                    pass
    return sorted(kxs)


def analytic_slab(problem: Waveguide) -> List[dict]:
    """Guided ``E_{pq}`` modes from the separable slab product ``kx * ky``."""
    kx = _slab_kx(problem.core_w / 2, problem.k0, problem.NA)
    ky = _slab_kx(problem.core_h / 2, problem.k0, problem.NA)
    kco = problem.k0 * problem.n_core
    kcl = problem.k0 * problem.n_clad
    out: List[dict] = []
    for p, kxi in enumerate(kx, 1):
        for q, kyj in enumerate(ky, 1):
            beta2 = kco ** 2 - kxi ** 2 - kyj ** 2
            if beta2 > kcl ** 2:                         # still guided
                out.append(dict(p=p, q=q, n_eff=np.sqrt(beta2) / problem.k0))
    return sorted(out, key=lambda d: -d["n_eff"])


# --------------------------------------------------------------------------- #
# 5. Plot the guided mode-field gallery
# --------------------------------------------------------------------------- #
def plot_modes(res: Dict, problem: Waveguide, mesh: Mesh, save_path: str,
               labels: Optional[List[str]] = None, n_show: int = 6) -> None:
    """Gallery of the guided transverse mode fields ``E(x, y)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.tri import Triangulation

    pts = mesh.points.cpu().numpy()
    tris = mesh.cells["triangle"].cpu().numpy()[:, :3]
    tri = Triangulation(pts[:, 0] * 1e6, pts[:, 1] * 1e6, triangles=tris)

    n_show = min(n_show, len(res["fields"]))
    ncol = 3
    nrow = int(np.ceil(n_show / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.5 * nrow),
                             squeeze=False)
    w_um, h_um = problem.core_w * 1e6, problem.core_h * 1e6
    lim = 1.6 * max(w_um, h_um) / 2
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n_show:
            ax.axis("off"); continue
        e = res["fields"][k]
        ax.tripcolor(tri, e, shading="gouraud", cmap="RdBu_r",
                     vmin=-1, vmax=1, rasterized=True)
        ax.add_patch(Rectangle((-w_um / 2, -h_um / 2), w_um, h_um,
                               fill=False, ec="k", lw=0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        tag = f"${labels[k]}$" if labels and k < len(labels) and labels[k] else ""
        ax.set_title(f"mode {k + 1}:  {tag}\n$n_{{eff}}$ = {res['n_eff'][k]:.5f}",
                     fontsize=10)
        ax.set_xlabel("x (µm)", fontsize=8); ax.set_ylabel("y (µm)", fontsize=8)
    fig.suptitle("Rectangular waveguide modes", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 6. run_demo / main
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, field_path: Optional[str] = None,
             problem: Optional[Waveguide] = None) -> Dict:
    """Solve the guided modes, save the field gallery, print a summary."""
    torch.set_default_dtype(torch.float64)
    problem = problem or Waveguide()

    mesh = build_mesh(problem)
    A, B, free = assemble(problem, mesh)
    print(f"mesh: {mesh.n_points} nodes, {mesh.n_elements} triangles; "
          f"Vx = {problem.Vx:.3f}, Vy = {problem.Vy:.3f}", flush=True)

    modes = solve_modes(problem, A, B, free)
    res = dict(mesh=mesh, **modes)
    ana = analytic_slab(problem)

    print(f"\nguided modes: {len(modes['n_eff'])} FEM  /  "
          f"{len(ana)} analytic (Marcatili)", flush=True)
    print(f"{'#':>3} {'n_eff (FEM)':>12} {'Marcatili':>11}  mode", flush=True)
    for i, ne in enumerate(modes["n_eff"]):
        tag = f"{ana[i]['n_eff']:11.5f}  E{ana[i]['p']}{ana[i]['q']}" \
            if i < len(ana) else ""
        print(f"{i + 1:>3} {ne:12.5f} {tag}", flush=True)

    labels = [f"E_{{{d['p']}{d['q']}}}" for d in ana]
    if make_plot:
        plot_modes(res, problem, mesh,
                   field_path or str(HERE / "waveguide_modes.png"), labels=labels)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-w-um", type=float, default=18.0, help="core width (µm)")
    parser.add_argument("--core-h-um", type=float, default=14.0, help="core height (µm)")
    parser.add_argument("--n-core", type=float, default=1.450)
    parser.add_argument("--n-clad", type=float, default=1.444)
    parser.add_argument("--lam0-nm", type=float, default=1550.0)
    parser.add_argument("--n-modes", type=int, default=12)
    parser.add_argument("--mesh-h-um", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    problem = Waveguide(core_w=args.core_w_um * 1e-6, core_h=args.core_h_um * 1e-6,
                        n_core=args.n_core, n_clad=args.n_clad,
                        lam0=args.lam0_nm * 1e-9, n_modes=args.n_modes,
                        mesh_h=(args.mesh_h_um * 1e-6) if args.mesh_h_um else None)
    run_demo(make_plot=not args.no_plot, field_path=args.output, problem=problem)


if __name__ == "__main__":
    main()

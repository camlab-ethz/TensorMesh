"""2D photonic waveguide + microdisk coupler on TensorMesh — TM / E_z.

A silicon bus waveguide evanescently couples light into a silicon microdisk
(both n = 3.48) embedded in silica (n = 1.44); a stretched-coordinate PML frame
makes the domain open.  In TM polarization the out-of-plane field E_z obeys the
scalar Helmholtz equation

    div( Lambda grad E_z ) + k0^2 eps_r sx sy E_z = source ,

where the PML gives the complex diagonal tensor ``Lambda = diag(sy/sx, sx/sy)``
and mass scaling ``sx sy`` (both 1 outside the PML).  This is a plain scalar
Helmholtz with a spatially-varying complex coefficient, assembled natively by a
custom :class:`~tensormesh.ElementAssembler` whose ``forward`` reads the
quadrature coordinate and evaluates ``eps_r`` (Si in the waveguide + disk, SiO2
elsewhere) and the PML stretch on the fly.  A line-current source in the
waveguide launches the guided mode; on resonance it feeds a whispering-gallery
mode (WGM) of the disk — a bright ring of azimuthal lobes on the rim.

TensorMesh now ships a native PML (``tensormesh.cartesian_pml`` +
``AnisotropicLaplaceElementAssembler`` + ``ScaledMassElementAssembler``); this
example keeps an equivalent inline assembler that evaluates ``eps_r`` and the
stretch *per quadrature point*, which resolves the sharp Si/SiO2 interfaces a
touch better than nodal coefficients.  No public API is added here.

The disk WGM is a sharp resonance, so the drive wavelength must sit on it: with
linear (P1) elements the physical ~1.55 um resonance blue-shifts to ~1.512 um
(numerical dispersion), which is the default ``lam0``.  Off resonance the disk
fills with a messy radial mix instead of a clean rim mode; refining the mesh or
using ``mesh_order=2`` moves the numerical resonance back toward 1.55 um.

Workflow:  mesh (gmsh, conforming) -> assemble (K_PML - k0^2 M) -> complex solve
           -> E_z field (guided mode + disk WGM).

Run (env with torch + tensormesh + gmsh):
    python examples/wave/optical_ring_resonator/optical_ring_resonator.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from tensormesh import ElementAssembler, MassElementAssembler, Mesh
from tensormesh.sparse.matrix import SparseMatrix

HERE = Path(__file__).resolve().parent
C0 = 299792458.0


# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Coupler:
    r"""All parameters of the waveguide + microdisk coupler (SI, meters).

    Fields
    ------
    domain, pml : full domain size and PML-frame thickness (m).
    wg_width, wg_x : bus-waveguide width and center x (m).
    disk_r, disk_x, disk_y : microdisk radius and center (m).
    src_r, src_y : source-disk radius and y-position on the waveguide (m).
    n_core, n_clad : Si and SiO2 refractive indices.
    lam0 : drive wavelength (m).  Default 1.512 um — the disk's whispering-gallery
        resonance for the default mesh.  The ~1.55 um physical resonance
        blue-shifts numerically with linear (P1) elements; drive at the numerical
        resonance (or refine / use ``mesh_order=2``) to see a clean rim mode.
    pml_strength : imaginary coordinate-stretch amplitude of the PML.
    mesh_h : target element edge length (m); default lam0/n_core/12.
    mesh_order : 1 (linear) or 2 (quadratic ``triangle6``) elements.
    """

    domain: float = 6.0e-6
    pml: float = 0.6e-6
    wg_width: float = 0.25e-6
    wg_x: float = 1.6e-6
    disk_r: float = 1.0e-6
    disk_x: float = 2.75e-6
    disk_y: float = 3.0e-6
    src_r: float = 0.06e-6
    src_y: float = 1.2e-6
    n_core: float = 3.48
    n_clad: float = 1.44
    lam0: float = 1.512e-6          # disk WGM resonance for the default P1 mesh
    pml_strength: float = 10.0
    mesh_h: Optional[float] = None
    mesh_order: int = 1

    @property
    def k0(self) -> float:
        """Vacuum wavenumber ``2 pi / lam0`` (rad/m)."""
        return 2.0 * np.pi / self.lam0

    @property
    def freq(self) -> float:
        """Frequency ``c0 / lam0`` (Hz)."""
        return C0 / self.lam0

    @property
    def h(self) -> float:
        """Resolved mesh edge length (m)."""
        return self.mesh_h or self.lam0 / self.n_core / 12.0


# --------------------------------------------------------------------------- #
# Custom TM-Maxwell (scalar Helmholtz) assembler with an on-the-fly PML
# --------------------------------------------------------------------------- #
class MaxwellTMAssembler(ElementAssembler):
    r"""Integrand ``(Lambda grad u).grad v - k0^2 eps_r sx sy u v`` per point.

    ``eps_r`` and the PML stretch are evaluated from the quadrature coordinate
    ``x``, so material interfaces and the PML profile are resolved at quadrature
    level.  Geometry / PML parameters are stashed on the instance by
    :meth:`__post_init__`.
    """

    def __post_init__(self, geo: dict):
        for k, v in geo.items():
            setattr(self, k, v)

    def _stretch(self, t):
        """Complex SC-PML stretch ``s(t)``: 1 inside, ``1 - i*strength*xi^2`` in the PML."""
        d, L = self.pml, self.domain
        xi = torch.clamp((d - t) / d, min=0.0) + torch.clamp((t - (L - d)) / d, min=0.0)
        return 1.0 - 1j * self.pml_strength * xi * xi

    def forward(self, gradu, gradv, u, v, x):
        xx, yy = x[0].real, x[1].real
        in_core = (torch.abs(xx - self.wg_x) <= self.wg_width / 2) | \
                  ((xx - self.disk_x) ** 2 + (yy - self.disk_y) ** 2 <= self.disk_r ** 2)
        eps = torch.where(in_core,
                          torch.as_tensor(self.eps_core, dtype=torch.complex128),
                          torch.as_tensor(self.eps_clad, dtype=torch.complex128))
        sx, sy = self._stretch(xx), self._stretch(yy)
        z = torch.zeros_like(sx)
        lam = torch.stack([torch.stack([sy / sx, z]), torch.stack([z, sx / sy])])
        eps_s = (self.k0 ** 2) * eps * sx * sy
        return gradu.to(torch.complex128) @ lam @ gradv.to(torch.complex128) - eps_s * u * v


# --------------------------------------------------------------------------- #
# 1. Mesh: square domain conforming to the waveguide / disk interfaces
# --------------------------------------------------------------------------- #
def build_mesh(problem: Coupler, msh_path: Optional[str] = None) -> Mesh:
    """gmsh mesh conforming to the waveguide/disk/source interfaces."""
    import gmsh

    if msh_path is None:
        msh_path = str(HERE / "_coupler.msh")
    L, h = problem.domain, problem.h
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        sq = occ.addRectangle(0, 0, 0, L, L)
        wg = occ.addRectangle(problem.wg_x - problem.wg_width / 2, 0, 0, problem.wg_width, L)
        dk = occ.addDisk(problem.disk_x, problem.disk_y, 0, problem.disk_r, problem.disk_r)
        sc = occ.addDisk(problem.wg_x, problem.src_y, 0, problem.src_r, problem.src_r)
        occ.fragment([(2, sq)], [(2, wg), (2, dk), (2, sc)])
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)
        if problem.mesh_order >= 2:
            gmsh.model.mesh.setOrder(problem.mesh_order)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()
    return Mesh.from_file(msh_path, reorder=False)


# --------------------------------------------------------------------------- #
# 2. Assemble the PML Helmholtz operator, add the source, solve
# --------------------------------------------------------------------------- #
def solve(problem: Coupler, verbose: bool = True) -> Dict:
    """Solve the TM Helmholtz coupler; return the complex field ``E_z``."""
    torch.set_default_dtype(torch.float64)
    mesh = build_mesh(problem)
    pts = mesh.points.to(torch.float64)
    geo = dict(k0=problem.k0, eps_core=problem.n_core ** 2, eps_clad=problem.n_clad ** 2,
               wg_x=problem.wg_x, wg_width=problem.wg_width, disk_x=problem.disk_x,
               disk_y=problem.disk_y, disk_r=problem.disk_r, domain=problem.domain,
               pml=problem.pml, pml_strength=problem.pml_strength)

    asm = MaxwellTMAssembler.from_mesh(mesh, quadrature_order=4, geo=geo)
    asm.type(torch.float64)
    H = asm(points=pts)

    # source = uniform current over the source disk, applied as the consistent
    # FEM load  rhs_i = int_srcdisk phi_i  (= M @ indicator).
    Masm = MassElementAssembler.from_mesh(mesh, quadrature_order=4)
    Msp = Masm(pts)
    src = (((pts[:, 0] - problem.wg_x) ** 2 + (pts[:, 1] - problem.src_y) ** 2)
           < problem.src_r ** 2).to(torch.complex128)
    rhs = Msp @ src

    if verbose:
        print(f"mesh: {mesh.n_points} nodes, {mesh.n_elements} triangles; "
              f"{int(src.real.sum())} source nodes; lam0 = {problem.lam0*1e9:.0f} nm",
              flush=True)

    ez = SparseMatrix(H.values, H.row, H.col, H.shape).solve(rhs)
    return dict(mesh=mesh, points=pts.cpu().numpy(), Ez=ez.cpu().numpy())


# --------------------------------------------------------------------------- #
# 3. Plot the field
# --------------------------------------------------------------------------- #
def plot_field(res: Dict, problem: Coupler, save_path: str) -> None:
    """Draw ``Re(E_z)`` and ``|E_z|`` of the coupler (guided mode + disk WGM)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    pts = res["points"]
    cells = res["mesh"].cells
    key = "triangle6" if "triangle6" in list(cells.keys()) else "triangle"
    tris = cells[key].cpu().numpy()[:, :3]
    tri = Triangulation(pts[:, 0] * 1e6, pts[:, 1] * 1e6, triangles=tris)
    ez = res["Ez"]
    ez = ez / np.abs(ez).max()                          # normalize (source strength arbitrary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [("Re($E_z$)", ez.real, "RdBu_r", dict(vmin=-1, vmax=1)),
              ("|$E_z$|", np.abs(ez), "inferno", dict(vmin=0, vmax=0.5))]
    for ax, (title, data, cmap, norm) in zip(axes, panels):
        tpc = ax.tripcolor(tri, data, shading="gouraud", cmap=cmap, rasterized=True, **norm)
        ax.set_aspect("equal")
        ax.set_title(f"{title}   @ {problem.lam0*1e9:.0f} nm")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        cb = plt.colorbar(tpc, ax=ax, shrink=0.88)
        cb.ax.set_title("$E_z$\n(norm.)", fontsize=9)
    fig.suptitle("Waveguide-coupled silicon microdisk", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 4. run_demo / main
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, field_path: Optional[str] = None,
             problem: Optional[Coupler] = None) -> Dict:
    """Solve the coupler, save the field figure, return diagnostics."""
    problem = problem or Coupler()
    res = solve(problem)
    ez = res["Ez"]
    disk = (((res["points"][:, 0] - problem.disk_x) ** 2
             + (res["points"][:, 1] - problem.disk_y) ** 2) < problem.disk_r ** 2)
    print(f"|E_z| max {np.abs(ez).max():.2e}; mean |E_z| in disk / whole = "
          f"{np.abs(ez[disk]).mean() / np.abs(ez).mean():.2f} (coupling into the WGM)",
          flush=True)
    if make_plot:
        plot_field(res, problem, field_path or str(HERE / "optical_ring_resonator.png"))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam0-nm", type=float, default=1512.0,
                        help="drive wavelength (nm); default = disk WGM resonance")
    parser.add_argument("--mesh-h-nm", type=float, default=None, help="mesh edge (nm)")
    parser.add_argument("--order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--pml-strength", type=float, default=10.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    problem = Coupler(lam0=args.lam0_nm * 1e-9,
                      mesh_h=(args.mesh_h_nm * 1e-9) if args.mesh_h_nm else None,
                      mesh_order=args.order, pml_strength=args.pml_strength)
    run_demo(make_plot=not args.no_plot, field_path=args.output, problem=problem)


if __name__ == "__main__":
    main()

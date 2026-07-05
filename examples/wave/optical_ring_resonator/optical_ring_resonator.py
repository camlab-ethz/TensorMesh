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
elsewhere) and the PML stretch on the fly.  The waveguide is fed by a **modal
soft source**: the analytic fundamental transverse profile of the bus slab is
imprinted on a thin launch plane and phased as a two-plane directional launch,
so it injects the guided mode travelling toward the disk (not an isotropic
current blob).  On resonance it feeds a whispering-gallery mode (WGM) of the
disk — a bright ring of azimuthal lobes on the rim.

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
    src_y : y of the modal launch plane on the waveguide (m).
    launch_sigma : longitudinal 1/e half-width of the soft-source window (m);
        defaults to ``1.5 * h``.
    directional : if True, use a two-plane launch (a second plane a quarter
        guided-wavelength ahead, phased by +i) so the mode radiates toward the
        disk (+y) and the backward wave cancels; if False, a single symmetric
        plane launches both ways.
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
    src_y: float = 0.6e-6           # launch at the PML/cladding interface (= pml)
    launch_sigma: Optional[float] = None
    directional: bool = True
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

    @property
    def sigma(self) -> float:
        """Resolved longitudinal 1/e half-width of the launch window (m)."""
        return self.launch_sigma or 1.5 * self.h


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
        occ.fragment([(2, sq)], [(2, wg), (2, dk)])
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
# 1b. Modal soft source: fundamental transverse mode of the bus waveguide
# --------------------------------------------------------------------------- #
def slab_mode(problem: Coupler) -> Tuple[float, float, float]:
    r"""Fundamental even :math:`E_z` mode of the bus waveguide (symmetric slab).

    Solves the even-mode dispersion :math:`k_x \tan(k_x w/2) = \gamma` for the
    guided effective index :math:`n_\mathrm{eff}\in(n_\mathrm{clad},n_\mathrm{core})`,
    with transverse core wavenumber :math:`k_x = k_0\sqrt{n_\mathrm{core}^2-n_\mathrm{eff}^2}`
    and cladding decay :math:`\gamma = k_0\sqrt{n_\mathrm{eff}^2-n_\mathrm{clad}^2}`.
    Returns ``(n_eff, kx, gamma)`` (``kx, gamma`` in rad/m).
    """
    import math

    k0, w = problem.k0, problem.wg_width
    n1, n2 = problem.n_core, problem.n_clad

    def f(neff: float) -> float:
        kx = k0 * math.sqrt(max(n1 * n1 - neff * neff, 0.0))
        ga = k0 * math.sqrt(max(neff * neff - n2 * n2, 0.0))
        # even-mode condition times cos(kx w/2): avoids the tan asymptote.
        return kx * math.sin(kx * w / 2) - ga * math.cos(kx * w / 2)

    # Scan from just below n_core downward for the first (fundamental) root.
    lo, hi, steps = n2 + 1e-9, n1 - 1e-9, 4000
    prev_n, prev_f, root = hi, f(hi), None
    for i in range(1, steps + 1):
        n = hi - (hi - lo) * i / steps
        fn = f(n)
        if prev_f * fn <= 0.0:
            a, fa, b = n, fn, prev_n
            for _ in range(100):                      # bisection
                m = 0.5 * (a + b)
                if fa * f(m) <= 0.0:
                    b = m
                else:
                    a, fa = m, f(m)
            root = 0.5 * (a + b)
            break
        prev_n, prev_f = n, fn
    if root is None:
        raise RuntimeError("no guided even mode found for the bus waveguide")

    kx = k0 * math.sqrt(n1 * n1 - root * root)
    ga = k0 * math.sqrt(root * root - n2 * n2)
    return root, kx, ga


def launch_field(problem: Coupler, pts: torch.Tensor,
                 neff: float, kx: float, ga: float) -> torch.Tensor:
    r"""Complex nodal soft source imprinting the fundamental bus mode.

    The transverse profile :math:`\psi(x)` (cosine in the core, evanescent tails
    in the cladding) is imprinted on a thin Gaussian window in ``y`` at the
    launch plane ``src_y``.  When ``directional`` is set, a second window a
    quarter guided-wavelength ahead and phased by ``+i`` makes the ``+y``
    (toward-disk) wave add and the backward wave cancel.  The consistent FEM
    load is then ``rhs = M @ launch_field``.
    """
    import math

    x = pts[:, 0] - problem.wg_x
    y = pts[:, 1]
    w, sig = problem.wg_width, problem.sigma
    ax = torch.abs(x)
    psi = torch.where(ax <= w / 2,
                      torch.cos(kx * x),
                      math.cos(kx * w / 2) * torch.exp(-ga * (ax - w / 2)))

    def band(yc: float) -> torch.Tensor:
        return torch.exp(-((y - yc) / sig) ** 2)

    lam_g = 2.0 * math.pi / (problem.k0 * neff)        # guided wavelength
    win = band(problem.src_y).to(torch.complex128)
    if problem.directional:
        # second plane a quarter guided-wavelength ahead, phased by -i so the
        # outgoing wave (e^{-i beta y} for this solver's convention) adds in +y
        # (toward the disk) and cancels in -y.
        win = win - 1j * band(problem.src_y + lam_g / 4.0)
    return psi.to(torch.complex128) * win


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

    # Modal soft source: imprint the fundamental bus-waveguide mode on a thin
    # launch plane and apply it as the consistent FEM load  rhs = M @ psi.
    neff, kx, ga = slab_mode(problem)
    Masm = MassElementAssembler.from_mesh(mesh, quadrature_order=4)
    Msp = Masm(pts)
    src = launch_field(problem, pts, neff, kx, ga)
    rhs = Msp @ src

    if verbose:
        print(f"mesh: {mesh.n_points} nodes, {mesh.n_elements} triangles; "
              f"bus mode n_eff = {neff:.4f} ({'directional' if problem.directional else 'symmetric'} "
              f"launch); lam0 = {problem.lam0*1e9:.0f} nm", flush=True)

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
# 3b. Plot the setup: materials, PML frame, source, boundary conditions
# --------------------------------------------------------------------------- #
def plot_setup(problem: Coupler, save_path: str) -> None:
    """Draw the physical setup: materials (Si / SiO2), the PML frame, the
    line-current launch, and the applied boundary conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.lines import Line2D

    # Wong colorblind-safe qualitative palette.
    c_clad = "#56B4E9"      # SiO2 cladding (sky blue)
    c_core = "#E69F00"      # Si core (orange)
    c_pml = "#999999"       # PML frame (grey)
    c_src = "#D55E00"       # source (vermillion)

    L, d = problem.domain * 1e6, problem.pml * 1e6
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # SiO2 cladding fills the whole domain.
    ax.add_patch(Rectangle((0, 0), L, L, facecolor=c_clad, edgecolor="none",
                           alpha=0.35, zorder=0))
    # PML frame: solid grey border of thickness `pml` on all four sides.
    for xy, w, h in [((0, 0), L, d), ((0, L - d), L, d),
                     ((0, d), d, L - 2 * d), ((L - d, d), d, L - 2 * d)]:
        ax.add_patch(Rectangle(xy, w, h, facecolor=c_pml, edgecolor="none",
                               alpha=0.30, zorder=1))

    # Si core: bus waveguide (full-height strip) + microdisk, with a thin edge.
    core_edge = "#8a6100"
    wg_x0 = (problem.wg_x - problem.wg_width / 2) * 1e6
    ax.add_patch(Rectangle((wg_x0, 0), problem.wg_width * 1e6, L,
                           facecolor=c_core, edgecolor=core_edge, linewidth=0.8,
                           alpha=0.95, zorder=2))
    ax.add_patch(Circle((problem.disk_x * 1e6, problem.disk_y * 1e6),
                        problem.disk_r * 1e6, facecolor=c_core, edgecolor=core_edge,
                        linewidth=0.8, alpha=0.95, zorder=2))

    # Modal launch plane (thin line across the waveguide) + toward-disk arrow.
    xc, ys = problem.wg_x * 1e6, problem.src_y * 1e6
    hw = problem.wg_width * 1e6 / 2       # span exactly the waveguide width
    ax.plot([xc - hw, xc + hw], [ys, ys], color=c_src, linewidth=2.4, zorder=4)
    if problem.directional:
        ax.annotate("", xy=(xc, ys + 0.55), xytext=(xc, ys),
                    arrowprops=dict(arrowstyle="-|>", color=c_src, linewidth=2.0),
                    zorder=5)

    # Labels.
    ax.annotate("mode\nlaunch", (xc + hw + 0.15, ys), ha="left", va="center",
                fontsize=8, color=c_src, zorder=5)

    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("Simulation setup: materials & PML")

    legend = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c_core,
                     markersize=11, label="Si core ($n=%.2f$)" % problem.n_core),
              Line2D([0], [0], marker="s", color="none", markerfacecolor=c_clad,
                     markersize=11, alpha=0.5, label="SiO$_2$ ($n=%.2f$)" % problem.n_clad),
              Line2D([0], [0], marker="s", color="none", markerfacecolor=c_pml,
                     markersize=11, alpha=0.5, label="PML (radiating BC)"),
              Line2D([0], [0], color=c_src, linewidth=2.4, label="modal soft source")]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9,
              labelspacing=1.0, handletextpad=0.9, borderpad=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# 4. run_demo / main
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, field_path: Optional[str] = None,
             setup_path: Optional[str] = None,
             problem: Optional[Coupler] = None) -> Dict:
    """Solve the coupler, save the setup + field figures, return diagnostics."""
    problem = problem or Coupler()
    if make_plot:
        plot_setup(problem, setup_path or str(HERE / "optical_ring_resonator_setup.png"))
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
    parser.add_argument("--output", type=str, default=None, help="field figure path")
    parser.add_argument("--setup-output", type=str, default=None,
                        help="setup (materials/BC) figure path")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    problem = Coupler(lam0=args.lam0_nm * 1e-9,
                      mesh_h=(args.mesh_h_nm * 1e-9) if args.mesh_h_nm else None,
                      mesh_order=args.order, pml_strength=args.pml_strength)
    run_demo(make_plot=not args.no_plot, field_path=args.output,
             setup_path=args.setup_output, problem=problem)


if __name__ == "__main__":
    main()

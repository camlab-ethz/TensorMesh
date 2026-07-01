"""2D acoustic band structure of a triangular-lattice phononic crystal.

Demonstrates the Bloch-Floquet periodic boundary condition
(:class:`tensormesh.BlochReducer`) on a **non-orthogonal** lattice and a
**two-medium** (spatially varying material) scalar Helmholtz eigenproblem: a
triangular lattice of penetrable steel cylinders in water. Both media are
acoustic, coupled by pressure/velocity continuity at the interface.

It is deliberately example-only: no public API is added. The workflow is

    mesh -> per-element-weighted assembler -> BlochReducer -> generalized eig

with material-dependent operators
``K_ij = int (1/rho) grad phi_i . grad phi_j`` and
``M_ij = int (1/(rho c^2)) phi_i phi_j``, giving ``K p = omega^2 M p`` reduced
per wavevector to the master DOFs, then ``f = sqrt(omega^2) / (2 pi)`` along
M-Gamma-K-M.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse.linalg as sla
import torch

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh import ElementAssembler, BlochReducer  # noqa: E402  -- after shim

torch.set_default_dtype(torch.float64)

PATH_TICKS = [0, 1, 2, 3]
PATH_LABELS = ["M", r"$\Gamma$", "K", "M"]


@dataclass(frozen=True)
class TriangularCrystal:
    """Parameters for the triangular-lattice band-structure example.

    Parameters
    ----------
    a:
        Lattice constant ``|a1| = |a2|``, in metres.
    r_over_a:
        Steel-cylinder radius as a fraction of ``a`` (penetrable scatterer).
    c_water, rho_water:
        Matrix-fluid sound speed (m/s) and density (kg/m^3).
    c_steel, rho_steel:
        Scatterer sound speed (m/s) and density (kg/m^3).
    n_bands:
        Number of bands to compute.
    n_seg:
        k-points per Brillouin-zone segment (markers in the plot).
    chara_per_a:
        Mesh elements per lattice constant (mesh density).
    """

    a: float = 0.01
    r_over_a: float = 0.4
    c_water: float = 1481.44
    rho_water: float = 998.202
    c_steel: float = 5800.0
    rho_steel: float = 7850.0
    n_bands: int = 12
    n_seg: int = 12
    chara_per_a: float = 45.0

    @property
    def r(self) -> float:
        return self.r_over_a * self.a

    @property
    def h(self) -> float:
        return self.a / self.chara_per_a

    @property
    def a1(self) -> np.ndarray:
        return np.array([self.a / 2.0, -self.a * math.sqrt(3) / 2.0])

    @property
    def a2(self) -> np.ndarray:
        return np.array([self.a / 2.0, self.a * math.sqrt(3) / 2.0])


# --------------------------------------------------------------------------- #
# Per-element-weighted operators (two acoustic media)
# --------------------------------------------------------------------------- #
class _WeightedStiffness(ElementAssembler):
    r"""Stiffness with a per-element weight: :math:`\int \tfrac1\rho \nabla u \cdot \nabla v`.

    ``inv_rho`` is an ``element_data`` coefficient (one value per element), so the
    steel disk and the water matrix contribute different :math:`1/\rho`.
    """

    def forward(self, gradu, gradv, inv_rho):
        return inv_rho * (gradu @ gradv)


class _WeightedMass(ElementAssembler):
    r"""Mass with a per-element weight: :math:`\int \tfrac1{\rho c^2} u v`.

    ``inv_kappa`` (= :math:`1/(\rho c^2)`) is an ``element_data`` coefficient.
    """

    def forward(self, u, v, inv_kappa):
        return inv_kappa * (u * v)


# --------------------------------------------------------------------------- #
# Conformal periodic mesh of the rhombic cell (water matrix + steel disk)
# --------------------------------------------------------------------------- #
def build_mesh(problem: TriangularCrystal):
    """Mesh both domains conformally (gmsh ``fragment``) with edge-periodic nodes."""
    import gmsh

    a1, a2, r = problem.a1, problem.a2, problem.r
    a_lat = problem.a
    msh = os.path.join(os.path.dirname(__file__), "_cell.msh")
    mids = {"bottom": a1 / 2.0, "right": a1 + a2 / 2.0,
            "top": a2 + a1 / 2.0, "left": a2 / 2.0}

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("triangular_cell")
        occ = gmsh.model.occ
        p0 = occ.addPoint(0, 0, 0); p1 = occ.addPoint(a1[0], a1[1], 0)
        p2 = occ.addPoint(a1[0] + a2[0], a1[1] + a2[1], 0)
        p3 = occ.addPoint(a2[0], a2[1], 0)
        loop = occ.addCurveLoop([occ.addLine(p0, p1), occ.addLine(p1, p2),
                                 occ.addLine(p2, p3), occ.addLine(p3, p0)])
        surf = occ.addPlaneSurface([loop])
        cx, cy = (a1 + a2) / 2.0
        disk = occ.addDisk(cx, cy, 0.0, r, r)
        occ.fragment([(2, surf)], [(2, disk)])     # keep BOTH domains, share interface
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], 1)

        edges, tol = {}, 0.02 * a_lat
        for _, tag in gmsh.model.getEntities(1):
            com = np.array(gmsh.model.occ.getCenterOfMass(1, tag)[:2])
            for name, mid in mids.items():
                if np.linalg.norm(com - mid) < tol:
                    edges[name] = tag

        def tr(v):
            return [1, 0, 0, v[0], 0, 1, 0, v[1], 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, [edges["top"]], [edges["bottom"]], tr(a2))
        gmsh.model.mesh.setPeriodic(1, [edges["right"]], [edges["left"]], tr(a1))

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


def assemble(mesh, problem: TriangularCrystal):
    """Two-medium K, M; element material chosen by centroid (inside disk = steel)."""
    pts = mesh.points.cpu().numpy()
    cen = pts[mesh.elements().cpu().numpy()].mean(axis=1)         # element centroids
    inside = np.linalg.norm(cen - (problem.a1 + problem.a2) / 2.0, axis=1) < problem.r
    inv_rho = torch.tensor(np.where(inside, 1 / problem.rho_steel, 1 / problem.rho_water))
    inv_kappa = torch.tensor(np.where(
        inside, 1 / (problem.rho_steel * problem.c_steel ** 2),
        1 / (problem.rho_water * problem.c_water ** 2)))
    K = _WeightedStiffness.from_mesh(mesh, quadrature_order=2)(
        mesh.points, element_data={"inv_rho": inv_rho})
    M = _WeightedMass.from_mesh(mesh, quadrature_order=2)(
        mesh.points, element_data={"inv_kappa": inv_kappa})
    return K, M, int(inside.sum())


# --------------------------------------------------------------------------- #
# Hexagonal IBZ path M-Gamma-K-M and the generalized eig
# --------------------------------------------------------------------------- #
def k_path(problem: TriangularCrystal) -> Tuple[np.ndarray, np.ndarray]:
    """M-Gamma-K-M over the hexagonal Brillouin zone, parameter m in [0, 3]."""
    A = np.array([[problem.a1[0], problem.a2[0]], [problem.a1[1], problem.a2[1]]])
    B = 2 * math.pi * np.linalg.inv(A)            # rows b1, b2 (a_i . b_j = 2 pi)
    b1, b2 = B[0], B[1]
    M, K, G = 0.5 * b2, (b1 + b2) / 3.0, np.zeros(2)
    segs = [(M, G), (G, K), (K, M)]
    ms = np.linspace(0.0, 3.0, 3 * problem.n_seg + 1)
    ks = [(1 - (m - s)) * segs[s][0] + (m - s) * segs[s][1]
          for m in ms for s in [min(int(m), 2)]]
    return np.array(ks), ms


def generalized_eigh(Kr, Mr, n_bands: int) -> np.ndarray:
    """Lowest ``n_bands`` eigenvalues of the complex-Hermitian generalized
    problem ``K_r u = omega^2 M_r u``, via shift-inverted ARPACK
    (:func:`scipy.sparse.linalg.eigsh`) on the sparse reduced operators.

    The reduced ``K_r`` is singular at ``Gamma`` (the lowest band tends to 0),
    so the shift ``sigma`` is set just below the spectrum
    (``-1e-4 * trace(K_r)/trace(M_r)``, the ``omega^2`` scale) to regularize the
    null space while still returning the lowest modes.
    """
    Ks = Kr.to_scipy_coo().tocsc()
    Ms = Mr.to_scipy_coo().tocsc()
    sigma = -1.0e-4 * float(abs(Ks.diagonal()).sum() / abs(Ms.diagonal()).sum())
    w2 = sla.eigsh(Ks, k=n_bands, M=Ms, sigma=sigma, which="LM",
                   return_eigenvectors=False)
    return np.sort(w2.real)


def _sweep_bands(K, M, bloch: BlochReducer, ks: np.ndarray,
                 n_bands: int) -> np.ndarray:
    """Reduce ``(K, M)`` with the Bloch phase at every ``k`` and return the
    lowest ``n_bands`` frequencies ``f = sqrt(omega^2) / (2 pi)`` per k-point
    (the weighted operators already carry the material, so the eigenvalue is
    ``omega^2`` directly)."""
    freqs = np.zeros((len(ks), n_bands))
    for i, k in enumerate(ks):
        Kr, Mr = bloch.reduce_system(K, M, k)
        w2 = np.clip(generalized_eigh(Kr, Mr, n_bands), 0.0, None)
        freqs[i] = np.sqrt(w2) / (2 * math.pi)
    return freqs


# --------------------------------------------------------------------------- #
# Band-structure sweep with the native BlochReducer
# --------------------------------------------------------------------------- #
def compute_bands(problem: TriangularCrystal, verbose: bool = True,
                  ref: Dict[str, Any] | None = None) -> Dict[str, Any]:
    mesh = build_mesh(problem)
    K, M, n_steel = assemble(mesh, problem)
    bloch = BlochReducer(mesh.points, [problem.a1.tolist(), problem.a2.tolist()],
                         dofs_per_node=1)
    if verbose:
        print(f"mesh: {mesh.points.shape[0]} nodes ({n_steel} steel elems) -> "
              f"{bloch.n_masters} Bloch master DOFs", flush=True)

    ks, ms = k_path(problem)
    freqs = _sweep_bands(K, M, bloch, ks, problem.n_bands)
    out = dict(ks=ks, ms=ms, freqs=freqs,
               n_nodes=int(mesh.points.shape[0]), n_masters=int(bloch.n_masters),
               n_steel=n_steel)

    # Re-solve at COMSOL's exact wavevectors for a k-point-by-k-point error.
    if ref is not None:
        ref_ks = np.stack([ref["kx"], ref["ky"]], axis=1)
        out["freqs_at_ref"] = _sweep_bands(K, M, bloch, ref_ks, ref["n_compare"])
    return out


# --------------------------------------------------------------------------- #
# COMSOL reference (committed, offline): load + relative-error comparison
# --------------------------------------------------------------------------- #
def load_comsol_reference(path=None, n_compare: int = 10):
    """Load the committed COMSOL band table (``comsol_reference_triangle.npz``).

    The penetrable two-medium model returns a variable mode count per k-point;
    group the flat table by the path parameter ``m`` and keep the lowest
    ``n_compare`` frequencies per k. Returns ``None`` if the file is absent.
    """
    if path is None:
        path = Path(__file__).with_name("comsol_reference_triangle.npz")
    if not Path(path).exists():
        return None
    d = np.load(path)
    m, kx, ky, f = d["m"], d["kx"], d["ky"], d["freq_hz"]
    mvals = np.unique(m)
    n_compare = min(n_compare, *(int((m == u).sum()) for u in mvals))
    F = np.zeros((len(mvals), n_compare))
    KX = np.zeros(len(mvals)); KY = np.zeros(len(mvals))
    for i, u in enumerate(mvals):
        sel = m == u
        F[i] = np.sort(f[sel])[:n_compare]
        KX[i], KY[i] = kx[sel][0], ky[sel][0]
    return dict(m=mvals, kx=KX, ky=KY, freq=F, n_compare=n_compare,
                c=float(d["c_fluid"]))


def compare_to_comsol(freqs_at_ref: np.ndarray, ref: Dict[str, Any],
                      f_floor: float = 1.0e3) -> Dict[str, float]:
    """Per-mode nearest-frequency relative error of TensorMesh vs COMSOL at the
    same wavevectors; near-zero acoustic modes (``f < f_floor``) are skipped."""
    errs = []
    for i in range(ref["freq"].shape[0]):
        ft = np.sort(freqs_at_ref[i])
        for fc in ref["freq"][i]:
            if fc < f_floor:
                continue
            errs.append(abs(ft[np.argmin(np.abs(ft - fc))] - fc) / fc)
    errs = np.array(errs)
    return dict(mean=float(errs.mean()), p95=float(np.percentile(errs, 95)),
                max=float(errs.max()), n=int(errs.size))


# --------------------------------------------------------------------------- #
# Plot: rhombic cell | band structure
# --------------------------------------------------------------------------- #
def plot_bands(result: Dict[str, Any], problem: TriangularCrystal, save_path,
               ref: Dict[str, Any] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle

    a1, a2, r = problem.a1, problem.a2, problem.r
    F = result["freqs"] / 1e3
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 2.4], wspace=0.28)

    # -- Left panel: rhombic cell --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    mm = 1e3
    corners = np.array([[0, 0], a1, a1 + a2, a2]) * mm
    ctr = (a1 + a2) / 2.0
    ax.add_patch(Polygon(corners, closed=True, fc="#cfe8ff", ec="k"))
    ax.add_patch(Circle((ctr * mm), r * mm, fc="0.4", ec="k"))
    ax.set_xlim(corners[:, 0].min() - 1, corners[:, 0].max() + 1)
    ax.set_ylim(corners[:, 1].min() - 1, corners[:, 1].max() + 1)
    ax.set_aspect("equal"); ax.set_title("Rhombic cell (steel in water)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")

    # -- Right panel: TensorMesh bands (+ COMSOL reference if available) -------
    ax = fig.add_subplot(gs[0, 1])
    if ref is not None:
        for b in range(ref["freq"].shape[1]):
            ax.plot(ref["m"], ref["freq"][:, b] / 1e3, "o", ms=5.5, mfc="none",
                    mec="#D55E00", mew=1.0,
                    label="COMSOL" if b == 0 else None)
    for b in range(F.shape[1]):
        ax.plot(result["ms"], F[:, b], ".", color="#0072B2", ms=4.0,
                label="TensorMesh" if b == 0 else None)
    ax.set_xticks(PATH_TICKS); ax.set_xticklabels(PATH_LABELS)
    ax.set_xlim(result["ms"].min(), result["ms"].max()); ax.set_ylim(0, None)
    ax.set_xlabel("wavevector"); ax.set_ylabel("frequency [kHz]")
    title = "Triangular-lattice phononic crystal"
    if "stats" in result:
        title += f"  (mean err {100 * result['stats']['mean']:.2f}% vs COMSOL)"
    ax.set_title(title)
    if ref is not None:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Demo runner
# --------------------------------------------------------------------------- #
def run_demo(*, make_plot: bool = True, output_path=None,
             chara_per_a: float = 45.0, n_bands: int = 12,
             verbose: bool = True) -> Dict[str, Any]:
    """Run the triangular band-structure example and return diagnostics."""
    problem = TriangularCrystal(chara_per_a=chara_per_a, n_bands=n_bands)
    ref = load_comsol_reference()
    result = compute_bands(problem, verbose=verbose, ref=ref)
    if ref is not None:
        result["stats"] = compare_to_comsol(result["freqs_at_ref"], ref)

    if make_plot:
        if output_path is None:
            output_path = Path(__file__).with_name("band_structure_triangular.png")
        plot_bands(result, problem, output_path, ref=ref)

    F = result["freqs"]
    print("Triangular-lattice phononic-crystal band structure")
    print(f"  nodes: {result['n_nodes']} ({result['n_steel']} steel elems)  ->  "
          f"{result['n_masters']} master DOFs")
    print(f"  bands: {problem.n_bands}  k-points: {len(result['ks'])}")
    print(f"  frequency range: [{F.min()/1e3:.1f}, {F.max()/1e3:.1f}] kHz")
    if "stats" in result:
        s = result["stats"]
        print(f"  vs COMSOL ({ref['n_compare']} bands x {len(ref['m'])} k-pts): "
              f"mean {100*s['mean']:.2f}%  p95 {100*s['p95']:.2f}%  "
              f"max {100*s['max']:.2f}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip PNG generation")
    parser.add_argument("--chara-per-a", type=float, default=45.0,
                        help="mesh elements per lattice constant")
    parser.add_argument("--n-bands", type=int, default=12, help="number of bands")
    parser.add_argument("--output", type=str, default=None, help="output PNG path")
    args = parser.parse_args()
    run_demo(make_plot=not args.no_plot, output_path=args.output,
             chara_per_a=args.chara_per_a, n_bands=args.n_bands)


if __name__ == "__main__":
    main()

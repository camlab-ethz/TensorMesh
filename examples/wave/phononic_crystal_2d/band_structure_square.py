"""2D acoustic band structure of a square-lattice phononic crystal.

Demonstrates the Bloch-Floquet periodic boundary condition
(:class:`tensormesh.BlochReducer`) on a scalar Helmholtz eigenproblem: a square
lattice of rigid (sound-hard) cylinders in water. The cylinder is meshed as a
hole, so its surface carries the natural Neumann condition; the cell edges obey
``p(r + R) = exp(i k.R) p(r)``.

It is deliberately example-only: no public API is added. The workflow is

    mesh -> Laplace/Mass assembler -> BlochReducer -> generalized eig

giving ``K p = (omega/c)^2 M p`` reduced per wavevector to the master DOFs,
``K_r(k) = T(k)^H K T(k)``, then ``f = c sqrt(mu) / (2 pi)`` along M-Gamma-X-M.
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

from tensormesh import (  # noqa: E402  -- import after the sys.path shim
    LaplaceElementAssembler, MassElementAssembler, BlochReducer)

torch.set_default_dtype(torch.float64)

PATH_TICKS = [0, 1, 2, 3]
PATH_LABELS = ["M", r"$\Gamma$", "X", "M"]


@dataclass(frozen=True)
class SquareCrystal:
    """Parameters for the square-lattice band-structure example.

    Parameters
    ----------
    a:
        Lattice constant, in metres.
    r_over_a:
        Rigid-cylinder radius as a fraction of ``a``.
    c:
        Speed of sound in water, in m/s.
    n_bands:
        Number of bands to compute.
    n_seg:
        k-points per Brillouin-zone segment (markers in the plot).
    chara_per_a:
        Mesh elements per lattice constant (mesh density).
    """

    a: float = 0.01
    r_over_a: float = 0.45
    c: float = 1481.0
    n_bands: int = 12
    n_seg: int = 12
    chara_per_a: float = 44.0

    @property
    def r(self) -> float:
        return self.r_over_a * self.a

    @property
    def h(self) -> float:
        return self.a / self.chara_per_a


# --------------------------------------------------------------------------- #
# Periodic mesh of the fluid domain (square minus a centred disk)
# --------------------------------------------------------------------------- #
def build_mesh(problem: SquareCrystal):
    """Mesh the fluid domain with gmsh ``setPeriodic`` so opposite edges carry
    matching nodes (the precondition the Bloch node pairing needs)."""
    import gmsh

    a, r = problem.a, problem.r
    msh = os.path.join(os.path.dirname(__file__), "_cell.msh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("square_cell")
        occ = gmsh.model.occ
        occ.addRectangle(0.0, 0.0, 0.0, a, a)
        occ.addDisk(a / 2.0, a / 2.0, 0.0, r, r)
        occ.cut([(2, 1)], [(2, 2)])
        occ.synchronize()

        eps = 1e-9
        edges = {}
        for _, tag in gmsh.model.getEntities(1):
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(1, tag)
            if abs(cx) < eps:
                edges["left"] = tag
            elif abs(cx - a) < eps:
                edges["right"] = tag
            elif abs(cy) < eps:
                edges["bottom"] = tag
            elif abs(cy - a) < eps:
                edges["top"] = tag
        tx = [1, 0, 0, a, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        ty = [1, 0, 0, 0, 0, 1, 0, a, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(1, [edges["right"]], [edges["left"]], tx)
        gmsh.model.mesh.setPeriodic(1, [edges["top"]], [edges["bottom"]], ty)

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
# Wavevector path M-Gamma-X-M and the generalized eig
# --------------------------------------------------------------------------- #
def k_path(problem: SquareCrystal) -> Tuple[np.ndarray, np.ndarray]:
    """Square-lattice M-Gamma-X-M path, parameter m in [0, 3]."""
    g = math.pi / problem.a                       # Brillouin-zone edge
    ms = np.linspace(0.0, 3.0, 3 * problem.n_seg + 1)
    ks = []
    for m in ms:
        if m <= 1.0:
            ks.append([(1 - m) * g, (1 - m) * g])   # M -> Gamma
        elif m <= 2.0:
            ks.append([(m - 1) * g, 0.0])            # Gamma -> X
        else:
            ks.append([g, (m - 2) * g])              # X -> M
    return np.array(ks), ms


def generalized_eigh(Kr, Mr, n_bands: int) -> np.ndarray:
    """Lowest ``n_bands`` eigenvalues of the complex-Hermitian generalized
    problem ``K_r u = mu M_r u``, via shift-inverted ARPACK
    (:func:`scipy.sparse.linalg.eigsh`) on the sparse reduced operators.

    The reduced ``K_r`` is singular at ``Gamma`` (the lowest band tends to 0),
    so the shift ``sigma`` is set just below the spectrum
    (``-1e-4 * trace(K_r)/trace(M_r)``, the ``omega^2`` scale) to regularize the
    null space while still returning the lowest modes.
    """
    Ks = Kr.to_scipy_coo().tocsc()
    Ms = Mr.to_scipy_coo().tocsc()
    sigma = -1.0e-4 * float(abs(Ks.diagonal()).sum() / abs(Ms.diagonal()).sum())
    mu = sla.eigsh(Ks, k=n_bands, M=Ms, sigma=sigma, which="LM",
                   return_eigenvectors=False)
    return np.sort(mu.real)


def _sweep_bands(K, M, bloch: BlochReducer, ks: np.ndarray, n_bands: int,
                 c: float) -> np.ndarray:
    """Reduce ``(K, M)`` with the Bloch phase at every ``k`` and return the
    lowest ``n_bands`` frequencies ``f = c sqrt(mu) / (2 pi)`` per k-point."""
    freqs = np.zeros((len(ks), n_bands))
    for i, k in enumerate(ks):
        Kr, Mr = bloch.reduce_system(K, M, k)
        mu = np.clip(generalized_eigh(Kr, Mr, n_bands), 0.0, None)
        freqs[i] = c * np.sqrt(mu) / (2 * math.pi)
    return freqs


# --------------------------------------------------------------------------- #
# Band-structure sweep with the native BlochReducer
# --------------------------------------------------------------------------- #
def compute_bands(problem: SquareCrystal, verbose: bool = True,
                  ref: Dict[str, Any] | None = None) -> Dict[str, Any]:
    mesh = build_mesh(problem)
    K = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    bloch = BlochReducer(mesh.points, [[problem.a, 0.0], [0.0, problem.a]],
                         dofs_per_node=1)
    if verbose:
        print(f"mesh: {mesh.points.shape[0]} nodes -> {bloch.n_masters} "
              f"Bloch master DOFs", flush=True)

    ks, ms = k_path(problem)
    freqs = _sweep_bands(K, M, bloch, ks, problem.n_bands, problem.c)
    out = dict(ks=ks, ms=ms, freqs=freqs,
               n_nodes=int(mesh.points.shape[0]), n_masters=int(bloch.n_masters))

    # Re-solve at COMSOL's exact wavevectors so the relative error is measured
    # k-point by k-point (same path, same media) rather than across samplings.
    if ref is not None:
        ref_ks = np.stack([ref["kx"], ref["ky"]], axis=1)
        out["freqs_at_ref"] = _sweep_bands(K, M, bloch, ref_ks,
                                           ref["n_compare"], problem.c)
    return out


# --------------------------------------------------------------------------- #
# COMSOL reference (committed, offline): load + relative-error comparison
# --------------------------------------------------------------------------- #
def load_comsol_reference(path=None, n_compare: int = 10):
    """Load the committed COMSOL band table (``comsol_reference_square.npz``).

    The table is flat (one row per (k-point, mode)); group it by the path
    parameter ``m`` and keep the lowest ``n_compare`` frequencies per k-point.
    Returns ``None`` if the file is absent so the example still runs offline.
    """
    if path is None:
        path = Path(__file__).with_name("comsol_reference_square.npz")
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
    """Per-mode relative error of TensorMesh vs COMSOL at the same wavevectors.

    Nearest-frequency match per COMSOL mode (robust to the spurious Lagrange
    modes COMSOL's weak Floquet constraint adds at the top of its eigen-window);
    near-zero acoustic modes (``f < f_floor``) are skipped to avoid 0/0.
    """
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
# Plot: unit cell | band structure
# --------------------------------------------------------------------------- #
def plot_bands(result: Dict[str, Any], problem: SquareCrystal, save_path,
               ref: Dict[str, Any] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    a, r = problem.a, problem.r
    F = result["freqs"] / 1e3
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 2.4], wspace=0.28)

    # -- Left panel: unit cell -----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), a * 1e3, a * 1e3, fc="#cfe8ff", ec="k"))
    ax.add_patch(Circle((a / 2 * 1e3, a / 2 * 1e3), r * 1e3, fc="0.4", ec="k"))
    ax.set_xlim(-0.5, a * 1e3 + 0.5); ax.set_ylim(-0.5, a * 1e3 + 0.5)
    ax.set_aspect("equal"); ax.set_title("Unit cell (rigid cylinders in water)")
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
    title = "Square-lattice phononic crystal"
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
             chara_per_a: float = 44.0, n_bands: int = 12,
             verbose: bool = True) -> Dict[str, Any]:
    """Run the square band-structure example and return diagnostics."""
    problem = SquareCrystal(chara_per_a=chara_per_a, n_bands=n_bands)
    ref = load_comsol_reference()
    result = compute_bands(problem, verbose=verbose, ref=ref)
    if ref is not None:
        result["stats"] = compare_to_comsol(result["freqs_at_ref"], ref)

    if make_plot:
        if output_path is None:
            output_path = Path(__file__).with_name("band_structure_square.png")
        plot_bands(result, problem, output_path, ref=ref)

    F = result["freqs"]
    print("Square-lattice phononic-crystal band structure")
    print(f"  nodes: {result['n_nodes']}  ->  {result['n_masters']} master DOFs")
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
    parser.add_argument("--chara-per-a", type=float, default=44.0,
                        help="mesh elements per lattice constant")
    parser.add_argument("--n-bands", type=int, default=12, help="number of bands")
    parser.add_argument("--output", type=str, default=None, help="output PNG path")
    args = parser.parse_args()
    run_demo(make_plot=not args.no_plot, output_path=args.output,
             chara_per_a=args.chara_per_a, n_bands=args.n_bands)


if __name__ == "__main__":
    main()

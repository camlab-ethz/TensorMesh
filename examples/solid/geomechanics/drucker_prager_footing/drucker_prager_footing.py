"""Drucker-Prager strip-footing geomechanics example.

This example combines the two existing example-only geomechanics building
blocks into one nonlinear boundary-value problem:

* the local Drucker-Prager triaxial constitutive driver
  (``examples/solid/geomechanics/drucker_prager_triaxial``), and
* the elastic strip-footing setup
  (``examples/solid/geomechanics/elastic_footing``).

A rectangular soil block is loaded by a centered strip footing.  The footing
pressure is ramped in load steps, and at each converged step the per-quadrature
Drucker-Prager history is committed, exactly like the triaxial example.

It is deliberately example-only: no public geomechanics API is added.  The
constitutive code is copied locally (Torch only) rather than imported from the
other example, so this file is self-contained.

TensorMesh keeps the internal solid-mechanics convention tension-positive.  For
geomechanics reporting, settlement is shown positive downward: settlement
``= -u_y``.  This is a compact educational example, not a foundation-design
method.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh.assemble import DruckerPragerPlasticity
from tensormesh.material import FrictionalMaterial
from tensormesh.dataset.mesh import gen_rectangle


# ---------------------------------------------------------------------------
# Footing geometry, boundary conditions and load (from the elastic_footing
# example).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FootingProblem:
    """Parameters for the Drucker-Prager strip-footing example.

    The geometry matches the elastic footing example.  ``footing_pressure`` is
    the *final* compression-positive footing pressure that the load stepping
    ramps up to.
    """

    left: float = -4.0
    right: float = 4.0
    bottom: float = -4.0
    top: float = 0.0
    footing_width: float = 1.2
    footing_pressure: float = 300.0e3
    thickness: float = 1.0
    E: float = 50.0e6
    nu: float = 0.30
    chara_length: float = 0.35


def _move_mesh_to_dtype_device(mesh, dtype: torch.dtype, device: torch.device):
    """Move a TensorMesh mesh to dtype/device by moving its point tensor."""
    mesh.points = mesh.points.to(device=device, dtype=dtype)
    return mesh


def build_mesh(
    problem: FootingProblem,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
):
    """Build the 2D soil-block mesh."""
    device = torch.device(device)
    mesh = gen_rectangle(
        chara_length=problem.chara_length,
        left=problem.left,
        right=problem.right,
        bottom=problem.bottom,
        top=problem.top,
    )
    return _move_mesh_to_dtype_device(mesh, dtype=dtype, device=device)


def make_boundary_mask(mesh, problem: FootingProblem) -> torch.Tensor:
    """Return a vector-valued Dirichlet mask of shape [n_points, 2].

    Bottom boundary: ``uy = 0``; left and right boundaries: ``ux = 0``.
    """
    points = mesh.points
    dim = mesh.dim
    if dim != 2:
        raise ValueError(f"drucker_prager_footing expects a 2D mesh, got dim={dim}")

    tol = max(1.0e-9, 0.05 * problem.chara_length)
    x = points[:, 0]
    y = points[:, 1]

    left = torch.abs(x - problem.left) <= tol
    right = torch.abs(x - problem.right) <= tol
    bottom = torch.abs(y - problem.bottom) <= tol

    fixed = torch.zeros((mesh.n_points, dim), dtype=torch.bool, device=points.device)
    fixed[left | right, 0] = True
    fixed[bottom, 1] = True
    return fixed


def make_load_vector(
    mesh,
    problem: FootingProblem,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the full-pressure nodal load vector, loaded nodes, and total load.

    The footing pressure is lumped equally over top-surface nodes inside the
    footing patch, matching the elastic footing example.
    """
    points = mesh.points
    dim = mesh.dim
    tol = max(1.0e-9, 0.05 * problem.chara_length)

    x = points[:, 0]
    y = points[:, 1]
    footing_half_width = 0.5 * problem.footing_width

    top = torch.abs(y - problem.top) <= tol
    in_footing = torch.abs(x) <= footing_half_width + tol
    loaded_nodes = torch.where(top & in_footing)[0]

    if loaded_nodes.numel() < 2:
        raise RuntimeError(
            "Footing patch found too few top nodes. "
            "Use a smaller chara_length or a wider footing."
        )

    total_vertical_load = (
        -problem.footing_pressure * problem.footing_width * problem.thickness
    )

    rhs = torch.zeros((mesh.n_points, dim), dtype=points.dtype, device=points.device)
    rhs[loaded_nodes, 1] = total_vertical_load / loaded_nodes.numel()
    return rhs, loaded_nodes, float(total_vertical_load)


# ---------------------------------------------------------------------------
# Nonlinear load-stepped solver.
# ---------------------------------------------------------------------------
def solve_drucker_prager_footing(
    problem: FootingProblem,
    material: FrictionalMaterial,
    n_steps: int = 10,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    lbfgs_max_iter: int = 50,
) -> Dict[str, Any]:
    """Solve the footing problem with Drucker-Prager plasticity and load stepping.

    The total potential energy ``internal - external`` is minimized over the free
    displacement DOFs with L-BFGS at each load step.  After each converged step
    the per-quadrature Drucker-Prager history is committed with ``update_state``.
    """
    device = torch.device(device)
    mesh = build_mesh(problem, dtype=dtype, device=device)
    dp = DruckerPragerPlasticity.from_mesh(mesh, material=material)

    dim = mesh.dim
    n_points = mesh.n_points
    n_dof = n_points * dim

    fixed = make_boundary_mask(mesh, problem)
    rhs_full, loaded_nodes, total_vertical_load = make_load_vector(mesh, problem)
    rhs_full_flat = rhs_full.flatten()

    free_flat = (~fixed).flatten()
    free_idx = torch.where(free_flat)[0]
    base_zeros = torch.zeros(n_dof, dtype=dtype, device=device)

    free_u = torch.zeros(free_idx.numel(), dtype=dtype, device=device, requires_grad=True)

    def recover_full(free_values: torch.Tensor) -> torch.Tensor:
        """Scatter the free DOFs into a full [n_points, dim] displacement field."""
        u_flat = base_zeros.index_add(0, free_idx, free_values)
        return u_flat.reshape(n_points, dim)

    load_factors: List[float] = []
    pressures_kpa: List[float] = []
    footing_settlements: List[float] = []
    max_settlements: List[float] = []
    max_alphas: List[float] = []
    mean_alphas: List[float] = []

    for step in range(1, n_steps + 1):
        load_factor = step / n_steps
        f_ext_flat = load_factor * rhs_full_flat

        optimizer = torch.optim.LBFGS(
            [free_u],
            max_iter=lbfgs_max_iter,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-14,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            u_full = recover_full(free_u)
            internal = dp.energy(
                point_data={"displacement": u_full},
                element_data=dp.element_data_from_history(),
            )
            external = torch.dot(f_ext_flat, u_full.flatten())
            loss = internal - external
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            u_full = recover_full(free_u)
            dp.update_state(u_full)

            settlement = -u_full[:, 1]
            load_factors.append(float(load_factor))
            pressures_kpa.append(float(load_factor * problem.footing_pressure / 1.0e3))
            footing_settlements.append(float(settlement[loaded_nodes].mean()))
            max_settlements.append(float(settlement.max()))
            max_alphas.append(float(dp.max_alpha()))
            mean_alphas.append(float(dp.mean_alpha()))

    with torch.no_grad():
        u_final = recover_full(free_u).detach()

    return {
        "mesh": mesh,
        "dp": dp,
        "u": u_final,
        "fixed": fixed,
        "loaded_nodes": loaded_nodes,
        "total_vertical_load_N_per_m": total_vertical_load,
        "load_factors": load_factors,
        "pressures_kpa": pressures_kpa,
        "footing_settlements_m": footing_settlements,
        "max_settlements_m": max_settlements,
        "max_alphas": max_alphas,
        "mean_alphas": mean_alphas,
        "n_nodes": int(n_points),
        "n_steps": int(n_steps),
    }


# ---------------------------------------------------------------------------
# Post-processing: plastic-history fields (NumPy allowed here only).
# ---------------------------------------------------------------------------
def _element_alpha_and_centroids(out: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-element mean committed alpha and element centroids."""
    dp = out["dp"]
    points = out["mesh"].points
    alphas: List[torch.Tensor] = []
    centroids: List[torch.Tensor] = []
    for etype, trans in dp.transformation.items():
        elements = trans.elements
        alphas.append(dp.history[etype]["alpha"].mean(dim=1))
        centroids.append(points[elements].mean(dim=1)[:, :2])
    return torch.cat(alphas), torch.cat(centroids, dim=0)


def _nodal_alpha(out: Dict[str, Any]) -> torch.Tensor:
    """Average per-element committed alpha onto mesh nodes for contouring."""
    dp = out["dp"]
    mesh = out["mesh"]
    nodal_sum = torch.zeros(mesh.n_points, dtype=mesh.points.dtype, device=mesh.points.device)
    nodal_cnt = torch.zeros_like(nodal_sum)
    for etype, trans in dp.transformation.items():
        elements = trans.elements
        elem_alpha = dp.history[etype]["alpha"].mean(dim=1)
        n_nodes = elements.shape[1]
        flat_nodes = elements.reshape(-1)
        flat_vals = elem_alpha[:, None].expand(-1, n_nodes).reshape(-1)
        nodal_sum.index_add_(0, flat_nodes, flat_vals)
        nodal_cnt.index_add_(0, flat_nodes, torch.ones_like(flat_vals))
    return nodal_sum / torch.clamp(nodal_cnt, min=1.0)


def _plastic_centroid(out: Dict[str, Any]) -> tuple[float, float]:
    """Return the alpha-weighted centroid of plastic activity (x, y)."""
    elem_alpha, centroids = _element_alpha_and_centroids(out)
    total = float(elem_alpha.sum())
    if total <= 1.0e-12:
        return float("nan"), float("nan")
    cx = float((elem_alpha * centroids[:, 0]).sum() / total)
    cy = float((elem_alpha * centroids[:, 1]).sum() / total)
    return cx, cy


def _triangles_from_mesh(mesh) -> "object":
    """Return triangle connectivity for matplotlib, splitting quads if needed."""
    import numpy as np

    triangles = []
    for cells in mesh.cells.values():
        arr = cells.detach().cpu().numpy()
        if arr.shape[1] == 3:
            triangles.append(arr)
        elif arr.shape[1] == 4:
            # TensorMesh stores quads in FEniCS CCW order [BL, BR, TR, TL]
            # = node indices [0, 1, 3, 2]. The two non-overlapping triangles
            # along the BL-TR diagonal are [0, 1, 3] and [0, 3, 2]; the naive
            # [0,1,2]+[0,2,3] split produces the bowtie pattern reported in
            # plots of these examples.
            triangles.append(arr[:, [0, 1, 3]])
            triangles.append(arr[:, [0, 3, 2]])
    if not triangles:
        raise RuntimeError("No triangular or quadrilateral cells found for plotting.")
    return np.vstack(triangles)


def plot_solution(
    out: Dict[str, Any],
    problem: FootingProblem,
    save_path: str | os.PathLike[str],
) -> None:
    """Three-panel figure: settlement field, plastic history, load-settlement."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = out["mesh"]
    u = out["u"]
    fixed = out["fixed"]
    loaded_nodes = out["loaded_nodes"]

    points = mesh.points.detach().cpu()
    u_cpu = u.detach().cpu()

    width = problem.right - problem.left
    depth = problem.top - problem.bottom
    footing_left = -0.5 * problem.footing_width
    footing_right = 0.5 * problem.footing_width

    max_disp = torch.linalg.norm(u_cpu, dim=1).max().item()
    scale = (0.08 * max(width, depth)) / max_disp if max_disp > 0.0 else 1.0
    deformed = points[:, :2] + scale * u_cpu[:, :2]
    triangles = _triangles_from_mesh(mesh)
    triang = mtri.Triangulation(deformed[:, 0].numpy(), deformed[:, 1].numpy(), triangles)

    settlement_mm = (-u_cpu[:, 1] * 1.0e3).numpy()
    nodal_alpha = _nodal_alpha(out).detach().cpu().numpy()

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(17.0, 4.8), constrained_layout=True
    )

    def _draw_supports_and_footing(ax):
        outline_x = [problem.left, problem.right, problem.right, problem.left, problem.left]
        outline_y = [problem.bottom, problem.bottom, problem.top, problem.top, problem.bottom]
        ax.plot(outline_x, outline_y, color="0.55", linewidth=1.0, linestyle="--")
        footing_y = problem.top + 0.06 * depth
        ax.plot(
            [footing_left, footing_right], [footing_y, footing_y],
            color="tab:red", linewidth=6.0, solid_capstyle="butt", label="footing load patch",
        )
        for xx in np.linspace(footing_left, footing_right, 5):
            ax.arrow(
                xx, problem.top + 0.28 * depth, 0.0, -0.18 * depth,
                head_width=0.05 * width, head_length=0.04 * depth,
                length_includes_head=True, color="tab:red", alpha=0.9,
            )
        bottom_fixed = fixed[:, 1].detach().cpu()
        side_fixed = fixed[:, 0].detach().cpu()
        ax.scatter(points[bottom_fixed, 0], points[bottom_fixed, 1], marker="s", s=9,
                   color="tab:blue", label="uy fixed", zorder=5)
        ax.scatter(points[side_fixed, 0], points[side_fixed, 1], marker="|", s=26,
                   color="tab:purple", label="ux fixed", zorder=5)

    # -- Panel 1: settlement field -----------------------------------------
    contour0 = ax0.tricontourf(triang, settlement_mm, levels=18)
    ax0.triplot(triang, linewidth=0.2, color="0.25", alpha=0.3)
    _draw_supports_and_footing(ax0)
    loaded = loaded_nodes.detach().cpu()
    ax0.scatter(points[loaded, 0], points[loaded, 1], marker="v", s=22, color="tab:red", zorder=6)
    cbar0 = fig.colorbar(contour0, ax=ax0)
    cbar0.set_label("settlement, -u_y [mm]")
    ax0.set_title(f"Settlement field (deformation scale {scale:.0f}x)")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_aspect("equal", adjustable="box")
    ax0.legend(loc="lower right", frameon=True, fontsize=8)

    # -- Panel 2: committed plastic history --------------------------------
    contour1 = ax1.tricontourf(triang, nodal_alpha, levels=18, cmap="magma")
    ax1.triplot(triang, linewidth=0.2, color="0.6", alpha=0.25)
    _draw_supports_and_footing(ax1)
    cbar1 = fig.colorbar(contour1, ax=ax1)
    cbar1.set_label(r"committed plastic history $\alpha$")
    ax1.set_title("Drucker-Prager plastic history")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_aspect("equal", adjustable="box")

    # -- Panel 3: load-settlement curve ------------------------------------
    settle_mm = [s * 1.0e3 for s in out["footing_settlements_m"]]
    pressures = out["pressures_kpa"]
    ax2.plot(settle_mm, pressures, marker="o", markersize=4, color="tab:blue",
             label="Drucker-Prager")
    # Initial elastic tangent (first-step secant): the curve peels away from it
    # as plasticity accumulates, which is the nonlinearity to look for.
    if settle_mm[0] > 0.0:
        k0 = pressures[0] / settle_mm[0]
        ax2.plot(settle_mm, [k0 * s for s in settle_mm], linestyle="--", color="0.5",
                 label="initial elastic tangent")
    ax2.set_title("Load-settlement response")
    ax2.set_xlabel("footing settlement, -u_y [mm]")
    ax2.set_ylabel("footing pressure [kPa]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def run_demo(
    *,
    make_plot: bool = True,
    output_path: str | os.PathLike[str] | None = None,
    n_steps: int = 10,
    chara_length: float = 0.35,
    final_pressure: float = 300.0e3,
    cohesion: float = 10.0e3,
    friction_angle_deg: float = 30.0,
    H: float = 1.0e6,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    lbfgs_max_iter: int = 50,
) -> Dict[str, Any]:
    """Run the Drucker-Prager footing example and return diagnostics for tests."""
    problem = FootingProblem(chara_length=chara_length, footing_pressure=final_pressure)
    # Associated Drucker-Prager soil (dilatancy_angle defaults to the friction
    # angle), matching the parameters this example has always used.
    material = FrictionalMaterial(
        name="ExampleSoil",
        E=problem.E,
        nu=problem.nu,
        cohesion=cohesion,
        friction_angle=friction_angle_deg,
        H=H,
    )

    out = solve_drucker_prager_footing(
        problem, material, n_steps=n_steps, dtype=dtype, device=device,
        lbfgs_max_iter=lbfgs_max_iter,
    )

    plastic_centroid_x, plastic_centroid_y = _plastic_centroid(out)

    if make_plot:
        if output_path is None:
            output_path = Path(__file__).with_name("drucker_prager_footing.png")
        plot_solution(out, problem, output_path)

    result: Dict[str, Any] = {
        "load_factors": out["load_factors"],
        "pressures_kpa": out["pressures_kpa"],
        "footing_settlements_m": out["footing_settlements_m"],
        "max_settlements_m": out["max_settlements_m"],
        "max_alphas": out["max_alphas"],
        "mean_alphas": out["mean_alphas"],
        "final_footing_settlement_m": out["footing_settlements_m"][-1],
        "final_max_settlement_m": out["max_settlements_m"][-1],
        "final_max_alpha": out["max_alphas"][-1],
        "final_mean_alpha": out["mean_alphas"][-1],
        "plastic_centroid_x": plastic_centroid_x,
        "plastic_centroid_y": plastic_centroid_y,
        "n_nodes": out["n_nodes"],
        "n_steps": out["n_steps"],
    }

    print("Drucker-Prager strip-footing example")
    print(f"  nodes: {result['n_nodes']}  load steps: {result['n_steps']}")
    print(f"  final footing pressure: {out['pressures_kpa'][-1]:.1f} kPa")
    print(f"  final footing settlement: {result['final_footing_settlement_m'] * 1e3:.4f} mm")
    print(f"  final max settlement: {result['final_max_settlement_m'] * 1e3:.4f} mm")
    print(f"  final max alpha: {result['final_max_alpha']:.6e}")
    print(f"  final mean alpha: {result['final_mean_alpha']:.6e}")
    print(f"  plastic centroid: ({result['plastic_centroid_x']:.3f}, {result['plastic_centroid_y']:.3f}) m")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip PNG generation")
    parser.add_argument("--steps", type=int, default=10, help="number of load steps")
    parser.add_argument("--chara-length", type=float, default=0.35, help="target mesh size")
    parser.add_argument("--pressure-kpa", type=float, default=300.0,
                        help="final compression-positive footing pressure in kPa")
    parser.add_argument("--cohesion-kpa", type=float, default=10.0, help="cohesion in kPa")
    parser.add_argument("--friction-deg", type=float, default=30.0, help="friction angle in degrees")
    parser.add_argument("--output", type=str, default=None, help="optional output PNG path")
    args = parser.parse_args()

    run_demo(
        make_plot=not args.no_plot,
        output_path=args.output,
        n_steps=args.steps,
        chara_length=args.chara_length,
        final_pressure=args.pressure_kpa * 1.0e3,
        cohesion=args.cohesion_kpa * 1.0e3,
        friction_angle_deg=args.friction_deg,
    )


if __name__ == "__main__":
    main()

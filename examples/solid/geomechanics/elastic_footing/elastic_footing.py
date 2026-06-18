"""Elastic strip-footing geomechanics example.

This example solves a small-strain linear-elastic boundary-value problem for a
soil block loaded by a centered strip footing.  It is deliberately example-only:
no public geomechanics API is added.

The model is a 2D plane-strain-style educational problem with unit out-of-plane
thickness.  TensorMesh keeps the standard solid-mechanics sign convention
internally.  For geomechanics reporting, settlement is reported as positive
downward: settlement = -u_y.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh import Condenser
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.dataset.mesh import gen_rectangle


@dataclass(frozen=True)
class FootingProblem:
    """Parameters for the elastic strip-footing example.

    Parameters
    ----------
    left, right:
        Horizontal soil-domain bounds, in metres.
    bottom, top:
        Vertical soil-domain bounds, in metres.  The top surface is y = 0.
    footing_width:
        Strip-footing width, in metres.
    footing_pressure:
        Compression-positive footing pressure, in Pa.
    thickness:
        Unit out-of-plane thickness used to convert pressure to line load.
    E:
        Young's modulus, in Pa.
    nu:
        Poisson's ratio.
    chara_length:
        Target mesh size.
    """

    left: float = -4.0
    right: float = 4.0
    bottom: float = -4.0
    top: float = 0.0
    footing_width: float = 1.2
    footing_pressure: float = 80.0e3
    thickness: float = 1.0
    E: float = 50.0e6
    nu: float = 0.30
    chara_length: float = 0.35


def _move_mesh_to_dtype_device(mesh, dtype: torch.dtype, device: torch.device):
    """Move a TensorMesh mesh to dtype/device.

    The TensorMesh ``Mesh`` stores geometry in ``mesh.points``; the assembler
    reads its dtype/device from there.  ``Mesh`` does not expose a ``.to(...)``
    method, so we move the point tensor directly (cell connectivity stays
    integer).  This mirrors the existing Drucker-Prager geomechanics example.
    """
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
    mesh = _move_mesh_to_dtype_device(mesh, dtype=dtype, device=device)
    return mesh


def make_boundary_mask(mesh, problem: FootingProblem) -> torch.Tensor:
    """Return a vector-valued Dirichlet mask of shape [n_points, 2].

    Boundary conditions:

    - bottom boundary: uy = 0,
    - left and right boundaries: ux = 0.
    """
    points = mesh.points
    dim = mesh.dim
    if dim != 2:
        raise ValueError(f"elastic_footing expects a 2D mesh, got dim={dim}")

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
    """Return the nodal load vector, loaded node indices, and total vertical load.

    The footing pressure is lumped equally over top-surface nodes inside the
    footing patch.  This follows the same simple educational style as the
    cantilever example, where the total face load is distributed over boundary
    nodes rather than assembled with a dedicated facet traction form.
    """
    points = mesh.points
    dim = mesh.dim
    tol = max(1.0e-9, 0.05 * problem.chara_length)

    x = points[:, 0]
    y = points[:, 1]
    footing_half_width = 0.5 * problem.footing_width

    top = torch.abs(y - problem.top) <= tol
    in_footing = torch.abs(x) <= footing_half_width + tol
    loaded_mask = top & in_footing
    loaded_nodes = torch.where(loaded_mask)[0]

    if loaded_nodes.numel() < 2:
        raise RuntimeError(
            "Footing patch found too few top nodes. "
            "Use a smaller chara_length or a wider footing."
        )

    total_vertical_load = (
        -problem.footing_pressure * problem.footing_width * problem.thickness
    )

    rhs = torch.zeros(
        (mesh.n_points, dim),
        dtype=points.dtype,
        device=points.device,
    )
    rhs[loaded_nodes, 1] = total_vertical_load / loaded_nodes.numel()

    return rhs, loaded_nodes, float(total_vertical_load)


def solve_footing(
    problem: FootingProblem,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Assemble and solve the elastic footing boundary-value problem."""
    mesh = build_mesh(problem, dtype=dtype, device=device)

    assembler = LinearElasticityElementAssembler.from_mesh(
        mesh,
        E=problem.E,
        nu=problem.nu,
    )
    K = assembler()

    fixed = make_boundary_mask(mesh, problem)
    rhs, loaded_nodes, total_vertical_load = make_load_vector(mesh, problem)
    rhs_flat = rhs.flatten()

    condenser = Condenser(fixed.flatten())
    K_cond, rhs_cond = condenser(K, rhs_flat)

    u_cond = K_cond.solve(rhs_cond)
    u_flat = condenser.recover(u_cond).to(dtype=rhs_flat.dtype)
    u = u_flat.reshape(mesh.n_points, mesh.dim)

    # Reactions at constrained DOFs: r = K u - f.
    # SparseMatrix inherits @ from torch-sla SparseTensor.
    residual = K @ u_flat - rhs_flat
    reactions = residual.reshape(mesh.n_points, mesh.dim)

    vertical_fixed = fixed[:, 1]
    vertical_reaction = reactions[vertical_fixed, 1].sum()

    uy = u[:, 1]
    settlement = -uy
    footing_settlement = settlement[loaded_nodes].mean()
    max_settlement = settlement.max()
    min_uy = uy.min()

    load_balance_error = torch.abs(
        vertical_reaction + torch.as_tensor(
            total_vertical_load,
            dtype=vertical_reaction.dtype,
            device=vertical_reaction.device,
        )
    ) / max(abs(total_vertical_load), 1.0)

    result = {
        "mesh": mesh,
        "K": K,
        "rhs": rhs,
        "u": u,
        "fixed": fixed,
        "loaded_nodes": loaded_nodes,
        "total_vertical_load_N_per_m": total_vertical_load,
        "vertical_reaction_N_per_m": float(vertical_reaction.detach().cpu()),
        "load_balance_relative_error": float(load_balance_error.detach().cpu()),
        "max_settlement_m": float(max_settlement.detach().cpu()),
        "footing_settlement_m": float(footing_settlement.detach().cpu()),
        "min_vertical_displacement_m": float(min_uy.detach().cpu()),
        "n_nodes": int(mesh.n_points),
        "n_loaded_nodes": int(loaded_nodes.numel()),
        "n_fixed_dofs": int(fixed.sum().detach().cpu()),
        "n_free_dofs": int(fixed.numel() - fixed.sum().detach().cpu()),
    }
    return result


def _triangles_from_mesh(mesh) -> "object":
    """Return triangle connectivity for matplotlib plotting.

    Uses NumPy only in visualization/post-processing, never inside an assembler.
    Handles triangular cells directly and splits quadrilateral cells if needed.
    """
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


def _surface_settlement_profile(
    result: Dict[str, Any],
    problem: FootingProblem,
):
    """Return sorted (x, settlement_mm) along the top ground surface."""
    import numpy as np

    mesh = result["mesh"]
    u = result["u"]
    points = mesh.points.detach().cpu()
    u_cpu = u.detach().cpu()

    tol = max(1.0e-9, 0.05 * problem.chara_length)
    surface = torch.abs(points[:, 1] - problem.top) <= tol
    surface_idx = torch.where(surface)[0]

    x_surface = points[surface_idx, 0].numpy()
    settlement_mm = (-u_cpu[surface_idx, 1] * 1.0e3).numpy()

    order = np.argsort(x_surface)
    return x_surface[order], settlement_mm[order]


def plot_solution(
    result: Dict[str, Any],
    problem: FootingProblem,
    save_path: str | os.PathLike[str],
) -> None:
    """Plot the deformed settlement field and the surface settlement profile."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = result["mesh"]
    u = result["u"]
    fixed = result["fixed"]
    loaded_nodes = result["loaded_nodes"]

    points = mesh.points.detach().cpu()
    u_cpu = u.detach().cpu()

    width = problem.right - problem.left
    depth = problem.top - problem.bottom

    max_disp = torch.linalg.norm(u_cpu, dim=1).max().item()
    target_visual_disp = 0.08 * max(width, depth)
    scale = target_visual_disp / max_disp if max_disp > 0.0 else 1.0

    deformed = points[:, :2] + scale * u_cpu[:, :2]
    triangles = _triangles_from_mesh(mesh)

    triang = mtri.Triangulation(
        deformed[:, 0].numpy(),
        deformed[:, 1].numpy(),
        triangles,
    )

    settlement_mm = (-u_cpu[:, 1] * 1.0e3).numpy()

    fig, (ax, ax_profile) = plt.subplots(
        1, 2, figsize=(13.0, 4.8), constrained_layout=True
    )

    # -- Left panel: deformed mesh + settlement contour ----------------------
    contour = ax.tricontourf(triang, settlement_mm, levels=18)
    ax.triplot(triang, linewidth=0.25, color="0.25", alpha=0.35)

    # Plot undeformed outline.
    outline_x = [
        problem.left,
        problem.right,
        problem.right,
        problem.left,
        problem.left,
    ]
    outline_y = [
        problem.bottom,
        problem.bottom,
        problem.top,
        problem.top,
        problem.bottom,
    ]
    ax.plot(outline_x, outline_y, color="0.55", linewidth=1.0, linestyle="--")

    # Footing patch and downward load arrows.
    footing_left = -0.5 * problem.footing_width
    footing_right = 0.5 * problem.footing_width
    footing_y = problem.top + 0.06 * depth
    ax.plot(
        [footing_left, footing_right],
        [footing_y, footing_y],
        color="tab:red",
        linewidth=6.0,
        solid_capstyle="butt",
        label="footing load patch",
    )

    arrow_x = np.linspace(footing_left, footing_right, 5)
    for xx in arrow_x:
        ax.arrow(
            xx,
            problem.top + 0.28 * depth,
            0.0,
            -0.18 * depth,
            head_width=0.05 * width,
            head_length=0.04 * depth,
            length_includes_head=True,
            color="tab:red",
            alpha=0.9,
        )

    # Mark supports.
    bottom_fixed = fixed[:, 1].detach().cpu()
    side_fixed = fixed[:, 0].detach().cpu()
    ax.scatter(
        points[bottom_fixed, 0],
        points[bottom_fixed, 1],
        marker="s",
        s=10,
        color="tab:blue",
        label="uy fixed",
        zorder=5,
    )
    ax.scatter(
        points[side_fixed, 0],
        points[side_fixed, 1],
        marker="|",
        s=28,
        color="tab:purple",
        label="ux fixed",
        zorder=5,
    )

    loaded = loaded_nodes.detach().cpu()
    ax.scatter(
        points[loaded, 0],
        points[loaded, 1],
        marker="v",
        s=24,
        color="tab:red",
        zorder=6,
    )

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("settlement, -u_y [mm]")

    ax.set_title(
        "Elastic strip footing on a soil block "
        f"(deformation scale {scale:.0f}x)"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=True)

    # -- Right panel: ground-surface settlement profile ----------------------
    x_surface, surface_settlement_mm = _surface_settlement_profile(result, problem)
    ax_profile.plot(
        x_surface,
        surface_settlement_mm,
        marker="o",
        markersize=3,
        color="tab:blue",
    )
    ax_profile.axvspan(
        footing_left,
        footing_right,
        color="tab:red",
        alpha=0.15,
        label="footing width",
    )
    ax_profile.set_title("Ground-surface settlement profile")
    ax_profile.set_xlabel("x [m]")
    ax_profile.set_ylabel("settlement, -u_y [mm]")
    ax_profile.grid(True, alpha=0.3)
    # Settlement is positive downward; invert so the settlement bowl dips down.
    ax_profile.invert_yaxis()
    ax_profile.legend(loc="lower right", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run_demo(
    *,
    make_plot: bool = True,
    output_path: str | os.PathLike[str] | None = None,
    chara_length: float = 0.35,
    footing_pressure: float = 80.0e3,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Run the footing example and return diagnostics for tests."""
    problem = FootingProblem(
        chara_length=chara_length,
        footing_pressure=footing_pressure,
    )

    result = solve_footing(problem, dtype=dtype, device=device)

    if make_plot:
        if output_path is None:
            output_path = Path(__file__).with_name("elastic_footing.png")
        plot_solution(result, problem, output_path)

    print("Elastic strip-footing example")
    print(f"  nodes: {result['n_nodes']}")
    print(f"  loaded top nodes: {result['n_loaded_nodes']}")
    print(f"  fixed DOFs: {result['n_fixed_dofs']}")
    print(f"  total vertical load: {result['total_vertical_load_N_per_m'] / 1e3:.3f} kN/m")
    print(f"  vertical reaction: {result['vertical_reaction_N_per_m'] / 1e3:.3f} kN/m")
    print(f"  reaction/load relative error: {result['load_balance_relative_error']:.3e}")
    print(f"  footing settlement: {result['footing_settlement_m'] * 1e3:.4f} mm")
    print(f"  max settlement: {result['max_settlement_m'] * 1e3:.4f} mm")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip PNG generation")
    parser.add_argument(
        "--chara-length",
        type=float,
        default=0.35,
        help="target mesh size",
    )
    parser.add_argument(
        "--pressure-kpa",
        type=float,
        default=80.0,
        help="compression-positive footing pressure in kPa",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="optional output PNG path",
    )
    args = parser.parse_args()

    run_demo(
        make_plot=not args.no_plot,
        output_path=args.output,
        chara_length=args.chara_length,
        footing_pressure=args.pressure_kpa * 1.0e3,
    )


if __name__ == "__main__":
    main()

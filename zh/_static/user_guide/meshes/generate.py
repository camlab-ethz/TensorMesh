"""Generate user-guide / Meshes figures.

Run from repo root:
    source ~/venvs/tensorgalerkin/bin/activate
    python docs/source/_static/user_guide/meshes/generate.py

Outputs PNGs into the same directory.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensormesh as tm

OUT = Path(__file__).parent


def _draw_wireframe(ax, mesh, color="0.6", lw=0.6):
    """Draw mesh edges on ax (2D)."""
    points = mesh.points.cpu().numpy()
    for etype, conn in mesh.cells.items():
        if etype == "line":
            continue
        c = conn.cpu().numpy()
        # corner indices only (skip high-order mid-edge nodes for the wireframe)
        if etype.startswith("triangle"):
            corners = c[:, :3]
            edges = [(0, 1), (1, 2), (2, 0)]
        elif etype.startswith("quad"):
            corners = c[:, :4]
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        else:
            continue
        for tri in corners:
            for i, j in edges:
                p = points[tri[i]]
                q = points[tri[j]]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=1)


def boundary_per_side():
    """Highlight every boundary side in its own colour, plus the union."""
    mesh = tm.Mesh.gen_rectangle(chara_length=0.08)
    pts = mesh.points.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # left panel: is_boundary union
    ax = axes[0]
    _draw_wireframe(ax, mesh)
    interior = ~mesh.boundary_mask.cpu().numpy()
    boundary = mesh.boundary_mask.cpu().numpy()
    ax.scatter(pts[interior, 0], pts[interior, 1], s=14, c="0.7", label="interior")
    ax.scatter(pts[boundary, 0], pts[boundary, 1], s=28, c="C3", label="is_boundary",
               zorder=3, edgecolors="black", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_title("mesh.boundary_mask")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    # right panel: per-side colouring
    ax = axes[1]
    _draw_wireframe(ax, mesh)
    ax.scatter(pts[interior, 0], pts[interior, 1], s=14, c="0.85", zorder=2)
    side_colors = {
        "is_left_boundary":   "C0",
        "is_right_boundary":  "C2",
        "is_bottom_boundary": "C1",
        "is_top_boundary":    "C4",
    }
    for key, color in side_colors.items():
        mask = mesh.point_data[key].cpu().numpy()
        ax.scatter(pts[mask, 0], pts[mask, 1], s=28, c=color,
                   label=key, zorder=3, edgecolors="black", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_title("per-side masks set by gen_rectangle()")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)

    fig.suptitle("Boundary identification on Mesh.gen_rectangle(chara_length=0.08)")
    fig.tight_layout()
    fig.savefig(OUT / "boundary_per_side.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'boundary_per_side.png'}")


def plot_multi_panel():
    """Demonstrate mesh.plot({'u': ..., 'v': ...}) side-by-side rendering."""
    mesh = tm.Mesh.gen_rectangle(chara_length=0.04)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    u = torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)
    v = torch.cos(3 * np.pi * x) * torch.cos(3 * np.pi * y)

    out = OUT / "plot_multi_panel.png"
    mesh.plot({"sin(2πx) sin(2πy)": u, "cos(3πx) cos(3πy)": v},
              save_path=str(out), show=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    boundary_per_side()
    plot_multi_panel()

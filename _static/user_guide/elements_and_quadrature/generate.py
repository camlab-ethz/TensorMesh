"""Figures for User Guide / Elements and Quadrature.

Run from repo root after activating the environment:
    source ~/venvs/tensorgalerkin/bin/activate
    python docs/source/_static/user_guide/elements_and_quadrature/generate.py
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import tensormesh as tm
from tensormesh import Triangle

OUT = Path(__file__).parent


def ref_to_physical_triangle():
    """Reference triangle vs a stretched/sheared physical triangle, with
    quadrature points mapped through the affine map and basis-node
    indices labelled."""

    ref_pts = Triangle.get_basis(order=1).numpy()              # [3, 2]
    _, q_ref = Triangle.get_quadrature(order=2)
    q_ref = q_ref.numpy()                                      # [3, 2]

    phys_vertices = np.array([[1.0, 0.6], [3.0, 0.8], [2.0, 2.6]])  # [3, 2]
    # affine map: x = a + (B - A) * xi + (C - A) * eta
    A, B, C = phys_vertices
    Jmat = np.stack([B - A, C - A], axis=1)
    q_phys = q_ref @ Jmat.T + A

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- reference panel ---
    ax = axes[0]
    tri = plt.Polygon(ref_pts, fill=True, facecolor="#E8F0FF",
                      edgecolor="0.2", lw=1.6, zorder=1)
    ax.add_patch(tri)
    ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=70, c="C0",
               edgecolors="black", linewidths=0.6, zorder=3, label="basis nodes")
    for i, (x, y) in enumerate(ref_pts):
        ax.annotate(f" v{i}", (x, y), fontsize=12, zorder=4)
    ax.scatter(q_ref[:, 0], q_ref[:, 1], marker="*", s=170, c="C3",
               edgecolors="black", linewidths=0.6, zorder=3,
               label=r"quadrature pts $\xi_q$")
    for i, (x, y) in enumerate(q_ref):
        ax.annotate(f" q{i}", (x, y), fontsize=10, color="C3", zorder=4)
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.15, 1.2)
    ax.set_aspect("equal")
    ax.set_title(r"Reference element $\hat{K}$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\eta$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)

    # --- physical panel ---
    ax = axes[1]
    tri = plt.Polygon(phys_vertices, fill=True, facecolor="#FFF3E8",
                      edgecolor="0.2", lw=1.6, zorder=1)
    ax.add_patch(tri)
    ax.scatter(phys_vertices[:, 0], phys_vertices[:, 1], s=70, c="C0",
               edgecolors="black", linewidths=0.6, zorder=3, label="basis nodes")
    for i, (x, y) in enumerate(phys_vertices):
        ax.annotate(f" v{i}", (x, y), fontsize=12, zorder=4)
    ax.scatter(q_phys[:, 0], q_phys[:, 1], marker="*", s=170, c="C3",
               edgecolors="black", linewidths=0.6, zorder=3,
               label=r"mapped pts $\mathbf{x}_q$")
    for i, (x, y) in enumerate(q_phys):
        ax.annotate(f" q{i}", (x, y), fontsize=10, color="C3", zorder=4)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.2, 3.0)
    ax.set_aspect("equal")
    ax.set_title(r"Physical element $K$ —  $\mathbf{x} = \mathbf{F}_K(\boldsymbol{\xi})$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.suptitle(
        r"Reference $\to$ physical mapping for a single triangle "
        r"(order-2 quadrature, 3 Gauss points)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "ref_to_physical.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'ref_to_physical.png'}")


def shape_functions_p1_triangle():
    """Three P1 hat functions over the reference triangle, side-by-side."""

    fns = Triangle.get_basis_fns(order=1)
    nodes = Triangle.get_basis(order=1).numpy()

    n = 70
    s = np.linspace(0, 1, n)
    xi, eta = np.meshgrid(s, s)
    mask = xi + eta <= 1.0
    xy = np.stack([xi[mask], eta[mask]], axis=-1)
    xy_t = torch.tensor(xy, dtype=torch.float32)
    values = fns.map(xy_t).numpy()  # [n_pts, 3]

    fig = plt.figure(figsize=(13, 4.2))
    titles = [r"$\phi_0 = 1 - \xi - \eta$", r"$\phi_1 = \xi$", r"$\phi_2 = \eta$"]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        Z = np.full_like(xi, np.nan)
        Z[mask] = values[:, i]
        ax.plot_surface(xi, eta, Z, cmap="viridis", edgecolor="none", alpha=0.95)
        ax.scatter(nodes[:, 0], nodes[:, 1], np.eye(3)[i],
                   c="red", s=60, edgecolors="black", linewidths=0.6, zorder=10)
        ax.set_title(titles[i])
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        ax.set_zlabel(r"$\phi$")
        ax.set_zlim(-0.05, 1.05)
        ax.view_init(elev=22, azim=-65)
    fig.suptitle("Linear (P1) triangle shape functions on the reference element")
    fig.tight_layout()
    fig.savefig(OUT / "p1_shape_functions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'p1_shape_functions.png'}")


if __name__ == "__main__":
    ref_to_physical_triangle()
    shape_functions_p1_triangle()

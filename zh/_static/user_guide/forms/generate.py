"""Figures for User Guide / Forms.

Run from repo root after activating the environment:
    source ~/venvs/tensorgalerkin/bin/activate
    python docs/source/_static/user_guide/forms/generate.py
"""
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from tensormesh import (
    Mesh,
    LaplaceElementAssembler,
    LinearElasticityElementAssembler,
    MassElementAssembler,
    NodeAssembler,
    FacetAssembler,
)

OUT = Path(__file__).parent


def sparsity_scalar_vs_vector():
    """Side-by-side spy plots: scalar Laplace vs 2D linear elasticity."""

    mesh = Mesh.gen_rectangle(chara_length=0.15)
    K_scalar = LaplaceElementAssembler.from_mesh(mesh)()
    K_vector = LinearElasticityElementAssembler.from_mesh(mesh, E=1.0, nu=0.3)()

    Ks = K_scalar.to_dense().numpy()
    Kv = K_vector.to_dense().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.spy(Ks, markersize=3, color="C0")
    ax.set_title(
        rf"$K_{{ij}} = \int_\Omega \nabla\phi_i\cdot\nabla\phi_j\,\mathrm{{d}}\Omega$"
        "\n"
        f"Laplace, scalar — shape {Ks.shape}, nnz {int((Ks != 0).sum())}"
    )
    ax.set_xlabel("col"); ax.set_ylabel("row")

    ax = axes[1]
    ax.spy(Kv, markersize=3, color="C3")
    ax.set_title(
        r"$K_{ij}^{\alpha\beta} = \int_\Omega \sigma_{\alpha\beta}(\phi_i):\varepsilon(\phi_j)\,\mathrm{d}\Omega$"
        "\n"
        f"2D elasticity, vector — shape {Kv.shape}, nnz {int((Kv != 0).sum())}"
    )
    ax.set_xlabel("col"); ax.set_ylabel("row")

    fig.suptitle(
        "Assembled stiffness sparsity: scalar form (left) vs vector form "
        "(right) on the same triangulated unit square"
    )
    fig.tight_layout()
    fig.savefig(OUT / "sparsity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'sparsity.png'}")


def l2_projection_demo():
    """L2 projection of f(x,y) = sin(πx) sin(πy) onto P1 triangles.

    b_i = ∫ f(x) φ_i dΩ with f evaluated at quadrature points (so this is a
    real L2 projection, not just a round-trip of nodal samples). M is SPD so
    no Dirichlet BC is needed.
    """

    mesh = Mesh.gen_rectangle(chara_length=0.05)
    points = mesh.points.double()

    M = MassElementAssembler.from_mesh(mesh)()

    class L2Source(NodeAssembler):
        # `x` is auto-supplied at every quadrature point as [dim]; `v` is
        # the (scalar) shape value of the test basis at that point.
        def forward(self, x, v):
            return torch.sin(math.pi * x[0]) * torch.sin(math.pi * x[1]) * v

    b = L2Source.from_mesh(mesh)(points)
    u = M.solve(b)

    f_nodal = torch.sin(math.pi * points[:, 0]) * torch.sin(math.pi * points[:, 1])
    err_l2 = ((u - f_nodal) ** 2).sum().sqrt().item() / (f_nodal ** 2).sum().sqrt().item()

    pts_np = points.numpy()
    tris = mesh.cells["triangle"].numpy()
    triang = Triangulation(pts_np[:, 0], pts_np[:, 1], tris)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax = axes[0]
    tcf = ax.tricontourf(triang, b.numpy(), levels=20, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.triplot(triang, color="0.8", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_title(
        r"NodeAssembler output: $b_i = \int_\Omega f(\mathbf{x})\,\phi_i\,\mathrm{d}\Omega$"
        f"\n(load vector, shape {tuple(b.shape)})"
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[1]
    tcf = ax.tricontourf(triang, u.numpy(), levels=20, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.triplot(triang, color="0.8", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_title(
        r"$L^2$ projection $u_h = M^{-1} b$"
        f"\n($\\Vert u_h - f \\Vert_2 / \\Vert f \\Vert_2 = {err_l2:.2e}$ at nodes)"
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle(
        r"Forms in action: assemble $M$ (mass) and $b$ (load), solve $M u_h = b$ "
        r"to project $f = \sin\pi x\,\sin\pi y$ into the FE space"
    )
    fig.savefig(OUT / "l2_projection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'l2_projection.png'} (relative L2 nodal error = {err_l2:.4e})")


def facet_integral_demo():
    """Per-node boundary-integral weights on the unit square."""

    mesh = Mesh.gen_rectangle(chara_length=0.1)
    points = mesh.points.numpy()

    class IntegrateOne(FacetAssembler):
        def forward(self, v):
            return v

    b = IntegrateOne.from_mesh(mesh)().numpy()
    perimeter = b.sum()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    tris = mesh.cells["triangle"].numpy()
    triang = Triangulation(points[:, 0], points[:, 1], tris)
    ax.triplot(triang, color="0.85", linewidth=0.3)

    mask = b > 1e-12
    sc = ax.scatter(
        points[mask, 0], points[mask, 1], c=b[mask],
        s=60, cmap="plasma", edgecolors="black", linewidths=0.4, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label(r"$b_i = \int_{\partial\Omega} \phi_i\,\mathrm{d}S$")

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "FacetAssembler output: per-node boundary-integral weights\n"
        rf"$\sum_i b_i = {perimeter:.6f}$  (exact perimeter of $[0,1]^2$ is 4)"
    )
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUT / "facet_integral.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'facet_integral.png'} (sum = {perimeter:.10f})")


if __name__ == "__main__":
    sparsity_scalar_vs_vector()
    l2_projection_demo()
    facet_integral_demo()

"""Generate the static figures used in docs/source/user_guide/differentiability.rst.

Run from the repo root, after activating the tensorgalerkin venv:

    python docs/scripts/differentiability_figures.py

Three figures, two worked autograd examples:

* ``param_id_loss.png`` -- loss curve and observation mismatch for the
  coefficient-field identification example: recover ``kappa(x)`` at
  every mesh node by gradient descent through a Poisson FEM solve.
* ``param_id_fields.png`` -- three-panel triangulation plot of the
  ground-truth, recovered, and error coefficient fields after the
  Adam loop above.
* ``topology_thermal.png`` -- four-snapshot evolution of a 2D thermal
  topology-optimization driver: minimize compliance over per-element
  density ``rho``, with SIMP penalization and the
  :class:`tensormesh.optimizer.OCOptimizer` Optimality Criteria update.
"""

import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri

import torch

from tensormesh import (
    Mesh,
    ElementAssembler,
    NodeAssembler,
    Condenser,
    LaplaceElementAssembler,
    MassElementAssembler,
)
from tensormesh.optimizer import OCOptimizer


warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="float64 recommended")


OUT = os.path.join(os.path.dirname(__file__), "..", "source",
                   "_static", "user_guide", "differentiability")
os.makedirs(OUT, exist_ok=True)

torch.set_default_dtype(torch.float64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _triangles_of(mesh):
    cells = mesh.cells
    keys = list(cells.keys()) if hasattr(cells, "keys") else list(cells)
    for preferred in ("triangle", "tri"):
        if preferred in keys:
            return cells[preferred].cpu().numpy()
    return cells[keys[0]].cpu().numpy()


# ---------------------------------------------------------------------------
# Example 1: parameter identification.
#
# Forward map:  given kappa(x) at every mesh node, solve
#   -div( kappa grad u ) = f   in (0,1)^2,  u = 0 on the boundary.
# Inverse: recover kappa from a noisy observation u_obs, by Adam on the
# nodal kappa with a small smoothness penalty.
# ---------------------------------------------------------------------------

class WeightedLaplace(ElementAssembler):
    def forward(self, gradu, gradv, kappa):
        return kappa * (gradu @ gradv)


class Source(NodeAssembler):
    def forward(self, v, f):
        return v * f


def fig_parameter_identification():
    print("[fig 1+2] parameter identification...")
    mesh = Mesh.gen_rectangle(chara_length=0.04).to(DEVICE)
    cond = Condenser(mesh.boundary_mask)
    print(f"        mesh: {mesh.n_points} nodes on {DEVICE}")

    # Ground-truth coefficient field: smooth, bounded, strictly positive.
    pts = mesh.points
    x, y = pts[:, 0], pts[:, 1]
    kappa_true = 1.0 + 0.6 * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)
    f_vals = torch.ones(mesh.n_points, dtype=torch.float64, device=DEVICE)

    def fem_solve(kappa, f_vals):
        K = WeightedLaplace.from_mesh(mesh)(point_data={"kappa": kappa}).double()
        b = Source.from_mesh(mesh)(point_data={"f": f_vals}).double()
        K_, b_ = cond(K, b)
        return cond.recover(K_.solve(b_))

    # Synthetic observation.
    with torch.no_grad():
        u_obs = fem_solve(kappa_true, f_vals)
    u_norm = u_obs.abs().max().item()

    # Adam loop. Parameterise kappa as 1 + tanh(theta) so kappa stays in
    # (0, 2); this keeps the FEM matrix SPD throughout and shrinks the
    # null space of the inverse problem.
    torch.manual_seed(0)
    theta = torch.zeros(mesh.n_points, dtype=torch.float64,
                        device=DEVICE, requires_grad=True)
    optim = torch.optim.Adam([theta], lr=3e-2)
    n_iter = 400

    losses, u_errs, k_errs = [], [], []
    for step in range(n_iter):
        optim.zero_grad()
        kappa = 1.0 + torch.tanh(theta)
        u = fem_solve(kappa, f_vals)
        loss_data = ((u - u_obs) ** 2).sum()
        # mild smoothness prior, helps with the high-frequency null modes.
        # (theta lives at nodes; finite-difference roughness is approximated
        # by squared difference along incident edges.)
        loss = loss_data
        loss.backward()
        optim.step()
        with torch.no_grad():
            losses.append(loss_data.item())
            u_errs.append(((u - u_obs).abs().max() / u_norm).item())
            k_errs.append(((kappa - kappa_true).abs().max()
                           / kappa_true.abs().max()).item())
        if step % 50 == 0 or step == n_iter - 1:
            print(f"        step {step:4d}  loss={loss_data.item():.3e}  "
                  f"u_rel={u_errs[-1]:.3e}  k_rel={k_errs[-1]:.3e}")

    with torch.no_grad():
        kappa_final = 1.0 + torch.tanh(theta)

    # ---- Figure A: loss + observation mismatch over iterations.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    iters = np.arange(1, n_iter + 1)
    ax.semilogy(iters, np.array(losses), color="#c0392b", linewidth=2,
                label=r"data loss  $\|u_\theta - u_{\rm obs}\|_2^2$")
    ax2 = ax.twinx()
    ax2.semilogy(iters, np.array(u_errs), color="#2980b9", linewidth=2,
                 linestyle="--",
                 label=r"$\|u_\theta - u_{\rm obs}\|_\infty\,/\,\|u_{\rm obs}\|_\infty$")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("data loss", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylabel("relative max-norm error in $u$", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax.set_title("Coefficient identification: gradient flow through FEM solve")
    ax.grid(True, which="both", alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT, "param_id_loss.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")

    # ---- Figure B: three-panel kappa fields.
    triang = mtri.Triangulation(pts.cpu().numpy()[:, 0],
                                pts.cpu().numpy()[:, 1],
                                _triangles_of(mesh))

    kappa_true_np = kappa_true.cpu().numpy()
    kappa_rec_np = kappa_final.cpu().numpy()
    err_np = np.abs(kappa_rec_np - kappa_true_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    vmin, vmax = 0.4, 1.6
    levels = np.linspace(vmin, vmax, 21)
    for ax, data, title in zip(
        axes,
        [kappa_true_np, kappa_rec_np, err_np],
        [r"true $\kappa(x)$", r"recovered $\kappa(x)$",
         r"$|\kappa_{\rm rec} - \kappa_{\rm true}|$"],
    ):
        if "error" in title.lower() or "|" in title:
            cs = ax.tricontourf(triang, data, levels=21, cmap="Reds")
        else:
            cs = ax.tricontourf(triang, data, levels=levels,
                                cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, shrink=0.82)
    fig.suptitle(
        f"After {n_iter} Adam steps:  "
        f"data loss {losses[-1]:.1e},  "
        f"max-rel error in $u$  {u_errs[-1]:.1e},  "
        f"max-rel error in $\\kappa$  {k_errs[-1]:.1e}",
        fontsize=11, y=1.02,
    )
    out = os.path.join(OUT, "param_id_fields.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Example 2: thermal-compliance topology optimization on a square plate.
#
#   minimize       J(rho) = b^T u
#   subject to     K(rho) u = b,   mean(rho) <= V,   rho_min <= rho <= 1.
# Material law: kappa(rho) = kappa_min + (1 - kappa_min) * rho^p   (SIMP).
# rho lives per element; passed through ``element_data``.
# ---------------------------------------------------------------------------

class SIMPLaplace(ElementAssembler):
    def forward(self, gradu, gradv, kappa):
        # kappa has been interpolated to quadrature points; broadcast in.
        return kappa * (gradu @ gradv)


def fig_topology_thermal():
    print("[fig 3] topology optimization...")
    mesh = Mesh.gen_rectangle(chara_length=0.025).to(DEVICE)
    cond = Condenser(mesh.boundary_mask)

    # Element type and per-element density. gen_rectangle yields both
    # ``"line"`` boundary segments and ``"triangle"`` body elements; we only
    # need a density per body element.
    etype = "triangle"
    elements = mesh.cells[etype]
    n_elem = elements.shape[0]
    print(f"        mesh: {mesh.n_points} nodes, {n_elem} elements")

    # Volume fraction & SIMP params.
    V = 0.4
    p_simp = 3.0
    kmin = 1e-3

    # Initialize rho uniformly at V.
    rho = torch.full((n_elem,), V, dtype=torch.float64,
                     device=DEVICE, requires_grad=True)
    optim = OCOptimizer(rho, vf=V, move_limit=0.15)

    # Source: localised hot pad in the centre. Use NodeAssembler against a
    # gaussian source field, dirichlet u=0 on boundary -> "cold sink at edges".
    pts = mesh.points
    x, y = pts[:, 0], pts[:, 1]
    sigma = 0.08
    f_field = torch.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * sigma ** 2))

    def fem_solve(rho):
        kappa_elem = kmin + (1.0 - kmin) * rho ** p_simp     # [n_elem]
        K = SIMPLaplace.from_mesh(mesh)(
            element_data={"kappa": kappa_elem}
        ).double()
        b = Source.from_mesh(mesh)(point_data={"f": f_field}).double()
        K_, b_ = cond(K, b)
        u = cond.recover(K_.solve(b_))
        return u, b

    n_iter = 60
    snap_iters = [0, 2, 5, n_iter - 1]
    snapshots = {}
    history = []
    t_start = time.time()
    for step in range(n_iter):
        optim.zero_grad()
        u, b = fem_solve(rho)
        compliance = (b * u).sum()
        compliance.backward()
        info = optim.step()
        vol = info["volume"]
        history.append((compliance.item(), vol))
        if step in snap_iters:
            snapshots[step] = rho.detach().clone().cpu().numpy()
        if step % 10 == 0 or step == n_iter - 1:
            print(f"        step {step:3d}  C={compliance.item():.4e}  "
                  f"V={vol:.3f}  lambda={info['lambda']:.3e}")
    print(f"        OC loop took {time.time() - t_start:.1f}s")

    # ---- Plot.
    triang = mtri.Triangulation(pts.cpu().numpy()[:, 0],
                                pts.cpu().numpy()[:, 1],
                                elements.cpu().numpy())

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), constrained_layout=True)
    for ax, k in zip(axes, snap_iters):
        cs = ax.tripcolor(triang, facecolors=snapshots[k],
                          cmap="gray_r", vmin=0, vmax=1, shading="flat")
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        comp_k, vol_k = history[k]
        ax.set_title(f"iter {k}\n$C={comp_k:.3e}$, $V={vol_k:.2f}$",
                     fontsize=10)
    cbar = fig.colorbar(cs, ax=axes, shrink=0.82, ticks=[0, 0.5, 1.0])
    cbar.set_label(r"density $\rho$")
    fig.suptitle("Topology optimization: thermal compliance "
                 r"$\downarrow$ under volume constraint",
                 fontsize=12, y=1.04)
    out = os.path.join(OUT, "topology_thermal.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Optional: gradient-check against finite differences (small problem).
# Sanity result, not committed as a figure; printed to console.
# ---------------------------------------------------------------------------

def gradient_check():
    print("[grad check] vs finite differences...")
    mesh = Mesh.gen_rectangle(chara_length=0.2).to(DEVICE)
    cond = Condenser(mesh.boundary_mask)
    pts = mesh.points
    f_vals = torch.ones(mesh.n_points, dtype=torch.float64, device=DEVICE)

    def loss_of(theta):
        kappa = 1.0 + torch.tanh(theta)
        K = WeightedLaplace.from_mesh(mesh)(point_data={"kappa": kappa}).double()
        b = Source.from_mesh(mesh)(point_data={"f": f_vals}).double()
        K_, b_ = cond(K, b)
        u = cond.recover(K_.solve(b_))
        return (u ** 2).sum()

    torch.manual_seed(0)
    theta = torch.randn(mesh.n_points, dtype=torch.float64,
                        device=DEVICE, requires_grad=True)
    L = loss_of(theta)
    L.backward()
    g_auto = theta.grad.detach().clone()

    # Central differences on three random coords.
    eps = 1e-6
    idx = torch.randperm(mesh.n_points)[:3]
    rows = []
    for i in idx.tolist():
        theta_p = theta.detach().clone(); theta_p[i] += eps
        theta_m = theta.detach().clone(); theta_m[i] -= eps
        with torch.no_grad():
            g_fd = (loss_of(theta_p) - loss_of(theta_m)) / (2 * eps)
        rows.append((i, g_auto[i].item(), g_fd.item(),
                     abs(g_auto[i].item() - g_fd.item())))
    print("        node    autograd          fd                  |diff|")
    for r in rows:
        print(f"        {r[0]:4d}    {r[1]:+.6e}   {r[2]:+.6e}   {r[3]:.2e}")


if __name__ == "__main__":
    gradient_check()
    fig_parameter_identification()
    fig_topology_thermal()
    print("done.")

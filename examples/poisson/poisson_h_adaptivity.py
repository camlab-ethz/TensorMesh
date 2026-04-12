"""
H-Adaptivity for the L-Domain Laplace Problem.

The re-entrant corner of an L-shaped domain produces a gradient singularity
(u ~ r^{2/3}), making it the canonical test case for adaptive mesh refinement.
Uniform refinement yields sub-optimal convergence; h-adaptivity recovers the
optimal rate by concentrating elements near the corner.

    -Δu = 0   in Ω  (L-shaped domain)
     u  = g   on ∂Ω (Dirichlet, from the exact singular solution)

Algorithm:  Solve -> Estimate error -> Dörfler mark -> Gmsh remesh -> repeat

Outputs:
    poisson_h_adaptivity.png — convergence (adaptive vs uniform) + final mesh

Requirements: gmsh >= 4.8 (for setSizeCallback)
"""

import sys
sys.path.append("../..")

import os
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

import gmsh

from tensormesh import Mesh, LaplaceElementAssembler, MassElementAssembler, Condenser


# ============================================================================
# Exact singular solution on the L-domain
# ============================================================================

CORNER = (0.5, 0.5)          # re-entrant corner of the default Mesh.gen_L()
ALPHA = 2.0 / 3.0            # singularity exponent  (π / ω,  ω = 3π/2)


def singular_solution(points, corner=CORNER):
    """u = r^{2/3} sin(2θ/3) in polar coords centred at the re-entrant corner.

    θ is measured counter-clockwise from the *upward* edge of the notch
    (the edge x = corner_x, y > corner_y) into the interior of the domain.
    """
    dx = points[:, 0] - corner[0]
    dy = points[:, 1] - corner[1]
    r = torch.sqrt(dx ** 2 + dy ** 2)
    theta_std = torch.atan2(dy, dx)                       # ∈ (−π, π]
    phi = (theta_std - torch.pi / 2) % (2 * torch.pi)    # 0 at upward edge
    return r.pow(ALPHA) * torch.sin(ALPHA * phi)


# ============================================================================
# FEM solve  (Laplace with non-zero Dirichlet BC)
# ============================================================================

def solve_laplace(mesh):
    """Solve -Δu = 0 with u = g on ∂Ω where g is the singular solution."""
    K_asm = LaplaceElementAssembler.from_mesh(mesh)
    K = K_asm(mesh.points)

    g = singular_solution(mesh.points)
    condenser = Condenser(mesh.boundary_mask, g)          # non-zero Dirichlet
    K_, f_ = condenser(K, torch.zeros(mesh.n_points))
    u_ = K_.solve(f_)
    return condenser.recover(u_)


# ============================================================================
# Error estimation
# ============================================================================

def global_l2_error(mesh, u_fem, u_exact):
    """Mass-weighted relative L2 error."""
    M = MassElementAssembler.from_mesh(mesh)(mesh.points)
    e = u_fem - u_exact
    err2 = torch.abs((e * (M @ e)).sum())
    ref2 = torch.abs((u_exact * (M @ u_exact)).sum())
    return (err2.sqrt() / (ref2.sqrt() + 1e-30)).item()


def element_error_and_sizes(mesh, u_fem, u_exact):
    """Per-element error indicator (RMS nodal error) and diameter.

    Returns  centroids [E,2],  eta [E],  h [E]   as numpy arrays.
    """
    err = (u_fem - u_exact).abs()
    cc, ee, hh = [], [], []
    for _, cells in mesh.cells.items():
        coords = mesh.points[cells]
        cc.append(coords.mean(dim=1))
        ee.append(err[cells].pow(2).mean(dim=1).sqrt())
        diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        hh.append(diffs.norm(dim=-1).amax(dim=(1, 2)))
    return (torch.cat(cc).numpy(), torch.cat(ee).numpy(), torch.cat(hh).numpy())


# ============================================================================
# Dörfler marking  →  desired element sizes
# ============================================================================

def doerfler_sizes(h, eta, theta=0.5,
                   refine_factor=0.5, coarsen_factor=1.3,
                   h_min=0.002, h_max=0.15):
    """Mark elements whose error > θ · max(η)  and halve their size."""
    eta_max = eta.max()
    h_new = h.copy()
    h_new[eta > theta * eta_max] *= refine_factor
    h_new[eta < 0.05 * eta_max] *= coarsen_factor
    return np.clip(h_new, h_min, h_max)


# ============================================================================
# Gmsh adaptive remeshing of the L-domain
# ============================================================================

def remesh_L(centroids, sizes, h_min=0.002, h_max=0.15):
    """Re-generate L-shaped triangle mesh with a spatially varying size field."""
    tree = cKDTree(centroids)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L_adaptive")

    # geometry: [0,1]^2  minus  [0.5,1] × [0.5,1]
    r_out = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    r_cut = gmsh.model.occ.addRectangle(0.5, 0.5, 0, 0.5, 0.5)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.cut([(2, r_out)], [(2, r_cut)])
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.getEntities(2)
    s = surfs[0][1]
    bnd = gmsh.model.getBoundary([(2, s)], oriented=False)
    gmsh.model.addPhysicalGroup(1, [l[1] for l in bnd], name="boundary")
    gmsh.model.addPhysicalGroup(2, [s], name="domain")

    def size_cb(dim, tag, x, y, z, lc):
        _, idx = tree.query([x, y])
        return max(h_min, min(h_max, float(sizes[idx])))

    gmsh.model.mesh.setSizeCallback(size_cb)
    gmsh.model.mesh.generate(2)

    tmp = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmp.close()
    gmsh.write(tmp.name)
    gmsh.finalize()

    mesh = Mesh.from_file(tmp.name, reorder=True)
    os.unlink(tmp.name)

    # boundary detection for the L-domain
    pts = mesh.points
    eps = 1e-10
    is_boundary = (
        (pts[:, 0] < eps)
        | (pts[:, 0] > 1 - eps)
        | (pts[:, 1] < eps)
        | (pts[:, 1] > 1 - eps)
        | ((pts[:, 0] - 0.5).abs() < eps) & (pts[:, 1] > 0.5 - eps)
        | ((pts[:, 1] - 0.5).abs() < eps) & (pts[:, 0] > 0.5 - eps)
    )
    mesh.register_point_data("is_boundary", is_boundary)
    return mesh


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    h0 = 0.08
    max_levels = 10
    target_error = 5e-4

    # ---- adaptive refinement ------------------------------------------------
    mesh = Mesh.gen_L(chara_length=h0, element_type="tri")
    adapt_dofs, adapt_errs = [], []
    plot_mesh, plot_u = mesh, None

    print(f"{'Level':>5} {'DOFs':>8} {'Elems':>8} {'Rel L2':>14}")
    print("-" * 40)

    for level in range(max_levels):
        u_fem = solve_laplace(mesh)
        u_exact = singular_solution(mesh.points)
        rel_err = global_l2_error(mesh, u_fem, u_exact)

        adapt_dofs.append(mesh.n_points)
        adapt_errs.append(rel_err)
        plot_mesh, plot_u = mesh, u_fem

        print(f"{level:>5} {mesh.n_points:>8} {mesh.n_elements:>8} {rel_err:>14.4e}")
        if rel_err < target_error:
            print(f"\nConverged at level {level}.")
            break

        centroids, eta, h = element_error_and_sizes(mesh, u_fem, u_exact)
        h_new = doerfler_sizes(h, eta, theta=0.5, h_min=0.002, h_max=h0)
        mesh = remesh_L(centroids, h_new, h_min=0.002, h_max=h0)

    # ---- uniform refinement for comparison ----------------------------------
    print("\nUniform refinement:")
    uniform_hs = [0.08, 0.04, 0.02, 0.013, 0.01, 0.007]
    uni_dofs, uni_errs = [], []
    for uh in uniform_hs:
        m = Mesh.gen_L(chara_length=uh, element_type="tri")
        u = solve_laplace(m)
        ue = singular_solution(m.points)
        err = global_l2_error(m, u, ue)
        uni_dofs.append(m.n_points)
        uni_errs.append(err)
        print(f"  h={uh:.3f}  DOFs={m.n_points:>8}  Rel L2={err:.4e}")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) convergence
    ax = axes[0]
    ax.loglog(adapt_dofs, adapt_errs, "o-", color="#E74C3C", lw=2, ms=7, label="Adaptive")
    ax.loglog(uni_dofs, uni_errs, "s--", color="#3498DB", lw=2, ms=7, label="Uniform")
    d = np.array(uni_dofs, dtype=float)
    c = uni_errs[0] * d[0]
    ax.loglog(d, c / d, ":", color="gray", lw=1, label=r"$\mathcal{O}(N^{-1})$")
    ax.set_xlabel("Number of DOFs")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_title("Convergence: L-domain singularity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # (b) final adaptive mesh
    ax = axes[1]
    for _, cells in plot_mesh.cells.items():
        verts = plot_mesh.points.numpy()[cells.numpy()]
        poly = PolyCollection(verts, edgecolors="black", facecolors="white", lw=0.2)
        ax.add_collection(poly)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(f"Adaptive mesh ({plot_mesh.n_points} DOFs)")

    # (c) FEM solution
    ax = axes[2]
    for _, cells in plot_mesh.cells.items():
        c_np = cells.numpy()
        verts = plot_mesh.points.numpy()[c_np]
        vals = plot_u.detach().numpy()[c_np].mean(axis=1)
        poly = PolyCollection(verts, array=vals, cmap="RdBu_r", edgecolors="none")
        ax.add_collection(poly)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title("FEM solution")
    fig.colorbar(poly, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__) or ".", "poisson_h_adaptivity.png")
    fig.savefig(out, dpi=200)
    print(f"\nSaved: {out}")

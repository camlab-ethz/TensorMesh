import os
import sys
sys.path.append("../..")

import numpy as np
import torch

from tensormesh import Mesh, LaplaceElementAssembler, MassElementAssembler, NodeAssembler, Condenser
from tensormesh.visualization import setup_headless


class LoadAssembler(NodeAssembler):
    """Assemble RHS f_i = ∫ f v_i dx."""

    def forward(self, v, f):
        return v * f


def main():
    torch.manual_seed(0)
    setup_headless()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------
    # Mesh (unit cube)
    # --------------------
    # NOTE: Mesh.gen_cube uses gmsh and may generate tetra/hex depending on settings.
    # We keep it as-is but enforce 3D Poisson correctness independent of element type.
    # Smaller => denser mesh (clearer visualization, slower mesh generation/solve)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chara_length = 0.05
    order = 1
    mesh = Mesh.gen_cube(chara_length=chara_length, order=order).to(device=device).double()

    x = mesh.points  # [n_points, 3]

    # --------------------
    # Analytic solution (zero Dirichlet on boundary)
    # u = sin(pi x) sin(pi y) sin(pi z)
    # -Δu = 3*pi^2 * u
    # --------------------
    pi = torch.pi
    u_exact = torch.sin(pi * x[:, 0]) * torch.sin(pi * x[:, 1]) * torch.sin(pi * x[:, 2])
    f = 3.0 * (pi ** 2) * u_exact

    # Dirichlet boundary: u = 0 on boundary
    # Mesh has boundary_mask already for gmsh meshes
    if hasattr(mesh, "boundary_mask"):
        boundary_mask = mesh.boundary_mask
    else:
        # Fallback: infer boundary nodes from coordinates
        eps = 1e-12
        boundary_mask = (
            (x[:, 0] < eps)
            | (x[:, 0] > 1 - eps)
            | (x[:, 1] < eps)
            | (x[:, 1] > 1 - eps)
            | (x[:, 2] < eps)
            | (x[:, 2] > 1 - eps)
        )

    dirichlet_value = torch.zeros_like(x[:, 0])
    condenser = Condenser(boundary_mask, dirichlet_value)

    # --------------------
    # Assemble system
    # --------------------
    K_asm = LaplaceElementAssembler.from_mesh(mesh)
    K = K_asm(mesh.points)

    F_asm = LoadAssembler.from_mesh(mesh)
    rhs = F_asm(mesh.points, point_data={"f": f})

    Kc, rhsc = condenser(K, rhs)
    uc = Kc.solve(rhsc, verbose=True)
    u = condenser.recover(uc)

    # --------------------
    # Error (mass-weighted L2)
    # --------------------
    e = u - u_exact
    M_asm = MassElementAssembler.from_mesh(mesh)
    M = M_asm(mesh.points)
    l2_err = torch.sqrt((e * (M @ e)).sum())
    l2_ref = torch.sqrt((u_exact * (M @ u_exact)).sum())
    rel_l2 = (l2_err / (l2_ref + 1e-30)).item()

    print(f"[poisson_3d] n_points={mesh.points.shape[0]}  rel_L2={rel_l2:.3e}")

    # --------------------
    # Save outputs (in this folder)
    # --------------------
    mesh.register_point_data("u_fem", u)
    mesh.register_point_data("u_exact", u_exact)
    mesh.register_point_data("error", e)
    mesh.register_point_data("f", f)
    # gen_cube already provides "is_boundary" in point_data for gmsh meshes; don't overwrite it.
    # NOTE: mesh.point_data is a BufferDict; use .keys() for safe membership check.
    if not hasattr(mesh, "point_data") or ("is_boundary" not in mesh.point_data.keys()):
        mesh.register_point_data("is_boundary", boundary_mask.to(torch.int32))

    out_vtu = os.path.join(out_dir, "poisson_3d.vtu")
    mesh.save(out_vtu)

    # Optional: quick slice + 3D screenshots for visual sanity (headless-friendly)
    try:
        import pyvista as pv

        grid = pv.read(out_vtu)

        # True 3D render: keep HALF of the volume (not just a slice).
        # For this analytic u, the boundary is ~0, so we clip at x=0.5 and color the kept-half surface,
        # which includes the cut-face (interior) where values are non-zero.
        p3 = pv.Plotter(off_screen=True, window_size=(1400, 900))
        origin = (0.5, 0.5, 0.5)
        # Keep the OTHER half: x <= 0.5 (so the removed side is x > 0.5)
        kept = grid.clip(normal="x", origin=origin, invert=True)
        kept_surf = kept.extract_surface()
        surf_full = grid.extract_surface()

        # Contrast limits from kept half (more informative than full boundary)
        umin = float(kept_surf["u_fem"].min())
        umax = float(kept_surf["u_fem"].max())

        # Context: faint full cube wireframe
        p3.add_mesh(surf_full, color="gray", opacity=0.40, style="wireframe", line_width=1)

        # Main: kept-half colored surface (includes cut-face)
        p3.add_mesh(
            kept_surf,
            scalars="u_fem",
            cmap="turbo",
            clim=(umin, umax),
            opacity=1.0,
            show_scalar_bar=True,
            smooth_shading=True,
        )

        p3.add_title(f"3D Poisson (half-domain cut x=0.5, view from cut side)")
        # View FROM the removed side (x > 0.5) looking towards the cut face
        p3.camera_position = [(3,3,3), (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)]
        # p3.camera.zoom(1.2)

        out_png3d = os.path.join(out_dir, "poisson_3d_half_from_cut.png")
        p3.screenshot(out_png3d)
        p3.close()
        print(f"[poisson_3d] wrote: {out_png3d}")
    except Exception as ex:
        print(f"[poisson_3d] skip pyvista screenshot: {type(ex).__name__}: {ex}")

    print(f"[poisson_3d] wrote: {out_vtu}")


if __name__ == "__main__":
    main()



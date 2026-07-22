"""3D lid-driven cavity — steady incompressible Navier-Stokes.

The 3D extension of ``cavity.py``: Taylor-Hood P2-P1 on tetrahedra. The
scalar weak-form integrand in ``forward`` is dimension-generic — the only
changes from 2D are ``components=3``, the mesh, and the volumetric output.
The gmsh mesh is linear; the quadratic velocity space is generated
**topologically** (one extra DOF per unique edge of the tet mesh), so the
same script pattern works without an order-2 mesh.
"""
import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.visualization import setup_headless


class NavierStokesAssembler(MixedElementAssembler):
    r"""Picard-linearized steady Navier-Stokes (identical to ``cavity.py``):

    .. math::

        \rho\,(w\cdot\nabla)u\cdot v + \mu\,\nabla u : \nabla v
        - p\,\nabla\cdot v - q\,\nabla\cdot u.

    ``gradu`` is the ``[3, 3]`` velocity Jacobian here, but the integrand
    reads the dimension off its operands, so the expression is unchanged.
    """

    fields = [
        Field(trial="u", test="v", order=2, components=3),
        Field(trial="p", test="q", order=1),
    ]

    def __post_init__(self, rho=1.0, mu=0.01):
        self.rho = rho
        self.mu = mu

    def forward(self, gradu, p, v, gradv, q, w):
        convection = self.rho * (gradu @ w).dot(v)
        diffusion = self.mu * (gradu * gradv).sum()
        return convection + diffusion \
            - p * gradv.diagonal().sum() \
            - q * gradu.diagonal().sum()


def solve_cavity_3d(re=100, chara_length=0.1, max_iter=25, tol=1e-4):
    setup_headless()
    print(f"Solving 3D lid-driven cavity at Re={re}, chara_length={chara_length}...")

    # --- Mesh (linear tets; the P2 velocity space is built topologically) ---
    mesh = Mesh.gen_cube(chara_length=chara_length).double()

    assembler = NavierStokesAssembler.from_mesh(mesh, rho=1.0, mu=1.0 / re)
    layout = assembler.layout
    print(f"  Mesh: {mesh.points.shape[0]} P1 points, "
          f"{layout.n_nodes('u')} P2 velocity nodes, {layout.n_dofs} DOFs")

    # --- Boundary conditions on the P2 velocity space ---
    x_u = layout.points("u")
    is_boundary = layout.split(layout.boundary_mask("u"))["u"][:, 0]  # node level
    is_top = is_boundary & (x_u[:, 1] > 1.0 - 1e-6)

    bc_mask = layout.dof_mask("u", node_mask=is_boundary)  # no-slip on every wall
    bc_mask[layout.dof_index("p", int(layout.node_ids("p")[0]))] = True  # pressure pin

    bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
    bc_val[layout.dof_mask("u", node_mask=is_top, component=0)] = 1.0  # moving lid

    condenser = Condenser(bc_mask, bc_val[bc_mask])

    # --- Picard iteration ---
    sol = torch.zeros(layout.n_dofs, dtype=torch.float64)
    sol[bc_mask] = bc_val[bc_mask]

    for i in range(max_iter):
        w = layout.split(sol)["u"]  # previous-iterate velocity on the P2 DOFs
        K = assembler(field_data={"w": ("u", w)})
        f = torch.zeros(layout.n_dofs, dtype=torch.float64)

        K_, f_ = condenser(K, f)
        sol_new = condenser.recover(K_.solve(f_))

        diff = torch.norm(sol_new - sol) / (torch.norm(sol_new) + 1e-8)
        print(f"  Picard {i:2d}: relative diff = {diff:.6e}")
        sol = sol_new
        if diff < tol:
            print("Converged!")
            break

    # --- Post-processing (fields interpolated back to the P1 mesh points) ---
    fields = layout.split(sol)
    velocity = layout.prolong("u", fields["u"])  # [n_points, 3]
    pressure = fields["p"]  # P1 == mesh points
    speed = torch.norm(velocity, dim=1)
    print(f"  Max speed: {speed.max().item():.4f}, "
          f"pressure range: [{pressure.min().item():.4f}, {pressure.max().item():.4f}]")

    # Volumetric output: VTU for ParaView (full field) + a quick PyVista slice.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mesh.register_point_data("speed", speed)
    mesh.register_point_data("pressure", pressure)
    mesh.register_point_data("velocity", velocity)
    vtu_path = os.path.join(out_dir, "cavity_3d.vtu")
    mesh.save(vtu_path)
    print(f"Saved: {vtu_path}")

    try:
        import pyvista as pv

        grid = pv.read(vtu_path)
        slice_z = grid.slice(normal="z", origin=(0.5, 0.5, 0.5))  # mid-depth x-y plane

        p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 700))
        p.subplot(0, 0)
        p.add_mesh(slice_z, scalars="speed", cmap="jet", show_scalar_bar=True)
        p.add_text("Speed (z=0.5 slice)", font_size=10, position="upper_edge")
        p.view_xy()
        p.subplot(0, 1)
        p.add_mesh(slice_z, scalars="pressure", cmap="coolwarm", show_scalar_bar=True)
        p.add_text("Pressure (z=0.5 slice)", font_size=10, position="upper_edge")
        p.view_xy()

        png_path = os.path.join(out_dir, "cavity_3d.png")
        p.screenshot(png_path)
        p.close()
        print(f"Saved: {png_path}")
    except Exception as ex:
        print(f"Skip PyVista visualization: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    solve_cavity_3d(re=100, chara_length=0.1, max_iter=25)

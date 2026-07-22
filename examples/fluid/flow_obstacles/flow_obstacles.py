"""Steady channel flow past multiple obstacles — incompressible Navier-Stokes.

Taylor-Hood P2-P1 mixed discretization with Picard linearization. The
gmsh/MeshGen mesh is linear (order 1); the quadratic velocity space is
generated **topologically** on top of it by ``MixedElementAssembler``
(one extra DOF per unique edge), so no re-meshing at order 2 is needed.
The pair is LBB-stable — no SUPG/PSPG stabilization, no ``tau`` tuning —
and the do-nothing outlet fixes the pressure gauge, so nothing is pinned.
"""
import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, MeshGen, MixedElementAssembler


class NavierStokesAssembler(MixedElementAssembler):
    r"""Picard-linearized steady Navier-Stokes (same weak form as cavity.py):

    .. math::

        \rho\,(w\cdot\nabla)u\cdot v + \mu\,\nabla u : \nabla v
        - p\,\nabla\cdot v - q\,\nabla\cdot u,

    with ``w`` the previous velocity iterate on the P2 field's own DOFs,
    passed via ``field_data``.
    """

    fields = [
        Field(trial="u", test="v", order=2, components=2),
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


OBSTACLES = [
    (0.5, 0.5, 0.10),
    (1.0, 0.3, 0.08),
    (1.0, 0.7, 0.08),
    (1.5, 0.5, 0.12),
    (2.0, 0.4, 0.07),
    (2.0, 0.6, 0.07),
]
LENGTH, HEIGHT = 3.0, 1.0


def build_obstacle_mesh(chara_length):
    gen = MeshGen(chara_length=chara_length)
    gen.add_rectangle(0.0, 0.0, LENGTH, HEIGHT)
    for ox, oy, orad in OBSTACLES:
        gen.remove_circle(ox, oy, orad)
    return gen.gen().double()


def solve_flow_obstacles(re=150, n_grid=40, max_iter=30, tol=5e-4):
    print("Generating mesh with multiple obstacles...")
    mesh = build_obstacle_mesh(chara_length=1.0 / n_grid)

    rho = 1.0
    mu = 1.0 / re

    assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu)
    layout = assembler.layout
    print(f"  Mesh: {mesh.points.shape[0]} P1 points, "
          f"{layout.n_nodes('u')} P2 velocity nodes, {layout.n_dofs} DOFs")

    # --- Boundary conditions on the P2 velocity space ---
    # MeshGen meshes carry no ``is_boundary`` point data; the topological
    # boundary_mask (facet incidence) needs none, and works for the
    # edge-generated P2 nodes too.
    x_u = layout.points("u")
    is_boundary = layout.split(layout.boundary_mask("u"))["u"][:, 0]  # node level
    eps = 1e-6
    is_inlet = is_boundary & (x_u[:, 0] < eps)
    is_outlet = x_u[:, 0] > LENGTH - eps
    no_slip = is_boundary & ~is_inlet & ~is_outlet  # channel walls + obstacles

    bc_mask = layout.dof_mask("u", node_mask=is_inlet | no_slip)
    bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
    y_in = x_u[is_inlet, 1]
    bc_val[layout.dof_mask("u", node_mask=is_inlet, component=0)] = \
        4.0 * y_in * (HEIGHT - y_in) / (HEIGHT * HEIGHT)  # parabolic inlet

    condenser = Condenser(bc_mask, bc_val[bc_mask])

    # --- Picard iteration ---
    sol = torch.zeros(layout.n_dofs, dtype=torch.float64)
    sol[bc_mask] = bc_val[bc_mask]

    print(f"Solving flow past obstacles at Re={re}...")
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

    # --- Post-processing (interpolate the P2 fields back to mesh points) ---
    fields = layout.split(sol)
    speed = layout.prolong("u", fields["u"]).norm(dim=1)  # on mesh points
    pressure = fields["p"]  # P1 == mesh points

    mesh.plot(
        {"Speed": speed, "Pressure": pressure},
        save_path="flow_obstacles.png",
        show_mesh=False,
        cmap="jet",
    )
    print("Done! Results saved to flow_obstacles.png")


if __name__ == "__main__":
    solve_flow_obstacles(re=150, n_grid=40)

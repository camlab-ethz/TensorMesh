"""2D lid-driven cavity — steady incompressible Navier-Stokes.

Taylor-Hood P2-P1 mixed discretization (quadratic velocity, linear
pressure on the same triangulation), linearized by Picard iteration.
The pair is inf-sup (LBB) stable, so no SUPG/PSPG stabilization is
needed, and ``MixedElementAssembler`` handles the block DOF layout —
the weak form below is the whole discretization.
"""
import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler


class NavierStokesAssembler(MixedElementAssembler):
    r"""Picard-linearized steady Navier-Stokes:

    .. math::

        \rho\,(w\cdot\nabla)u\cdot v + \mu\,\nabla u : \nabla v
        - p\,\nabla\cdot v - q\,\nabla\cdot u,

    with ``w`` the previous velocity iterate (passed via ``point_data``).
    Trial fields ``(u, p)`` index columns, test fields ``(v, q)`` rows;
    ``gradu`` is the velocity Jacobian ``[2, 2]``, so the convection term
    is ``gradu @ w`` and the divergence is its trace.
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


def solve_cavity(re=100, n_grid=30, max_iter=20, tol=1e-4):
    print(f"Solving 2D lid-driven cavity at Re={re} on a {n_grid}x{n_grid} grid...")

    # --- Mesh (P2 geometry for the quadratic velocity space) ---
    mesh = Mesh.gen_rectangle(chara_length=1.0 / n_grid, order=2).double()

    # The default quadrature (2 * max field order = 4) is one degree shy of
    # the w.grad(u).v convection term — the standard, harmless choice; pass
    # quadrature_order= to from_mesh to integrate it exactly.
    assembler = NavierStokesAssembler.from_mesh(mesh, rho=1.0, mu=1.0 / re)
    layout = assembler.layout

    # --- Boundary conditions ---
    is_top = mesh.points[:, 1] > 1.0 - 1e-6

    bc_mask = layout.dof_mask("u", mesh.boundary_mask)  # no-slip on every wall
    bc_mask[layout.dof_index("p", int(layout.node_ids("p")[0]))] = True  # pin the pressure constant

    bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
    bc_val[layout.dof_mask("u", is_top, component=0)] = 1.0  # moving lid: u_x = 1 on top

    condenser = Condenser(bc_mask, bc_val[bc_mask])

    # --- Picard iteration ---
    sol = torch.zeros(layout.n_dofs, dtype=torch.float64)
    sol[bc_mask] = bc_val[bc_mask]

    for i in range(max_iter):
        w = layout.split(sol)["u"]  # previous-iterate velocity, [n_points, 2]
        K = assembler(point_data={"w": w})
        f = torch.zeros(layout.n_dofs, dtype=torch.float64)

        K_, f_ = condenser(K, f)
        sol_new = condenser.recover(K_.solve(f_))

        diff = torch.norm(sol_new - sol) / (torch.norm(sol_new) + 1e-8)
        print(f"  Picard {i:2d}: relative diff = {diff:.6e}")
        sol = sol_new
        if diff < tol:
            print("Converged!")
            break

    # --- Post-processing ---
    fields = layout.split(sol)
    speed = torch.norm(fields["u"], dim=1)
    pressure = layout.prolong("p", fields["p"])  # P1 pressure on all P2 mesh points

    mesh.plot(
        {"speed": speed, "pressure": pressure},
        save_path="cavity_results.png",
        show_mesh=False,
        cmap="jet",
    )
    print("Saved: cavity_results.png")


if __name__ == "__main__":
    solve_cavity(re=100, n_grid=30)

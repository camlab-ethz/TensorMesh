"""2D flow past a cylinder — transient incompressible Navier-Stokes.

Taylor-Hood P2-P1 with semi-implicit backward Euler and Picard
sub-iterations. The quadratic velocity space is generated
**topologically on the linear gmsh mesh** (no re-meshing at order 2),
the pair is LBB-stable so no SUPG/PSPG stabilization or ``tau`` tuning
is needed, and ``MixedElementAssembler`` owns the block layout:

* matrix and load vector come from ``forward`` / ``forward_vector``;
* the previous velocity iterates live on the P2 field's own DOFs and
  enter through ``field_data``;
* boundary nodes are found topologically via ``layout.boundary_mask``
  (MeshGen meshes carry no ``is_boundary`` point data — the
  facet-incidence criterion needs none);
* the do-nothing outlet fixes the pressure gauge, so nothing is pinned.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, MeshGen, MixedElementAssembler
from tensormesh.assemble import MassElementAssembler


class NavierStokesTransientAssembler(MixedElementAssembler):
    r"""Backward-Euler, Picard-linearized incompressible Navier-Stokes:

    .. math::

        \rho\,\frac{u - u_{\mathrm{prev}}}{\Delta t}
        + \rho\,(w\cdot\nabla)u - \mu\,\Delta u + \nabla p = 0,
        \qquad \nabla\cdot u = 0.

    ``forward`` is the bilinear (matrix) side, ``forward_vector`` the
    load vector :math:`\rho/\Delta t\,\int u_{\mathrm{prev}}\cdot v`;
    both take the lagged velocities via ``field_data``.
    """

    fields = [
        Field(trial="u", test="v", order=2, components=2),
        Field(trial="p", test="q", order=1),
    ]

    def __post_init__(self, rho=1.0, mu=0.01, dt=1e-3):
        self.rho = rho
        self.mu = mu
        self.dt = dt

    def forward(self, u, gradu, p, v, gradv, q, w):
        mass = self.rho / self.dt * u.dot(v)
        convection = self.rho * (gradu @ w).dot(v)
        diffusion = self.mu * (gradu * gradv).sum()
        return mass + convection + diffusion \
            - p * gradv.diagonal().sum() \
            - q * gradu.diagonal().sum()

    def forward_vector(self, v, uprev):
        return self.rho / self.dt * uprev.dot(v)


def build_cylinder_channel_mesh(chara_length: float = 0.02):
    length = 2.2
    height = 0.41
    cx, cy, radius = 0.2, 0.2, 0.05

    gen = MeshGen(chara_length=chara_length)
    gen.add_rectangle(0.0, 0.0, length, height)
    gen.remove_circle(cx, cy, radius)
    mesh = gen.gen().double()
    return mesh, (length, height, cx, cy, radius)


def solve_cylinder_flow(
    re: float = 100.0,
    chara_length: float = 0.02,
    dt: float = 1.0e-3,
    n_steps: int = 700,
    picard_iter: int = 2,
    picard_tol: float = 1.0e-5,
    save_every: int = 10,
):
    torch.random.manual_seed(0)

    rho = 1.0
    diameter = 0.1
    u_mean = 1.0
    u_max = 1.5
    mu = rho * u_mean * diameter / re

    mesh, (length, height, cx, cy, radius) = build_cylinder_channel_mesh(chara_length)

    assembler = NavierStokesTransientAssembler.from_mesh(mesh, rho=rho, mu=mu, dt=dt)
    layout = assembler.layout
    print(f"Start cylinder flow: Re={re}, mesh points={mesh.points.shape[0]}, "
          f"dofs={layout.n_dofs}, dt={dt}, steps={n_steps}")

    # --- Boundary conditions on the P2 velocity space ---
    x_u = layout.points("u")
    is_boundary = layout.split(layout.boundary_mask("u"))["u"][:, 0]  # node level
    eps = 2.0e-3
    is_inlet = is_boundary & (x_u[:, 0] <= eps)
    is_outlet = x_u[:, 0] >= length - eps
    no_slip = is_boundary & ~is_inlet & ~is_outlet  # channel walls + cylinder

    bc_mask = layout.dof_mask("u", node_mask=is_inlet | no_slip)
    bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
    y_in = x_u[is_inlet, 1]
    bc_val[layout.dof_mask("u", node_mask=is_inlet, component=0)] = \
        4.0 * u_max * y_in * (height - y_in) / (height * height)

    condenser = Condenser(bc_mask, bc_val[bc_mask])

    sol = torch.zeros(layout.n_dofs, dtype=torch.float64)
    sol[bc_mask] = bc_val[bc_mask]
    # small asymmetric v_y perturbation to trigger vortex shedding faster
    sol[layout.dof_mask("u", component=1)] += \
        1e-3 * torch.sin(2.0 * torch.pi * x_u[:, 0] / length)

    # P1 mass matrix (the pressure space = mesh points) for the vorticity projection
    m_mat = MassElementAssembler.from_mesh(mesh)()

    speed_frames, pressure_frames, vorticity_frames = [], [], []

    for step in tqdm(range(n_steps), desc="Time marching"):
        u_prev = layout.split(sol)["u"]  # [n_u, 2] on the P2 field DOFs
        u_iter = sol.clone()

        for _ in range(picard_iter):
            w = layout.split(u_iter)["u"]
            k_mat = assembler(field_data={"w": ("u", w)})
            f_vec = assembler.assemble_vector(field_data={"uprev": ("u", u_prev)})

            k_cond, f_cond = condenser(k_mat, f_vec)
            u_new = condenser.recover(k_cond.solve(f_cond))

            picard_err = torch.norm(u_new - u_iter) / (torch.norm(u_new) + 1e-12)
            u_iter = u_new
            if float(picard_err) < picard_tol:
                break

        sol = u_iter

        if step % save_every == 0 or step == n_steps - 1:
            velocity = layout.split(sol)["u"]
            speed = layout.prolong("u", velocity).norm(dim=1)  # on mesh points
            pressure = layout.split(sol)["p"]  # P1 == mesh points

            # vorticity: L2-project curl(u) onto P1, using the exact P2 gradient
            omega_rhs = assembler.assemble_vector(
                func=lambda q, gradw: (gradw[1, 0] - gradw[0, 1]) * q,
                field_data={"w": ("u", velocity)},
            )
            omega = m_mat.solve(layout.split(omega_rhs)["p"])

            speed_frames.append(speed.detach().cpu())
            pressure_frames.append(pressure.detach().cpu())
            vorticity_frames.append(omega.detach().cpu())

    os.makedirs("frames", exist_ok=True)
    for i, (vort, spd, pres) in enumerate(zip(vorticity_frames, speed_frames, pressure_frames)):
        mesh.plot(
            {"vorticity": vort, "speed": spd, "pressure": pres},
            save_path=f"frames/frame_{i:04d}.png",
            show_mesh=False,
        )
    print(f"Saved {len(vorticity_frames)} frames to frames/")

    mesh.plot(
        {"vorticity": vorticity_frames[-1], "speed": speed_frames[-1], "pressure": pressure_frames[-1]},
        save_path="cylinder_flow_final.png",
        show_mesh=False,
    )
    print("Saved: cylinder_flow_final.png")


if __name__ == "__main__":
    solve_cylinder_flow(
        re=100.0,
        chara_length=0.02,
        dt=1.0e-3,
        n_steps=5000,
        picard_iter=2,
        picard_tol=1.0e-5,
        save_every=50,
    )

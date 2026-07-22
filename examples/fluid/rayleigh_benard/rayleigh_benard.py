"""Rayleigh-Bénard convection — transient Boussinesq Navier-Stokes.

A three-field mixed problem: Taylor-Hood P2-P1 velocity/pressure plus a
P2 temperature, all declared on one ``MixedElementAssembler``. The
buoyancy term couples the trial temperature to the velocity test field
(an off-diagonal block the one-hot pass extracts automatically), and the
energy equation rides along as a third row of the same block system.

The conductive state (linear temperature, zero velocity) is a steady
solution at every Rayleigh number, so the script marches backward Euler
in time from a slightly perturbed profile: above the critical Ra the
instability grows into steady convection rolls, which is where the run
stops.
"""
import math
import os
import sys

import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler


class RayleighBenardAssembler(MixedElementAssembler):
    r"""Backward-Euler Boussinesq system, Picard-linearized in ``w``:

    .. math::

        \rho\,\frac{u - u_{\mathrm{prev}}}{\Delta t}
        + \rho\,(w\cdot\nabla)u - \mu\,\Delta u + \nabla p
        &= \rho\, g\, \beta\, T\, \hat{e}_y,
        \qquad \nabla\cdot u = 0, \\
        \frac{T - T_{\mathrm{prev}}}{\Delta t}
        + w\cdot\nabla T &= \kappa\, \Delta T.

    Trial fields ``(u, p, T)`` index columns, test fields ``(v, q, s)``
    rows. Every term pairs one trial with one test factor — the buoyancy
    ``-rho g beta T v_y`` lands in the (v, T) block, keeping temperature
    fully coupled (only the advecting velocity ``w`` is lagged).
    """

    fields = [
        Field(trial="u", test="v", order=2, components=2),
        Field(trial="p", test="q", order=1),
        Field(trial="T", test="s", order=2),
    ]

    def __post_init__(self, rho=1.0, mu=0.1, kappa=0.1, g=10.0, beta=1.0, dt=1e-2):
        self.rho = rho
        self.mu = mu
        self.kappa = kappa
        self.g = g
        self.beta = beta  # thermal expansion coefficient
        self.dt = dt

    def forward(self, u, gradu, p, T, gradT, v, gradv, q, s, grads, w):
        momentum = self.rho / self.dt * u.dot(v) \
            + self.rho * (gradu @ w).dot(v) \
            + self.mu * (gradu * gradv).sum() \
            - p * gradv.diagonal().sum() \
            - self.rho * self.g * self.beta * T * v[1]  # buoyancy
        continuity = -q * gradu.diagonal().sum()
        energy = T * s / self.dt \
            + w.dot(gradT) * s \
            + self.kappa * gradT.dot(grads)
        return momentum + continuity + energy

    def forward_vector(self, v, s, uprev, Tprev):
        return self.rho / self.dt * uprev.dot(v) + Tprev * s / self.dt


def solve_rayleigh_benard(ra=2e4, aspect_ratio=2, n_grid=30, dt=5e-3,
                          n_steps=400, steady_tol=1e-5):
    # Rayleigh number Ra = (g * beta * deltaT * L^3) / (nu * alpha);
    # with deltaT = L = rho = 1 and nu = mu, alpha = kappa:
    rho, mu, kappa, g = 1.0, 0.1, 0.1, 10.0
    beta = ra * mu * kappa / (g * 1.0 * 1.0**3)

    print(f"Solving Rayleigh-Bénard convection at Ra={ra:.1e} (Pr={mu / kappa:.1f})...")
    mesh = Mesh.gen_rectangle(
        left=0, right=aspect_ratio, bottom=0, top=1.0,
        chara_length=1.0 / n_grid, element_type="tri", order=2,
    ).double()
    points = mesh.points
    n_points = points.shape[0]

    assembler = RayleighBenardAssembler.from_mesh(
        mesh, rho=rho, mu=mu, kappa=kappa, g=g, beta=beta, dt=dt)
    layout = assembler.layout
    print(f"  Mesh: {n_points} P2 points, {layout.n_dofs} DOFs "
          f"(u {layout.n_nodes('u') * 2}, p {layout.n_nodes('p')}, T {layout.n_nodes('T')})")

    # --- Boundary conditions ---
    is_boundary = mesh.boundary_mask
    is_bottom = points[:, 1] < 1e-6
    is_top = points[:, 1] > 1.0 - 1e-6

    bc_mask = layout.dof_mask("u", is_boundary)      # no-slip on all walls
    bc_mask |= layout.dof_mask("T", is_bottom | is_top)  # heated floor, cooled lid
    bc_mask[layout.dof_index("p", int(layout.node_ids("p")[0]))] = True  # pressure pin

    bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
    bc_val[layout.dof_mask("T", is_bottom)] = 1.0    # T=1 bottom, T=0 top
    condenser = Condenser(bc_mask, bc_val[bc_mask])

    # --- Initial state: conductive profile + a small perturbation ---
    T0 = (1.0 - points[:, 1]) + 0.01 * torch.sin(
        math.pi * points[:, 0] / aspect_ratio) * torch.sin(math.pi * points[:, 1])
    sol = layout.cat(u=0.0, p=0.0, T=T0)
    sol[bc_mask] = bc_val[bc_mask]

    # --- Time marching (backward Euler, temperature fully coupled) ---
    for step in tqdm(range(n_steps), desc="Time marching"):
        fields = layout.split(sol)
        w, u_prev, T_prev = fields["u"], fields["u"], fields["T"]

        K = assembler(point_data={"w": w})
        f = assembler.assemble_vector(point_data={"uprev": u_prev, "Tprev": T_prev})

        K_, f_ = condenser(K, f)
        sol_new = condenser.recover(K_.solve(f_))

        diff = torch.norm(sol_new - sol) / (torch.norm(sol_new) + 1e-8)
        sol = sol_new
        if diff < steady_tol:
            print(f"\nReached steady state at step {step} (t={step * dt:.3f}).")
            break

    # --- Post-processing (P2 fields live on the mesh points directly) ---
    fields = layout.split(sol)
    T = fields["T"]
    V = torch.norm(fields["u"], dim=1)
    print(f"  Max speed: {V.max().item():.4f}, "
          f"T range: [{T.min().item():.3f}, {T.max().item():.3f}]")

    mesh.plot(
        {"Temperature": T, "Velocity": V},
        save_path="rayleigh_benard.png",
        show_mesh=False,
        cmap="inferno",
    )
    print("Done! Results saved to rayleigh_benard.png")


if __name__ == "__main__":
    solve_rayleigh_benard(ra=2e4, aspect_ratio=2, n_grid=30)

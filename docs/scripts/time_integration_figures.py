"""Generate the static figures used in docs/source/user_guide/time_integration.rst.

Run from the repo root, after activating the tensorgalerkin venv:

    python docs/scripts/time_integration_figures.py

Three figures, three independent problems:

* ``convergence.png`` -- scalar ODE ``du/dt = -lambda u`` integrated with
  :class:`tensormesh.ode.ImplicitLinearEuler` and
  :class:`tensormesh.ode.MidPointLinearEuler` at decreasing ``dt``; shows
  the expected order-1 and order-2 slopes against the analytic
  ``u(t) = exp(-lambda t)``.
* ``stability.png`` -- 2D heat equation by hand-rolled time loop, comparing
  forward Euler (with lumped mass) at safe vs unsafe ``dt`` against
  backward Euler at the unsafe ``dt``.
* ``heat_snapshots.png`` -- snapshot triptych of the 2D heat equation
  driven by the same backward-Euler hand-rolled loop.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri as mtri

from tensormesh import Mesh, ElementAssembler, Condenser
from tensormesh.ode import ImplicitLinearEuler, MidPointLinearEuler


OUT = os.path.join(os.path.dirname(__file__), "..", "source",
                   "_static", "user_guide", "time_integration")
os.makedirs(OUT, exist_ok=True)

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Figure 1: scalar-ODE convergence.
#
# du/dt = -lambda u,  u(0) = 1,  exact u(t) = exp(-lambda t).
# We feed lambda into the linear-implicit form M du/dt = A u + B as
# M = 1, A = -lambda, B = 0. The integrator class accepts scalar
# `forward_M` / `forward_A` / `forward_B` and lifts them to identity-like
# operators internally, so this is the smallest possible example.
# ---------------------------------------------------------------------------

class ScalarImplicit(ImplicitLinearEuler):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward_M(self, t): return 1.0
    def forward_A(self, t): return -self.lam
    def forward_B(self, t): return 0.0


class ScalarMidpoint(MidPointLinearEuler):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward_M(self, t): return 1.0
    def forward_A(self, t): return -self.lam
    def forward_B(self, t): return 0.0


def fig_convergence():
    print("[fig 1] convergence...")
    lam = float(np.pi ** 2)
    T = 0.05
    dts = np.array([5e-3, 2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4])

    err_ie, err_mp = [], []
    for dt in dts:
        n = int(round(T / dt))
        T_actual = n * dt

        ie = ScalarImplicit(lam)
        u = torch.ones(1)
        for k in range(n):
            u = ie.step(k * dt, u, dt)
        err_ie.append(abs(u.item() - np.exp(-lam * T_actual)))

        mp = ScalarMidpoint(lam)
        u = torch.ones(1)
        for k in range(n):
            u = mp.step(k * dt, u, dt)
        err_mp.append(abs(u.item() - np.exp(-lam * T_actual)))

    err_ie = np.array(err_ie)
    err_mp = np.array(err_mp)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.loglog(dts, err_ie, 'o-', label='ImplicitLinearEuler (order 1)',
              color='#c0392b', linewidth=2, markersize=7)
    ax.loglog(dts, err_mp, 's-', label='MidPointLinearEuler (order 2)',
              color='#2980b9', linewidth=2, markersize=7)
    # Reference slope guides anchored at the coarsest dt.
    ax.loglog(dts, err_ie[0] * (dts / dts[0]) ** 1, '--',
              color='#c0392b', alpha=0.4, linewidth=1, label=r'$\propto \Delta t$')
    ax.loglog(dts, err_mp[0] * (dts / dts[0]) ** 2, '--',
              color='#2980b9', alpha=0.4, linewidth=1, label=r'$\propto \Delta t^{2}$')

    ax.set_xlabel(r'time step $\Delta t$')
    ax.set_ylabel(r'$|u_{\Delta t}(T) - e^{-\lambda T}|$')
    ax.set_title(r'Temporal convergence on $\dot u = -\pi^{2}\,u$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT, 'convergence.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Common FEM setup for figures 2 and 3:
# u_t = Delta u on [0,1]^2 with Dirichlet u=0 and
# u(x,y,0) = sin(pi x) sin(pi y).
# ---------------------------------------------------------------------------

class GradGrad(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class UV(ElementAssembler):
    def forward(self, u, v):
        return u * v


def build_heat(h):
    mesh = Mesh.gen_rectangle(chara_length=h, order=1)
    M = UV.from_mesh(mesh)().double()
    K = GradGrad.from_mesh(mesh)().double()
    cond = Condenser(mesh.boundary_mask)
    x, y = mesh.points.double()[:, 0], mesh.points.double()[:, 1]
    u0 = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return mesh, M, K, cond, u0


def implicit_euler_loop(M, K, cond, u0, dt, n_steps):
    """Manual backward-Euler loop with static condensation."""
    A = M + dt * K
    A_in, _ = cond(A, torch.zeros_like(u0))
    u = u0.clone()
    history = [u.clone()]
    for _ in range(n_steps):
        f = M @ u
        f_in = cond.condense_rhs(f)
        u_in = A_in.solve(f_in)
        u = cond.recover(u_in)
        history.append(u.clone())
    return history


def explicit_euler_lumped_loop(M, K, mask, u0, dt, n_steps):
    """Manual forward-Euler loop with row-sum lumped mass.

    On each step ``u <- u + dt * M_lump^{-1}(-K u)``; the Dirichlet mask
    pins boundary nodes to zero after each update.
    """
    M_lump = (M @ torch.ones(u0.shape[0], dtype=u0.dtype)).clone()
    u = u0.clone()
    history = [u.clone()]
    for _ in range(n_steps):
        rhs = -(K @ u)
        u = u + dt * rhs / M_lump
        u[mask] = 0.0
        history.append(u.clone())
    return history


# ---------------------------------------------------------------------------
# Figure 2: stability of explicit vs implicit on the heat equation.
# ---------------------------------------------------------------------------

def fig_stability():
    print("[fig 2] stability...")
    mesh, M, K, cond, u0 = build_heat(h=0.1)
    # Add a high-frequency seed so the unstable mode has something to grow.
    # Without it, ``sin(pi x) sin(pi y)`` projects almost entirely onto the
    # smoothest eigenmode and roundoff has to amplify on its own.
    x, y = mesh.points.double()[:, 0], mesh.points.double()[:, 1]
    u0 = u0 + 1e-3 * torch.sin(10 * torch.pi * x) * torch.sin(10 * torch.pi * y)
    u0[mesh.boundary_mask] = 0.0

    T = 0.05
    dt_safe = 1.0e-3
    dt_unsafe = 4.0e-3

    def maxabs(history):
        return np.array([h.abs().max().item() for h in history])

    n_safe = int(round(T / dt_safe))
    n_unsafe = int(round(T / dt_unsafe))

    hist_ex_safe = explicit_euler_lumped_loop(M, K, mesh.boundary_mask, u0, dt_safe, n_safe)
    hist_ex_unsafe = explicit_euler_lumped_loop(M, K, mesh.boundary_mask, u0, dt_unsafe, n_unsafe)
    hist_im = implicit_euler_loop(M, K, cond, u0, dt_unsafe, n_unsafe)

    t_safe = np.linspace(0, T, len(hist_ex_safe))
    t_unsafe = np.linspace(0, T, len(hist_ex_unsafe))
    t_im = np.linspace(0, T, len(hist_im))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.semilogy(t_safe, maxabs(hist_ex_safe), '-',
                label=fr'forward Euler, $\Delta t = {dt_safe:.0e}$ (stable)',
                color='#27ae60', linewidth=2)
    ax.semilogy(t_im, maxabs(hist_im), '-',
                label=fr'backward Euler, $\Delta t = {dt_unsafe:.0e}$ (stable)',
                color='#2980b9', linewidth=2)
    ax.semilogy(t_unsafe, np.clip(maxabs(hist_ex_unsafe), 1e-20, 1e30), '-',
                label=fr'forward Euler, $\Delta t = {dt_unsafe:.0e}$ (unstable)',
                color='#c0392b', linewidth=2)
    ax.set_xlabel(r'time $t$')
    ax.set_ylabel(r'$\max_x |u(x, t)|$')
    ax.set_title('Stability on 2D heat: forward vs backward Euler')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUT, 'stability.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"        -> {out}")


# ---------------------------------------------------------------------------
# Figure 3: snapshot triptych.
# ---------------------------------------------------------------------------

def _triangles_of(mesh):
    cells = mesh.cells
    keys = list(cells.keys()) if hasattr(cells, 'keys') else list(cells)
    for preferred in ('triangle', 'tri'):
        if preferred in keys:
            return cells[preferred].cpu().numpy()
    return cells[keys[0]].cpu().numpy()


def fig_heat_snapshots():
    print("[fig 3] heat snapshots...")
    mesh, M, K, cond, u0 = build_heat(h=0.025)
    dt = 5e-4
    T = 0.05
    n = int(round(T / dt))
    history = implicit_euler_loop(M, K, cond, u0, dt, n)

    pts = mesh.points.cpu().numpy()
    tris = _triangles_of(mesh)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    target = [0, n // 2, n]
    vmin, vmax = -1.0, 1.0
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, k in zip(axes, target):
        cs = ax.tricontourf(triang, history[k].cpu().numpy(),
                            levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(f"$t = {k * dt:.3f}$")
    cbar = fig.colorbar(cs, ax=axes, shrink=0.85, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label(r'$u(x, y, t)$')
    fig.suptitle('2D heat equation, backward Euler', fontsize=12, y=1.02)
    out = os.path.join(OUT, 'heat_snapshots.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"        -> {out}")


if __name__ == '__main__':
    fig_convergence()
    fig_stability()
    fig_heat_snapshots()
    print('done.')

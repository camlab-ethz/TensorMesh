"""Decaying 2D Taylor-Green vortex — transient incompressible Navier-Stokes.

Taylor-Hood P2-P1 mixed discretization (backward Euler + Picard) with the
exact analytical solution for verification. The velocity lives on the
order-2 mesh points, the pressure on the corner vertices; the pair is
LBB-stable so the assembler is the four-line weak form — no SUPG/PSPG
terms, no ``tau`` schedule. Backward Euler is first-order in time, so
the convergence study couples ``dt ~ h^2`` to expose the O(h^2) spatial
accuracy of the pressure (the P2 velocity is even more accurate).
"""
import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
import torch
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.assemble import MassElementAssembler
from tensormesh.visualization import draw_mesh_2d_static, draw_mesh_2d_stream


# ---------------------------------------------------------------------------
# Analytical solution for the 2D Taylor-Green vortex on [0, 2pi]^2
# ---------------------------------------------------------------------------

def exact_velocity(points, t, nu):
    x, y = points[:, 0], points[:, 1]
    decay = math.exp(-2.0 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    return torch.stack([u, v], dim=1)


def exact_pressure(points, t, nu):
    x, y = points[:, 0], points[:, 1]
    decay = math.exp(-4.0 * nu * t)
    return -0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay


# ---------------------------------------------------------------------------
# Assembler (same pattern as cylinder_flow.py)
# ---------------------------------------------------------------------------

class NavierStokesTransientAssembler(MixedElementAssembler):
    r"""Backward-Euler, Picard-linearized incompressible Navier-Stokes:

    .. math::

        \rho\,\frac{u - u_{\mathrm{prev}}}{\Delta t}
        + \rho\,(w\cdot\nabla)u - \mu\,\Delta u + \nabla p = 0,
        \qquad \nabla\cdot u = 0.

    ``forward`` is the bilinear (matrix) side, ``forward_vector`` the load
    vector; the lagged velocities enter via ``point_data`` (the order-2
    mesh points coincide with the P2 velocity nodes).
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


# ---------------------------------------------------------------------------
# L2 error computation via mass matrix
# ---------------------------------------------------------------------------

def compute_l2_error(M, numerical, exact):
    """Compute L2 error = sqrt( sum_d (e_d @ M @ e_d) ) for vector/scalar fields."""
    if numerical.dim() == 1:
        e = numerical - exact
        return float(torch.sqrt((e * (M @ e)).sum()).item())
    else:
        err_sq = 0.0
        for d in range(numerical.shape[1]):
            e = numerical[:, d] - exact[:, d]
            err_sq += float((e * (M @ e)).sum().item())
        return math.sqrt(err_sq)


def compute_l2_norm(M, field):
    if field.dim() == 1:
        return float(torch.sqrt((field * (M @ field)).sum()).item())
    else:
        norm_sq = 0.0
        for d in range(field.shape[1]):
            norm_sq += float((field[:, d] * (M @ field[:, d])).sum().item())
        return math.sqrt(norm_sq)


# ---------------------------------------------------------------------------
# Vortex visualization (vorticity + streamlines / quiver)
# ---------------------------------------------------------------------------

def _interpolate_to_grid(points, velocity, grid_density=200):
    """Interpolate FEM velocity field onto a regular grid and compute vorticity."""
    pts = points.detach().cpu().numpy()
    x, y = pts[:, 0], pts[:, 1]

    xi = np.linspace(x.min(), x.max(), grid_density)
    yi = np.linspace(y.min(), y.max(), grid_density)
    Xi, Yi = np.meshgrid(xi, yi)

    u_np = velocity[:, 0].detach().cpu().numpy()
    v_np = velocity[:, 1].detach().cpu().numpy()

    Ui = griddata((x, y), u_np, (Xi, Yi), method='cubic', fill_value=0)
    Vi = griddata((x, y), v_np, (Xi, Yi), method='cubic', fill_value=0)

    # Vorticity: dv/dx - du/dy
    dvdx = np.gradient(Vi, xi, axis=1)
    dudy = np.gradient(Ui, yi, axis=0)
    vorticity = dvdx - dudy
    speed = np.sqrt(Ui ** 2 + Vi ** 2)

    return xi, yi, Xi, Yi, Ui, Vi, vorticity, speed


def save_vortex_snapshot(points, velocity, t=0.0,
                         filename="taylor_green_vortex.png",
                         grid_density=200):
    """Static image: vorticity + streamlines (left) and speed + quiver (right)."""
    xi, yi, Xi, Yi, Ui, Vi, vort, spd = _interpolate_to_grid(
        points, velocity, grid_density)

    vort_max = np.abs(vort).max()
    speed_max = spd.max()
    if vort_max < 1e-10:
        vort_max = 1.0
    if speed_max < 1e-10:
        speed_max = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: vorticity + streamlines ---
    levels_v = np.linspace(-vort_max, vort_max, 50)
    cf1 = ax1.contourf(Xi, Yi, vort, levels=levels_v, cmap='RdBu_r', extend='both')
    fig.colorbar(cf1, ax=ax1, label='Vorticity')
    ax1.streamplot(Xi, Yi, Ui, Vi, color='k',
                   linewidth=0.7, density=2.0, arrowsize=1.2)
    ax1.set_title(f'Vorticity + Streamlines  (t={t:.3f})', fontsize=12)
    ax1.set_aspect('equal')

    # --- Right: speed + quiver ---
    levels_s = np.linspace(0, speed_max, 50)
    cf2 = ax2.contourf(Xi, Yi, spd, levels=levels_s, cmap='coolwarm')
    fig.colorbar(cf2, ax=ax2, label='Speed')
    skip = max(grid_density // 25, 1)
    ax2.quiver(Xi[::skip, ::skip], Yi[::skip, ::skip],
               Ui[::skip, ::skip], Vi[::skip, ::skip],
               color='black', alpha=0.7)
    ax2.set_title(f'Speed + Velocity Vectors  (t={t:.3f})', fontsize=12)
    ax2.set_aspect('equal')

    fig.suptitle('Taylor-Green Vortex', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved vortex snapshot: {filename}")


def save_vortex_video(points, snapshots_velocity, snap_dt,
                      filename="taylor_green_vortex.mp4",
                      grid_density=200, fps=10):
    """Animated video: vorticity + streamlines (left) and speed + quiver (right)."""
    pts = points.detach().cpu().numpy()
    x, y = pts[:, 0], pts[:, 1]

    xi = np.linspace(x.min(), x.max(), grid_density)
    yi = np.linspace(y.min(), y.max(), grid_density)
    Xi, Yi = np.meshgrid(xi, yi)

    # Pre-interpolate all frames
    print("  Interpolating frames for vortex video ...")
    frames = []
    for vel in snapshots_velocity:
        u_np = vel[:, 0].detach().cpu().numpy()
        v_np = vel[:, 1].detach().cpu().numpy()

        Ui = griddata((x, y), u_np, (Xi, Yi), method='cubic', fill_value=0)
        Vi = griddata((x, y), v_np, (Xi, Yi), method='cubic', fill_value=0)

        dvdx = np.gradient(Vi, xi, axis=1)
        dudy = np.gradient(Ui, yi, axis=0)
        vort = dvdx - dudy
        spd = np.sqrt(Ui ** 2 + Vi ** 2)
        frames.append((Ui, Vi, vort, spd))

    # Fixed colorbar ranges across all frames
    vort_max = max(np.abs(f[2]).max() for f in frames)
    speed_max = max(f[3].max() for f in frames)
    if vort_max < 1e-10:
        vort_max = 1.0
    if speed_max < 1e-10:
        speed_max = 1.0

    levels_v = np.linspace(-vort_max, vort_max, 40)
    levels_s = np.linspace(0, speed_max, 40)
    skip = max(grid_density // 25, 1)

    # Create figure with persistent colorbars using ScalarMappable
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    vort_norm = mcolors.TwoSlopeNorm(vmin=-vort_max, vcenter=0, vmax=vort_max)
    speed_norm = mcolors.Normalize(vmin=0, vmax=speed_max)
    sm_vort = plt.cm.ScalarMappable(norm=vort_norm, cmap='RdBu_r')
    sm_speed = plt.cm.ScalarMappable(norm=speed_norm, cmap='coolwarm')
    fig.colorbar(sm_vort, ax=ax1, label='Vorticity')
    fig.colorbar(sm_speed, ax=ax2, label='Speed')

    writer = animation.writers['ffmpeg'](fps=fps, bitrate=2000)
    print(f"  Rendering {len(frames)} frames ...")
    with writer.saving(fig, filename, dpi=150):
        for t_idx, (Ui, Vi, vort, spd) in enumerate(frames):
            ax1.clear()
            ax2.clear()
            t_val = t_idx * snap_dt

            # Vorticity + streamlines
            ax1.contourf(Xi, Yi, vort, levels=levels_v,
                         cmap='RdBu_r', extend='both')
            ax1.streamplot(Xi, Yi, Ui, Vi, color='k',
                           linewidth=0.7, density=2.0, arrowsize=1.2)
            ax1.set_title('Vorticity + Streamlines', fontsize=11)
            ax1.set_aspect('equal')
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(y.min(), y.max())

            # Speed + quiver
            ax2.contourf(Xi, Yi, spd, levels=levels_s, cmap='coolwarm')
            ax2.quiver(Xi[::skip, ::skip], Yi[::skip, ::skip],
                       Ui[::skip, ::skip], Vi[::skip, ::skip],
                       color='black', alpha=0.7)
            ax2.set_title('Speed + Velocity Vectors', fontsize=11)
            ax2.set_aspect('equal')
            ax2.set_xlim(x.min(), x.max())
            ax2.set_ylim(y.min(), y.max())

            fig.suptitle(f'Taylor-Green Vortex  t = {t_val:.3f}s',
                         fontsize=14, fontweight='bold')
            writer.grab_frame()

    plt.close(fig)
    print(f"  Saved vortex video: {filename}")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_taylor_green(nu=0.01, n_grid=30, dt=0.01, t_final=1.0, picard_iter=2,
                       picard_tol=1e-5, verbose=True, save_video=False,
                       video_filename="taylor_green.mp4", snapshot_interval=1):
    """Solve decaying Taylor-Green vortex and return L2 errors at final time."""
    L = 2.0 * math.pi
    h = L / n_grid

    if verbose:
        print(f"Taylor-Green: nu={nu}, grid={n_grid}, h={h:.4f}, dt={dt}, t_final={t_final}")

    # Order-2 mesh: its points are exactly the P2 velocity nodes.
    mesh = Mesh.gen_rectangle(
        left=0.0, right=L, bottom=0.0, top=L,
        chara_length=h, element_type="tri", order=2,
    ).double()
    points = mesh.points
    n_points = points.shape[0]
    n_steps = int(round(t_final / dt))
    rho = 1.0

    assembler = NavierStokesTransientAssembler.from_mesh(mesh, rho=rho, mu=nu, dt=dt)
    layout = assembler.layout

    if verbose:
        print(f"  Mesh: {n_points} P2 points, {layout.n_nodes('p')} P1 pressure "
              f"nodes, {layout.n_dofs} DOFs, {n_steps} time steps")

    # --- Boundary conditions: exact velocity on the boundary + pressure pin ---
    pin = layout.dof_index("p", int(layout.node_ids("p")[0]))
    x_p = layout.points("p")

    def dirichlet_values(t):
        bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
        bc_val[layout.dof_mask("u")] = exact_velocity(points, t, nu).reshape(-1)
        bc_val[pin] = exact_pressure(x_p, t, nu)[0]
        return bc_val

    bc_mask = layout.dof_mask("u", mesh.boundary_mask)
    bc_mask[pin] = True
    condenser = Condenser(bc_mask, dirichlet_values(0.0))

    # --- Initial condition: exact fields at t=0 ---
    sol = layout.cat(
        u=exact_velocity(points, 0.0, nu),
        p=exact_pressure(x_p, 0.0, nu),
    )

    # Mass matrix of the P2 space (order-2 mesh) for L2 errors.
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=4)()

    # --- Snapshot collection for video ---
    if save_video:
        fields0 = layout.split(sol)
        snapshots_speed = [torch.norm(fields0["u"], dim=1).float()]
        snapshots_pressure = [layout.prolong("p", fields0["p"]).float()]
        snapshots_velerr = [torch.zeros(n_points, dtype=torch.float32)]
        snapshots_velocity = [fields0["u"].clone()]

    # --- Time stepping ---
    iterator = tqdm(range(1, n_steps + 1), desc="Time stepping") if verbose else range(1, n_steps + 1)
    for step in iterator:
        condenser.update_dirichlet(dirichlet_values(step * dt))

        u_prev = layout.split(sol)["u"]
        u_iter = sol.clone()

        for _ in range(picard_iter):
            w = layout.split(u_iter)["u"]
            K = assembler(point_data={"w": w})
            f = assembler.assemble_vector(point_data={"uprev": u_prev})

            K_cond, f_cond = condenser(K, f)
            u_new = condenser.recover(K_cond.solve(f_cond))

            picard_err = torch.norm(u_new - u_iter) / (torch.norm(u_new) + 1e-12)
            u_iter = u_new
            if float(picard_err) < picard_tol:
                break

        sol = u_iter

        # Collect snapshots
        if save_video and step % snapshot_interval == 0:
            fields_snap = layout.split(sol)
            vel_snap = fields_snap["u"]
            snapshots_speed.append(torch.norm(vel_snap, dim=1).float())
            snapshots_pressure.append(layout.prolong("p", fields_snap["p"]).float())
            vel_ex_snap = exact_velocity(points, step * dt, nu)
            snapshots_velerr.append(torch.norm(vel_snap - vel_ex_snap, dim=1).float())
            snapshots_velocity.append(vel_snap.clone())

    # --- Generate video ---
    if save_video:
        elements = mesh.elements()
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}
        snap_dt = dt * snapshot_interval
        draw_mesh_2d_stream(
            points=mesh.points,
            elements=elements,
            point_values={
                "speed": torch.stack(snapshots_speed),
                "pressure": torch.stack(snapshots_pressure),
                "velocity error": torch.stack(snapshots_velerr),
            },
            dt=snap_dt,
            show_mesh=False,
            fix_colorbar=True,
            filename=video_filename,
            cmap="viridis",
        )
        if verbose:
            print(f"  Saved scalar video: {video_filename}")

        # Vortex visualization: vorticity + streamlines + quiver
        vortex_video = video_filename.replace(".mp4", "_vortex.mp4")
        save_vortex_video(points, snapshots_velocity, snap_dt,
                          filename=vortex_video, grid_density=200, fps=10)

        # Also save a static snapshot at t=0
        save_vortex_snapshot(points, snapshots_velocity[0], t=0.0,
                             filename=video_filename.replace(".mp4", "_t0.png"))

    # --- Compute errors (P1 pressure prolonged into the P2 space is exact) ---
    fields = layout.split(sol)
    vel_num = fields["u"]
    p_num = layout.prolong("p", fields["p"])

    vel_ex = exact_velocity(points, t_final, nu)
    p_ex = exact_pressure(points, t_final, nu)

    l2_vel = compute_l2_error(M, vel_num, vel_ex)
    l2_p = compute_l2_error(M, p_num, p_ex)
    l2_vel_ref = compute_l2_norm(M, vel_ex)
    l2_p_ref = compute_l2_norm(M, p_ex)

    rel_vel = l2_vel / (l2_vel_ref + 1e-30)
    rel_p = l2_p / (l2_p_ref + 1e-30)

    if verbose:
        print(f"  L2 velocity error: {l2_vel:.6e} (relative: {rel_vel:.6e})")
        print(f"  L2 pressure error: {l2_p:.6e} (relative: {rel_p:.6e})")

    return {
        "mesh": mesh, "sol": sol, "h": h, "n_points": n_points,
        "l2_vel": l2_vel, "l2_p": l2_p,
        "rel_vel": rel_vel, "rel_p": rel_p,
        "vel_num": vel_num, "vel_ex": vel_ex, "p_num": p_num, "p_ex": p_ex,
    }


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def convergence_study(nu=0.01, grids=None, t_final=0.5):
    if grids is None:
        grids = [10, 20, 40]

    print("=" * 70)
    print(f"Taylor-Green Convergence Study  (nu={nu}, t_final={t_final})")
    print("=" * 70)

    results = []
    for n in grids:
        L = 2.0 * math.pi
        h = L / n
        # Backward Euler is O(dt); couple dt ~ h^2 so the time error shrinks
        # at the same rate as the O(h^2) pressure error.
        dt = 0.25 * h * h
        res = solve_taylor_green(nu=nu, n_grid=n, dt=dt, t_final=t_final)
        results.append(res)

    # Print convergence table
    print("\n{:<6s} {:<10s} {:<14s} {:<14s} {:<10s} {:<10s}".format(
        "Grid", "h", "L2_vel", "L2_pres", "Rate_vel", "Rate_pres"))
    print("-" * 64)
    for i, (n, res) in enumerate(zip(grids, results)):
        if i == 0:
            rate_v, rate_p = "-", "-"
        else:
            rate_v = f"{math.log(results[i-1]['l2_vel'] / res['l2_vel']) / math.log(results[i-1]['h'] / res['h']):.2f}"
            rate_p = f"{math.log(results[i-1]['l2_p'] / (res['l2_p'] + 1e-30)) / math.log(results[i-1]['h'] / res['h']):.2f}"
        print(f"{n:<6d} {res['h']:<10.4f} {res['l2_vel']:<14.6e} {res['l2_p']:<14.6e} {rate_v:<10s} {rate_p:<10s}")

    # Convergence plot
    hs = [r["h"] for r in results]
    l2_vels = [r["l2_vel"] for r in results]
    l2_ps = [r["l2_p"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, l2_vels, "o-", label="velocity L2 error")
    ax.loglog(hs, l2_ps, "s-", label="pressure L2 error")
    # Reference slopes
    h_ref = np.array(hs)
    ax.loglog(h_ref, l2_ps[0] * (h_ref / hs[0]) ** 2, "k--", alpha=0.4, label="O(h$^2$)")
    ax.set_xlabel("h")
    ax.set_ylabel("L2 error")
    ax.set_title("Taylor-Green Vortex Convergence (Taylor-Hood P2-P1)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("taylor_green_convergence.png", dpi=200)
    plt.close(fig)
    print("\nSaved: taylor_green_convergence.png")

    # Visualize finest result
    res = results[-1]
    mesh = res["mesh"]
    elements = mesh.elements()
    if isinstance(elements, torch.Tensor):
        elements = {mesh.default_element_type: elements}

    speed = torch.norm(res["vel_num"], dim=1)
    vel_err = torch.norm(res["vel_num"] - res["vel_ex"], dim=1)

    draw_mesh_2d_static(
        mesh.points, elements,
        {"speed": speed, "pressure": res["p_num"], "velocity error": vel_err},
        filename="taylor_green_results.png",
        show_mesh=False, cmap="viridis",
    )
    print("Saved: taylor_green_results.png")

    # Vortex visualization of the finest grid result
    save_vortex_snapshot(mesh.points, res["vel_num"], t=t_final,
                         filename="taylor_green_vortex_final.png")


if __name__ == "__main__":
    convergence_study(nu=0.01, grids=[10, 20, 40], t_final=0.5)

    # Visualization run: low viscosity to keep vortices visible
    # exp(-2*0.01*5) ≈ 0.90, so vortices retain ~90% strength at t=5
    print("\n" + "=" * 70)
    print("Generating Taylor-Green vortex visualization ...")
    print("=" * 70)
    solve_taylor_green(
        nu=0.01, n_grid=40, dt=0.02, t_final=5.0,
        save_video=True, video_filename="taylor_green.mp4",
        snapshot_interval=5,
    )

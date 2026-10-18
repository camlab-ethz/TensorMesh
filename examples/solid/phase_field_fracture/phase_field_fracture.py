"""Phase-field fracture: single-edge-notched tension.

A square plate with a horizontal slit from the left edge to the centre is
pulled at its top edge until a crack runs through the ligament. This is the
standard brittle phase-field benchmark (Miehe, Hofacker and Welschinger, 2010),
in plane strain, with the AT1 or AT2 crack model and the Amor (volumetric /
deviatoric) energy split, so the crack does not form in compression.

The damage field alpha only grows, alpha_n >= alpha_{n-1}, and stays below 1.
These are bounds, so each load step is solved by alternate
minimization with two bound-constrained Newton solves (torch_sla.vi_solve),
whose residuals are autograd gradients of the energy and whose Jacobians are
assembled bilinear forms:

- displacement, alpha fixed: the Dirichlet values enter as fixed bounds
  (lower = upper = prescribed value), every other component is unbounded;
- damage, u fixed: bounds [alpha_prev, 1].

Units: kN and mm (E in kN/mm^2 = GPa / 1000, Gc in kN/mm).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import meshio
import numpy as np
import torch
from torch_sla import vi_solve

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh import ElementAssembler, Mesh  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class PhaseFieldModel:
    """Energy densities at one quadrature point (plane strain or 3D).

    psi(eps, alpha) = g(alpha) psi_D(eps) + psi_R(eps),  g = (1 - alpha)^2 + g0,
    psi_R = kappa/2 <tr eps>_-^2 (never degraded), psi_D = lam/2 (tr eps)^2 + mu eps:eps - psi_R,
    fracture density Gc/c_w (w(alpha)/ell + ell |grad alpha|^2), with
    AT1: w = alpha, c_w = 8/3;  AT2: w = alpha^2, c_w = 2.

    The bulk modulus is the 3D one (kappa = lam + 2 mu / 3), which keeps the
    split exact in plane strain where eps_zz = 0.
    """

    def __init__(self, E: float, nu: float, Gc: float, ell: float, model: str = "at1", g0: float = 1e-6):
        if model not in ("at1", "at2"):
            raise ValueError(f"model must be 'at1' or 'at2', got {model!r}")
        self.E, self.nu, self.Gc, self.ell, self.model, self.g0 = E, nu, Gc, ell, model, g0
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = self.lam + 2 * self.mu / 3
        self.cw = 8 / 3 if model == "at1" else 2.0
        self.ddw = 0.0 if model == "at1" else 2.0  # w''

    def g(self, a):
        return (1 - a) ** 2 + self.g0

    def w(self, a):
        return a if self.model == "at1" else a**2

    @staticmethod
    def strain(grad_u):
        return 0.5 * (grad_u + grad_u.transpose(-1, -2))

    def psi_R(self, eps):
        tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
        return 0.5 * self.kappa * tr.clamp(max=0) ** 2

    def psi_D(self, eps):
        tr = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
        return 0.5 * self.lam * tr**2 + self.mu * (eps * eps).sum((-2, -1)) - self.psi_R(eps)

    def psi_el(self, eps, a):
        return self.g(a) * self.psi_D(eps) + self.psi_R(eps)

    def psi_frac(self, a, grad_a):
        return self.Gc / self.cw * (self.w(a) / self.ell + self.ell * (grad_a * grad_a).sum(-1))


# --------------------------------------------------------------------------- #
# Energy and Hessians (point data: ``disp`` [n, dim] and ``alpha`` [n])       #
# --------------------------------------------------------------------------- #

class PhaseFieldEnergy(ElementAssembler):
    """The elastic and fracture energies. The residuals are their gradients,
    taken by autograd, so the model is written down once."""

    def __post_init__(self, model: PhaseFieldModel):
        self.model = model

    def elastic(self, disp, alpha):
        m = self.model
        return self.energy(func=lambda graddisp, alpha: m.psi_el(m.strain(graddisp), alpha),
                           point_data={"disp": disp, "alpha": alpha})

    def fracture(self, alpha):
        m = self.model
        return self.energy(func=lambda alpha, gradalpha: m.psi_frac(alpha, gradalpha),
                           point_data={"alpha": alpha})


class DisplacementTangent(ElementAssembler):
    """d2E/du2 with the split's tangent C = g C_0 + (1 - g) kappa [tr eps < 0] I x I."""

    def __post_init__(self, model: PhaseFieldModel):
        self.model = model

    def forward(self, gradu, gradv, graddisp, alpha):
        m = self.model
        g = m.g(alpha)
        compressed = (graddisp.diagonal().sum() < 0).to(gradu.dtype)
        lam = g * m.lam + (1 - g) * m.kappa * compressed
        I = torch.eye(gradu.shape[0], dtype=gradu.dtype, device=gradu.device)
        return lam * torch.outer(gradu, gradv) + g * m.mu * ((gradu * gradv).sum() * I + torch.outer(gradv, gradu))


class PhaseFieldHessian(ElementAssembler):
    """d2E/dalpha2, which does not depend on alpha for AT1 and AT2 (g'' = 2)."""

    def __post_init__(self, model: PhaseFieldModel):
        self.model = model

    def forward(self, u, v, gradu, gradv, graddisp):
        m = self.model
        reaction = 2 * m.psi_D(m.strain(graddisp)) + m.Gc / (m.cw * m.ell) * m.ddw
        return reaction * u * v + 2 * m.Gc * m.ell / m.cw * (gradu * gradv).sum()


# --------------------------------------------------------------------------- #
# Staggered solver                                                            #
# --------------------------------------------------------------------------- #

class StaggeredPhaseField:
    """Alternate minimization of E(u, alpha) on a mesh of linear elements.

    Parameters
    ----------
    mesh : tensormesh.Mesh
    model : PhaseFieldModel
    u_fixed : torch.Tensor
        Bool mask over the displacement dofs ``[n_points * dim]`` (node-major)
        that carry Dirichlet values.
    alpha_fixed : torch.Tensor, optional
        Bool mask over the nodes where alpha is prescribed, with values
        ``alpha_value`` (e.g. 1 on a pre-crack).
    tol : float
        Staggered stopping criterion: the displacement residual on the free
        dofs, after the damage update, relative to the largest reaction (on
        the fixed dofs) seen in the load step.
    max_staggered : int
        Staggered iterations allowed per load step.
    relaxation : float
        Over-relaxation factor w of the damage update,
        alpha <- alpha + w (alpha_new - alpha), kept within the bounds
        (Farrell and Maurini, 2017). 1 is plain alternate minimization; values
        around 1.6 need fewer iterations while a crack runs.
    linear_solver : str
        torch-sla backend for the Newton systems.
    """

    def __init__(self, mesh, model, u_fixed, alpha_fixed=None, alpha_value=None, tol=1e-4,
                 max_staggered=10000, relaxation=1.0, quadrature_order=2, linear_solver="auto"):
        self.model = model
        self.dim = mesh.points.shape[1]
        n = mesh.n_points
        dev, dtype = mesh.points.device, mesh.points.dtype
        self.E = PhaseFieldEnergy.from_mesh(mesh, quadrature_order=quadrature_order, model=model)
        self.K = DisplacementTangent.from_assembler(self.E, model=model)
        self.J = PhaseFieldHessian.from_assembler(self.E, model=model)
        self.u_fixed = u_fixed
        self.alpha_fixed = torch.zeros(n, dtype=torch.bool, device=dev) if alpha_fixed is None else alpha_fixed
        self.alpha_value = torch.zeros(n, dtype=dtype, device=dev) if alpha_value is None else alpha_value
        self.u = torch.zeros(n * self.dim, dtype=dtype, device=dev)
        self.alpha = torch.where(self.alpha_fixed, self.alpha_value, torch.zeros_like(self.alpha_value))
        self.tol, self.max_staggered, self.linear_solver = tol, max_staggered, linear_solver
        self.relaxation = relaxation

    def _disp(self, u):
        return u.reshape(-1, self.dim)

    def residual_u(self, u, alpha):
        """dE/du (internal forces)."""
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            return torch.autograd.grad(self.E.elastic(self._disp(u), alpha.detach()), u)[0]

    def tangent_u(self, u, alpha):
        return self.K(point_data={"disp": self._disp(u), "alpha": alpha})

    def residual_alpha(self, alpha, u):
        """dE/dalpha."""
        with torch.enable_grad():
            a = alpha.detach().requires_grad_(True)
            energy = self.E.elastic(self._disp(u.detach()), a) + self.E.fracture(a)
            return torch.autograd.grad(energy, a)[0]

    def hessian_alpha(self, u):
        return self.J(point_data={"disp": self._disp(u)})

    def solve_step(self, u_bc):
        """One load step with the Dirichlet values ``u_bc`` (full dof vector);
        returns the number of staggered iterations."""
        inf = torch.full_like(self.u, math.inf)
        u_lower = torch.where(self.u_fixed, u_bc, -inf)
        u_upper = torch.where(self.u_fixed, u_bc, inf)
        a_lower = torch.where(self.alpha_fixed, self.alpha_value, self.alpha)  # irreversibility
        a_upper = torch.where(self.alpha_fixed, self.alpha_value, torch.ones_like(self.alpha))
        scale = 1e-12
        for it in range(1, self.max_staggered + 1):
            alpha = self.alpha
            self.u = vi_solve(lambda u: self.residual_u(u, alpha), self.u, lower=u_lower, upper=u_upper,
                              jacobian_fn=lambda u: self.tangent_u(u, alpha), line_search=True,
                              tol=1e-9, atol=1e-11, linear_solver=self.linear_solver)
            u = self.u
            hessian = self.hessian_alpha(u)  # constant during the alpha solve
            alpha_new = vi_solve(lambda a: self.residual_alpha(a, u), alpha, lower=a_lower, upper=a_upper,
                                 jacobian_fn=lambda a: hessian, tol=1e-9, atol=1e-11,
                                 linear_solver=self.linear_solver)
            r = self.residual_u(u, alpha_new)
            free = float(torch.linalg.vector_norm(r[~self.u_fixed]))
            scale = max(scale, float(torch.linalg.vector_norm(r[self.u_fixed])))
            if free <= self.tol * scale:
                self.alpha = alpha_new
                return it
            if self.relaxation != 1.0:
                alpha_new = (alpha + self.relaxation * (alpha_new - alpha)).clamp(a_lower, a_upper)
            self.alpha = alpha_new
        raise RuntimeError(f"staggered scheme did not converge in {self.max_staggered} iterations")

    def reaction(self, dofs):
        """Sum of the internal forces on the given dofs (the reaction they carry)."""
        return float(self.residual_u(self.u, self.alpha)[dofs].sum())

    def energy(self):
        """(elastic, fracture) energies."""
        disp = self._disp(self.u)
        return float(self.E.elastic(disp, self.alpha)), float(self.E.fracture(self.alpha))


# --------------------------------------------------------------------------- #
# Single-edge-notched tension                                                 #
# --------------------------------------------------------------------------- #

def structured_triangle_mesh(x, y, slit=None):
    """Union-jack triangles on the tensor grid of the 1D coordinates ``x`` and
    ``y``: the cell diagonals alternate like a checkerboard, so the mesh is
    mirror-symmetric about every grid line and does not favour one crack
    direction.

    ``slit=(y0, x_tip)`` cuts the mesh along the grid line y = y0 for
    x < x_tip: the elements above the line get their own copies of its nodes,
    so the two faces of the slit move independently.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    nid = np.arange(points.shape[0]).reshape(len(x), len(y))
    a, b, c, d = nid[:-1, :-1], nid[1:, :-1], nid[1:, 1:], nid[:-1, 1:]  # cell corners, counter-clockwise
    i, j = np.meshgrid(np.arange(len(x) - 1), np.arange(len(y) - 1), indexing="ij")
    even = (i + j) % 2 == 0  # diagonal a-c, otherwise b-d
    cells = np.concatenate([
        np.stack([a[even], b[even], c[even]], 1), np.stack([a[even], c[even], d[even]], 1),
        np.stack([a[~even], b[~even], d[~even]], 1), np.stack([b[~even], c[~even], d[~even]], 1),
    ])
    if slit is not None:
        y0, x_tip = slit
        cut = nid[x < x_tip - 1e-12, int(np.argmin(np.abs(y - y0)))]  # slit nodes, tip excluded
        remap = np.arange(len(points))
        remap[cut] = len(points) + np.arange(len(cut))
        points = np.concatenate([points, points[cut]])
        above = points[cells].mean(axis=1)[:, 1] > y0
        cells[above] = remap[cells[above]]
    return Mesh(meshio.Mesh(points, [("triangle", cells)]))


def graded_coordinates(h_fine: float, h_coarse: float, band: float):
    """Coordinates on [0, 1] with spacing h_fine within ``band`` of 0.5 and
    about h_coarse outside; 0.5 is a grid line."""
    m = max(1, round(band / h_fine))
    fine = 0.5 + h_fine * np.arange(0, m + 1)
    n_out = max(1, math.ceil((1.0 - fine[-1]) / h_coarse))
    upper = np.concatenate([fine, np.linspace(fine[-1], 1.0, n_out + 1)[1:]])
    return np.concatenate([1.0 - upper[::-1], upper[1:]])


@dataclass(frozen=True)
class SENTProblem:
    """Single-edge-notched tension in plane strain (Miehe et al. 2010).

    Unit square with a slit along y = 0.5 for x < 0.5, bottom clamped,
    top pulled by u_y = delta with u_x = 0. Units: kN, mm.
    """

    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    ell: float = 0.015
    model: str = "at1"
    g0: float = 1e-6
    h_fine: float = 0.0075      # mesh size near the crack path (ell / 2)
    h_coarse: float = 0.03
    band: float = 0.06          # half-width of the refined band around y = 0.5
    increment: float = 1e-4     # displacement per load step
    max_displacement: float = 8e-3
    stop_ratio: float = 0.02    # stop once the reaction falls below this fraction of its peak
    staggered_tol: float = 1e-4  # free-dof residual / reaction at which a load step is converged
    relaxation: float = 1.0     # over-relaxation of the damage update (1: plain alternate minimization)


def default_linear_solver(device: str) -> str:
    """cuDSS on a GPU when torch-sla has it, SciPy's direct solver otherwise
    (the iterative fallback struggles with the stiffness contrast of a crack)."""
    from torch_sla import is_cudss_available

    if str(device).startswith("cuda") and not is_cudss_available():
        print("cuDSS is not available: the Newton systems are solved with SciPy on the host")
        return "scipy"
    return "auto"


def run_demo(problem: SENTProblem = SENTProblem(), device: str = "cpu", make_plot: bool = True,
             verbose: bool = True, linear_solver: str | None = None):
    """Run the benchmark; returns a dict with the load-displacement curve,
    energies, iteration counts and the final fields."""
    if linear_solver is None:
        linear_solver = default_linear_solver(device)
    x = np.linspace(0.0, 1.0, round(1.0 / problem.h_fine) + 1)
    y = graded_coordinates(problem.h_fine, problem.h_coarse, problem.band)
    mesh = structured_triangle_mesh(x, y, slit=(0.5, 0.5)).to(device)
    pts = mesh.points
    n = mesh.n_points
    bottom, top = pts[:, 1] < 1e-12, pts[:, 1] > 1 - 1e-12
    u_fixed = (bottom | top)[:, None].expand(-1, 2).reshape(-1)
    top_y = torch.zeros(n, 2, dtype=torch.bool, device=device)
    top_y[top, 1] = True
    top_y = top_y.reshape(-1)
    model = PhaseFieldModel(problem.E, problem.nu, problem.Gc, problem.ell, problem.model, problem.g0)
    solver = StaggeredPhaseField(mesh, model, u_fixed, tol=problem.staggered_tol, relaxation=problem.relaxation,
                                 linear_solver=linear_solver)
    if verbose:
        print(f"SENT, {problem.model.upper()}, ell = {problem.ell} mm: {n} nodes, {mesh.n_elements} triangles, "
              f"device {device}")

    displacement, reaction, elastic, fracture, staggered = [0.0], [0.0], [0.0], [0.0], []
    snapshots = {0: solver.alpha.cpu().numpy()}
    t0 = time.time()
    k = 0
    while (k + 1) * problem.increment <= problem.max_displacement + 1e-15:
        k += 1
        delta = k * problem.increment
        u_bc = torch.where(top_y, torch.full_like(solver.u, delta), torch.zeros_like(solver.u))
        its = solver.solve_step(u_bc)
        e_el, e_fr = solver.energy()
        displacement.append(delta)
        reaction.append(solver.reaction(top_y))
        elastic.append(e_el)
        fracture.append(e_fr)
        staggered.append(its)
        snapshots[k] = solver.alpha.cpu().numpy()
        if verbose:
            print(f"step {k:3d}  delta = {delta:.2e} mm  F = {reaction[-1]:.4f} kN  "
                  f"E_frac = {e_fr:.4e}  max alpha = {float(solver.alpha.max()):.3f}  staggered its = {its}")
        if reaction[-1] < problem.stop_ratio * max(reaction):
            break
    result = {
        "displacement": np.array(displacement), "reaction": np.array(reaction),
        "elastic_energy": np.array(elastic), "fracture_energy": np.array(fracture),
        "staggered": np.array(staggered), "points": pts.cpu().numpy(),
        "triangles": mesh.cells["triangle"].cpu().numpy(), "alpha": solver.alpha.cpu().numpy(),
        "u": solver.u.reshape(-1, 2).cpu().numpy(), "snapshots": snapshots,
        "model": model, "seconds": time.time() - t0,
    }
    if verbose:
        print(f"peak load {result['reaction'].max():.4f} kN at delta = "
              f"{result['displacement'][result['reaction'].argmax()]:.2e} mm; "
              f"{len(staggered)} load steps in {result['seconds']:.1f} s")
    if make_plot:
        plot(result, problem)
    return result


def plot(result, problem: SENTProblem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    OUTPUT_DIR.mkdir(exist_ok=True)
    pts, cells, u = result["points"], result["triangles"], result["u"]
    F, d = result["reaction"], result["displacement"]
    peak, last = int(F.argmax()), len(F) - 1
    deformed = pts + 0.04 / max(np.abs(u).max(), 1e-30) * u  # displacements magnified
    style = dict(shading="gouraud", cmap="inferno", vmin=0, vmax=1)

    fig = plt.figure(figsize=(12, 3.9), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax = fig.add_subplot(grid[0, 0])
    ax.tripcolor(Triangulation(pts[:, 0], pts[:, 1], cells), result["snapshots"][peak], **style)
    ax.plot([0, 0.5], [0.5, 0.5], color="w", lw=1)
    ax.set(xlim=(0.42, 0.62), ylim=(0.4, 0.6), xticks=[], yticks=[], aspect="equal")
    ax.set_title(f"peak load: process zone at the slit tip\nδ = {d[peak] * 1e3:.1f}·10⁻³ mm, "
                 f"F = {F[peak]:.3f} kN", fontsize=10)
    ax = fig.add_subplot(grid[0, 1])
    im = ax.tripcolor(Triangulation(deformed[:, 0], deformed[:, 1], cells), result["snapshots"][last], **style)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"broken (displacements magnified)\nδ = {d[last] * 1e3:.1f}·10⁻³ mm, "
                 f"F = {F[last]:.3f} kN", fontsize=10)
    fig.colorbar(im, ax=fig.axes[:2], shrink=0.8, label="damage α")
    ax = fig.add_subplot(grid[0, 2])
    ax.plot(d * 1e3, F, "k.-", lw=1.2, ms=3)
    ax.plot(d[[peak, last]] * 1e3, F[[peak, last]], "o", color="tab:orange")
    ax.set_xlabel("top displacement δ [10⁻³ mm]")
    ax.set_ylabel("reaction force F [kN]")
    ax.set_title(f"{problem.model.upper()}, ℓ = {problem.ell} mm", fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "phase_field_fracture.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4.2), constrained_layout=True)
    ax.tripcolor(Triangulation(deformed[:, 0], deformed[:, 1], cells), result["snapshots"][last], **style)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "phase_field_fracture_crack.png", dpi=150)
    plt.close(fig)
    print(f"figures: {OUTPUT_DIR / 'phase_field_fracture.png'}, {OUTPUT_DIR / 'phase_field_fracture_crack.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["at1", "at2"], default="at1")
    p.add_argument("--ell", type=float, default=0.015, help="phase-field length scale [mm]")
    p.add_argument("--h", type=float, default=None, help="mesh size near the crack path (default ell / 2)")
    p.add_argument("--increment", type=float, default=1e-4, help="displacement per load step [mm]")
    p.add_argument("--relaxation", type=float, default=1.0,
                   help="over-relaxation of the damage update (about 1.6 needs fewer iterations)")
    p.add_argument("--max_displacement", type=float, default=8e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--linear_solver", default=None,
                   help="torch-sla backend for the Newton systems (default: cuDSS on GPU if available, else SciPy)")
    a = p.parse_args()
    problem = SENTProblem(model=a.model, ell=a.ell, increment=a.increment, max_displacement=a.max_displacement,
                          relaxation=a.relaxation)
    h = a.ell / 2 if a.h is None else a.h
    problem = replace(problem, h_fine=h, band=4 * a.ell, h_coarse=max(4 * h, problem.h_coarse))
    run_demo(problem, device=a.device, linear_solver=a.linear_solver)


if __name__ == "__main__":
    main()

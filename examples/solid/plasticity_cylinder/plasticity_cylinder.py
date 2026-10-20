"""J2 plasticity with Newton's method: a thick-walled cylinder under internal pressure.

A long tube (inner radius a, outer radius b) is loaded by an internal pressure
until the whole wall flows plastically. Plane strain, small strain, von Mises
(J2) plasticity with linear isotropic hardening (perfect plasticity by default).
The textbook solution (Hill, 1950) gives two checks:

- elastic range (Lame): the inner-wall displacement grows linearly with the
  pressure, and the inner wall first yields at the pressure p_e;
- collapse: an elastic-perfectly plastic tube cannot carry more than the limit
  pressure p_lim = (2 / sqrt(3)) sigma_y ln(b / a).

Every load step is solved by Newton's method with the consistent (algorithmic)
tangent of the radial return map, so the iterations converge quadratically.
The plastic state lives at the quadrature points and enters the assemblers as
``quadrature_data``: the stress for the internal forces, the tangent moduli for
the stiffness matrix. The pressure acts along the outward facet normal ``n``
of a ``FacetAssembler``. One quarter of the tube is modelled with symmetry
conditions, meshed with quadratic triangles (no gmsh needed).

Units: N and mm (stresses in MPa).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch

# Allow running this file directly from the source tree.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tensormesh import Condenser, ElementAssembler, FacetAssembler, Mesh, NodeAssembler  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


# --------------------------------------------------------------------------- #
# Material: J2 plasticity, radial return and its consistent tangent           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class J2Material:
    """Isotropic elasticity, von Mises yield, linear isotropic hardening.

    The yield stress grows with the equivalent plastic strain p as
    sigma_y + H p; H = 0 is perfect plasticity.
    """

    E: float = 200e3        # MPa
    nu: float = 0.3
    sigma_y: float = 250.0  # MPa
    H: float = 0.0          # MPa

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def kappa(self):
        return self.E / (3 * (1 - 2 * self.nu))


def radial_return(eps, eps_p, p, mat: J2Material):
    """Stress update of J2 plasticity at every quadrature point.

    Parameters
    ----------
    eps : torch.Tensor
        Total strain ``[..., 3, 3]`` at the end of the step (plane strain:
        the third row and column are zero).
    eps_p, p : torch.Tensor
        Plastic strain ``[..., 3, 3]`` and equivalent plastic strain ``[...]``
        at the start of the step.

    Returns
    -------
    sigma : torch.Tensor
        Stress ``[..., 3, 3]``.
    eps_p, p : torch.Tensor
        Updated plastic state.
    C : torch.Tensor
        Consistent tangent d sigma / d eps ``[..., 3, 3, 3, 3]``:
        kappa 1x1 + 2 mu (1 - 3 mu dp / q) I_dev + 6 mu^2 (dp / q - 1 / (3 mu + H)) N x N
        on plastic points, kappa 1x1 + 2 mu I_dev on elastic ones, with q the
        trial von Mises stress and N the unit trial deviator
        (de Souza Neto, Peric and Owen, 2008, box 7.4).
    """
    mu, kappa, H = mat.mu, mat.kappa, mat.H
    one = torch.eye(3, dtype=eps.dtype, device=eps.device)
    ee = eps - eps_p                                                # trial elastic strain
    tr = ee.diagonal(dim1=-2, dim2=-1).sum(-1)
    s = 2 * mu * (ee - tr[..., None, None] / 3 * one)              # trial deviatoric stress
    norm_s = torch.sqrt((s * s).sum((-2, -1)))
    q = math.sqrt(1.5) * norm_s                                     # trial von Mises stress
    f = q - (mat.sigma_y + H * p)
    plastic = f > 0
    dp = torch.where(plastic, f, torch.zeros_like(f)) / (3 * mu + H)
    N = s / torch.where(plastic, norm_s, torch.ones_like(norm_s))[..., None, None]
    flow = math.sqrt(1.5) * dp[..., None, None] * N                 # plastic strain increment
    sigma = kappa * tr[..., None, None] * one + s - 2 * mu * flow

    q_safe = torch.where(plastic, q, torch.ones_like(q))
    a = torch.where(plastic, 3 * mu * dp / q_safe, torch.zeros_like(q))
    b = torch.where(plastic, 6 * mu**2 * (dp / q_safe - 1 / (3 * mu + H)), torch.zeros_like(q))
    ones = torch.einsum("ij,kl->ijkl", one, one)
    sym = 0.5 * (torch.einsum("ik,jl->ijkl", one, one) + torch.einsum("il,jk->ijkl", one, one))
    C = (kappa * ones + (2 * mu * (1 - a))[..., None, None, None, None] * (sym - ones / 3)
         + b[..., None, None, None, None] * torch.einsum("...ij,...kl->...ijkl", N, N))
    return sigma, eps_p + flow, p + dp, C


def von_mises(sigma):
    s = sigma - sigma.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3 * torch.eye(
        3, dtype=sigma.dtype, device=sigma.device)
    return torch.sqrt(1.5 * (s * s).sum((-2, -1)))


# --------------------------------------------------------------------------- #
# Assemblers: the plastic state enters as quadrature data                     #
# --------------------------------------------------------------------------- #

class InternalForce(NodeAssembler):
    """f_i = int sigma . grad N_i dx, sigma given at the quadrature points."""

    def forward(self, gradv, sigma):
        return sigma @ gradv


class Tangent(ElementAssembler):
    """K_ij = int grad N_i . C . grad N_j dx, C given at the quadrature points."""

    def forward(self, gradu, gradv, C):
        return torch.einsum("j,ajbl,l->ab", gradu, C, gradv)


class Pressure(FacetAssembler):
    """Nodal forces of a unit pressure, f_i = -int N_i n ds (n: outward normal)."""

    def forward(self, v, n):
        return -v[:, None] * n


# --------------------------------------------------------------------------- #
# Newton solver for the load steps                                            #
# --------------------------------------------------------------------------- #

class J2NewtonSolver:
    """Small-strain J2 plasticity in plane strain on quadratic triangles.

    Each call of :meth:`solve_step` finds the displacement in equilibrium with
    the external forces for prescribed values on the ``fixed`` dofs, by
    Newton's method with the consistent tangent, and then commits the plastic
    state.

    Parameters
    ----------
    mesh : tensormesh.Mesh
    material : J2Material
    fixed : torch.Tensor
        Bool mask over the displacement dofs ``[n_points * 2]`` (node-major)
        with prescribed values.
    quadrature_order : int
        Volume quadrature (2: three points per triangle).
    tol : float
        Newton stops once the residual on the free dofs is below ``tol`` times
        the larger of the external and reaction forces.
    max_iterations : int
        Newton updates allowed per load step.
    linear_solver : str
        torch-sla backend for the Newton systems.
    """

    def __init__(self, mesh, material: J2Material, fixed, quadrature_order=2, tol=1e-10, max_iterations=25,
                 linear_solver="auto"):
        self.material, self.fixed = material, fixed
        self.tol, self.max_iterations, self.linear_solver = tol, max_iterations, linear_solver
        self.tangent = Tangent.from_mesh(mesh, quadrature_order=quadrature_order)
        self.internal = InternalForce.from_mesh(mesh, quadrature_order=quadrature_order)
        element_type = mesh.default_element_type
        trans = self.tangent.transformation[element_type]
        self.elements = self.tangent.elements[element_type]   # [n_elements, 6]
        self.shape_grad = trans.shape_grad                     # [n_elements, n_quadrature, 6, 2]
        self.JxW = trans.JxW                                   # [n_elements, n_quadrature]
        shape = self.JxW.shape
        dev, dtype = mesh.points.device, mesh.points.dtype
        self.eps_p = torch.zeros(*shape, 3, 3, dtype=dtype, device=dev)
        self.p = torch.zeros(shape, dtype=dtype, device=dev)
        self.u = torch.zeros(mesh.n_points * 2, dtype=dtype, device=dev)
        self.condenser = Condenser(fixed)   # the fixed dofs move only in the first iteration of a step
        self.sigma = torch.zeros(*shape, 3, 3, dtype=dtype, device=dev)

    def strain(self, u):
        """Plane-strain strain [n_elements, n_quadrature, 3, 3] of the nodal displacement u."""
        grad = torch.einsum("eqbd,ebi->eqid", self.shape_grad, u.reshape(-1, 2)[self.elements])
        eps = torch.zeros(*grad.shape[:2], 3, 3, dtype=u.dtype, device=u.device)
        eps[..., :2, :2] = 0.5 * (grad + grad.transpose(-1, -2))
        return eps

    def residual(self, u, f_ext):
        """Internal minus external forces, and the trial state of the return map."""
        sigma, eps_p, p, C = radial_return(self.strain(u), self.eps_p, self.p, self.material)
        R = self.internal(quadrature_data={"sigma": sigma[..., :2, :2]}).reshape(-1) - f_ext
        return R, (sigma, eps_p, p, C)

    def solve_step(self, f_ext, prescribed=None):
        """Equilibrium for the external forces ``f_ext`` [n_points * 2] and the
        values ``prescribed`` on the fixed dofs (default: keep them).

        The first Newton iteration starts from the converged state and carries
        the change of the prescribed values into the whole body (the linear
        predictor); the later ones keep them. Returns the relative residual
        after every Newton update.
        """
        jump = torch.zeros_like(self.u)
        if prescribed is not None:
            jump = torch.where(self.fixed, prescribed - self.u, jump)
        history = []
        for it in range(self.max_iterations + 1):
            R, (sigma, eps_p, p, C) = self.residual(self.u, f_ext)
            if it > 0:
                scale = max(float(torch.linalg.vector_norm(f_ext)), float(torch.linalg.vector_norm(R[self.fixed])),
                            1e-300)
                history.append(float(torch.linalg.vector_norm(R[~self.fixed])) / scale)
                if history[-1] <= self.tol:
                    break
            if it == self.max_iterations:
                raise RuntimeError(f"Newton did not converge: relative residuals {history}")
            K = self.tangent(quadrature_data={"C": C[..., :2, :2, :2, :2]})
            self.condenser.update_dirichlet(jump if it == 0 else torch.zeros_like(jump))
            K_free, rhs = self.condenser(K, -R)
            du = K_free.solve(rhs, backend=self.linear_solver).to(rhs.device)   # SciPy answers on the host
            self.u = self.u + self.condenser.recover(du)
        self.eps_p, self.p, self.sigma = eps_p, p, sigma   # commit the converged state
        self.reaction = R
        return history


# --------------------------------------------------------------------------- #
# Thick-walled cylinder                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CylinderProblem:
    """Quarter of a tube a <= r <= b in plane strain, loaded by internal pressure."""

    inner: float = 100.0     # a, mm
    outer: float = 200.0     # b, mm
    material: J2Material = J2Material()
    n_radial: int = 12       # quadratic triangles: 2 per cell of the polar grid
    n_angular: int = 16
    control: str = "displacement"   # or "pressure"
    steps: int = 30
    max_load: float = 6.0    # last load: inner displacement / first-yield displacement,
    #                          or (pressure control) pressure / limit pressure


def quarter_ring_mesh(inner, outer, n_radial, n_angular):
    """Six-node triangles on the quarter ring inner <= r <= outer, 0 <= theta <= pi/2.

    The radial spacing grows with r (r_i = inner (outer / inner)^(i / n_radial)),
    the angular spacing is uniform, and the mid-edge nodes of the arcs lie on
    the circles, so the curved boundaries are second-order accurate. The cell
    diagonals alternate. Returns the mesh and boolean node masks of the inner,
    outer and two straight edges.
    """
    r_v = inner * (outer / inner) ** (np.arange(n_radial + 1) / n_radial)
    r = np.empty(2 * n_radial + 1)
    r[0::2], r[1::2] = r_v, 0.5 * (r_v[:-1] + r_v[1:])
    t = np.linspace(0.0, 0.5 * np.pi, 2 * n_angular + 1)
    R, T = np.meshgrid(r, t, indexing="ij")
    points = np.stack([R * np.cos(T), R * np.sin(T)], -1)
    points[:, 0, 1] = 0.0    # theta = 0: exactly on the x axis
    points[:, -1, 0] = 0.0   # theta = pi/2: exactly on the y axis
    nid = np.arange(R.size).reshape(R.shape)
    cells = []
    for i in range(n_radial):
        for j in range(n_angular):
            A, B, C, D = nid[2 * i, 2 * j], nid[2 * i + 2, 2 * j], nid[2 * i + 2, 2 * j + 2], nid[2 * i, 2 * j + 2]
            AB, BC, CD, DA = nid[2 * i + 1, 2 * j], nid[2 * i + 2, 2 * j + 1], nid[2 * i + 1, 2 * j + 2], nid[2 * i, 2 * j + 1]
            M = nid[2 * i + 1, 2 * j + 1]
            if (i + j) % 2 == 0:   # diagonal A-C; Gmsh/VTK order: corners, then edges 01, 12, 20
                cells += [[A, B, C, AB, BC, M], [A, C, D, M, CD, DA]]
            else:                  # diagonal B-D
                cells += [[A, B, D, AB, M, DA], [B, C, D, BC, CD, M]]
    mesh = Mesh(meshio.Mesh(points.reshape(-1, 2), [("triangle6", np.array(cells))]), reorder=True)
    masks = {name: np.zeros(R.shape, dtype=bool) for name in ("inner", "outer", "x_axis", "y_axis")}
    masks["inner"][0, :], masks["outer"][-1, :] = True, True
    masks["x_axis"][:, 0], masks["y_axis"][:, -1] = True, True
    return mesh, {k: torch.from_numpy(v.reshape(-1)) for k, v in masks.items()}


def analytical(problem: CylinderProblem):
    """Elastic stiffness p / u_r(a) (Lame, plane strain), the first-yield pressure
    p_e (von Mises at r = a) and the limit pressure of perfect plasticity."""
    a, b, m = problem.inner, problem.outer, problem.material
    stiffness = m.E * (b**2 - a**2) / ((1 + m.nu) * a * ((1 - 2 * m.nu) * a**2 + b**2))
    hoop = (b**2 + a**2) / (b**2 - a**2)           # sigma_tt / p at r = a
    axial = 2 * m.nu * a**2 / (b**2 - a**2)        # sigma_zz / p
    p_e = m.sigma_y / math.sqrt(0.5 * ((1 + hoop) ** 2 + (hoop - axial) ** 2 + (1 + axial) ** 2))
    p_lim = 2 / math.sqrt(3) * m.sigma_y * math.log(b / a)
    return {"stiffness": stiffness, "p_e": p_e, "u_e": p_e / stiffness, "p_lim": p_lim}


def default_linear_solver(device: str) -> str:
    """cuDSS on a GPU when torch-sla has it, SciPy's direct solver otherwise."""
    from torch_sla import is_cudss_available

    if str(device).startswith("cuda") and not is_cudss_available():
        print("cuDSS is not available: the Newton systems are solved with SciPy on the host")
        return "scipy"
    return "auto"


def run_demo(problem: CylinderProblem = CylinderProblem(), device: str = "cpu", make_plot: bool = True,
             verbose: bool = True, linear_solver: str | None = None):
    """Load the tube step by step; returns the pressure-displacement curve, the
    Newton histories, the analytical references and the final fields."""
    if problem.control not in ("displacement", "pressure"):
        raise ValueError(f"control must be 'displacement' or 'pressure', got {problem.control!r}")
    if linear_solver is None:
        linear_solver = default_linear_solver(device)
    ref = analytical(problem)
    mesh, masks = quarter_ring_mesh(problem.inner, problem.outer, problem.n_radial, problem.n_angular)
    mesh = mesh.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}
    pts = mesh.points
    n = mesh.n_points

    # symmetry: u_y = 0 on the x axis, u_x = 0 on the y axis; displacement control also fixes the inner wall
    fixed = torch.zeros(n, 2, dtype=torch.bool, device=device)
    fixed[masks["x_axis"], 1] = True
    fixed[masks["y_axis"], 0] = True
    if problem.control == "displacement":
        fixed[masks["inner"]] = True
    fixed = fixed.reshape(-1)
    radial = (pts / torch.linalg.vector_norm(pts, dim=1, keepdim=True)).reshape(-1)   # e_r at every node

    unit = Pressure.from_mesh(mesh, boundary_mask=masks["inner"])().reshape(-1)       # forces of p = 1
    inner_dofs = masks["inner"][:, None].expand(-1, 2).reshape(-1)

    def pressure_of(reaction):
        """Pressure on the inner wall: the reaction projected on the unit-pressure forces."""
        return float((reaction[inner_dofs] @ unit[inner_dofs]) / (unit[inner_dofs] @ unit[inner_dofs]))

    solver = J2NewtonSolver(mesh, problem.material, fixed, linear_solver=linear_solver)
    if verbose:
        print(f"thick cylinder a = {problem.inner} mm, b = {problem.outer} mm, {problem.control} control: "
              f"{n} nodes, {mesh.n_elements} quadratic triangles, device {device}")
        print(f"Lame stiffness {ref['stiffness']:.2f} MPa/mm, first yield p_e = {ref['p_e']:.2f} MPa, "
              f"limit pressure p_lim = {ref['p_lim']:.2f} MPa")

    displacement, pressure, histories, plastic_fraction = [0.0], [0.0], [], [0.0]
    area = float(solver.JxW.sum())
    t0 = time.time()
    for k in range(1, problem.steps + 1):
        load = problem.max_load * k / problem.steps
        if problem.control == "displacement":
            delta = load * ref["u_e"]
            history = solver.solve_step(torch.zeros_like(solver.u), prescribed=delta * radial)
            p_k = pressure_of(solver.reaction)
        else:
            p_k = load * ref["p_lim"]
            history = solver.solve_step(p_k * unit)
        u_inner = float((solver.u * radial)[inner_dofs].reshape(-1, 2).sum(1).mean())
        displacement.append(u_inner)
        pressure.append(p_k)
        histories.append(history)
        plastic_fraction.append(float((solver.JxW * (solver.p > 0)).sum()) / area)
        if verbose:
            print(f"step {k:3d}  u_r(a) = {u_inner:.4f} mm  p = {p_k:7.2f} MPa  plastic area {plastic_fraction[-1]:6.1%}"
                  f"  Newton: {len(history) - 1} its, residuals " + " ".join(f"{h:.0e}" for h in history))
    result = {
        "displacement": np.array(displacement), "pressure": np.array(pressure), "newton": histories,
        "plastic_fraction": np.array(plastic_fraction), "analytical": ref, "seconds": time.time() - t0,
        "points": pts.cpu().numpy(), "cells": mesh.cells[mesh.default_element_type].cpu().numpy(),
        "u": solver.u.reshape(-1, 2).cpu().numpy(), "p": solver.p.cpu().numpy(),
        "von_mises": von_mises(solver.sigma).cpu().numpy(), "solver": solver, "mesh": mesh, "masks": masks,
    }
    if verbose:
        print(f"{problem.steps} load steps in {result['seconds']:.1f} s; final pressure {pressure[-1]:.2f} MPa "
              f"= {pressure[-1] / ref['p_lim']:.4f} p_lim")
    if make_plot:
        plot(result, problem)
    return result


def plot(result, problem: CylinderProblem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    OUTPUT_DIR.mkdir(exist_ok=True)
    ref = result["analytical"]
    pts, u = result["points"], result["u"]
    # each quadratic triangle drawn as four linear ones, coloured by its mean plastic strain
    cells = result["mesh"].to_meshio(reorder=True).cells_dict["triangle6"]
    sub = np.concatenate([cells[:, [0, 3, 5]], cells[:, [3, 1, 4]], cells[:, [5, 4, 2]], cells[:, [3, 4, 5]]])
    p_cell = np.tile(result["p"].mean(1), 4)
    magnify = round(5.0 / max(np.abs(u).max(), 1e-30))     # the largest displacement drawn as about 5 mm
    deformed = pts + magnify * u

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1])
    ax = fig.add_subplot(grid[0, 0])
    im = ax.tripcolor(Triangulation(deformed[:, 0], deformed[:, 1], sub), facecolors=p_cell * 100, cmap="magma_r")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="equivalent plastic strain p [%]")
    ax.set_title(f"end of loading: {result['plastic_fraction'][-1]:.0%} of the wall plastic\n"
                 f"(displacements x{magnify})", fontsize=10)

    ax = fig.add_subplot(grid[0, 1])
    d, P = result["displacement"], result["pressure"]
    ax.plot(d, P, "k.-", lw=1.2, ms=4, label="TensorMesh")
    d_el = np.linspace(0, ref["u_e"], 2)
    ax.plot(d_el, ref["stiffness"] * d_el, color="tab:blue", lw=2, label="Lamé (elastic)")
    ax.plot([ref["u_e"]], [ref["p_e"]], "o", color="tab:blue", label="first yield $p_e$")
    ax.axhline(ref["p_lim"], color="tab:orange", lw=1.5, label=r"limit pressure $\frac{2}{\sqrt{3}}\sigma_y\ln\frac{b}{a}$")
    ax.set_xlabel("inner-wall displacement $u_r(a)$ [mm]")
    ax.set_ylabel("pressure $p$ [MPa]")
    ax.set_ylim(0, 1.15 * max(ref["p_lim"], P.max()))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    fraction = result["plastic_fraction"][1:]
    plastic = np.flatnonzero(fraction > 0)
    shown = {0, len(fraction) - 1}
    if len(plastic):
        shown |= {int(plastic[0]), int(plastic[np.argmin(np.abs(fraction[plastic] - 0.5))])}
    for k in sorted(shown):
        h = np.array(result["newton"][k])
        h = h[h > 0]
        ax.semilogy(np.arange(1, len(h) + 1), h, "o-", ms=4, label=f"step {k + 1}, p = {P[k + 1]:.0f} MPa")
    ax.axhline(1e-10, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("Newton update")
    ax.set_ylabel("relative residual")
    ax.set_title("consistent tangent: quadratic convergence", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "plasticity_cylinder.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.tripcolor(Triangulation(deformed[:, 0], deformed[:, 1], sub), facecolors=p_cell * 100, cmap="magma_r")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "plasticity_cylinder_field.png", dpi=150)
    plt.close(fig)
    print(f"figures: {OUTPUT_DIR / 'plasticity_cylinder.png'}, {OUTPUT_DIR / 'plasticity_cylinder_field.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", choices=["displacement", "pressure"], default="displacement")
    p.add_argument("--hardening", type=float, default=0.0, help="isotropic hardening modulus H [MPa]")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--max_load", type=float, default=None,
                   help="last inner displacement in units of the first-yield displacement (default 6), "
                        "or the last pressure in units of the limit pressure under pressure control (default 0.98)")
    p.add_argument("--n_radial", type=int, default=12)
    p.add_argument("--n_angular", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--linear_solver", default=None,
                   help="torch-sla backend for the Newton systems (default: cuDSS on GPU if available, else SciPy)")
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    max_load = a.max_load if a.max_load is not None else (6.0 if a.control == "displacement" else 0.98)
    problem = CylinderProblem(material=J2Material(H=a.hardening), n_radial=a.n_radial, n_angular=a.n_angular,
                              control=a.control, steps=a.steps, max_load=max_load)
    run_demo(problem, device=a.device, linear_solver=a.linear_solver)


if __name__ == "__main__":
    main()

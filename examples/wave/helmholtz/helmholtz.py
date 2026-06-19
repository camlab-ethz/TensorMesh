"""Complex-valued Helmholtz on the unit square — manufactured solution.

Validates the complex-coefficient FEM assembly path landed in the
ROADMAP-item-2 unblock PR. We solve the interior Helmholtz problem

    -∇²u(x,y) - k² u(x,y) = 0      in Ω = (0,1)²
                  u(x,y)  = g(x,y) on ∂Ω

with the analytic plane-wave solution

    u_exact(x,y) = exp(i k x)

so that g = u_exact|_{∂Ω}. The body force is zero because
``-∇²(e^{ikx}) = k² e^{ikx}``, cancelling the ``-k² u`` mass term.

What this exercises end-to-end:

* Complex-valued ``point_data`` flowing through ``ElementAssembler``
  (the assembler is dtype-agnostic; complex enters through ``k_sq``).
* Complex Dirichlet condensation in ``Condenser`` — boundary values
  ``g`` are complex; the inner-system RHS becomes complex.
* Complex linear solve via ``SparseMatrix.solve`` (delegates to
  torch-sla, which carries the matching complex adjoint and
  LDLᵀ / LDLᴴ factorisations).
* End-to-end convergence: L2 error against ``u_exact`` should
  decrease at the expected FEM rate under mesh refinement.

Run directly to see the L2 error + write the convergence plot:

    python examples/wave/helmholtz/helmholtz.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch

from tensormesh import ElementAssembler, Condenser, Mesh
from tensormesh.dataset.mesh import gen_rectangle


# --------------------------------------------------------------------- #
# Bilinear form
# --------------------------------------------------------------------- #
class HelmholtzAssembler(ElementAssembler):
    r"""Complex Helmholtz stiffness :math:`a(u,v) = \int_\Omega \nabla u \cdot \nabla v - k^2 u v\, d\Omega`.

    The ``k_sq`` point_data carries :math:`k^2` (per-node, may be complex
    to model damping / PML in follow-ups). Keeping it as a point_data
    coefficient lets the underlying ``__call__`` dispatch broadcast
    over elements and quadrature, and is the same mechanism PML
    coefficients will use.
    """

    def forward(self, gradu, gradv, u, v, k_sq):
        return gradu @ gradv - k_sq * u * v


# --------------------------------------------------------------------- #
# Manufactured solution + setup
# --------------------------------------------------------------------- #
def u_exact(points: torch.Tensor, k: float, dtype: torch.dtype) -> torch.Tensor:
    """Plane wave ``exp(i k x)`` evaluated at ``points`` (shape ``[n, dim]``)."""
    x = points[..., 0].to(dtype=torch.float64)
    return torch.exp(1j * k * x).to(dtype)


def _solve_one_mesh(chara_length: float, k: float, dtype: torch.dtype, device: str
                    ) -> Tuple[float, int]:
    """Build, condense, and solve Helmholtz on a mesh; return L2 error + n_dofs."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    mesh = gen_rectangle(chara_length=chara_length, element_type="tri",
                         left=0.0, right=1.0, bottom=0.0, top=1.0)
    points = mesh.points.to(real_dtype).to(device)

    # k_sq is a complex per-node coefficient (constant here; PML follow-up
    # would make it anisotropic + spatially varying inside the PML layer).
    k_sq_field = torch.full((mesh.n_points,), k * k + 0j, dtype=dtype, device=device)

    asm = HelmholtzAssembler.from_mesh(mesh, quadrature_order=3)
    asm.type(dtype).to(device)

    H = asm(points=points, point_data={"k_sq": k_sq_field})

    # Dirichlet ``g`` from u_exact on every boundary node.
    g = u_exact(mesh.points.to(device), k=k, dtype=dtype)
    boundary_mask = mesh.boundary_mask.to(device)

    rhs = torch.zeros(mesh.n_points, dtype=dtype, device=device)
    condenser = Condenser(boundary_mask, dirichlet_value=g[boundary_mask])
    H_inner, rhs_inner = condenser(H, rhs)
    u_inner = H_inner.solve(rhs_inner)
    u = condenser.recover(u_inner)

    # L2 error via the mass matrix on interior dofs (cheap reusable proxy).
    diff = u - g
    err2 = (diff.conj() * diff).real.sum().item()
    err_l2 = (err2 / mesh.n_points) ** 0.5

    return err_l2, mesh.n_points


def run_convergence(k: float, chara_lengths: List[float],
                    dtype: torch.dtype, device: str
                    ) -> List[Tuple[float, int, float]]:
    """Run one solve per ``chara_length``; return (h, n_dofs, L2 err) triples."""
    results = []
    for h in chara_lengths:
        err, n = _solve_one_mesh(chara_length=h, k=k, dtype=dtype, device=device)
        results.append((h, n, err))
        print(f"  h={h:.3f}  n_dofs={n:5d}  L2 err = {err:.3e}")
    return results


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
def plot_solution(k: float, chara_length: float, dtype: torch.dtype,
                  device: str, out_path: str | os.PathLike) -> None:
    """Solve on one mesh and write a 3-panel figure (Re, Im, |u-u_exact|)."""
    import matplotlib.pyplot as plt
    import numpy as np

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    mesh = gen_rectangle(chara_length=chara_length, element_type="tri",
                         left=0.0, right=1.0, bottom=0.0, top=1.0)
    points = mesh.points.to(real_dtype).to(device)
    k_sq_field = torch.full((mesh.n_points,), k * k + 0j, dtype=dtype, device=device)

    asm = HelmholtzAssembler.from_mesh(mesh, quadrature_order=3)
    asm.type(dtype).to(device)
    H = asm(points=points, point_data={"k_sq": k_sq_field})

    g = u_exact(mesh.points.to(device), k=k, dtype=dtype)
    boundary_mask = mesh.boundary_mask.to(device)
    rhs = torch.zeros(mesh.n_points, dtype=dtype, device=device)
    condenser = Condenser(boundary_mask, dirichlet_value=g[boundary_mask])
    H_inner, rhs_inner = condenser(H, rhs)
    u_inner = H_inner.solve(rhs_inner)
    u = condenser.recover(u_inner).detach().cpu().numpy()
    g_np = g.detach().cpu().numpy()
    pts = mesh.points.cpu().numpy()

    # Triangulate for matplotlib.tri.
    import matplotlib.tri as mtri
    tris = mesh.cells["triangle"].cpu().numpy()
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles=tris)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("Re(u)", u.real),
        ("Im(u)", u.imag),
        ("|u - u_exact|", np.abs(u - g_np)),
    ]
    for ax, (title, data) in zip(axes, panels):
        tcf = ax.tricontourf(triang, data, levels=20, cmap="RdBu_r" if "Re" in title or "Im" in title else "viridis")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(tcf, ax=ax, shrink=0.85)

    fig.suptitle(f"Helmholtz manufactured solution  k={k:.2f}  h={chara_length:.2f}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2 * torch.pi,
                        help="wave number (default 2π)")
    parser.add_argument("--chara-length", type=float, default=0.1,
                        help="characteristic mesh size for the plotted solve")
    parser.add_argument("--dtype", choices=["complex64", "complex128"],
                        default="complex128")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).with_name("helmholtz.png")))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    print(f"device={device}  dtype={dtype}  k={args.k:.4f}")

    print("Convergence study:")
    run_convergence(
        k=args.k,
        chara_lengths=[0.2, 0.1, 0.05, 0.025],
        dtype=dtype,
        device=device,
    )

    if not args.no_plot:
        plot_solution(k=args.k, chara_length=args.chara_length,
                      dtype=dtype, device=device, out_path=args.output)


if __name__ == "__main__":
    main()

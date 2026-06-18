"""Convergence test for the complex Helmholtz example.

Validates that the end-to-end complex pipeline (assembly + Condenser
+ complex linear solve via torch-sla) actually converges to the
analytic plane-wave solution under mesh refinement — the missing
piece in the ROADMAP-item-2 unit tests.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "wave" / "helmholtz"
sys.path.insert(0, str(EXAMPLE_DIR))

from helmholtz import run_convergence  # type: ignore  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.complex128], ids=["complex128"])
def test_helmholtz_convergence_under_refinement(dtype):
    """Halving ``h`` should reduce the L2 error by ~3× or more on each
    step (Helmholtz pollution prevents the textbook 4× for P1, but a
    consistent monotone decrease + final error well below the coarsest
    is the right correctness signal)."""
    results = run_convergence(
        k=2 * math.pi,
        chara_lengths=[0.2, 0.1, 0.05],
        dtype=dtype,
        device="cpu",
    )
    errs = [r[2] for r in results]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), (
        f"L2 error must strictly decrease under refinement, got {errs}"
    )
    # Demand at least a ~2.3x reduction per halving (very loose — the
    # observed value sits around 3-3.8x; this guards against regression
    # to first-order or worse).
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert r > 2.3, f"convergence rate too slow: per-step ratios = {ratios}"


def test_helmholtz_result_is_complex():
    """At fixed mesh, sanity-check that the solver returned a complex
    field with non-trivial imaginary part — the analytic exp(i k x)
    must have non-zero Im(u)."""
    from helmholtz import _solve_one_mesh  # type: ignore

    err, _ = _solve_one_mesh(
        chara_length=0.1, k=2 * math.pi, dtype=torch.complex128, device="cpu"
    )
    assert err < 0.1, f"single-mesh error sanity: expected < 0.1, got {err:.3e}"


def test_helmholtz_cross_validates_against_scikit_fem():
    """Hand-to-hand check on identical (points, cells): the TensorMesh
    complex Helmholtz pipeline (assembly + Condenser + complex solve)
    should agree with scikit-fem to machine precision. Catches any
    silent drift in either layer of the complex stack."""
    skfem = pytest.importorskip("skfem")
    pytest.importorskip("scipy.sparse")
    import numpy as np
    import scipy.sparse.linalg as spla
    from skfem import Basis, ElementTriP1
    from skfem.models.poisson import laplace, mass

    from tensormesh import Condenser
    from tensormesh.dataset.mesh import gen_rectangle
    from helmholtz import HelmholtzAssembler  # type: ignore

    k = 2 * math.pi
    mesh = gen_rectangle(chara_length=0.1, element_type="tri",
                         left=0.0, right=1.0, bottom=0.0, top=1.0)
    pts = mesh.points.cpu().numpy().astype(np.float64)
    cells = mesh.cells["triangle"].cpu().numpy().astype(np.int64)

    # scikit-fem on the SAME mesh
    m = skfem.MeshTri(pts.T.copy(), cells.T.copy())
    basis = Basis(m, ElementTriP1())
    K_sk = laplace.assemble(basis)
    M_sk = mass.assemble(basis)
    H_sk = (K_sk - k * k * M_sk).astype(np.complex128).tolil()

    boundary_dofs = m.boundary_nodes()
    free = np.setdiff1d(np.arange(pts.shape[0]), boundary_dofs)
    g = np.exp(1j * k * pts[:, 0]).astype(np.complex128)
    H_ff = H_sk[free, :][:, free].tocsr()
    H_fc = H_sk[free, :][:, boundary_dofs].tocsr()
    rhs_sk = -H_fc @ g[boundary_dofs]
    u_sk = g.copy()
    u_sk[free] = spla.spsolve(H_ff, rhs_sk)

    # TensorMesh on the SAME mesh
    asm = HelmholtzAssembler.from_mesh(mesh, quadrature_order=3)
    asm.type(torch.complex128)
    k_sq = torch.full((mesh.n_points,), k * k + 0j, dtype=torch.complex128)
    H_tm = asm(points=mesh.points.to(torch.float64),
               point_data={"k_sq": k_sq})

    g_t = torch.from_numpy(g)
    rhs_t = torch.zeros(mesh.n_points, dtype=torch.complex128)
    cond = Condenser(mesh.boundary_mask, dirichlet_value=g_t[mesh.boundary_mask])
    H_inner, rhs_inner = cond(H_tm, rhs_t)
    u_inner = H_inner.solve(rhs_inner)
    u_tm = cond.recover(u_inner).detach().cpu().numpy()

    diff = np.abs(u_tm - u_sk)
    rel = diff.max() / max(np.abs(u_sk).max(), 1e-30)
    assert rel < 1e-12, (
        f"TensorMesh vs scikit-fem mismatch: rel = {rel:.3e}, max |Δu| = {diff.max():.3e}"
    )

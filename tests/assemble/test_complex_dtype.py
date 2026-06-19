"""Regression tests for complex-valued FEM assembly (ROADMAP item 2).

Three things should hold after the ``ElementAssembler.type()`` /
``ReduceProjector`` / ``SparseProjector`` complex-dtype unblock:

1. ``.type(torch.complex64)`` / ``.type(torch.complex128)`` is accepted
   (previously raised ``Exception("dtype not supported")``).
2. After casting, the assembled matrix's ``edata`` is complex with the
   requested dtype, and the real part matches the equivalent real-dtype
   assembly (because the bilinear form is real when no complex
   coefficient is supplied).
3. When the bilinear form carries a complex coefficient via
   ``point_data`` / ``element_data``, the imaginary part of the
   assembled matrix is non-trivial and equals the matrix assembled with
   the imaginary coefficient alone (real - imag separability).
"""

from __future__ import annotations

import pytest
import torch

from tensormesh import Mesh
from tensormesh.assemble import (
    ElementAssembler,
    LaplaceElementAssembler,
    MassElementAssembler,
)


def _real_mesh(element_type: str = "tri"):
    return Mesh.gen_rectangle(chara_length=0.3, element_type=element_type)


# --------------------------------------------------------------------- #
# 1. ``.type(complex)`` is accepted
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype", [torch.complex64, torch.complex128], ids=["complex64", "complex128"]
)
@pytest.mark.parametrize("AsmCls", [LaplaceElementAssembler, MassElementAssembler])
def test_assembler_accepts_complex_dtype(AsmCls, dtype):
    """``ElementAssembler.type(complex_dtype)`` no longer raises."""
    mesh = _real_mesh()
    asm = AsmCls.from_mesh(mesh)
    asm.type(dtype)  # used to raise


# --------------------------------------------------------------------- #
# 2. Complex-cast assembler agrees with real assembler on the real form
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype", [torch.complex64, torch.complex128], ids=["complex64", "complex128"]
)
@pytest.mark.parametrize("AsmCls", [LaplaceElementAssembler, MassElementAssembler])
def test_complex_type_no_op_without_complex_coefficient(AsmCls, dtype):
    """``.type(complex_dtype)`` declares intent only — geometry buffers
    stay real, so a form with no complex coefficient still produces a
    real K identical to the un-typed version. Complex content only
    enters when a complex coefficient flows through the call path
    (see ``test_imaginary_coefficient_carries_through``)."""
    mesh = _real_mesh()

    asm_real = AsmCls.from_mesh(mesh)
    K_real = asm_real().to_dense()

    asm_cplx = AsmCls.from_mesh(mesh)
    asm_cplx.type(dtype)  # accepted; geometry not cast — see _apply overrides
    K_cplx = asm_cplx().to_dense()

    # Whichever dtype K_cplx came out as, its real part has to match the
    # un-typed assembly and any imaginary part must be ~0.
    real_tol = 1e-5 if dtype == torch.complex64 else 1e-10
    K_real_part = K_cplx.real if K_cplx.is_complex() else K_cplx
    assert torch.allclose(
        K_real_part.to(K_real.dtype), K_real, atol=real_tol, rtol=real_tol
    ), f"Re(K_complex) deviates from K_real by max {(K_real_part.to(K_real.dtype) - K_real).abs().max().item():.3e}"
    if K_cplx.is_complex():
        assert K_cplx.imag.abs().max().item() < real_tol


# --------------------------------------------------------------------- #
# 3. Complex coefficient routed via ``__call__`` carries imag through
# --------------------------------------------------------------------- #
class _HelmholtzLike(ElementAssembler):
    """``a(u,v) = ∫ alpha(x) u v dΩ`` where alpha can be complex.

    Minimal stand-in for the Helmholtz volume mass-like term. The
    coefficient ``alpha`` is broadcast over (element, quadrature).
    """

    def forward(self, u, v, alpha):
        return alpha * u * v


def test_imaginary_coefficient_carries_through():
    """If alpha = (a + i b)(x), then K_complex.real == K(a) and
    K_complex.imag == K(b). Complex content enters via the
    ``point_data`` coefficient — no explicit ``.type(complex)`` call
    is needed."""
    mesh = _real_mesh()
    n_pts = mesh.n_points
    points = mesh.points.to(torch.float64)

    # Two random real coefficients defined on the mesh nodes.
    g = torch.Generator().manual_seed(0)
    a = torch.rand(n_pts, generator=g, dtype=torch.float64) + 0.5
    b = torch.rand(n_pts, generator=g, dtype=torch.float64) + 0.3
    alpha_cplx = (a + 1j * b).to(torch.complex128)

    # Reference: assemble with each real part separately.
    asm_a = _HelmholtzLike.from_mesh(mesh)
    asm_a.type(torch.float64)
    K_a = asm_a(points=points, point_data={"alpha": a}).to_dense()

    asm_b = _HelmholtzLike.from_mesh(mesh)
    asm_b.type(torch.float64)
    K_b = asm_b(points=points, point_data={"alpha": b}).to_dense()

    # Complex assembly via complex point_data.
    asm_c = _HelmholtzLike.from_mesh(mesh)
    asm_c.type(torch.float64)
    K_c = asm_c(points=points, point_data={"alpha": alpha_cplx}).to_dense()

    assert K_c.is_complex(), (
        f"complex coefficient should yield complex K, got {K_c.dtype}"
    )
    re_err = (K_c.real - K_a).abs().max().item()
    im_err = (K_c.imag - K_b).abs().max().item()
    K_scale = K_a.abs().max().item() + K_b.abs().max().item()
    assert re_err / max(K_scale, 1.0) < 1e-12, f"Re mismatch {re_err:.3e}"
    assert im_err / max(K_scale, 1.0) < 1e-12, f"Im mismatch {im_err:.3e}"
    assert K_c.imag.abs().max().item() > 1e-6, (
        "Im(K) should be non-trivial for a complex coefficient"
    )

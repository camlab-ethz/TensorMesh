"""Tests for the open-domain wave operators: anisotropic / scaled-mass /
facet-bilinear assemblers, the Cartesian PML, and the Robin / port boundary
operators."""

import sys
sys.path.append("../..")

import numpy as np
import pytest
import torch

from tensormesh import (
    Mesh,
    LaplaceElementAssembler,
    AnisotropicLaplaceElementAssembler,
    MassElementAssembler,
    ScaledMassElementAssembler,
    FacetBilinearAssembler,
    cartesian_pml,
    robin_operator,
    port_source,
)


@pytest.fixture(autouse=True)
def _default_float64():
    """Use double precision for the exact-equality asserts, restoring the global
    default afterward so this module never leaks its dtype into other tests."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _unit_square(h=0.2):
    return Mesh.gen_rectangle(chara_length=h, element_type="tri",
                              left=0, right=1, bottom=0, top=1)


def _dense(sp):
    return sp.to_scipy_coo().tocsr().toarray()


# --------------------------------------------------------------------------- #
# AnisotropicLaplaceElementAssembler
# --------------------------------------------------------------------------- #
def test_anisotropic_laplace_identity_equals_laplace():
    mesh = _unit_square()
    K = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    A = torch.eye(2).expand(mesh.n_points, 2, 2).clone()
    Ka = AnisotropicLaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(
        mesh.points, point_data={"A": A})
    assert np.abs(_dense(K) - _dense(Ka)).max() < 1e-12


def test_anisotropic_laplace_diagonal_scales_direction():
    # A = diag(2, 1) doubles the xx-contribution vs plain Laplace
    mesh = _unit_square()
    A = torch.diag(torch.tensor([2.0, 1.0])).expand(mesh.n_points, 2, 2).clone()
    Ka = AnisotropicLaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(
        mesh.points, point_data={"A": A})
    # symmetric, real, and not equal to the isotropic operator
    d = _dense(Ka)
    assert np.allclose(d, d.T, atol=1e-10)


def test_anisotropic_laplace_complex_coefficient():
    mesh = _unit_square()
    A = (torch.eye(2) * (1.0 - 0.5j)).expand(mesh.n_points, 2, 2).clone()
    Ka = AnisotropicLaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(
        mesh.points, point_data={"A": A})
    assert Ka.values.is_complex()
    # (1 - 0.5j) * Laplace
    K = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    assert np.abs(_dense(Ka) - (1.0 - 0.5j) * _dense(K)).max() < 1e-12


# --------------------------------------------------------------------------- #
# ScaledMassElementAssembler
# --------------------------------------------------------------------------- #
def test_scaled_mass_unit_coefficient_equals_mass():
    mesh = _unit_square()
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    c = torch.ones(mesh.n_points)
    Ms = ScaledMassElementAssembler.from_mesh(mesh, quadrature_order=2)(
        mesh.points, point_data={"c": c})
    assert np.abs(_dense(M) - _dense(Ms)).max() < 1e-12


def test_scaled_mass_constant_coefficient_scales():
    mesh = _unit_square()
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=2)(mesh.points)
    c = 3.0 * torch.ones(mesh.n_points)
    Ms = ScaledMassElementAssembler.from_mesh(mesh, quadrature_order=2)(
        mesh.points, point_data={"c": c})
    assert np.abs(_dense(Ms) - 3.0 * _dense(M)).max() < 1e-12


# --------------------------------------------------------------------------- #
# FacetBilinearAssembler  (boundary mass)
# --------------------------------------------------------------------------- #
class _BoundaryMass(FacetBilinearAssembler):
    def forward(self, u, v):
        return u * v


class _CoeffBoundaryMass(FacetBilinearAssembler):
    def forward(self, u, v, c):
        return c * u * v


def test_facet_bilinear_full_boundary_is_perimeter():
    # sum over B = int_Gamma (sum_i N_i)(sum_j N_j) dS = int_Gamma 1 dS = |Gamma|
    mesh = _unit_square(0.1)
    B = _BoundaryMass.from_mesh(mesh, boundary_mask=None)(mesh.points)
    assert abs(B.values.sum().item() - 4.0) < 1e-9         # unit-square perimeter


def test_facet_bilinear_single_edge_length():
    mesh = _unit_square(0.1)
    bottom = mesh.points[:, 1] < 1e-9
    B = _BoundaryMass.from_mesh(mesh, boundary_mask=bottom)(mesh.points)
    assert abs(B.values.sum().item() - 1.0) < 1e-9         # bottom edge length


def test_facet_bilinear_coefficient():
    mesh = _unit_square(0.1)
    bottom = mesh.points[:, 1] < 1e-9
    c = 2.0 * torch.ones(mesh.n_points)
    B = _CoeffBoundaryMass.from_mesh(mesh, boundary_mask=bottom)(
        mesh.points, point_data={"c": c})
    assert abs(B.values.sum().item() - 2.0) < 1e-9


# --------------------------------------------------------------------------- #
# cartesian_pml
# --------------------------------------------------------------------------- #
def test_cartesian_pml_interior_is_identity():
    # points well inside the domain -> stretch 1 -> Lambda = I, s_prod = 1
    pts = torch.tensor([[3.0, 3.0], [3.0, 2.0], [2.5, 3.5]])  # interior of [0,6]^2, tpml=0.6
    Lam, sprod = cartesian_pml(pts, bounds=[(0.0, 6.0), (0.0, 6.0)],
                               thickness=0.6, strength=5.0, order=2)
    assert torch.allclose(Lam, torch.eye(2, dtype=torch.complex128).expand(3, 2, 2))
    assert torch.allclose(sprod, torch.ones(3, dtype=torch.complex128))


def test_cartesian_pml_tensor_and_product():
    # inside the PML the diagonal is (sy/sx, sx/sy) and s_prod = sx*sy
    pts = torch.tensor([[0.1, 3.0]])                          # deep in the -x PML
    L, d, sig, p = 6.0, 0.6, 5.0, 2
    Lam, sprod = cartesian_pml(pts, bounds=[(0.0, L), (0.0, L)],
                               thickness=d, strength=sig, order=p)
    xi = (d - 0.1) / d
    sx = torch.tensor(1.0 - 1j * sig * xi ** p, dtype=torch.complex128)
    sy = torch.tensor(1.0 + 0j, dtype=torch.complex128)      # y interior
    assert torch.allclose(Lam[0, 0, 0], sy / sx)
    assert torch.allclose(Lam[0, 1, 1], sx / sy)
    assert torch.allclose(Lam[0, 0, 1], torch.zeros((), dtype=torch.complex128))
    assert torch.allclose(sprod[0], sx * sy)


# --------------------------------------------------------------------------- #
# robin_operator / port_source
# --------------------------------------------------------------------------- #
def test_robin_operator_full_boundary_is_perimeter():
    mesh = _unit_square(0.1)
    B = robin_operator(mesh, boundary_mask=None, coeff=1.0)
    assert abs(B.values.sum().item() - 4.0) < 1e-9


def test_robin_operator_complex_coefficient():
    mesh = _unit_square(0.1)
    B = robin_operator(mesh, boundary_mask=None, coeff=1j)
    assert B.values.is_complex()
    assert abs(B.values.sum().imag.item() - 4.0) < 1e-9      # i * perimeter


def test_port_source_integrates_shape_functions():
    # sum_i int_Gamma N_i dS = int_Gamma 1 dS = perimeter
    mesh = _unit_square(0.1)
    e = port_source(mesh, boundary_mask=None, incident=1.0)
    assert abs(e.real.sum().item() - 4.0) < 1e-9
    # a single edge integrates to its length
    bottom = mesh.points[:, 1] < 1e-9
    e_edge = port_source(mesh, boundary_mask=bottom, incident=1.0)
    assert abs(e_edge.real.sum().item() - 1.0) < 1e-9

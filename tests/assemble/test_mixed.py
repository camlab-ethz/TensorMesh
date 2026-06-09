"""Tests for MixedElementAssembler (multi-field / block assembly).

All tests run on CPU in float64. References are the existing single-field
assemblers (themselves validated against scikit-fem in test_element.py)
and a manufactured Stokes solution that is exactly representable in the
Taylor-Hood P2-P1 pair.
"""
import sys

import meshio
import numpy as np
import pytest
import torch

sys.path.append("../..")

from tensormesh import Condenser, ElementAssembler, Field, Mesh, MixedElementAssembler
from tensormesh.assemble import LaplaceElementAssembler, MassElementAssembler

torch.manual_seed(0)


# --------------------------------------------------------------------- #
# assemblers shared across tests
# --------------------------------------------------------------------- #
class ScalarLaplaceMixed(MixedElementAssembler):
    fields = [Field(trial="u", test="v", order=1)]

    def forward(self, gradu, gradv):
        return (gradu * gradv).sum()


class ScalarMassMixed(MixedElementAssembler):
    fields = [Field(trial="u", test="v", order=1)]

    def forward(self, u, v):
        return u * v


class StokesAssembler(MixedElementAssembler):
    """Taylor-Hood Stokes: mu grad(u):grad(v) - p div(v) - q div(u)."""

    fields = [
        Field(trial="u", test="v", order=2, components=2),
        Field(trial="p", test="q", order=1),
    ]

    def __post_init__(self, mu=1.0):
        self.mu = mu

    def forward(self, gradu, p, gradv, q):
        return self.mu * (gradu * gradv).sum() \
             - p * gradv.diagonal(dim1=-2, dim2=-1).sum() \
             - q * gradu.diagonal(dim1=-2, dim2=-1).sum()


@pytest.fixture(scope="module")
def p1_mesh():
    return Mesh.gen_rectangle(chara_length=0.2, element_type="tri").double()


@pytest.fixture(scope="module")
def p2_mesh():
    return Mesh.gen_rectangle(chara_length=0.2, order=2, element_type="tri").double()


def _dense(K):
    return K.to_dense().cpu()


# --------------------------------------------------------------------- #
# 1. a single scalar P1 field reproduces the single-field machinery
# --------------------------------------------------------------------- #
def test_scalar_p1_matches_element_assembler(p1_mesh):
    K_mixed = _dense(ScalarLaplaceMixed.from_mesh(p1_mesh, quadrature_order=2)())
    K_ref = _dense(LaplaceElementAssembler.from_mesh(p1_mesh, quadrature_order=2)(p1_mesh.points))
    np.testing.assert_allclose(K_mixed, K_ref, rtol=0, atol=1e-14)

    M_mixed = _dense(ScalarMassMixed.from_mesh(p1_mesh, quadrature_order=3)())
    M_ref = _dense(MassElementAssembler.from_mesh(p1_mesh, quadrature_order=3)(p1_mesh.points))
    np.testing.assert_allclose(M_mixed, M_ref, rtol=0, atol=1e-14)


# --------------------------------------------------------------------- #
# 2. trial -> columns, test -> rows (non-symmetric form), with point_data
# --------------------------------------------------------------------- #
def test_row_col_convention_nonsymmetric(p1_mesh):
    w = torch.randn(p1_mesh.points.shape[0], 2, dtype=torch.float64)

    class ConvectionMixed(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, gradu, v, w):
            return (w * gradu).sum() * v  # (w . grad u) v: K_ij = ∫ (w.∇φ_j) ψ_i

    # in ElementAssembler the `u` argument is the TEST function (rows),
    # so the same matrix is written with the letters swapped:
    class ConvectionRef(ElementAssembler):
        def forward(self, u, gradv, w):
            return (w * gradv).sum() * u

    K_mixed = _dense(ConvectionMixed.from_mesh(p1_mesh)(point_data={"w": w}))
    K_ref = _dense(ConvectionRef.from_mesh(p1_mesh)(p1_mesh.points, point_data={"w": w}))
    assert (K_mixed - K_mixed.T).abs().max() > 1e-8  # genuinely non-symmetric
    np.testing.assert_allclose(K_mixed, K_ref, rtol=0, atol=1e-14)


# --------------------------------------------------------------------- #
# 3. data arguments: grad-of-point_data, element_data, scalar_data
# --------------------------------------------------------------------- #
def test_data_arguments_match_element_assembler(p1_mesh):
    n_points = p1_mesh.points.shape[0]
    n_elem = p1_mesh.elements().shape[0]
    kappa = torch.rand(n_points, dtype=torch.float64) + 0.5
    rho = torch.rand(n_elem, dtype=torch.float64) + 0.5

    class Mixed(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, u, v, gradu, gradv, kappa, gradkappa, rho, nu):
            return nu * rho * kappa * (gradu * gradv).sum() \
                 + (gradkappa * gradu).sum() * v + u * v

    class Ref(ElementAssembler):
        # nu is baked in: ElementAssembler's scalar_data dispatch cannot
        # take 0-d scalars (its in_dims broadcast assumes batched tensors)
        def forward(self, u, v, gradu, gradv, kappa, gradkappa, rho):
            # swap trial/test letters: u is the test (row) argument here
            return 2.0 * rho * kappa * (gradv * gradu).sum() \
                 + (gradkappa * gradv).sum() * u + v * u

    kw = dict(point_data={"kappa": kappa}, element_data={"rho": rho})
    K_mixed = _dense(Mixed.from_mesh(p1_mesh)(scalar_data={"nu": 2.0}, **kw))
    K_ref = _dense(Ref.from_mesh(p1_mesh)(p1_mesh.points, **kw))
    np.testing.assert_allclose(K_mixed, K_ref, rtol=0, atol=1e-14)


# --------------------------------------------------------------------- #
# 4. a single vector P2 field matches the block-COO (trailing-dim) path
# --------------------------------------------------------------------- #
def test_vector_p2_block_matches_block_assembler(p2_mesh):
    class VecLaplaceMixed(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=2, components=2)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

    class VecLaplaceRef(ElementAssembler):
        def forward(self, gradu, gradv):
            return torch.dot(gradu, gradv) * torch.eye(2, dtype=gradu.dtype)

    K_mixed = _dense(VecLaplaceMixed.from_mesh(p2_mesh, quadrature_order=4)())
    K_ref = _dense(VecLaplaceRef.from_mesh(p2_mesh, quadrature_order=4)(p2_mesh.points))
    # exact match also certifies the node-major DOF layout is identical
    np.testing.assert_allclose(K_mixed, K_ref, rtol=0, atol=1e-14)


# --------------------------------------------------------------------- #
# 5. Taylor-Hood Stokes with an exactly representable solution
# --------------------------------------------------------------------- #
def test_taylor_hood_poiseuille_exact(p2_mesh):
    mu = 0.7
    asm = StokesAssembler.from_mesh(p2_mesh, mu=mu)
    lay = asm.layout

    K = asm()
    assert K.shape == (lay.n_dofs, lay.n_dofs)
    Kd = _dense(K)
    assert (Kd - Kd.T).abs().max() == 0.0  # Stokes block system is symmetric

    # Poiseuille: u = (y(1-y), 0), p = -2 mu x + 1 — exact in P2/P1
    pts = p2_mesh.points
    u_exact = torch.stack([pts[:, 1] * (1 - pts[:, 1]),
                           torch.zeros_like(pts[:, 0])], dim=-1)
    p_exact = -2.0 * mu * lay.points("p")[:, 0] + 1.0

    bc_mask = lay.dof_mask("u", p2_mesh.boundary_mask)
    pin = lay.dof_index("p", int(lay.node_ids("p")[0]))
    bc_mask[pin] = True
    bc_val = torch.zeros(lay.n_dofs, dtype=torch.float64)
    bc_val[lay.dof_mask("u")] = u_exact.reshape(-1)
    bc_val[pin] = p_exact[0]

    condenser = Condenser(bc_mask, bc_val[bc_mask])
    K_, f_ = condenser(K, torch.zeros(lay.n_dofs, dtype=torch.float64))
    sol = condenser.recover(K_.solve(f_))

    parts = lay.split(sol)
    assert (parts["u"] - u_exact).abs().max() < 1e-10
    assert (parts["p"] - p_exact).abs().max() < 1e-9


# --------------------------------------------------------------------- #
# 6. BlockLayout helpers
# --------------------------------------------------------------------- #
def test_layout_helpers(p2_mesh):
    asm = StokesAssembler.from_mesh(p2_mesh)
    lay = asm.layout
    n_points = p2_mesh.points.shape[0]
    n_p = lay.n_nodes("p")

    assert lay.names == ["u", "p"]
    assert lay.n_nodes("u") == n_points and n_p < n_points
    assert lay.n_dofs == 2 * n_points + n_p
    assert lay.offsets == {"u": 0, "p": 2 * n_points}

    # split / cat roundtrip, scalar broadcast
    x = torch.randn(lay.n_dofs, dtype=torch.float64)
    assert torch.equal(lay.cat(**lay.split(x)), x)
    y = lay.cat(u=torch.zeros(n_points, 2, dtype=torch.float64), p=1.5)
    assert torch.equal(y[2 * n_points:], torch.full((n_p,), 1.5, dtype=torch.float64))

    # dof_mask: full field, mesh-point mask restriction, single component
    assert lay.dof_mask("u").sum() == 2 * n_points
    bnd = lay.dof_mask("p", p2_mesh.boundary_mask)
    assert bnd.sum() == p2_mesh.boundary_mask[lay.node_ids("p")].sum()
    comp0 = lay.dof_mask("u", component=0)
    assert comp0.sum() == n_points and not comp0[1]  # dof 1 is (node 0, comp 1)

    # dof_index agrees with dof_mask; midside nodes carry no pressure DOF
    node0 = int(lay.node_ids("p")[0])
    assert bool(lay.dof_mask("p")[lay.dof_index("p", node0)])
    midside = int(torch.nonzero(asm.field_g2l["p"] < 0)[0])
    with pytest.raises(ValueError):
        lay.dof_index("p", midside)

    # restrict / points / prolong (linear function is reproduced exactly)
    pts = p2_mesh.points
    lin = pts[:, 0] + 2.0 * pts[:, 1]
    assert torch.equal(lay.restrict("p", lin), lin[lay.node_ids("p")])
    assert torch.equal(lay.points("p"), pts[lay.node_ids("p")])
    pro = lay.prolong("p", lay.restrict("p", lin))
    assert (pro - lin).abs().max() < 1e-12
    assert torch.equal(lay.prolong("u", torch.randn(n_points, 2).double() * 0 + 1.0),
                       torch.ones(n_points, 2, dtype=torch.float64))


# --------------------------------------------------------------------- #
# 7. validation errors
# --------------------------------------------------------------------- #
def test_validation_errors(p2_mesh, p1_mesh):
    with pytest.raises(ValueError):  # trial == test
        Field(trial="u", test="u")
    with pytest.raises(ValueError):  # reserved coordinate name
        Field(trial="x", test="y")
    with pytest.raises(ValueError):  # grad prefix
        Field(trial="gradient", test="v")

    class BadOrder(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=3)]

        def forward(self, u, v):
            return u * v

    with pytest.raises(ValueError, match="order 3"):
        BadOrder.from_mesh(p2_mesh)

    class DupNames(MixedElementAssembler):
        fields = [Field(trial="u", test="v"), Field(trial="p", test="u")]

        def forward(self, u, v, p):
            return u * v

    with pytest.raises(ValueError, match="must all differ"):
        DupNames.from_mesh(p1_mesh)

    asm = ScalarMassMixed.from_mesh(p1_mesh)
    with pytest.raises(ValueError, match="collide"):  # point_data key == field name
        asm(point_data={"u": torch.zeros(p1_mesh.points.shape[0]).double()})

    class UnknownArg(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, u, v, mystery):
            return u * v

    with pytest.raises(ValueError, match="mystery"):
        UnknownArg.from_mesh(p1_mesh)()

    class NonScalarReturn(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, gradu, gradv):
            return gradu * gradv  # [D] instead of a 0-d scalar

    with pytest.raises(ValueError, match="0-d scalar"):
        NonScalarReturn.from_mesh(p1_mesh)()

    class NotBilinear(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, u, v):
            return u * v + 1.0  # constant term

    with pytest.raises(ValueError, match="bilinear"):
        NotBilinear.from_mesh(p1_mesh)()


# --------------------------------------------------------------------- #
# 8. mixed tri+quad mesh: per-type contributions sum correctly
# --------------------------------------------------------------------- #
def test_mixed_etype_tri_quad():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
    )
    quad = np.array([[0, 1, 2, 3]])
    tris = np.array([[1, 4, 5], [1, 5, 2]])
    mixed_mesh = Mesh.from_meshio(
        meshio.Mesh(points, [("quad", quad), ("triangle", tris)]), reorder=True
    ).double()
    # assembly is additive over disjoint element sets: the matrix on the
    # mixed mesh equals the sum of the single-type matrices on the same points
    quad_mesh = Mesh.from_meshio(meshio.Mesh(points, [("quad", quad)]), reorder=True).double()
    tri_mesh = Mesh.from_meshio(meshio.Mesh(points, [("triangle", tris)]), reorder=True).double()

    K_mixed = _dense(ScalarLaplaceMixed.from_mesh(mixed_mesh, quadrature_order=3)())
    K_sum = _dense(ScalarLaplaceMixed.from_mesh(quad_mesh, quadrature_order=3)()) \
          + _dense(ScalarLaplaceMixed.from_mesh(tri_mesh, quadrature_order=3)())
    np.testing.assert_allclose(K_mixed, K_sum, rtol=0, atol=1e-14)

    # ... and the (fixed) single-field path agrees on the mixed mesh
    K_single = _dense(LaplaceElementAssembler.from_mesh(mixed_mesh, quadrature_order=3)(mixed_mesh.points))
    np.testing.assert_allclose(K_mixed, K_single, rtol=0, atol=1e-14)


# --------------------------------------------------------------------- #
# 9. quadrature batching is exact
# --------------------------------------------------------------------- #
def test_batch_size_parity(p2_mesh):
    asm = StokesAssembler.from_mesh(p2_mesh)
    K_full = _dense(asm(batch_size=-1))
    K_batched = _dense(asm(batch_size=1))
    np.testing.assert_allclose(K_batched, K_full, rtol=0, atol=1e-13)

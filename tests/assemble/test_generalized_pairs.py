"""Tests for generalized order pairs (topological Lagrange DOF maps).

Field orders decoupled from the mesh order: quadratic/cubic fields on P1
gmsh meshes, P3-P2 generalized Taylor-Hood, oriented edge DOFs, the
topological boundary detection and the field_data channel. All tests run
on CPU in float64; the manufactured solutions are exactly representable
in the respective spaces, so solves recover them to machine precision —
a wrong edge orientation or a broken DOF map fails these loudly.
"""
import sys

import pytest
import torch

sys.path.append("../..")

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.assemble.topology import lagrange_boundary_mask, lagrange_dofmap

torch.manual_seed(0)


@pytest.fixture(scope="module")
def p1_mesh():
    return Mesh.gen_rectangle(chara_length=0.15, element_type="tri").double()


def _poisson_cls(k):
    class Poisson(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=k)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

    return Poisson


def _harmonic_solve(mesh, k, exact_fn):
    """Solve Laplace(u)=0 with exact Dirichlet data; return max nodal error."""
    asm = _poisson_cls(k).from_mesh(mesh)
    lay = asm.layout
    K = asm()
    exact = exact_fn(lay.points("u"))
    bnd = lay.boundary_mask("u")
    bc_val = torch.zeros(lay.n_dofs, dtype=torch.float64)
    bc_val[lay.dof_mask("u")] = exact
    condenser = Condenser(bnd, bc_val[bnd])
    K_, f_ = condenser(K, torch.zeros(lay.n_dofs, dtype=torch.float64))
    sol = condenser.recover(K_.solve(f_))
    return float((lay.split(sol)["u"] - exact).abs().max()), lay


# --------------------------------------------------------------------- #
# 1. the DOF map itself: counts, cross-element consistency, boundary
# --------------------------------------------------------------------- #
def test_dofmap_counts_and_shared_edge_consistency():
    # two triangles sharing edge (1, 2), traversed in opposite senses —
    # the classic case a wrong edge orientation breaks
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                       dtype=torch.float64)
    cells = {"triangle": torch.tensor([[0, 1, 2], [1, 3, 2]])}
    for k in [1, 2, 3, 4]:
        dofmap = lagrange_dofmap(cells, 4, k)
        n_interior = (k - 1) * (k - 2) // 2
        assert dofmap.n_dofs == 4 + 5 * (k - 1) + 2 * n_interior

        # every DOF must get the SAME coordinate from all elements that
        # reference it (affine map of the reference nodes)
        ref = dofmap.ref_nodes["triangle"]                   # [nb, 2]
        conn = dofmap.conn["triangle"]                       # [2, nb]
        coords = torch.full((dofmap.n_dofs, 2), torch.nan, dtype=torch.float64)
        for e in range(2):
            v = pts[cells["triangle"][e]]
            xy = v[0] + ref[:, :1] * (v[1] - v[0]) + ref[:, 1:] * (v[2] - v[0])
            for i in range(ref.shape[0]):
                d = conn[e, i]
                if torch.isnan(coords[d, 0]):
                    coords[d] = xy[i]
                else:
                    assert (coords[d] - xy[i]).norm() < 1e-12, \
                        f"P{k}: shared DOF {int(d)} disagrees across elements"
        assert not torch.isnan(coords).any()

        # topological boundary: only the diagonal edge + cell interiors are inner
        bnd = lagrange_boundary_mask(dofmap.conn, k, dofmap.n_dofs)
        assert int((~bnd).sum()) == (k - 1) + 2 * n_interior


def test_dofmap_3d_face_dofs_not_implemented():
    cells = {"tetra": torch.tensor([[0, 1, 2, 3]])}
    lagrange_dofmap(cells, 4, 2)  # P2 tetra: vertices + edges only — fine
    with pytest.raises(NotImplementedError, match="face"):
        lagrange_dofmap(cells, 4, 3)  # P3 tetra has face nodes


# --------------------------------------------------------------------- #
# 2. exact solves: harmonic polynomials representable in the space
# --------------------------------------------------------------------- #
def test_p2_field_on_p1_mesh_harmonic_exact(p1_mesh):
    err, lay = _harmonic_solve(p1_mesh, 2, lambda x: x[:, 0] ** 2 - x[:, 1] ** 2)
    assert err < 1e-10
    assert lay.n_dofs > p1_mesh.points.shape[0]  # edge DOFs beyond mesh nodes


def test_p3_field_on_p1_mesh_harmonic_exact(p1_mesh):
    # Re(z^3) = x^3 - 3xy^2 is harmonic and cubic: exact in P3. With two
    # DOFs per edge this is the sharp edge-orientation test — a single
    # flipped edge destroys C0 conformity and the solve fails loudly.
    err, _ = _harmonic_solve(p1_mesh, 3,
                             lambda x: x[:, 0] ** 3 - 3 * x[:, 0] * x[:, 1] ** 2)
    assert err < 1e-9


def test_3d_p2_field_on_p1_tetra_mesh():
    mesh = Mesh.gen_cube(chara_length=0.35).double()
    err, _ = _harmonic_solve(mesh, 2, lambda x: x[:, 0] ** 2 - x[:, 1] ** 2)
    assert err < 1e-9


# --------------------------------------------------------------------- #
# 3. Stokes: super-parametric and generalized Taylor-Hood
# --------------------------------------------------------------------- #
def _stokes_poiseuille(mesh, order_u, order_p, mu=0.7):
    class Stokes(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=order_u, components=2),
                  Field(trial="p", test="q", order=order_p)]

        def __post_init__(self, mu=1.0):
            self.mu = mu

        def forward(self, gradu, p, gradv, q):
            return self.mu * (gradu * gradv).sum() \
                 - p * gradv.diagonal(dim1=-2, dim2=-1).sum() \
                 - q * gradu.diagonal(dim1=-2, dim2=-1).sum()

    asm = Stokes.from_mesh(mesh, mu=mu)
    lay = asm.layout
    K = asm()
    Kd = K.to_dense()
    assert (Kd - Kd.T).abs().max() == 0.0

    xu, xp = lay.points("u"), lay.points("p")
    u_exact = torch.stack([xu[:, 1] * (1 - xu[:, 1]),
                           torch.zeros_like(xu[:, 0])], dim=-1)
    p_exact = -2.0 * mu * xp[:, 0] + 1.0

    bc_mask = lay.boundary_mask("u")
    pin = int(torch.nonzero(lay.dof_mask("p"))[0])  # pin the pressure constant
    bc_mask[pin] = True
    bc_val = torch.zeros(lay.n_dofs, dtype=torch.float64)
    bc_val[lay.dof_mask("u")] = u_exact.reshape(-1)
    bc_val[pin] = p_exact[0]

    condenser = Condenser(bc_mask, bc_val[bc_mask])
    K_, f_ = condenser(K, torch.zeros(lay.n_dofs, dtype=torch.float64))
    sol = condenser.recover(K_.solve(f_))
    parts = lay.split(sol)
    return (float((parts["u"] - u_exact).abs().max()),
            float((parts["p"] - p_exact).abs().max()))


def test_taylor_hood_p2_p1_on_p1_mesh(p1_mesh):
    # classical Taylor-Hood WITHOUT regenerating the mesh at order 2:
    # the velocity space is super-parametric on the linear gmsh mesh
    u_err, p_err = _stokes_poiseuille(p1_mesh, 2, 1)
    assert u_err < 1e-10 and p_err < 1e-9


def test_generalized_taylor_hood_p3_p2(p1_mesh):
    u_err, p_err = _stokes_poiseuille(p1_mesh, 3, 2)
    assert u_err < 1e-9 and p_err < 1e-8


# --------------------------------------------------------------------- #
# 4. layout helpers on generalized fields
# --------------------------------------------------------------------- #
def test_layout_generalized_field(p1_mesh):
    asm = _poisson_cls(3).from_mesh(p1_mesh)
    lay = asm.layout
    pts = lay.points("u")
    lin = pts[:, 0] + 2.0 * pts[:, 1]

    # restrict (mesh -> field) reproduces a linear function exactly
    mesh_lin = p1_mesh.points[:, 0] + 2.0 * p1_mesh.points[:, 1]
    assert float((lay.restrict("u", mesh_lin) - lin).abs().max()) < 1e-12
    # prolong (field -> mesh) is its inverse on the mesh nodes
    assert float((lay.prolong("u", lin) - mesh_lin).abs().max()) < 1e-12

    # topological boundary == geometric boundary on the unit square
    geo = ((pts[:, 0] - 0.0).abs() < 1e-9) | ((pts[:, 0] - 1.0).abs() < 1e-9) | \
          ((pts[:, 1] - 0.0).abs() < 1e-9) | ((pts[:, 1] - 1.0).abs() < 1e-9)
    assert torch.equal(lay.boundary_mask("u"), lay.dof_mask("u", node_mask=geo))
    # where= coordinate predicate
    assert torch.equal(lay.dof_mask("u", where=lambda x: x[:, 1] > 1 - 1e-9),
                       lay.dof_mask("u", node_mask=pts[:, 1] > 1 - 1e-9))

    # mesh-node semantics are not available for topological carriers
    with pytest.raises(ValueError, match="topological"):
        lay.node_ids("u")
    with pytest.raises(ValueError, match="topological"):
        lay.dof_index("u", 0)
    with pytest.raises(ValueError):
        lay.dof_mask("u", node_mask=torch.ones(p1_mesh.points.shape[0], dtype=torch.bool))


# --------------------------------------------------------------------- #
# 5. field_data: values on a field's DOFs, interpolated with its basis
# --------------------------------------------------------------------- #
def test_field_data_matches_point_data_on_mesh_order_field(p1_mesh):
    # for a mesh-order field the two channels must agree exactly
    w = torch.randn(p1_mesh.points.shape[0], 2, dtype=torch.float64)

    class ConvVal(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, gradu, v, w):
            return (w * gradu).sum() * v

    class ConvGrad(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=1)]

        def forward(self, u, v, gradw):
            return gradw.trace() * u * v

    for cls in (ConvVal, ConvGrad):
        asm = cls.from_mesh(p1_mesh)
        K_point = asm(point_data={"w": w}).to_dense()
        K_field = asm(field_data={"w": ("u", w)}).to_dense()
        assert float((K_point - K_field).abs().max()) < 1e-14


def test_field_data_on_generalized_field(p1_mesh):
    # Picard-style: previous velocity lives on the generalized P3 space
    class NS(MixedElementAssembler):
        fields = [Field(trial="u", test="v", order=3, components=2),
                  Field(trial="p", test="q", order=2)]

        def forward(self, gradu, p, v, gradv, q, w):
            return (gradu @ w).dot(v) + (gradu * gradv).sum() \
                 - p * gradv.diagonal().sum() - q * gradu.diagonal().sum()

    asm = NS.from_mesh(p1_mesh)
    lay = asm.layout
    # w = interpolant of a constant field -> convection term is exactly
    # the one of a constant advection velocity
    w = torch.zeros(lay.n_nodes("u"), 2, dtype=torch.float64)
    w[:, 0] = 1.0
    K_w = asm(field_data={"w": ("u", w)}).to_dense()

    def const_adv(gradu, p, v, gradv, q, x):
        c = torch.zeros(2, dtype=x.dtype, device=x.device)
        c[0] = 1.0
        return (gradu @ c).dot(v) + (gradu * gradv).sum() \
             - p * gradv.diagonal().sum() - q * gradu.diagonal().sum()

    K_c = asm(func=const_adv).to_dense()
    assert float((K_w - K_c).abs().max()) < 1e-12

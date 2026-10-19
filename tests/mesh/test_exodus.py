"""Exodus II I/O — :func:`tensormesh.mesh.exodus.read_exodus` /
:func:`~tensormesh.mesh.exodus.write_exodus`, :attr:`Mesh.point_sets`,
:attr:`Mesh.side_sets`, and side sets as ``FacetAssembler`` boundaries.

The fixtures in ``data/exodus`` were written by MOOSE (``moose-opt -i
<input>.i --mesh-only``; the inputs sit next to them), so they carry the
node ordering, side numbering and set layout of a real Exodus producer:

* ``<type>.e`` — one element type on [0,2]x[0,1](x[0,1]) with the side sets
  and node sets ``left/right/bottom/top(/back/front)``;
* ``mixed.e`` — blocks 1 (HEX8), 2 (WEDGE6), 3 (HEX8) side by side;
* ``interface.e`` — QUAD4 blocks 0 and 1, the interior side set
  ``interface`` between them, and an EDGE2 lower-dimensional block on it;
* ``quad8.e`` — a serendipity element TensorMesh does not support.
"""
import os

import pytest
import torch

pytest.importorskip("netCDF4")

from tensormesh import Condenser, ElementAssembler, FacetAssembler, Mesh
from tensormesh.element import element_type2element, element_type2order
from tensormesh.mesh.mesh import local_facets

DATA = os.path.join(os.path.dirname(__file__), "data", "exodus")

# fixture -> TensorMesh element type
TYPES = {
    "tri3": "triangle", "tri6": "triangle6", "quad4": "quad", "quad9": "quad9",
    "tet4": "tetra", "tet10": "tetra10", "hex8": "hexahedron", "hex27": "hexahedron27",
    "prism6": "wedge", "prism18": "wedge18", "pyramid5": "pyramid", "pyramid14": "pyramid14",
}
# side set -> (axis, coordinate) of the plane it lies on
PLANES = {"left": (0, 0.0), "right": (0, 2.0), "bottom": (1, 0.0),
          "top": (1, 1.0), "back": (2, 0.0), "front": (2, 1.0)}


def _read(name):
    return Mesh.read(os.path.join(DATA, f"{name}.e")).double()


class Mass(ElementAssembler):
    def forward(self, u, v):
        return u * v


class Laplace(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class IntegrateOne(FacetAssembler):
    def forward(self, v):
        return v


# ------------------------------------------------------------------ #
# reading
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("name", TYPES)
def test_exodus_read_cells_blocks_and_sets(name):
    mesh = _read(name)
    element_type = TYPES[name]
    dim = 2 if name[:3] in ("tri", "qua") else 3
    assert list(mesh.cells.keys()) == [element_type]
    assert mesh.dim == dim
    assert mesh.check_node_ordering(raise_on_error=False).get(element_type, 0) == 0
    assert mesh.cell_data["block_id"][element_type].unique().tolist() == [0]
    volume = Mass.from_mesh(mesh, quadrature_order=4)(mesh.points).to_dense().sum().item()
    assert volume == pytest.approx(2.0, abs=1e-12)

    names = ["bottom", "right", "top", "left"] if dim == 2 else \
            ["back", "bottom", "right", "top", "left", "front"]
    assert list(mesh.side_sets.keys()) == names
    assert list(mesh.point_sets.keys()) == names
    for side in names:
        axis, value = PLANES[side]
        mask = mesh.side_set_mask(side)
        assert torch.allclose(mesh.points[mask, axis], torch.tensor(value, dtype=torch.float64))
        # MOOSE writes each side set's nodes as the node set of the same name
        assert torch.equal(mask, mesh.point_set_mask(side))


@pytest.mark.parametrize("name", TYPES)
def test_exodus_node_positions(name):
    # the fixtures are straight-sided, so every node must sit at the image of
    # its TensorMesh reference position under the corners' linear map — this
    # pins the Exodus -> TensorMesh permutation of every higher-order node
    mesh = _read(name)
    element_type = TYPES[name]
    element, order = element_type2element(element_type), element_type2order[element_type]
    ref = element.get_basis(order, torch.float64)
    corners = [int(torch.cdist(v[None], ref).argmin()) for v in element.get_basis(1, torch.float64)]
    linear = element.eval_shape_val(ref, 1).reshape(ref.shape[0], -1)      # [n_basis, n_vertex]
    nodes = mesh.points[mesh.cells[element_type]]                          # [n_cell, n_basis, D]
    expected = torch.einsum("bc,ecd->ebd", linear, nodes[:, corners])
    assert torch.allclose(nodes, expected, atol=1e-12)


# pyramid14 is left out: TensorMesh's order-2 pyramid space is not continuous
# across the triangular faces two pyramids share, so no Galerkin solution is
# exact there (its node positions are covered by test_exodus_node_positions)
@pytest.mark.parametrize("name", [n for n in TYPES if n != "pyramid14"])
def test_exodus_side_set_patch_test(name):
    # Laplace(u) = 0 with u exact on every side set: a harmonic polynomial of
    # the element's degree is reproduced to round-off only if the node
    # ordering (geometry map) and the side sets are both right.
    mesh = _read(name)
    x = mesh.points
    if TYPES[name][-1].isdigit():            # quadratic cells
        exact = x[:, 0] ** 2 - x[:, 1] ** 2 + x[:, 0] * x[:, 1]
    else:
        exact = 1.0 + x[:, 0] + 2.0 * x[:, 1] + (3.0 * x[:, 2] if mesh.dim == 3 else 0.0)
    dirichlet = torch.zeros(mesh.n_points, dtype=torch.bool)
    for side in mesh.side_sets:
        dirichlet |= mesh.side_set_mask(side)
    assert not bool(dirichlet.all())         # there is something to solve for
    K = Laplace.from_mesh(mesh, quadrature_order=4)(mesh.points)
    condenser = Condenser(dirichlet, exact[dirichlet])
    K_, f_ = condenser(K, torch.zeros(mesh.n_points, dtype=torch.float64))
    u = condenser.recover(K_.solve(f_))
    assert float((u - exact).abs().max()) < 1e-9


def test_exodus_mixed_blocks():
    # element ids run HEX8 (block 1), WEDGE6 x2 (block 2), HEX8 (block 3):
    # the two hexahedron blocks are concatenated, so the third block's
    # element is hexahedron cell 1, not Exodus element 3
    mesh = _read("mixed")
    assert {k: v.shape[0] for k, v in mesh.cells.items()} == {"hexahedron": 2, "wedge": 2}
    assert mesh.cell_data["block_id"]["hexahedron"].tolist() == [1, 3]
    assert mesh.cell_data["block_id"]["wedge"].tolist() == [2, 2]
    x_center = {k: mesh.points[v].mean(1)[:, 0] for k, v in mesh.cells.items()}
    assert x_center["hexahedron"].tolist() == pytest.approx([0.5, 2.5])
    assert bool(((x_center["wedge"] > 1) & (x_center["wedge"] < 2)).all())

    # "bottom" (y = 0) spans both element types: both hexahedra and the one
    # prism of block 2 with a quadrilateral face on y = 0
    bottom = mesh.side_sets["bottom"]
    assert sorted(bottom.keys()) == ["hexahedron", "wedge"]
    assert bottom["hexahedron"][:, 0].tolist() == [0, 1]
    assert len(bottom["wedge"]) == 1
    for element_type, pairs in bottom.items():
        facets = local_facets(element_type)
        for cell, facet in pairs.tolist():
            y = mesh.points[mesh.cells[element_type][cell, facets[facet]], 1]
            assert torch.allclose(y, torch.zeros_like(y))


def test_exodus_interior_side_set_integrates_once():
    mesh = _read("interface")
    # block 0 / block 1 quads plus the EDGE2 lower-dimensional block 2
    assert mesh.cell_data["block_id"]["quad"].tolist() == [0] * 4 + [1] * 4
    assert mesh.cell_data["block_id"]["line"].tolist() == [2, 2]
    assert mesh.default_element_type == "quad"
    interface = mesh.side_sets["interface"]
    assert list(interface.keys()) == ["quad"]
    assert bool((mesh.cell_data["block_id"]["quad"][interface["quad"][:, 0]] == 0).all())

    # the side set selects the x = 1 facets of block 0 only: length 1
    exact = IntegrateOne.from_mesh(mesh, boundary_mask="interface")().sum().item()
    assert exact == pytest.approx(1.0, abs=1e-12)
    # a node mask cannot tell the two sides apart and integrates twice
    by_nodes = IntegrateOne.from_mesh(mesh, boundary_mask=mesh.point_set_mask("interface"))().sum().item()
    assert by_nodes == pytest.approx(2.0, abs=1e-12)


# (FacetAssembler does not integrate over prism/pyramid facets yet)
@pytest.mark.parametrize("name", ["tri6", "quad9", "tet10", "hex27"])
def test_exodus_side_set_matches_node_mask_on_exterior_boundary(name):
    mesh = _read(name)
    for side in mesh.side_sets:
        by_side_set = IntegrateOne.from_mesh(mesh, boundary_mask=side)()
        by_nodes = IntegrateOne.from_mesh(mesh, boundary_mask=mesh.side_set_mask(side))()
        assert torch.allclose(by_side_set, by_nodes, atol=1e-12)
        assert by_side_set.sum().item() == pytest.approx(2.0 if side in ("bottom", "top", "back", "front") else 1.0)


def test_exodus_unsupported_element_type():
    with pytest.raises(NotImplementedError, match="QUAD8"):
        Mesh.read(os.path.join(DATA, "quad8.e"))


# ------------------------------------------------------------------ #
# writing
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("name", list(TYPES) + ["mixed", "interface"])
def test_exodus_write_read_round_trip(name, tmp_path):
    mesh = _read(name)
    path = str(tmp_path / f"{name}.e")
    mesh.save(path)
    back = Mesh.read(path).double()
    assert torch.equal(back.points, mesh.points)
    assert list(back.cells.keys()) == list(mesh.cells.keys())
    for k in mesh.cells.keys():
        # blocks are regrouped by block id on write; compare cells as sets
        assert sorted(map(tuple, back.cells[k].tolist())) == sorted(map(tuple, mesh.cells[k].tolist()))
        assert sorted(back.cell_data["block_id"][k].tolist()) == sorted(mesh.cell_data["block_id"][k].tolist())
    assert list(back.point_sets.keys()) == list(mesh.point_sets.keys())
    for k in mesh.point_sets.keys():
        assert torch.equal(back.point_sets[k], mesh.point_sets[k])
    assert list(back.side_sets.keys()) == list(mesh.side_sets.keys())
    for k in mesh.side_sets.keys():
        assert torch.equal(back.side_set_mask(k), mesh.side_set_mask(k))
        n_facet = lambda m: sum(len(p) for p in m.side_sets[k].values())
        assert n_facet(back) == n_facet(mesh)


def test_exodus_mesh_without_sets_round_trip(tmp_path):
    mesh = Mesh.gen_rectangle(0.5, element_type="quad", order=2)
    path = str(tmp_path / "rect.exo")
    mesh.save(path)
    back = Mesh.read(path)
    assert torch.allclose(back.points, mesh.points.to(back.dtype))
    assert torch.equal(back.cells["quad9"], mesh.cells["quad9"])
    assert len(back.point_sets) == 0 and len(back.side_sets) == 0


# ------------------------------------------------------------------ #
# the set containers
# ------------------------------------------------------------------ #
def test_side_sets_survive_clone_and_to():
    mesh = _read("tri6")
    copy = mesh.clone()
    assert list(copy.side_sets.keys()) == list(mesh.side_sets.keys())
    assert torch.equal(copy.side_set_mask("left"), mesh.side_set_mask("left"))
    assert torch.equal(copy.point_sets["left"], mesh.point_sets["left"])
    mesh.float()
    assert mesh.side_sets["left"]["triangle6"].dtype == torch.long
    assert mesh.point_sets["left"].dtype == torch.long


def test_register_sets_by_hand():
    mesh = Mesh.gen_rectangle(0.5, element_type="quad")
    left = mesh.point_data["is_left_boundary"]
    mesh.register_point_set("left", left.nonzero().flatten())
    assert torch.equal(mesh.point_set_mask("left"), left)
    # the facets whose nodes are all on the left edge
    facets = torch.zeros(0, 2, dtype=torch.long)
    cells = mesh.cells["quad"]
    for f, nodes in enumerate(local_facets("quad")):
        on_left = left[cells[:, nodes]].all(-1).nonzero().flatten()
        facets = torch.cat([facets, torch.stack([on_left, torch.full_like(on_left, f)], 1)])
    mesh.register_side_set("left", {"quad": facets})
    assert torch.equal(mesh.side_set_mask("left"), left)
    length = IntegrateOne.from_mesh(mesh, boundary_mask="left")().sum().item()
    assert length == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(AssertionError):
        mesh.register_side_set("bad", {"quad": torch.tensor([[0, 4]])})
    with pytest.raises(KeyError):
        IntegrateOne.from_mesh(mesh, boundary_mask="no_such_set")

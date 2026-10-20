"""Boundary-facet selection by facet incidence, and the outward facet normal ``n``."""

import itertools

import meshio
import pytest
import torch

from tensormesh import FacetAssembler, FacetBilinearAssembler, Mesh, NodeAssembler, Transformation
from tensormesh.element import element_type2element, element_type2order

f64 = torch.float64


def _affine_mesh(element_type, frames):
    """Mesh of affine images ``x0 + xi @ E`` of the reference element, one per ``(x0, E)``.

    Coincident nodes are merged, so neighbouring cells share their facets."""
    ref = element_type2element(element_type).get_basis(element_type2order[element_type], f64)
    nodes = torch.cat([x0 + ref @ E for x0, E in frames])
    key, conn = torch.unique(torch.round(nodes * 1e9).long(), dim=0, return_inverse=True)
    points = torch.zeros(len(key), nodes.shape[1], dtype=f64).index_put_((conn,), nodes)
    return Mesh(meshio.Mesh(points.numpy(), [(element_type, conn.reshape(len(frames), -1).numpy())]))


def _square(element_type="triangle", k=4):
    """Unit square, k x k cells, two triangles each."""
    h, eye, frames = 1.0 / k, torch.eye(2, dtype=f64), []
    for i, j in itertools.product(range(k), repeat=2):
        o = torch.tensor([i, j], dtype=f64) * h
        frames.append((o, h * eye))
        frames.append((o + h, -h * eye))
    return _affine_mesh(element_type, frames)


def _cube(element_type="tetra", k=3):
    """Unit cube, k^3 cells: six Kuhn tetrahedra, two prisms or six pyramids each."""
    h, eye, frames = 1.0 / k, torch.eye(3, dtype=f64), []
    for corner in itertools.product(range(k), repeat=3):
        o = torch.tensor(corner, dtype=f64) * h
        if element_type.startswith("pyramid"):
            centre = o + h / 2
            for axis, side in itertools.product(range(3), (0, 1)):
                e1, e2 = (h * eye[d] for d in range(3) if d != axis)
                x0 = o + side * h * eye[axis]
                if torch.dot(torch.linalg.cross(e1, e2), centre - x0) < 0:
                    e1, e2 = e2, e1
                frames.append((x0, torch.stack([e1, e2, centre - x0])))
        elif element_type.startswith("tetra"):
            for p in itertools.permutations(range(3)):
                e1, e2, e3 = h * eye[p[0]], h * (eye[p[0]] + eye[p[1]]), h * eye.sum(0)
                if torch.linalg.det(torch.stack([e1, e2, e3])) < 0:
                    e1, e2 = e2, e1
                frames.append((o, torch.stack([e1, e2, e3])))
        else:
            frames.append((o, h * eye))
            frames.append((o + h * (eye[0] + eye[1]), torch.stack([-h * eye[0], -h * eye[1], h * eye[2]])))
    return _affine_mesh(element_type, frames)


def _box_boundary(mesh):
    """Nodes on the faces of the unit square / cube (before bending)."""
    return ((mesh.points < 1e-9) | (mesh.points > 1 - 1e-9)).any(1)


def _bend(mesh):
    """Map the nodes through a smooth non-affine map: higher-order cells become curved."""
    p = mesh.points
    shift = 0.1 * torch.sin(torch.pi * p.roll(1, dims=1)) * torch.sin(torch.pi * p.roll(-1, dims=1))
    element_type = mesh.default_element_type
    return Mesh(meshio.Mesh((p + shift).numpy(), [(element_type, mesh.cells[element_type].numpy())]))


class Area(FacetAssembler):
    def forward(self, v):
        return v


class Flux(FacetAssembler):
    """Columns: the components of n, then x . n."""

    def forward(self, v, x, n):
        return v[:, None] * torch.cat([n, (x * n).sum()[None]])


class Volume(NodeAssembler):
    def forward(self, v):
        return v


class NormalXSquaredMass(FacetBilinearAssembler):
    def forward(self, u, v, n):
        return n[0] ** 2 * u * v


@pytest.mark.parametrize("element_type, expected", [
    ("triangle", 4.0), ("triangle6", 4.0),
    ("tetra", 6.0), ("tetra10", 6.0),
    ("wedge", 6.0), ("pyramid", 6.0),
])
def test_whole_boundary_measure(element_type, expected):
    # interior facets whose nodes all lie on the boundary (the cell diagonals
    # touching two sides) used to be selected too, once from each side
    mesh = _square(element_type) if element_type.startswith("triangle") else _cube(element_type)
    area = Area.from_mesh(mesh, boundary_mask=_box_boundary(mesh), quadrature_order=2)().sum()
    assert abs(float(area) - expected) < 1e-12


def test_partial_mask_selects_one_face():
    mesh = _cube("tetra10")
    top = (mesh.points[:, 2] - 1.0).abs() < 1e-12
    assert abs(float(Area.from_mesh(mesh, boundary_mask=top)().sum()) - 1.0) < 1e-12


def test_selection_does_not_modify_the_connectivity():
    mesh = _cube("tetra")
    elements = {"tetra": mesh.cells["tetra"].clone()}
    Area.from_elements(mesh.points, elements, mesh.boundary_mask, dtype=f64)
    assert torch.equal(elements["tetra"], mesh.cells["tetra"])


@pytest.mark.parametrize("element_type", ["triangle", "triangle6", "tetra", "tetra10", "wedge",
                                          "pyramid", "pyramid14"])
def test_divergence_theorem(element_type):
    # int_dOmega n ds = 0 and int_dOmega x . n ds = D |Omega|; with a polynomial
    # isoparametric map n |F| is polynomial, so both hold to round-off on
    # curved cells as well (pyramid bases are rational: those cells stay affine)
    mesh = _square(element_type) if element_type.startswith("triangle") else _cube(element_type)
    boundary = _box_boundary(mesh)
    if not element_type.startswith("pyramid"):
        mesh = _bend(mesh)
    dim = mesh.points.shape[1]
    flux = Flux.from_mesh(mesh, boundary_mask=boundary, quadrature_order=5)()
    flux = flux.reshape(mesh.n_points, dim + 1).sum(0)
    volume = Volume.from_mesh(mesh, quadrature_order=5)().sum()
    assert float(flux[:dim].abs().max()) < 1e-12
    assert abs(float(flux[dim] - dim * volume)) < 1e-12


def test_default_mask_is_the_whole_boundary():
    mesh = _cube("tetra10")
    assert abs(float(Area.from_mesh(mesh)().sum()) - 6.0) < 1e-12


def test_normals_point_out_of_each_cell():
    # a reflected cell (negative Jacobian determinant) still gets outward normals
    mesh = _cube("tetra", k=1)
    cells = mesh.cells["tetra"].clone()
    cells[::2, [0, 1]] = cells[::2, [1, 0]]
    trans = Transformation(mesh.points, cells, "tetra", quadrature_order=2)
    x = torch.einsum("fqb,ebd->efqd", trans.facet_shape_val, mesh.points[cells])        # facet points
    centroid = mesh.points[cells].mean(1)[:, None, None, :]
    n = trans.facet_normal
    assert torch.allclose(n.norm(dim=-1), torch.ones((), dtype=f64))
    assert bool((((x - centroid) * n).sum(-1) > 0).all())


def test_normals_follow_moved_points():
    mesh = _cube("tetra10", k=2)
    trans = Transformation(mesh.points, mesh.cells["tetra10"], "tetra10", quadrature_order=2)
    before = trans.facet_normal.clone()
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=f64)
    trans.update_points(mesh.points @ rotation.T)
    assert torch.allclose(trans.facet_normal, before @ rotation.T, atol=1e-14)


def test_bilinear_assembler_gets_normals():
    mesh = _cube("tetra")
    B = NormalXSquaredMass.from_mesh(mesh, quadrature_order=2)()
    # sum_ij B_ij = int n_x^2 ds = area of the two faces normal to x
    assert abs(float(B.values.sum()) - 2.0) < 1e-12


def test_point_data_named_n_takes_precedence():
    class Coefficient(FacetAssembler):
        def forward(self, v, n):
            return n * v

    mesh = _square("triangle")
    value = Coefficient.from_mesh(mesh)(point_data={"n": torch.full((mesh.n_points,), 2.0, dtype=f64)})
    assert abs(float(value.sum()) - 8.0) < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    mesh = _bend(_cube("tetra10", k=2))
    cpu = Flux.from_mesh(mesh, quadrature_order=4)()
    gpu = Flux.from_mesh(mesh.to("cuda"), quadrature_order=4)()
    assert gpu.is_cuda
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-12)

import itertools
import math

import meshio
import pytest
import torch

from tensormesh import Mesh, Condenser, LaplaceElementAssembler, MassElementAssembler, element_type2order
from tensormesh.element import Pyramid

f64 = torch.float64
V = Pyramid.points.to(f64)          # reference vertices; apex (0, 0, 1) above vertex 0


def _on_face(points, face, tol=1e-10):
    a, b, c = V[list(face[:3])]
    n = torch.linalg.cross(b - a, c - a)
    return ((points - a) @ n).abs() < tol


def _face_points(face):
    s = torch.linspace(0.1, 0.9, 5, dtype=f64)
    u, v = (t.reshape(-1, 1) for t in torch.meshgrid(s, s, indexing="ij"))
    if len(face) == 3:
        keep = (u + v < 1).flatten()
        p0, p1, p2 = V[list(face)]
        return (p0 + u * (p1 - p0) + v * (p2 - p0))[keep]
    p0, p1, p2, p3 = V[list(face)]           # the base is numbered lexicographically
    return (1 - u) * (1 - v) * p0 + u * (1 - v) * p1 + (1 - u) * v * p2 + u * v * p3


def _interior_points(n=200):
    s = torch.rand(n, 3, generator=torch.Generator().manual_seed(0), dtype=f64)
    s[:, 2] *= 0.95                          # stay away from the apex, where gradients have no limit
    return torch.stack([s[:, 0] * (1 - s[:, 2]), s[:, 1] * (1 - s[:, 2]), s[:, 2]], 1)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pyramid_basis_is_conforming(order):
    # a shape function whose node is not on a face (edge) vanishes on that face (edge), so the
    # trace there depends on the face's own nodes only (#62)
    nodes = Pyramid.get_basis(order, f64)
    phi = Pyramid.get_basis_fns(order, f64)
    for face in Pyramid.face:
        off = ~_on_face(nodes, face)
        assert phi.map(_face_points(face))[:, off].abs().max() < 1e-10, f"face {face}"
    t = torch.linspace(0.1, 0.9, 5, dtype=f64)[:, None]
    for a, b in Pyramid.edge.tolist():
        f1, f2 = [f for f in Pyramid.face if a in f and b in f]
        off = ~(_on_face(nodes, f1) & _on_face(nodes, f2))
        assert phi.map(V[a] + t * (V[b] - V[a]))[:, off].abs().max() < 1e-10, f"edge ({a}, {b})"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pyramid_basis_reproduces_polynomials(order):
    # P_r is part of the space, so interpolating a monomial of degree <= r is exact
    # (the constant gives the partition of unity)
    nodes = Pyramid.get_basis(order, f64)
    phi = Pyramid.get_basis_fns(order, f64).map(_interior_points())
    for a, b, c in itertools.product(range(order + 1), repeat=3):
        if a + b + c <= order:
            f = lambda p: p[:, 0]**a * p[:, 1]**b * p[:, 2]**c
            assert torch.allclose(phi @ f(nodes), f(_interior_points()), atol=1e-12), (a, b, c)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_pyramid_basis_gradient(order):
    phi = Pyramid.get_basis_fns(order, f64)
    grad = Pyramid.get_basis_grad_fns(order, f64)
    assert grad.shape == (3, phi.shape[0])
    x, h = _interior_points(), 1e-6
    fd = torch.stack([(phi.map(x + h * e) - phi.map(x - h * e)) / (2 * h) for e in torch.eye(3, dtype=f64)], 1)
    assert torch.allclose(grad.map(x), fd, atol=1e-6)


@pytest.mark.parametrize("n", range(1, 7))
def test_pyramid_quadrature_is_exact(n):
    # \int x^a y^b z^c over the pyramid with cross-section [0, 1 - z]^2 is
    # c! (a + b + 2)! / ((a + b + c + 3)! (a + 1) (b + 1)); the n-point rule is exact for a + b + c <= 2n - 1
    w, q = Pyramid.get_quadrature(n, dtype=f64)
    for a, b, c in itertools.product(range(2 * n), repeat=3):
        if a + b + c <= 2 * n - 1:
            exact = math.factorial(c) * math.factorial(a + b + 2) / math.factorial(a + b + c + 3) / ((a + 1) * (b + 1))
            approx = (w * q[:, 0]**a * q[:, 1]**b * q[:, 2]**c).sum().item()
            assert approx == pytest.approx(exact, rel=1e-12), (a, b, c)


def _cube_of_pyramids(element_type, k=2):
    """The unit cube cut into k^3 cubes, each cut into six pyramids with their apex at its centre."""
    ref = Pyramid.get_basis(element_type2order[element_type], f64)
    eye, h = torch.eye(3, dtype=f64), 1.0 / k
    cells = []
    for corner in itertools.product(range(k), repeat=3):
        o = torch.tensor(corner, dtype=f64) * h
        centre = o + h / 2
        for axis, side in itertools.product(range(3), (0, 1)):
            e1, e2 = (h * eye[d] for d in range(3) if d != axis)
            x0 = o + side * h * eye[axis]
            if torch.dot(torch.linalg.cross(e1, e2), centre - x0) < 0:   # keep the reference orientation
                e1, e2 = e2, e1
            cells.append(x0 + ref @ torch.stack([e1, e2, centre - x0]))  # affine image of the reference nodes
    nodes = torch.cat(cells)
    key, conn = torch.unique(torch.round(nodes * 1e6).long(), dim=0, return_inverse=True)
    points = torch.zeros(len(key), 3, dtype=f64).index_put_((conn,), nodes)
    return Mesh(meshio.Mesh(points.numpy(), [(element_type, conn.reshape(len(cells), -1).numpy())]))


@pytest.mark.parametrize("element_type", ["pyramid", "pyramid14"])
def test_pyramid_patch_test(element_type):
    # Laplace(u) = 0 with exact Dirichlet data: a harmonic polynomial of the element's degree is
    # reproduced to round-off only if the space is conforming and the stiffness is integrated exactly
    order = element_type2order[element_type]
    mesh = _cube_of_pyramids(element_type)
    x, y, z = mesh.points.T
    exact = 1 + x + 2 * y + 3 * z if order == 1 else x**2 - y**2 + x * y + y * z + z * x + x
    boundary = ((mesh.points < 1e-12) | (mesh.points > 1 - 1e-12)).any(1)
    assert not bool(boundary.all())
    K = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=order + 1)(mesh.points)
    condenser = Condenser(boundary, exact[boundary])
    K_, f_ = condenser(K, torch.zeros(mesh.n_points, dtype=f64))
    u = condenser.recover(K_.solve(f_))
    assert float((u - exact).abs().max()) < 1e-10


@pytest.mark.parametrize("element_type", ["pyramid", "pyramid14"])
def test_pyramid_mass_matrix_is_exact(element_type):
    # u^T M u = \int u^2 for u in P_r, which checks the quadrature weights
    order = element_type2order[element_type]
    mesh = _cube_of_pyramids(element_type)
    M = MassElementAssembler.from_mesh(mesh, quadrature_order=order + 1)(mesh.points).to_dense()
    u = mesh.points[:, 0] ** order
    assert float(u @ M @ u) == pytest.approx(1 / (2 * order + 1), rel=1e-12)   # \int_0^1 x^(2 order) dx

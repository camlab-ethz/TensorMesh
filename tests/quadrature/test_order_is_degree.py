"""``order`` is the polynomial exactness degree on every reference shape.

Each rule is checked against analytic moments of every monomial in the space
it guarantees: total degree on lines, triangles, tetrahedra and pyramids,
degree in each coordinate on quadrilaterals and hexahedra, total degree in
the triangle and degree in the height on prisms.
"""
import itertools
import math

import pytest
import torch

from tensormesh.element import Hexahedron, Line, Prism, Pyramid, Quadrilateral, Tetrahedron, Triangle

f64 = torch.float64
ELEMENTS = [Line, Triangle, Tetrahedron, Quadrilateral, Hexahedron, Pyramid, Prism]


def _guaranteed_monomials(element, d):
    if element is Line:
        return [(a,) for a in range(d + 1)]
    if element in (Triangle, Tetrahedron, Pyramid):
        return [p for p in itertools.product(range(d + 1), repeat=element.dim) if sum(p) <= d]
    if element in (Quadrilateral, Hexahedron):
        return list(itertools.product(range(d + 1), repeat=element.dim))
    return [(a, b, c) for a in range(d + 1) for b in range(d + 1 - a) for c in range(d + 1)]   # prism


def _moment(element, p):
    f = math.factorial
    if element in (Line, Quadrilateral, Hexahedron):          # unit interval / square / cube
        return math.prod(1 / (a + 1) for a in p)
    if element in (Triangle, Tetrahedron):                    # unit simplex
        return math.prod(f(a) for a in p) / f(sum(p) + element.dim)
    a, b, c = p
    if element is Pyramid:                                    # cross-section [0, 1 - z]^2
        return f(c) * f(a + b + 2) / (f(a + b + c + 3) * (a + 1) * (b + 1))
    return f(a) * f(b) / (f(a + b + 2) * (c + 1))             # prism: triangle times [0, 1]


@pytest.mark.parametrize("d", range(1, 10))
@pytest.mark.parametrize("element", ELEMENTS)
def test_rule_is_exact_through_the_requested_degree(element, d):
    w, q = element.get_quadrature(d, f64)
    powers = _guaranteed_monomials(element, d)
    exact = torch.tensor([_moment(element, p) for p in powers], dtype=f64)
    approx = w @ (q[:, None, :] ** torch.tensor(powers)[None]).prod(-1)
    torch.testing.assert_close(approx / exact, torch.ones_like(exact), rtol=1e-11, atol=0)


@pytest.mark.parametrize("d", range(1, 16))
def test_gaussian_factors_use_the_fewest_points(d):
    n = d // 2 + 1                                            # n Gauss points reach degree 2n - 1 >= d
    assert len(Line.get_quadrature(d)[0]) == n
    assert len(Quadrilateral.get_quadrature(d)[0]) == n ** 2
    assert len(Hexahedron.get_quadrature(d)[0]) == n ** 3
    assert len(Pyramid.get_quadrature(d)[0]) == n ** 3
    assert len(Prism.get_quadrature(d)[0]) == len(Triangle.get_quadrature(d)[0]) * n


def test_default_order_two_keeps_its_rules():
    # order 2 selected two Gauss points per direction before order became a degree, and still does
    counts = {Line: 2, Triangle: 3, Tetrahedron: 4, Quadrilateral: 4, Hexahedron: 8, Pyramid: 8, Prism: 6}
    for element, count in counts.items():
        assert len(element.get_quadrature(2)[0]) == count


@pytest.mark.parametrize("d", [1, 2, 3, 4, 7, 9])
def test_facet_rules_take_the_same_degree(d):
    n = d // 2 + 1
    n_tri = len(Triangle.get_quadrature(d)[0])
    assert Triangle.get_facet_quadrature(d)[0].shape == (3, n)
    assert Quadrilateral.get_facet_quadrature(d)[0].shape == (4, n)
    assert Tetrahedron.get_facet_quadrature(d)[0].shape == (4, n_tri)
    assert Hexahedron.get_facet_quadrature(d)[0].shape == (6, n ** 2)
    for element in (Prism, Pyramid):
        tri_w, _, quad_w, _ = element.get_facet_quadrature(d)
        assert tri_w.shape[1] == n_tri and quad_w.shape[1] == n ** 2

"""Gaussian rules checked against analytic moments, not another FEM backend."""

import math

import pytest
import torch

from tensormesh.element import Line, Triangle, Tetrahedron, Quadrilateral, Hexahedron, Prism, Pyramid
from tensormesh.element import quadrature


@pytest.mark.parametrize("n", [1, 7, 8, 32])
def test_line_gaussian_exactness(n):
    weights, points = Line.get_quadrature(n, torch.float64)
    assert weights.shape == (n,) and points.shape == (n, 1)
    assert (weights > 0).all()
    assert ((points > 0) & (points < 1)).all()
    powers = torch.arange(2 * n, dtype=torch.float64)
    actual = (weights[:, None] * points ** powers).sum(0)
    exact = 1 / (powers + 1)
    torch.testing.assert_close(actual / exact, torch.ones_like(exact), atol=3e-12, rtol=0)


@pytest.mark.parametrize("order", [8, 12])
@pytest.mark.parametrize("element", [Quadrilateral, Hexahedron, Prism, Pyramid])
def test_inherited_high_order_rules(element, order):
    # Sample constant, axial and mixed moments at the exactness limits of the
    # newly available factors, rather than repeat the upstream low-order tests.
    limit = 2 * order - 1
    if element in (Quadrilateral, Hexahedron):
        powers = [(0,) * element.dim, (limit,) * element.dim]
        powers += [tuple(limit if d == axis else 0 for d in range(element.dim))
                   for axis in range(element.dim)]
        exact = [math.prod(1 / (a + 1) for a in p) for p in powers]
        count = order ** element.dim
    elif element is Prism:
        # Triangle degree is order; the height uses order Gaussian points.
        powers = [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, limit),
                  (order // 2, order - order // 2, limit)]
        exact = [math.factorial(a) * math.factorial(b)
                 / (math.factorial(a + b + 2) * (c + 1)) for a, b, c in powers]
        count = ((order + 2) // 2) ** 2 * order
    else:
        # Pyramid cross-sections are [0, 1-z]^2; integrate x and y first,
        # then use the beta integral in z, independently of the collapse rule.
        powers = [(0, 0, 0), (limit, 0, 0), (0, limit, 0), (0, 0, limit),
                  (2, 3, limit - 5)]
        exact = [math.factorial(c) * math.factorial(a + b + 2)
                 / (math.factorial(a + b + c + 3) * (a + 1) * (b + 1))
                 for a, b, c in powers]
        count = order ** 3

    weights, points = element.get_quadrature(order, dtype=torch.float64)
    assert weights.shape == (count,) and points.shape == (count, element.dim)
    assert (weights > 0).all()
    monomials = (points[:, None, :] ** torch.tensor(powers)[None, :, :]).prod(-1)
    exact = torch.tensor(exact, dtype=torch.float64)
    actual = weights @ monomials
    torch.testing.assert_close(actual / exact, torch.ones_like(exact), rtol=1e-12, atol=0)


@pytest.mark.parametrize("degree", [2, 7, 8, 9, 20])
def test_triangle_all_monomials_through_requested_degree(degree):
    weights, points = Triangle.get_quadrature(degree, torch.float64)
    powers = [(a, total - a) for total in range(degree + 1) for a in range(total + 1)]
    exponents = torch.tensor(powers)
    monomials = points[:, :1] ** exponents[:, 0] * points[:, 1:] ** exponents[:, 1]
    exact = torch.tensor([math.exp(math.lgamma(a + 1) + math.lgamma(b + 1)
                                   - math.lgamma(a + b + 3)) for a, b in powers], dtype=torch.float64)
    actual = torch.einsum("q,qi->i", weights, monomials)
    # Relative normalization is essential: tiny high-degree moments must
    # not pass merely because of a generous absolute tolerance.
    torch.testing.assert_close(actual / exact, torch.ones_like(exact), atol=4e-12, rtol=0)
    if degree > 7:
        assert len(weights) == ((degree + 2) // 2) ** 2
        assert (weights > 0).all()
        assert (points > 0).all() and (points.sum(-1) < 1).all()
    torch.testing.assert_close(weights.sum(), torch.tensor(0.5, dtype=torch.float64), atol=3e-15, rtol=0)


def test_original_low_degree_triangle_rules_retained():
    for degree, count in enumerate([1, 3, 4, 6, 7, 12, 13], start=1):
        weights, points = Triangle.get_quadrature(degree, torch.float64)
        assert len(weights) == count
        if degree in (3, 7):
            assert weights[0] < 0  # preserve the original nonpositive rules


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_rule_layouts_dtype_and_facets(dtype):
    for element, order, count, dim in [(Line, 12, 12, 1), (Triangle, 20, 121, 2),
                                      (Tetrahedron, 8, 125, 3),
                                      (Quadrilateral, 8, 64, 2)]:
        weights, points = element.get_quadrature(order, dtype)
        assert weights.dtype == points.dtype == dtype
        assert weights.shape == (count,) and points.shape == (count, dim)
        assert weights.device.type == points.device.type == "cpu"
    weights, points = Triangle.get_facet_quadrature(12, dtype=dtype)
    assert weights.shape == (3, 12) and points.shape == (3, 12, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, dtype=dtype))


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("rule", [quadrature.tri_quadrature, quadrature.tet_quadrature])
def test_generated_accelerator_rules(rule, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    weights, points = rule(20, torch.float32, torch.device(device))
    assert weights.device.type == points.device.type == device
    reference = rule(20, torch.float32)
    torch.testing.assert_close(weights.cpu(), reference[0])
    torch.testing.assert_close(points.cpu(), reference[1])

"""Tetrahedral quadrature checked against analytic moments."""

import math

import pytest
import torch

from tensormesh.element import Tetrahedron
from tensormesh.element import quadrature


def _assert_tetrahedron_moments(weights, points, degree, rtol=5e-13):
    # Factorial moments are independent of the Gaussian generator and its map.
    # Normalize relatively so tiny high-degree moments cannot pass on atol alone.
    powers = [
        (a, b, total - a - b)
        for total in range(degree + 1)
        for a in range(total + 1)
        for b in range(total - a + 1)
    ]
    for start in range(0, len(powers), 128):
        batch = powers[start:start + 128]
        exponents = torch.tensor(batch, device=points.device)
        monomials = (
            points[:, :1] ** exponents[:, 0]
            * points[:, 1:2] ** exponents[:, 1]
            * points[:, 2:] ** exponents[:, 2]
        )
        exact = torch.tensor(
            [math.factorial(a) * math.factorial(b) * math.factorial(c)
             / math.factorial(a + b + c + 3) for a, b, c in batch],
            dtype=points.dtype, device=points.device,
        )
        actual = torch.einsum("q,qi->i", weights, monomials)
        torch.testing.assert_close(actual / exact, torch.ones_like(exact), rtol=rtol, atol=0)


@pytest.mark.parametrize("degree, count", [
    (1, 1), (4, 11), (5, 15), (6, 24), (7, 64), (8, 125), (20, 1331),
])
def test_tetrahedron_all_analytic_moments(degree, count):
    weights, points = Tetrahedron.get_quadrature(degree, dtype=torch.float64)
    assert weights.shape == (count,)
    assert points.shape == (count, 3)
    assert weights.dtype == points.dtype == torch.float64
    assert weights.device.type == points.device.type == "cpu"
    assert torch.isfinite(weights).all() and torch.isfinite(points).all()
    if degree >= 7:
        assert (weights > 0).all()
        assert (points > 0).all() and (points.sum(-1) < 1).all()
    torch.testing.assert_close(
        weights.sum(), torch.tensor(1 / 6, dtype=torch.float64), rtol=0, atol=3e-15
    )
    _assert_tetrahedron_moments(weights, points, degree)


def test_generated_tetrahedron_cache_reuse_and_mutation_isolation(monkeypatch):
    quadrature._gaussian_interval_rule.cache_clear()
    quadrature._gaussian_tetrahedron_rule.cache_clear()
    calls = []
    original = quadrature.roots_jacobi

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(quadrature, "roots_jacobi", counted)
    weights, points = quadrature.tet_quadrature(20, dtype=torch.float64)
    reference = (weights.clone(), points.clone())
    assert len(calls) == 3
    weights.zero_()
    points.fill_(float("nan"))

    # Even/odd degrees share n=11; conversions must not expose cached storage.
    converted = quadrature.tet_quadrature(21, dtype=torch.float32)
    for actual, expected in zip(converted, reference):
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
        actual.zero_()
    fresh = quadrature.tet_quadrature(20, dtype=torch.float64)
    for actual, expected in zip(fresh, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 3

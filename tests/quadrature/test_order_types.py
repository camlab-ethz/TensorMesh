"""NumPy integer orders work with both cold and warm public rule caches."""

import numpy as np
import pytest
import torch

from tensormesh.element import Line, Tetrahedron, Triangle
from tensormesh.element import quadrature


@pytest.mark.parametrize("order", [np.int32(4), np.int64(8)])
@pytest.mark.parametrize("element, rule", [
    (Line, quadrature.lin_quadrature),
    (Triangle, quadrature.tri_quadrature),
    (Tetrahedron, quadrature.tet_quadrature),
])
def test_numpy_integer_orders_with_cold_and_warm_caches(element, rule, order):
    factory = element.get_quadrature
    expected = factory(int(order), dtype=torch.float64)
    direct = rule(order, dtype=torch.float64)
    for actual, reference in zip(direct, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for warm in (False, True):
        factory.cache_clear()
        if warm:
            factory(int(order), dtype=torch.float64)
        actual = factory(order, dtype=torch.float64)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

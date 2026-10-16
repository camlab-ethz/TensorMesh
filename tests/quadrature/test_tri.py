import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from tensormesh.element import Triangle

@pytest.mark.parametrize("n", range(1, 7))
def test_shape(n:int):

    weights, points = Triangle.get_quadrature(n)
    assert weights.dim() == 1
    assert points.dim() == 2 
    assert weights.shape[0] == points.shape[0]
    assert points.shape[-1] == 2

@pytest.mark.parametrize("n", range(1, 7))
def test_sum(n:int):
    r"""
        \int_0^1 \int_0^{1-x} 1 dx dy = 1/2
    """
    weights, points = Triangle.get_quadrature(n, dtype=torch.float64)
    np.testing.assert_allclose(weights.sum().item(), 0.5, rtol=1e-12)

import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from tensormesh.element import Line

@pytest.mark.parametrize("n", range(1, 7))
def test_shape(n:int):
    weights, points = Line.get_quadrature(n)
    assert weights.shape == (n,), f"weights.shape = {weights.shape} at n = {n}"
    assert points.shape == (n, 1), f"points.shape = {points.shape} at n = {n}"
    
@pytest.mark.parametrize("n", range(1, 7))
def test_sum(n:int):
    weights, points = Line.get_quadrature(n, dtype=torch.float64)
    np.testing.assert_allclose(weights.sum().item(), 1.0, rtol=1e-12)
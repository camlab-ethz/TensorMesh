import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from tensormesh.element import Quadrilateral

@pytest.mark.parametrize("n", range(1, 7))
def test_shape(n:int):
    weights, points = Quadrilateral.get_quadrature(n)
    assert weights.shape == ((n)**2,)
    assert points.shape == ((n)**2, 2)

@pytest.mark.parametrize("n", range(1, 7))
def test_sum(n:int):
    weights, points = Quadrilateral.get_quadrature(n, dtype=torch.float64)
    np.testing.assert_allclose(weights.sum().item(), 1, rtol=1e-12)
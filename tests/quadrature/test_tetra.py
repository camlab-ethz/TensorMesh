import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from tensormesh.element import Tetrahedron

@pytest.mark.parametrize("n", range(1, 7))
def test_shape(n:int):

    weights, points = Tetrahedron.get_quadrature(n)
    assert weights.dim() == 1
    assert points.dim() == 2 
    assert weights.shape[0] == points.shape[0]
    assert points.shape[-1] == 3

@pytest.mark.parametrize("n", range(1, 7))
def test_sum(n:int):

    weights, points = Tetrahedron.get_quadrature(n, dtype=torch.float64)
    np.testing.assert_allclose(weights.sum().item(), 1/6, rtol=1e-12)
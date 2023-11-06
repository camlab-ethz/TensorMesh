import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from torch_fem.quadrature import get_quadrature

def test_shape():
    for n in range(1, 24):
        weights, points = get_quadrature("quad",n)
        assert weights.shape == ((n+1)**2,)
        assert points.shape == ((n+1)**2, 2)

def test_sum():
    for n in range(1, 24):
        weights, points = get_quadrature("quad",n)
        np.testing.assert_allclose(weights.sum().item(), 1)
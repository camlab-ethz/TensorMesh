import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from torch_fem.quadrature import get_quadrature

def test_shape():
    for n in range(1, 20):
        weights, points = get_quadrature("triangle",n)
        assert weights.dim() == 1
        assert points.dim() == 2 
        assert weights.shape[0] == points.shape[0]
        assert points.shape[-1] == 2

def test_sum():
    """
        \int_0^1 \int_0^{1-x} 1 dx dy = 1/2
    """
    for n in range(1, 20):
        weights, points = get_quadrature("triangle",n)
        np.testing.assert_allclose( weights.sum().item() , 0.5)

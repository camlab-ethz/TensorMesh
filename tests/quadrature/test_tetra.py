import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from torch_fem.quadrature import get_quadrature

def test_shape():
    for n in range(1,10):
        weights, points = get_quadrature("tetra",n)
        assert weights.dim() == 1
        assert points.dim() == 2 
        assert weights.shape[0] == points.shape[0]
        assert points.shape[-1] == 3

def test_sum():
    for n in range(1,10):
        weights, points = get_quadrature("tetra",n)
        np.testing.assert_allclose(weights.sum().item(),1/6, rtol=1e-6)
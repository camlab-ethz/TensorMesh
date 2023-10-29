import pytest
import torch
import sys 
sys.path.append("../../..")
from torch_fem.quadrature.quad import gauss_points

def test_shape():
    for n in range(1, 8):
        weights, points = gauss_points(n**2)
        assert weights.shape == (n**2,)
        assert points.shape == (n**2, 2)

def test_sum():
    for n in range(1, 8):
        weights, points = gauss_points(n**2)
        assert weights.sum().item() == 4
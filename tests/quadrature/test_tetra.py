import pytest
import torch
import sys 
sys.path.append("../../..")
from torch_fem.quadrature.tetra import gauss_points

def test_shape():
    for n in [1, 4, 5]:
        weights, points = gauss_points(n)
        assert weights.shape == (n,)
        assert points.shape == (n, 3)

def test_sum():
    for n in [1, 4, 5]:
        weights, points = gauss_points(n)
        assert weights.sum().item() == 1/6
import pytest
import torch
import sys 
sys.path.append("../../..")
from torch_fem.quadrature.tri import gauss_points

def test_shape():
    for n in range(1, 4):
        weights, points = gauss_points(n)
        assert weights.shape == (n,)
        assert points.shape == (n, 2)

def test_sum():
    for n in range(1, 4):
        weights, points = gauss_points(n)
        assert weights.sum().item() == 0.5
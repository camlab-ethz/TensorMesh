import pytest
import torch
import sys 
sys.path.append("../../..")
from torch_fem.quadrature.line import gauss_points

def test_shape():
    for n in range(1, 8):
        weights, points = gauss_points(n)
        assert weights.shape == (n,), f"weights.shape = {weights.shape} at n = {n}"
        assert points.shape == (n, 1), f"points.shape = {points.shape} at n = {n}"
        
def test_sum():
    for n in range(1, 8):
        weights, points = gauss_points(n)
        assert weights.sum().item() == 2.0, f"weights.sum() = {weights.sum().item()} at n = {n}"
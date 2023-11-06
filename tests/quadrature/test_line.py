import pytest
import torch
import numpy as np
import sys 
sys.path.append("../../..")
from torch_fem.quadrature import get_quadrature

def test_shape():
    for n in range(1, 24):
        weights, points = get_quadrature("line",n)
        assert weights.shape == (n+1,), f"weights.shape = {weights.shape} at n = {n}"
        assert points.shape == (n+1, 1), f"points.shape = {points.shape} at n = {n}"
        
def test_sum():
    for n in range(1, 24):
        weights, points = get_quadrature("line",n)
        np.testing.assert_allclose(weights.sum().item(),1.0), f"weights.sum() = {weights.sum().item()} at n = {n}"
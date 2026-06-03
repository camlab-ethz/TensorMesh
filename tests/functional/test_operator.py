import sys
sys.path.append("../..")
import torch
import pytest
from tensormesh.functional.ops import skew, sym


def test_sym():
    # Test basic functionality
    x = torch.tensor([1., 2.])
    y = sym(x)
    assert y.shape == (2, 2)
    expected = torch.tensor([[1., 1.5],
                           [1.5, 2.]])
    assert torch.allclose(y, expected)

    # Test batched input
    x = torch.rand(10, 2)
    def fn(x):
        return sym(x)
    y = torch.vmap(fn)(x)
    assert y.shape == (10, 2, 2)
    # Check symmetry property
    assert torch.allclose(y, y.transpose(-2, -1))

    # Test 3D case
    x = torch.tensor([1., 2., 3.])
    y = sym(x) 
    assert y.shape == (3, 3)
    expected = torch.tensor([[1., 1.5, 2.],
                           [1.5, 2., 2.5],
                           [2., 2.5, 3.]])
    assert torch.allclose(y, expected)

def test_skew():
    # 2D case
    x = torch.tensor([1.,2.])
    y = skew(x)
    assert y.shape == (2,)
    assert torch.allclose(y, torch.tensor([-2., 1.])) # Check numerical values

    # Test sign=False for 2D
    y = skew(x, sign=False)
    assert torch.allclose(y, torch.tensor([2., 1.])) # With sign=False, first element is x[0]

    # Test at_least2d=True for 2D
    y = skew(x, at_least2d=True)
    assert y.shape == (1, 2)
    assert torch.allclose(y[0], torch.tensor([-2., 1.]))

    # Test both options for 2D
    y = skew(x, sign=False, at_least2d=True) 
    assert y.shape == (1, 2)
    assert torch.allclose(y[0], torch.tensor([2., 1.]))
    
    # 3D case
    x = torch.tensor([1.,2.,3.])
    y = skew(x)
    assert y.shape == (3,3)
    expected = torch.tensor([[0., -3., 2.],
                           [3., 0., -1.],
                           [-2., 1., 0.]])
    assert torch.allclose(y, expected) # Check numerical values

    # Test sign=False for 3D
    y = skew(x, sign=False)
    assert y.shape == (3,3)
    expected = torch.tensor([[0., 3., 2.],
                           [3., 0., 1.],
                           [2., 1., 0.]])
    assert torch.allclose(y, expected)



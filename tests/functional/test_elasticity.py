import sys 
sys.path.append("../..")
import torch
import pytest
from tensormesh.functional import elasticity

def test_isotropic_stress():

    # Test 2D case with scalar parameters
    strain_2d = torch.rand(10, 2, 2)
    E = 70.0
    nu = 0.3
    def fn(strain_2d):
        return elasticity.isotropic_stress(strain_2d, E, nu)
    stress_2d = torch.vmap(fn)(strain_2d)



def test_voigt_shape_grad():
    # Test 2D case
    gradu = torch.tensor([1., 2.])
    B = elasticity.voigt_shape_grad(gradu)
    assert B.shape == (3, 2)
    expected = torch.tensor([[1., 0.],
                           [0., 2.],
                           [2., 1.]])

    assert torch.allclose(B, expected)

    # Test 3D case 
    gradu = torch.tensor([1., 2., 3.])
    B = elasticity.voigt_shape_grad(gradu)
    assert B.shape == (6, 3)
    expected = torch.tensor([[1., 0., 0.],
                           [0., 2., 0.],
                           [0., 0., 3.],
                           [0., 3., 2.],
                           [3., 0., 1.],
                           [2., 1., 0.]])
    assert torch.allclose(B, expected)

    # Test invalid dimensions
    with pytest.raises(AssertionError):
        gradu = torch.tensor([1., 2., 3., 4.])
        elasticity.voigt_shape_grad(gradu)

    with pytest.raises(AssertionError):
        gradu = torch.rand(2, 2)
        elasticity.voigt_shape_grad(gradu)

def test_voigt_voigt_stiffness():
    # Test 2D case with scalar parameters
    E = 70.0
    nu = 0.3
    C = elasticity.voigt_stiffness(E, nu, dim=2)
    assert C.shape == (3, 3)
    
    # Calculate expected values
    mu = E/(2*(1 + nu))
    _lambda = E*nu/((1+nu)*(1-2*nu))
    expected = torch.zeros(3, 3)
    expected[0,0] = expected[1,1] = _lambda + 2*mu
    expected[0,1] = expected[1,0] = _lambda
    expected[2,2] = mu
    assert torch.allclose(C, expected)

    # Test 3D case with tensor parameters
    E = torch.tensor(70.0)
    nu = torch.tensor(0.3)
    C = elasticity.voigt_stiffness(E, nu, dim=3)
    assert C.shape == (6, 6)

    # Calculate expected values
    mu = E/(2*(1 + nu))
    _lambda = E*nu/((1+nu)*(1-2*nu))
    expected = torch.zeros(6, 6)
    expected[:3,:3] = _lambda
    expected[0,0] = expected[1,1] = expected[2,2] = _lambda + 2*mu
    expected[3,3] = expected[4,4] = expected[5,5] = mu
    assert torch.allclose(C, expected)

    # Test invalid dimensions
    with pytest.raises(Exception):
        elasticity.voigt_stiffness(E, nu, dim=4)


def test_voigt_shape_val():
    # Test 2D case
    u = torch.tensor(1.0)
    N = elasticity.voigt_shape_val(u, dim=2)
    assert N.shape == (2, 2)
    expected = torch.eye(2)
    assert torch.allclose(N, expected)

    # Test 3D case 
    N = elasticity.voigt_shape_val(u, dim=3)
    assert N.shape == (3, 3)
    expected = torch.eye(3)
    assert torch.allclose(N, expected)

    # Test invalid dimensions
    with pytest.raises(AssertionError):
        elasticity.voigt_shape_val(u, dim=4)
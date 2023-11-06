import pytest
import torch
import sys 

sys.path.append("../..")
from torch_fem.shape.quad import shape_val_p1,\
                                shape_val_p2,\
                                shape_grad_p1,\
                                shape_grad_p2,\
                                basis_p1,\
                                basis_p2


def test_sum_p1():
    x, y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = shape_val_p1(xy) # [n_quadrature, n_basis]
    phi  = phi.sum(dim=-1)
    assert torch.allclose(phi, torch.ones_like(phi))

def test_max_p1():
    x, y = torch.meshgrid(torch.linspace(0, 1, 200), torch.linspace(0, 1, 200))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = shape_val_p1(xy) # [n_quadrature, n_basis]
    phi  = phi.max(dim=0).values # [n_basis]
    assert torch.allclose(phi, torch.ones_like(phi))

def test_sum_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = shape_val_p2(xy) # [n_quadrature, n_basis]
    phi  = phi.sum(dim=-1)
    assert torch.allclose(phi, torch.ones_like(phi))

def test_max_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = shape_val_p2(xy) # [n_quadrature, n_basis]
    phi  = phi.max(dim=0).values # [n_basis]
    assert torch.allclose(phi, torch.ones_like(phi),  rtol=1e-4), f"phi = {phi}"

def test_integral_p1():
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    x  = x.reshape(-1)
    y  = y.reshape(-1)
    element_coords = basis_p1

    quadrature = torch.stack([x, y], dim=-1)
    grad = shape_grad_p1(torch.stack([x, y], dim=-1), element_coords[None, :, :], return_jac=False)[0] # [n_quadrature, n_basis, n_dim]

    n_basis = grad.shape[1]
    for i_basis in range(n_basis):

        integral_dx = torch.trapz(grad[:, i_basis, 0], quadrature[:, 0], dim=0) # []
        integral_dy = torch.trapz(grad[:, i_basis, 1], quadrature[:, 1], dim=0) # []

        torch.allclose(integral_dx, torch.zeros_like(integral_dx))
        torch.allclose(integral_dy, torch.zeros_like(integral_dy))

def test_integral_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    x  = x.reshape(-1)
    y  = y.reshape(-1)
   
    element_coords = basis_p2

    quadrature = torch.stack([x, y], dim=-1)
    grad = shape_grad_p2(torch.stack([x, y], dim=-1), element_coords[None, :, :], return_jac=False)[0] # [n_quadrature, n_basis, n_dim]

    n_basis = grad.shape[1]
    for i_basis in range(n_basis):

        integral_dx = torch.trapz(grad[:, i_basis, 0], quadrature[:, 0], dim=0) # []
        integral_dy = torch.trapz(grad[:, i_basis, 1], quadrature[:, 1], dim=0) # []

        torch.allclose(integral_dx, torch.zeros_like(integral_dx))
        torch.allclose(integral_dy, torch.zeros_like(integral_dy))
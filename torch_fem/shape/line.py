import torch 
import numpy as np
from numpy.polynomial.legendre import Legendre


basis_p1 = torch.tensor([[0.], [1.]], dtype=torch.float64)

def shape_val_p1(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 1 for line
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 2
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    assert quadrature.dim() == 2, f"quadrature must be 2D, but got {quadrature.dim()}"

    phi = torch.zeros((*quadrature.shape[:-1], 2), device=quadrature.device,  dtype=quadrature.dtype)
    x = quadrature[..., 0]
    phi[:, 0] = 1 - x
    phi[:, 1] = x

    return phi

def shape_grad_p1(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_basis, n_dim]
                        n_dim = 1 for line
            return_jac: bool
                        whether to return the jacobian
                        default is False
        Returns:
        --------
            grad_phi: torch.Tensor of shape [n_element, n_quadrature, n_basis, n_dim]
                the gradient of the base functions
            jac     : torch.Tensor of shape [n_element, n_quadrature, n_dim, n_dim]
                the jacobian of the base functions
                if return_jac is False, then jac is None
    """
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.shape[1:] == (2, 1), f"element_coords must be 3D of shape [n_element, 2, 1], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    assert n_basis == 2, f"n_basis must be 2 for line , but got {n_basis}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    grad_phi[:, 0, 0] = -1
    grad_phi[:, 1, 0] = 1
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi
    
basis_p2 = torch.tensor([[0.], [1.], [0.5]],dtype=torch.float64)


def shape_val_p2(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 1 for line
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 3
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    assert quadrature.dim() == 2, f"quadrature must be 2D, but got {quadrature.dim()}"

    phi = torch.zeros((*quadrature.shape[:-1], 3), device=quadrature.device,  dtype=quadrature.dtype)
    x = quadrature[..., 0]
    x2 = x ** 2
    phi[:, 0] = 1 - 3 * x + 2 * x2
    phi[:, 1] = -x + 2 * x2
    phi[:, 2] = 4 * x - 4 * x2

    return phi

def shape_grad_p2(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_basis, n_dim]
                        n_dim = 1 for line
            return_jac: bool
                        whether to return the jacobian
                        default is False
        Returns:
        --------
            grad_phi: torch.Tensor of shape [n_element, n_quadrature, n_basis, n_dim]
                the gradient of the base functions
            jac     : torch.Tensor of shape [n_element, n_quadrature, n_dim, n_dim]
                the jacobian of the base functions
                if return_jac is False, then jac is None
    """
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.shape[1:] == (3, 1), f"element_coords must be 3D of shape [n_element, 2, 1], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    assert n_basis == 3, f"n_basis must be 3 for line , but got {n_basis}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    x = quadrature[..., 0]
    grad_phi[:, 0, 0] = -3 + 4 * x
    grad_phi[:, 1, 0] = -1 + 4 * x
    grad_phi[:, 2, 0] = 4 - 8 * x
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi
    

"""
    Experimental !!!
"""

def shape_val_pn(quadrature, p):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 1 for line
            p:int   
                        the element order
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = n+1
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    assert quadrature.dim() == 2, f"quadrature must be 2D, but got {quadrature.dim()}"

    phi = torch.zeros((*quadrature.shape[:-1], p+1), device=quadrature.device,  dtype=quadrature.dtype)
    x   = quadrature[..., 0]
    phi[:, 0] = 1. - x
    phi[:, 1] = x

    y = 2. * x - 1.
    for i in range(2, p + 1):
        c = np.zeros(i)
        c[i - 1] = 1.
        s = Legendre(c).integ(lbnd=-1)
        scale = np.sqrt((2. * i - 1.) / 2.)
        phi[:, i] = s(y) * scale
        
    return phi 

def shape_grad_pn(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_basis, n_dim]
                        n_dim = 1 for line
            return_jac: bool
                        whether to return the jacobian
                        default is False
        Returns:
        --------
            grad_phi: torch.Tensor of shape [n_element, n_quadrature, n_basis, n_dim]
                the gradient of the base functions
            jac     : torch.Tensor of shape [n_element, n_quadrature, n_dim, n_dim]
                the jacobian of the base functions
                if return_jac is False, then jac is None
    """
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert element_coords.device == quadrature.device, f"element_coords.device must be {quadrature.device}, but got {element_coords.device}"
    assert element_coords.shape[1:] == (3, 1), f"element_coords must be 3D of shape [n_element, 2, 1], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 1, f"n_dim must be 1 for line , but got {n_dim}"
    p = n_basis - 1
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    x = quadrature[..., 0]
           
    grad_phi[:, 0, 0] = -1. + 0. * x
    grad_phi[:, 1, 0] = 1. + 0. * x

    y = 2. * x - 1.
    for i in range(2, p + 1):
        c = np.zeros(i)
        c[i - 1] = 1.
        s = Legendre(c).integ(lbnd=-1)
        scale = np.sqrt((2. * i - 1.) / 2.)
        grad_phi[:, i, 0] = 2 * s.deriv()(y) * scale
    
    jac  = torch.einsum("bhi,ghj->bgij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("gbi,ngji->ngbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    x = torch.linspace(0, 1, 100)[:, None]
    y = shape_val_pn(x, 2)
    fig, ax = plt.subplots()
    for i in range(y.shape[-1]):
        ax.plot(x, y[:, i])
    plt.show()

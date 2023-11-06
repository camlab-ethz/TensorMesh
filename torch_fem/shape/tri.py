import torch 

basis_p1 = torch.tensor([[0, 0],
                         [1, 0],
                         [0, 1]], dtype=torch.float)

def shape_val_p1(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 2 for triangle
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 3
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros((*quadrature.shape[:-1], 3), device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    phi[..., 0] = 1 - xi - eta
    phi[..., 1] = xi
    phi[..., 2] = eta
    return phi

def shape_grad_p1(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 2 for triangle
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
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 3, 2], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    grad_phi[..., 0, (0,1)] = -1
    grad_phi[..., 1, 0] = 1
    grad_phi[..., 2, 1] = 1
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi

basis_p2 = torch.tensor([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=torch.float32)

def shape_val_p2(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 2 for triangle
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 6
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros(*quadrature.shape[:-1], 6, device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    
    phi[..., 0] = (1 - xi - eta) * ( 1 - 2*xi - 2*eta)
    phi[..., 1] = xi * (2*xi - 1)
    phi[..., 2] = eta * (2*eta - 1)
    phi[..., 3] = 4*xi * (1 - xi - eta)
    phi[..., 4] = 4*xi * eta
    phi[..., 5] = 4*eta * (1 - xi - eta)
    # phi[..., 0] = 1 - 3*xi - 3*eta + 2*xi*xi + 4*xi*eta + 2*eta*eta
    # phi[..., 1] = 2*xi - 2*xi*xi - 2*xi*eta
    # phi[..., 2] = -xi + 2*xi*xi
    # phi[..., 3] = 2*xi*eta
    # phi[..., 4] = -eta + 2*xi*eta
    # phi[..., 5] = 2*eta*eta - 2*eta
    return phi

def shape_grad_p2(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 2 for triangle
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
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 6, 2], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_corner, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    n_basis = 6
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    grad_phi[..., 0, 0] = -3 + 4*xi + 4*eta
    grad_phi[..., 0, 1] = -3 + 4*xi + 4*eta
    grad_phi[..., 1, 0] = 2 - 4*xi - 2*eta
    grad_phi[..., 1, 1] = -2*xi
    grad_phi[..., 2, 0] = 2 - 4*xi - 2*eta
    grad_phi[..., 2, 1] = 0
    grad_phi[..., 3, 0] = 2*eta
    grad_phi[..., 3, 1] = 2*xi
    grad_phi[..., 4, 0] = -2*eta
    grad_phi[..., 4, 1] = 1 - 2*xi - 2*eta
    grad_phi[..., 5, 0] = 0
    grad_phi[..., 5, 1] = 4*eta - 2

    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi


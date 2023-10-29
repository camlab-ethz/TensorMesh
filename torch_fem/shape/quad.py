import torch 



basis_p1 = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float32) 

def shape_val_p1(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 2 for quadliteral
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 4
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    assert quadrature.dim() == 2, f"quadrature must be 2D, but got {quadrature.dim()}"

    phi = torch.zeros((*quadrature.shape[:-1], 4), device=quadrature.device,  dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    phi[:, 0] = 0.25 * (1 - xi) * (1 - eta)
    phi[:, 1] = 0.25 * (1 + xi) * (1 - eta)
    phi[:, 2] = 0.25 * (1 + xi) * (1 + eta)
    phi[:, 3] = 0.25 * (1 - xi) * (1 + eta)

    return phi

def shape_grad_p1(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_basis, n_dim]
                        n_dim = 2 for quadliteral
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
    assert element_coords.shape[1:] == (4, 2), f"element_coords must be 3D of shape [n_element, 4, 2], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    eta, xi  = quadrature[..., 0], quadrature[..., 1]
    grad_phi[:, 0, 0] = -0.25 * (1 - xi)
    grad_phi[:, 0, 1] = -0.25 * (1 - eta)
    grad_phi[:, 1, 0] = 0.25 * (1 - xi)
    grad_phi[:, 1, 1] = -0.25 * (1 + eta)
    grad_phi[:, 2, 0] = 0.25 * (1 + xi)
    grad_phi[:, 2, 1] = 0.25 * (1 + eta)
    grad_phi[:, 3, 0] = -0.25 * (1 + xi)
    grad_phi[:, 3, 1] = 0.25 * (1 - eta)
    
    
    jac  = torch.einsum("bhi,ghj->bgij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("gbi,ngji->ngbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi

basis_p2 = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]], dtype=torch.float32) 

def shape_val_p2(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 2 for triangle
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 9
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

    phi = torch.zeros(*quadrature.shape[:-1], 9, device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]

    phi[:, 0] = 0.25 * (xi * xi - xi) * (eta * eta - eta)
    phi[:, 1] = 0.25 * (xi * xi + xi) * (eta * eta - eta)
    phi[:, 2] = 0.25 * (xi * xi + xi) * (eta * eta + eta)
    phi[:, 3] = 0.25 * (xi * xi - xi) * (eta * eta + eta)
    phi[:, 4] = 0.5 * (1 - xi * xi) * (eta * eta - eta)
    phi[:, 5] = 0.5 * (1 - eta * eta) * (xi * xi + xi)
    phi[:, 6] = 0.5 * (1 - xi * xi) * (eta * eta + eta)
    phi[:, 7] = 0.5 * (1 - eta * eta) * (xi * xi - xi)
    phi[:, 8] = (1 - xi * xi) * (1 - eta * eta)


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
    assert element_coords.shape[1:] == (9,2), f"element_coords must be 3D of shape [n_element, 9, 2], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_corner, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    n_basis = 9
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device, dtype=quadrature.dtype)
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    
    grad_phi[:, 0, 0] = 0.25 * (2 * xi - 1) * (eta * eta - eta)
    grad_phi[:, 0, 1] = 0.25 * (xi * xi - xi) * (2 * eta - 1)
    grad_phi[:, 1, 0] = 0.25 * (2 * xi + 1) * (eta * eta - eta)
    grad_phi[:, 1, 1] = 0.25 * (xi * xi + xi) * (2 * eta - 1)
    grad_phi[:, 2, 0] = 0.25 * (2 * xi + 1) * (eta * eta + eta)
    grad_phi[:, 2, 1] = 0.25 * (xi * xi + xi) * (2 * eta + 1)
    grad_phi[:, 3, 0] = 0.25 * (2 * xi - 1) * (eta * eta + eta)
    grad_phi[:, 3, 1] = 0.25 * (xi * xi - xi) * (2 * eta + 1)
    grad_phi[:, 4, 0] = -xi * (eta * eta - eta)
    grad_phi[:, 4, 1] = 0.5 * (1 - xi * xi) * (2 * eta - 1)
    grad_phi[:, 5, 0] = 0.5 * (1 - eta * eta) * (2 * xi + 1)
    grad_phi[:, 5, 1] = -eta * (xi * xi + xi)
    grad_phi[:, 6, 0] = -xi * (eta * eta + eta)
    grad_phi[:, 6, 1] = 0.5 * (1 - xi * xi) * (2 * eta + 1)
    grad_phi[:, 7, 0] = 0.5 * (1 - eta * eta) * (2 * xi - 1)
    grad_phi[:, 7, 1] = -eta * (xi * xi - xi)
    grad_phi[:, 8, 0] = -2 * xi * (1 - eta * eta)
    grad_phi[:, 8, 1] = -2 * eta * (1 - xi * xi)

    jac  = torch.einsum("bhi,ghj->bgij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("gbi,ngji->ngbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi


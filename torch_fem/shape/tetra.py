import torch 

basis_p1 = torch.tensor([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])

def shape_val_p1(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 3 for tetra
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 4
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"

    phi = torch.zeros((*quadrature.shape[:-1], 4), device=quadrature.device, dtype=quadrature.dtype)
    x, y, z = quadrature[..., 0], quadrature[..., 1], quadrature[..., 2]
    phi[..., 0] = 1 - x - y - z
    phi[..., 1] = x
    phi[..., 2] = y 
    phi[..., 3] = z
    return phi

def shape_grad_p1(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 3 for tetra
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
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 4, 3], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"
    assert n_basis == 4, f"n_basis must be 4 for tetra , but got {n_basis}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    grad_phi[..., 0, :] = -1
    grad_phi[..., 1, 0] = 1
    grad_phi[..., 2, 1] = 1
    grad_phi[..., 3, 2] = 1
    
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi
    

basis_p2 = torch.tensor([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [.5, 0., 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., .0, .5],
                        [.5, .0, .5],
                        [.0, .5, .5]])

def shape_val_p2(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 3 for tetra
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 10
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"

    phi = torch.zeros((*quadrature.shape[:-1], 10), device=quadrature.device, dtype=quadrature.dtype)
    x, y, z = quadrature[..., 0], quadrature[..., 1], quadrature[..., 2]
    xy, xz, yz  = x * y, x * z, y * z
    x2, y2, z2  = x * x, y * y, z * z
    phi[..., 0] = 1. - 3.*x + 2.*x2 - 3.*y + 4.*xy + 2.*y2 - 3.*z + 4.*xz + 4.*yz + 2.*z2
    phi[..., 1] = - 1.*x + 2.*x2
    phi[..., 2] = - 1.*y + 2.*y2 
    phi[..., 3] = - 1.*z + 2.*z2
    phi[..., 4] = 4.*x - 4.*x2 - 4.*xy - 4*xz
    phi[..., 5] = 4.*xy
    phi[..., 6] = 4.*y - 4.*xy - 4.*y2 - 4.*yz
    phi[..., 7] = 4.*z - 4.*xz - 4.*yz - 4.*z2
    phi[..., 8] = 4.*xz
    phi[..., 9] = 4.*yz
    return phi

def shape_grad_p2(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 3 for tetra
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
    assert element_coords.dim() == 3, f"element_coords must be 3D of shape [n_element, 10, 3], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"
    assert n_basis == 10, f"n_basis must be 10 for tetra , but got {n_basis}"
    
    grad_phi = torch.zeros(n_quadrature, n_basis, n_dim, device=quadrature.device,dtype=quadrature.dtype)
    x, y, z  = quadrature[..., 0], quadrature[..., 1], quadrature[..., 2]
    -3. + 4.*x + 4.*y + 4.*z
    grad_phi[..., 0, :] = (-3. + 4.*x + 4.*y + 4.*z)[:,None]
    grad_phi[..., 1, 0] = -1  + 4.*x
    grad_phi[..., 2, 1] = -1. + 4.*y
    grad_phi[..., 3, 2] = -1. + 4.*z
    grad_phi[..., 4, 0] = 4. - 8.*x - 4.*y - 4.*z
    grad_phi[..., 4, 1:]= (-4.*x)[:,None]
    grad_phi[..., 5, 0] = 4.*y
    grad_phi[..., 5, 1] = 4.*x
    grad_phi[..., 6, (0,2)] = (-4.*y)[:,None]
    grad_phi[..., 6, 1] = 4. - 4.*x - 8.*y - 4.*z
    grad_phi[..., 7,:-1] = (-4.*z)[:,None] 
    grad_phi[..., 7, 2] = 4. - 4.*x - 4.*y - 8.*z
    grad_phi[..., 8, 0] = 4.*z
    grad_phi[..., 8, 2] = 4.*x
    grad_phi[..., 9, 1] = 4.*z
    grad_phi[..., 9, 2] = 4.*y
    
    jac  = torch.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac = torch.inverse(jac)
    grad_phi = torch.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi
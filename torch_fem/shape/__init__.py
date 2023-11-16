import os
import toml
import torch
import scipy.spatial
from .tri import basis_p1 as tri3_basis_p1
from .tri import basis_p2 as tri6_basis_p2
from .tri import shape_val_p1 as tri3_shape_val 
from .tri import shape_grad_p1 as tri3_shape_grad
from .tri import shape_val_p2 as tri6_shape_val
from .tri import shape_grad_p2 as tri6_shape_grad
from .quad import basis_p1 as quad4_basis_p1
from .quad import basis_p2 as quad9_basis_p2
from .quad import shape_val_p1 as quad4_shape_val
from .quad import shape_grad_p1 as quad4_shape_grad
from .quad import shape_val_p2 as quad9_shape_val
from .quad import shape_grad_p2 as quad9_shape_grad
from .tetra import basis_p1 as tetra4_basis_p1
from .tetra import basis_p2 as tetra10_basis_p2
from .tetra import shape_val_p1 as tetra4_shape_val
from .tetra import shape_grad_p1 as tetra4_shape_grad
from .tetra import shape_val_p2 as tetra10_shape_val
from .tetra import shape_grad_p2 as tetra10_shape_grad


with open(os.path.join(os.path.dirname(__file__), "dimension.toml"), "r") as f:
    element_type2dimension = toml.load(f)
with open(os.path.join(os.path.dirname(__file__), "order.toml"), "r") as f:
    element_type2order = toml.load(f)


def get_basis(element_type):
    find_basis = {
        "triangle":tri3_basis_p1,
        "triangle6":tri6_basis_p2,
        "tri3":tri3_basis_p1,
        "tri6":tri6_basis_p2,
        "quad":quad4_basis_p1,
        "quad4":quad4_basis_p1,
        "quad9":quad9_basis_p2,
        "tetra":tetra4_basis_p1,
        "tetra4":tetra4_basis_p1,
        "tetra10":tetra10_basis_p2,
    }
    if element_type not in find_basis:
        raise ValueError(f"element_type must be one of {list(find_basis.keys())}, but got {element_type}")
    
    return find_basis[element_type]

def get_boundary(element_type):
    """
        Parameters:
        -----------
            element_type: str
                the type of the element
        Returns:
        --------
            boundary elements: torch.Tensor [n_boundary, n_boundary_basis]

        Examples:
        >>> get_boundary("quad")
        tensor([[1, 0],
                [2, 1],
                [3, 0],
                [3, 2]])
    """
    basis  = get_basis(element_type)
    hull   = scipy.spatial.ConvexHull(basis)
    return torch.tensor(hull.simplices, dtype=torch.int64)

def get_shape_val(element_type, quadrature_points):
    """
        Parameters:
        -----------
            element_type: str
                the type of the element
                must be one of ["tri3", "tri6"]
            quadrature_points: torch.Tensor [n_quadrature, n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
        Returns:
        --------
            phi: torch.Tensor of shape [n_quadrature, n_basis]
                the base functions
    """
    find_shape_val = {
        "triangle":tri3_shape_val,
        "triangle6":tri6_shape_val,
        "tri3":tri3_shape_val,
        "tri6":tri6_shape_val,
        "quad":quad4_shape_val,
        "quad4":quad4_shape_val,
        "quad9":quad9_shape_val,
        "tetra":tetra4_shape_val,
        "tetra4":tetra4_shape_val,
        "tetra10":tetra10_shape_val,
    }
    if element_type not in find_shape_val:
        raise ValueError(f"element_type must be one of {list(find_shape_val.keys())}, but got {element_type}")

    return  find_shape_val[element_type](quadrature_points)
   
def get_shape_grad(element_type, quadrature_weights, quadrature_points, element_coords):
    """
        Parameters:
        -----------
            element_type: str
                the type of the element
                must be one of ["tri3", "tri6"]
            quadrature_weights: torch.Tensor [n_quadrature]
                the quadrature weights
            quadrature_points: torch.Tensor [n_quadrature, n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                n_dim = 2 for triangle
                the coordinates of the element corners

        Returns:
        --------
            grad_phi: torch.Tensor of shape [n_element, n_quadrature, n_basis, n_dim]
                the gradient of the base functions
            jxw     : torch.Tensor of shape [n_element, n_quadrature]
                the jacobian of the base functions multiplied by the quadrature weights
    """
    find_shape_grad = {
        "triangle":tri3_shape_grad,
        "triangle6":tri6_shape_grad, 
        "tri3":tri3_shape_grad,
        "tri6":tri6_shape_grad,
        "quad":quad4_shape_grad,
        "quad4":quad4_shape_grad,
        "quad9":quad9_shape_grad,
        "tetra":tetra4_shape_grad,
        "tetra4":tetra4_shape_grad,
        "tetra10":tetra10_shape_grad,
    }

    assert element_type in find_shape_grad, f"element_type must be one of {list(find_shape_grad.keys())}, but got {element_type}"

    shape_grad, jac = find_shape_grad[element_type](quadrature_points, element_coords, return_jac=True) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, n_dim, n_dim]

    jacdet = torch.abs(torch.det(jac))
    jxw    = jacdet * quadrature_weights
    return shape_grad, jxw
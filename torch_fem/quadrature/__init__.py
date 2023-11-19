from .line import gauss_points as gauss_points_line
from .quad import gauss_points as gauss_points_quad
from .tetra import gauss_points as gauss_points_tetra
from .tri import gauss_points as gauss_points_tri
import toml 
import torch
import numpy as np
from numpy.polynomial.legendre import leggauss
import os 

quadrature_lookup = toml.load(os.path.join(os.path.dirname(__file__),"./quadrature.toml"))

def get_quadrature(element_type, order:int=None):
    """get the quadrature information
    Parameters
    ----------
    element_type : str
        the type of the element
    order : int, optional
        the order of the quadrature, by default None
        
    Returns
    -------
    weights : torch.Tensor
        the weights of the quadrature
    points : torch.Tensor
        the points of the quadrature
    """
    if element_type.startswith("line"): # line
        if order is None:
            order = 1
        points, weights = leggauss(order+1)
        points = torch.from_numpy(0.5 * points + 0.5)[:, None]
        weights = torch.from_numpy(0.5 * weights)
    elif element_type.startswith("tri"): # triangle
        if order is None:
            order = 1
        order = str(order)
        assert order in quadrature_lookup["tri"], f"order must be one of {list(quadrature_lookup['tri'].keys())}, but got {order}"
        points = torch.tensor(quadrature_lookup["tri"][order]["points"])
        weights = torch.tensor(quadrature_lookup["tri"][order]["weights"])
    elif element_type.startswith("quad"): # quadrilateral
        if order is None:
            order = 2
        points, weights = leggauss(order+1)
        points = 0.5 * points + 0.5
        weights = 0.5 * weights
        points  = np.stack(np.meshgrid(points, points), -1).reshape(-1, 2)
        weights = np.outer(weights, weights).reshape(-1)
        points = torch.from_numpy(points)
        weights = torch.from_numpy(weights)
    elif element_type.startswith("tet"): # tetrahedron
        if order is None:
            order = 1
        order = str(order)
        assert order in quadrature_lookup["tetra"], f"order must be one of {list(quadrature_lookup['tetra'].keys())}, but got {order}"
        points = torch.tensor(quadrature_lookup["tetra"][order]["points"])
        weights = torch.tensor(quadrature_lookup["tetra"][order]["weights"])
    elif element_type.startswith("hex"): # hexahedron
        if order is None:
            order = 2
        points, weights = leggauss(int(np.ceil((order + 1.0) / 2.0)))
        points  = np.stack(np.meshgrid(points, points, points), -1).reshape(-1, 3)
        w1, w2, w3 = np.meshgrid(weights, weights, weights)
        weights = (w1 * w2 * w3).reshape(-1)
        points = torch.from_numpy(points)
        weights = torch.from_numpy(weights)
    else:
        raise ValueError(f"Unknown element type: {element_type}")
    return weights, points

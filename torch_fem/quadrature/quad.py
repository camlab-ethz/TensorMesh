import  torch 
import numpy as np
from .line import gauss_points as gauss_points_line


def gauss_points(n:int, device:str='cpu'):
    """
        Parameters:
        -----------
            n: int
                the number of quadrature points
                should be a square number like 1, 4, 9...
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [n]
                the quadrature weights
            points: torch.Tensor of shape [n, n_dim]
                the quadrature points
    """
    assert np.sqrt(n) % 1 == 0, "n must be a square number"
    weights, points = gauss_points_line(n)
    weights = torch.outer(weights, weights)
    points  = torch.stack(torch.meshgrid(points[:,0], points[:,0]), dim=-1)
    weights = weights.reshape(-1).to(device)
    points = points.reshape(-1, 2).to(device)
    return weights, points

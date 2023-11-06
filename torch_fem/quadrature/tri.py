"""
(0, 0) --- (1, 0)
|           /
|         /
|       /
|     /
|   /
(0, 1)
"""
import torch 





def gauss_points_n1(device:str='cpu'):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [1]
                the quadrature weights
            points: torch.Tensor of shape [1, 2]
                the quadrature points
    """
    weights = torch.tensor([
        1/2
    ], device=device)
    points = torch.tensor([
        [1/3, 1/3]
    ], device=device)
    return weights, points

def gauss_points_n3(device:str='cpu'):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [3]
                the quadrature weights
            points: torch.Tensor of shape [3, 2]
                the quadrature points
    """
    weights = torch.tensor([
        1/6, 1/6, 1/6
    ], device=device)
    points = torch.tensor([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ], device=device)
    return weights, points

def gauss_points_n4(device:str='cpu'):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [4]
                the quadrature weights
            points: torch.Tensor of shape [4, 2]
                the quadrature points
    """
    weights = torch.tensor([
        -27/96, 25/96, 25/96, 25/96
    ], device=device)
    points = torch.tensor([
        [1/3, 1/3],
        [1/5, 1/5],
        [3/5, 1/5],
        [1/5, 3/5]
    ], device=device)
    return weights, points


def gauss_points(n:int, device:str='cpu'):
    """
        Parameters:
        -----------
            n: int
                the number of quadrature points
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [n]
                the quadrature weights
            points: torch.Tensor of shape [n, 2]
                the quadrature points
    """
    
    find_points = {
        1: gauss_points_n1,
        3: gauss_points_n3,
        4: gauss_points_n4
    }

    assert n in find_points, f"n must be in {find_points.keys()}, but got {n}"
    return find_points[n](device=device)
   

    
    
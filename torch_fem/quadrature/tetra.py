"""
(0,0,0) ---- (1,0,0)
|     \     / |
|      \---\  |
|      /    \ |
(0,1,0) ---- (0,0,1)
    


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
        1/6
    ], device=device)
    points = torch.tensor([
        [1/4, 1/4, 1/4, 1/4]
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
        -1/30, 3/40, 3/40, 3/40
    ], device=device)
    points = torch.tensor([
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/6, 1/6, 1/6],
        [1/6, 1/2, 1/6, 1/6],
        [1/6, 1/6, 1/2, 1/6]
    ], device=device)
    return weights, points

def gauss_points_n5(device:str='cpu'):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [5]
                the quadrature weights
            points: torch.Tensor of shape [5, 2]
                the quadrature points
    """
    weights = torch.tensor([
        -4/30, 9/40, 9/40, 9/40, 9/40
    ], device=device)
    points = torch.tensor([
        [1/4, 1/4, 1/4, 1/4],
        [1/2, 1/6, 1/6, 1/6],
        [1/6, 1/2, 1/6, 1/6],
        [1/6, 1/6, 1/2, 1/6],
        [1/6, 1/6, 1/6, 1/2]
    ], device=device)
    return weights, points

def gauss_points(n:int, device:str='cpu'):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [n]
                the quadrature weights
            points: torch.Tensor of shape [n, 2]
                the quadrature points
    """
    return {
        1: gauss_points_n1,
        4: gauss_points_n4,
        5: gauss_points_n5
    }[n](device)
   
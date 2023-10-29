"""
Line element
    -1 ---- 1


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
            points: torch.Tensor of shape [1, 1]
                the quadrature points
    """
    weights = torch.tensor([
        2
    ], device=device)
    points = torch.tensor([
        [0]
    ], device=device)
    return weights, points

def gauss_points_n2(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [2]
                the quadrature weights
            points: torch.Tensor of shape [2, 1]
                the quadrature points
    """
    weights = torch.tensor([
        1, 1
    ], device=device)
    points = torch.tensor([
        [-1/3**0.5],
        [1/3**0.5]
    ], device=device)
    return weights, points

def gauss_points_n3(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [3]
                the quadrature weights
            points: torch.Tensor of shape [3, 1]
                the quadrature points
    """
    weights = torch.tensor([
        5/9, 8/9, 5/9
    ], device=device)
    points = torch.tensor([
        [-3/5**0.5],
        [0],
        [3/5**0.5]
    ], device=device)
    return weights, points

def gauss_points_n4(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [4]
                the quadrature weights
            points: torch.Tensor of shape [4, 1]
                the quadrature points
    """
    weights = torch.tensor([
        (18+30**0.5)/36, (18+30**0.5)/36, (18-30**0.5)/36, (18-30**0.5)/36
    ], device=device)
    points = torch.tensor([
        [-0.8611363115940526],
        [-0.3399810435848563],
        [0.3399810435848563],
        [0.8611363115940526]
    ], device=device)
    return weights, points

def gauss_points_n5(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [5]
                the quadrature weights
            points: torch.Tensor of shape [5, 1]
                the quadrature points
    """
    weights = torch.tensor([
        (322+13*70**0.5)/900, (322+13*70**0.5)/900, 128/225, (322-13*70**0.5)/900, (322-13*70**0.5)/900
    ], device=device)
    points = torch.tensor([
        [-0.9061798459386640],
        [-0.5384693101056831],
        [0],
        [0.5384693101056831],
        [0.9061798459386640]
    ], device=device)
    return weights, points

def gauss_points_n6(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [6]
                the quadrature weights
            points: torch.Tensor of shape [6, 1]
                the quadrature points
    """
    weights = torch.tensor([
        (0.1713244923791704+0.3607615730481386)/2, (0.1713244923791704+0.3607615730481386)/2, (0.1713244923791704+0.3607615730481386)/2,
        (0.1713244923791704-0.3607615730481386)/2, (0.1713244923791704-0.3607615730481386)/2, (0.1713244923791704-0.3607615730481386)/2
    ], device=device)
    points = torch.tensor([
        [-0.9324695142031521],
        [-0.6612093864662645],
        [-0.2386191860831969],
        [0.2386191860831969],
        [0.6612093864662645],
        [0.9324695142031521]
    ], device=device)
    return weights, points

def gauss_points_n7(device:str="cpu"):
    """
        Parameters:
        -----------
            device: str
                the device to store the points
        Returns:
        --------
            weights: torch.Tensor of shape [7]
                the quadrature weights
            points: torch.Tensor of shape [7, 1]
                the quadrature points
    """
    weights = torch.tensor([
        (0.1294849661688697+0.2797053914892766)/2, (0.1294849661688697+0.2797053914892766)/2, (0.1294849661688697+0.2797053914892766)/2,
        0.3818300505051189, (0.1294849661688697-0.2797053914892766)/2, (0.1294849661688697-0.2797053914892766)/2, (0.1294849661688697-0.2797053914892766)/2
    ], device=device)
    points = torch.tensor([
        [-0.9491079123427585],
        [-0.7415311855993945],
        [-0.4058451513773972],
        [0],
        [0.4058451513773972],
        [0.7415311855993945],
        [0.9491079123427585]
    ], device=device)
    return weights, points

def gauss_points(n:int, device:str="cpu"):
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
            points: torch.Tensor of shape [n, 1]
                the quadrature points
    """
    return [
        gauss_points_n1,
        gauss_points_n2,
        gauss_points_n3,
        gauss_points_n4,
        gauss_points_n5,
        gauss_points_n6,
        gauss_points_n7
    ][n-1](device=device)
    
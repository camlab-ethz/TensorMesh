import torch 
class MultiSinCos:
    """
        The heat velocity is 1
    """
    def __init__(self, mu=None, d=2):
        """
            Parameters:
            -----------
                mu: torch.Tensor (N, d) or (d,)
                    the coefficient of the heat equation
                d: int
                    the dimension of the domain
        """
        if mu is None:
            mu = torch.rand(d)
        else:
            d = mu.shape[-1]
        self.mu = mu
        self.d = d

    def initial_condition(self, points):
        """Generate the heat source function at each point in the domain
            Parameters:
            -----------
                points: torch.Tensor (n_points, 2)
                mu: torch.Tensor (N, d) or (d,)
                    the coefficient of the heat equation

            
            Returns:
            --------
                f: torch.Tensor (n_points) or (N, n_points)
        """
        mu = self.mu
        d  = self.d
        m = torch.arange(1, d+1)
        if len(mu.shape) == 1:
            mu = mu[None, :] # (1, d)
            m  = m[None, :]  # (1, d)
            x,y = points[:, 0][:, None], points[:, 1][:, None] # (n_points, 1)
        else:
            mu = mu[:, None, ...] # (N, 1, d)
            m  = m[None, None, ...] # (1, 1, d)
            x, y = points[:, 0][None, :, None], points[:, 1][None, :, None] # (1, n_points, 1)
        
        u0 = - (mu * torch.sin(torch.pi * m * x) * torch.sin(torch.pi * m * y) / torch.sqrt(m) / d).sum(-1)

        return u0
  
    def solution(self, points, t):
        """Generate the poisson solution function at each point in the domain
            Parameters:
            -----------
                points: torch.Tensor (n_points, 2)
                mu : torch.Tensor (N, d) or (d,)
                    the coefficient of the heat equation
                t: float
                    the time
            
            Returns:
            --------
                ut: torch.Tensor (n_points) or (N, n_points)
        """
        mu = self.mu
        d  = self.d
        m = torch.arange(1, d+1)
        if len(mu.shape) == 1:
            mu = mu[None, ...] # (1, d)
            m  = m[None, ...]  # (1, d)
            x,y = points[:, 0][:, None], points[:, 1][:, None] # (n_points, 1)
        else:
            mu = mu[:, None, ...] # (N, 1, d)
            m  = m[None, None, ...] # (1, 1, d)
            x, y = points[:, 0][None, :, None], points[:, 1][None, :, None] # (1, n_points, 1)
        ut = - (mu * torch.sin(torch.pi * m * x) * torch.sin(torch.pi * m * y) * torch.exp(-2 * m * m * torch.pi * torch.pi * t)/ torch.sqrt(m) / d).sum(-1)
        return ut
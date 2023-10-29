import torch 


class MultiSinCos:
    """
        The initial velocity is 0
    """
    def __init__(self, a=None, K=2, c=1.0, r=0.5 ):
        """
            Parameters:
            -----------
                a: torch.Tensor (N, K, K) or (K, K)
                    the coefficient of the wave equation
                K: int
                    the dimension of the domain
                c: float
                    the wave speed
                r: float
                    the coefficient of the wave equation
        """

        if a is None:
            assert K is not None, "K should be specified if a is None"
            a = torch.zeros((K, K)).uniform_(-1, 1)
        else:
            K = a.shape[-1]
            assert a.shape[-2:] == (K, K), f"the shape of a should be (N, {K}, {K}) or ({K}, {K}), but got {a.shape}"
        self.K = K
        self.a = a
        self.c = c 
        self.r = r

    def initial_condition(self, points):
        """Generate the wave initial function at each point in the domain
            Parameters:
            -----------
                points: torch.Tensor (n_points, 2)
            
            Returns:
            --------
                u0: torch.Tensor (n_points) or (N, n_points)
                v0: torch.Tensor (n_points) or (N, n_points)
        """
        K = self.K
       
        i, j = torch.meshgrid(torch.arange(1,K+1), torch.arange(1,K+1)) # (K, K)
        if len(self.a.shape) == 2:
            a  = self.a[None, :, :] # (1, K, K)
            i,j = i[None, :, :], j[None, :, :] # (1, K, K)
            x,y = points[:, 0][:, None, None], points[:, 1][:, None, None] # (n_points, 1)
        else:
            a  = self.a[:, None, :, :] # (N, 1, K, K)
            i,j = i[None, None, :, :], j[None, None, :, :] # (1, 1, K, K)
            x,y = points[:, 0][None, :, None, None], points[:, 1][None, :, None, None] # (1, n_points, 1, 1)
        
        u0 = torch.pi /K/K * (a * (i*i+j*j)**(-self.r) * torch.sin(torch.pi * i * x) * torch.sin(torch.pi * j * y)).sum((-2,  -1))
    
        return u0

    def solution(self, points, t=0.1):
        """Generate the wave solution function at each point in the domain
            Parameters:
            -----------
                points: torch.Tensor (n_points, 2)
                t: float    
                    the time
            
            Returns:
            --------
                ut: torch.Tensor (n_points) or (N, n_points)
        """
        K = self.K
        i,j  = torch.meshgrid(torch.arange(1, K+1), torch.arange(1,K+1)) # (K, K)
        if len(self.a.shape) == 2:
            a  = self.a[None, :, :] # (1, K, K)
            i,j = i[None, :, :], j[None, :, :] # (1, K, K)
            x,y = points[:, 0][:, None, None], points[:, 1][:, None, None] # (n_points, 1)
        else:
            a  = self.a[:, None, :, :] # (N, 1, K, K)
            i,j = i[None, None, :, :], j[None, None, :, :] # (1, 1, K, K)
            x,y = points[:, 0][None, :, None, None], points[:, 1][None, :, None, None] # (1, n_points, 1, 1)
        u0 = torch.pi /K/K * (a * (i*i+j*j)**(-self.r) * torch.sin(torch.pi * i * x) * torch.sin(torch.pi * j * y) * torch.cos(self.c * torch.pi * t * torch.sqrt(i*i + j*j))).sum((-2,  -1))
        return u0
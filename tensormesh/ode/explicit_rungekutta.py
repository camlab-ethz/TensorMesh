from abc import ABC, abstractmethod
import torch 
from tensormesh.sparse import SparseMatrix



class ExplicitRungeKutta:
    r"""
    .. math::

       \frac{\partial u}{\partial t} = f(t, u)

    Examples
    --------

    .. math::

        \frac{\text{d}u}{\text{d}t} = u 

    .. code-block:: python

        import torch
        from tensormesh.ode import ExplicitRungeKutta

        class MyExplicitRungeKutta(ExplicitRungeKutta):
            def forward(self, t, u):
                return u

        u0 = torch.rand(4)
        dt = 0.1
        ut_my = MyExplicitRungeKutta(a, b).step(0, u0, dt)

    Parameters
    ----------
    a : torch.Tensor
        2D tensor of shape [s, s]
        shoule be lower triangular

        .. math::

            a = \begin{bmatrix}
            0 & \cdots &0& 0 \\
            a_{21} & \cdots &0 & 0 \\
            \vdots & \ddots & \vdots  &\vdots\\
            a_{s1} & \cdots & a_{s{s-1}} &0
            \end{bmatrix}

    b : torch.Tensor
        1D tensor of shape [s], :math:`\sum_{b_i} = 1`
    """
    def __init__(self, a, b):
        assert a.dim() == 2, f"expected a to be 2D tensor, got {a.dim()}"
        assert b.dim() == 1, f"expected b to be 1D tensor, got {b.dim()}"
        assert a.shape[0] == a.shape[1], f"expected a to be square, got {a.shape}"
        assert a.shape[0] == b.shape[0], f"expected a and b to have same shape, got {a.shape} and {b.shape}"
        assert b.sum() == 1, f"expected b to sum to 1, got {b.sum()}"
        assert torch.allclose(a.tril(), a), f"expected a to be lower triangular, got {a}"
        
        self.a = a
        self.b = b
        self.c = a.sum(dim=1)
        self.s = b.shape[0]
        self.__post_init__()

    def __post_init__(self):
        """precompute something after the initialization of tensormesh.ode.builtin.ExplicitRungeKutta
        """
        pass


    def forward(self, t, u):
        r"""right side function :math:`f(t, u)`

        .. math::

           \frac{\partial u}{\partial t} = \text{forward}(t, u)

        default :math:`f(t, u) = u`

        Parameters
        ----------
        t : float
            time
        u : torch.Tensor
            value of shape :math:`[D]` where D is the dimension of the problem
        
        
        Returns
        -------
        torch.Tensor
            value of shape :math:`[D]` where D is the dimension of the problem
            the value of the right side function :math:`f(t, u)`
        """
        return u
    
    def step(self, t0, u0, dt):
        """one step of explicit Runge-Kutta method
        
        .. math::

            \\textbf k_i =\\textbf f(t+c_i\\tau, \\textbf u +\\tau \\sum_{j=1}^s a_{ij}\\textbf k_j)\\quad \\Psi^{t,t+\\tau}\\textbf u = \\textbf u+\\tau\\sum_{i=1}^s b_i \\textbf  k_i

        Parameters
        ----------
        t0 : float
            initial time
        u0 : torch.Tensor
            initial value of shape :math:`[D]` where D is the dimension of the problem
        dt : float
            time step

        Returns
        -------
        torch.Tensor
            value of shape :math:`[D]` where D is the dimension of the problem
            the value of the solution at time :math:`t_0 + \\text{d}t`
        """
        assert u0.dim() == 1, f"expected u0 to be 1D tensor, got {u0.dim()}"
        a = self.a.type(u0.dtype).to(u0.device)
        b = self.b.type(u0.dtype).to(u0.device)
        c = self.c.type(u0.dtype).to(u0.device)
        D = u0.shape[0]
        h = dt
        k = torch.zeros((self.s, D), dtype=u0.dtype, device=u0.device)
        for i in range(self.s):
            ci = c[i]

            if i == 0:
                f = self.forward(t0 + ci * h, u0)
            else:
                f = self.forward(t0 + ci * h, u0 + h * a[i, :i] @ k[:i])
            k[i] += f
        u = u0 + h * b @ k
        return u
    



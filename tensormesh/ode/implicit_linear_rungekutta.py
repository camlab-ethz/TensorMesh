from abc import ABC, abstractmethod
import torch 
from tensormesh.sparse import SparseMatrix



class ImplicitLinearRungeKutta:
    r"""
    
    .. math::
        M(t) \frac{\partial u}{\partial t} = A(t) u + B(t)

    * :math:`M\in \mathbb R^{n\times n}`
    * :math:`A\in \mathbb R^{n\times n}`
    * :math:`B\in \mathbb R^{n}`
    * :math:`u\in \mathbb R^n`

    .. math::

        \begin{bmatrix}
        M_0 - A_0\tau a_{0,0}& - A_0\tau a_{0,1}&\cdots  & - A_{0}\tau a_{0,{n-1}}\\
        -A_1\tau a_{1,0}& M_1-A_1\tau a_{1,1} & \cdots & - A_{1}\tau a_{1,{n-1}}\\
        \vdots & \vdots &\ddots & \vdots \\
        -A_{n-1}\tau a_{{n-1},0} & -A_{n-1}\tau a_{{n-1},1} & \cdots &  M_{n-1} - A_{n-1}\tau a_{n-1,n-1}
        \end{bmatrix}
        \begin{bmatrix}
        \textbf k_0\\ \textbf k_1 \\\vdots \\\textbf k_{n-1}
        \end{bmatrix}= 
        \begin{bmatrix}
        B_0 + A_0 u \\
        B_1 + A_1 u \\
        \vdots\\
        B_{n-1} + A_{n-1} u 
        \end{bmatrix}

    Parameters
    ----------
    a : torch.Tensor
        2D tensor of shape [s, s]
    b : torch.Tensor
        1D tensor of shape [s], :math:`\sum_{b_i} = 1`

    Examples
    --------

    .. math::

        M \frac{\text{d}u}{\text{d}t} = Au + B

    .. code-block:: python

        import torch
        from tensormesh.ode import ImplicitLinearRungeKutta

        class MyImplicitLinearRungeKutta(ImplicitLinearRungeKutta):
            def forward_M(self, t):
                return torch.eye(4)
            
            def forward_A(self, t):
                return torch.eye(4)
            
            def forward_B(self, t):
                return torch.zeros(4)

        u0 = torch.rand(4)
        dt = 0.1
        ut_my = MyImplicitLinearRungeKutta(a, b).step(0, u0, dt)

    """
    def __init__(self, a, b):
        assert a.dim() == 2, f"expected a to be 2D tensor, got {a.dim()}"
        assert b.dim() == 1, f"expected b to be 1D tensor, got {b.dim()}"
        assert a.shape[0] == a.shape[1], f"expected a to be square, got {a.shape}"
        assert a.shape[0] == b.shape[0], f"expected a and b to have same shape, got {a.shape} and {b.shape}"
        assert b.sum() == 1, f"expected b to sum to 1, got {b.sum()}"
       
        self.a = a
        self.b = b
        self.c = a.sum(dim=1)
        self.s = b.shape[0]

        self.__post_init__()

    def __post_init__(self):
        """precompute something after the initialization of tensormesh.ode.builtin.ImplicitLinearRungeKutta
        """
        pass

    def forward_M(self, t):
        r"""left side matrix 

        .. math::

            M \frac{\partial u}{\partial t} = A(t)u + B(t)

        Parameters
        ----------
        t : float
            time
        Returns
        -------
        tensormesh.sparse.SparseMatrix or torch.Tensor or float
            normally, 2D :class:`torch.Tensor` or :meth:`tensormesh.sparse.SparseMatrix` of shape :math:`[D, D]` where :math:`D` is the dimension of the problem;
            if return :obj:`int` or :obj:`float`, the left side matrix :math:`M` is assumed to be a diagonal matrix with the same value
        """
        return 1.0

    def forward_A(self, t):
        r"""compute the linear mapping term :math:`A(t)`

        .. math::
            
            M \frac{\partial u}{\partial t} = A(t)u + B(t)

        Parameters
        ----------
        t : float
            time
        Returns
        -------
        tensormesh.sparse.SparseMatrix or torch.Tensor or float
            2D :class:`torch.Tensor` or :meth:`tensormesh.sparse.SparseMatrix` of shape :math:`[D, D]` where :math:`D` is the dimension of the problem;
            if return :obj:`int` or :obj:`float`, the linear mapping term is assumed to be a diagonal matrix with the same value
        """
        return 1.0

    def forward_B(self, t):
        r"""compute the linear mapping term :math:`B(t)`

        .. math::
            
            M \frac{\partial u}{\partial t} = A(t)u + B(t)

        Parameters
        ----------
        t : float
            time
        Returns
        -------
        torch.Tensor or float
            1D :class:`torch.Tensor` of shape :math:`[D]` where :math:`D` is the dimension of the problem;
            if return :obj:`int` or :obj:`float`, the linear mapping term is assumed to be a vector with the same value
        """
        return 0.0
    
    def pre_solve_lhs(self, K):
        r"""precompute something before solving the linear system,
        for example, do the condensation

        Parameters
        ----------
        K : torch.Tensor or tensormesh.sparse.SparseMatrix
            the left side matrix

        Returns
        -------
        torch.Tensor or tensormesh.sparse.SparseMatrix
            the left side matrix after precompute
        """
        return K
    
    def pre_solve_rhs(self, f):
        r"""precompute something before solving the linear system,
        for example, do the condensation

        Parameters
        ----------
        f : torch.Tensor
            the right hand side vector

        Returns
        -------
        torch.Tensor
            the right hand side vector after precompute
        """
        return f

    def post_solve(self, u):
        r"""postprocess after solving the linear system,
        for example, do the condensation recovery

        Parameters
        ----------
        u: torch.Tensor
            the solution of the linear system

        Returns
        -------
        torch.Tensor
            the solution after postprocess
        """
        return u

    def step(self, t0, u0, dt):
        """

        .. math::

        Parameters
        ----------
        t0 : float
            initial time
        u0 : torch.Tensor
            initial value of shape :math:`[D]` where D is the dimension of the problem
        dt : float
            time step
        """
        assert u0.dim() == 1, f"expected u0 to be 1D tensor, got {u0.dim()}"
        a = self.a.type(u0.dtype).to(u0.device)
        b = self.b.type(u0.dtype).to(u0.device)
        c = self.c.type(u0.dtype).to(u0.device)
        D = u0.shape[0]
        h = dt 
        ts = t0 + dt * c
        lhs = [[None for _ in range(self.s)] for _ in range(self.s)]  
        rhs = [None for _ in range(self.s)]
        use_sparse = None
        for i in range(self.s):
            Ai = self.forward_A(ts[i])
            Bi = self.forward_B(ts[i])
            Mi = self.forward_M(ts[i])
            assert isinstance(Ai, (SparseMatrix, torch.Tensor, int, float)) , f"expected A to be SparseMatrix or torch.Tensor or float, got {type(Ai)}"
            assert isinstance(Bi, (torch.Tensor, int, float)), f"expected B to be torch.Tensor or float, got {type(Bi)}"
            assert isinstance(Mi, (SparseMatrix, torch.Tensor, int, float)) , f"expected M to be SparseMatrix or torch.Tensor or float, got {type(Mi)}"
            
            if i == 0: 
                use_sparse = isinstance(Mi, SparseMatrix) or isinstance(Ai, SparseMatrix)
            else: # check if all the matrices are of the same type
                if use_sparse: # if use_sparse, then all the matrices should be SparseMatrix or float
                    assert not isinstance(Ai, torch.Tensor), f"expected A to be SparseMatrix or None, got {type(Ai)}"
                    assert not isinstance(Mi, torch.Tensor), f"expected M to be SparseMatrix or None, got {type(Mi)}"
                else: # if not use_sparse, then all the matrices should be torch.Tensor or float
                    assert not isinstance(Ai, SparseMatrix), f"expected A to be torch.Tensor or None, got {type(Ai)}"
                    assert not isinstance(Mi, SparseMatrix), f"expected M to be torch.Tensor or None, got {type(Mi)}"
            
            # convert Mi, Ai to torch.Tensor or SparseMatrix
            if isinstance(Mi, (int, float)):
                Mi = float(Mi)
                Mi = (SparseMatrix.eye(D, value=Mi, device=u0.device, dtype=u0.dtype)
                      if use_sparse
                      else torch.eye(D, device=u0.device, dtype=u0.dtype) * Mi)
            if isinstance(Ai, (int, float)):
                Ai = float(Ai)
                Ai = (SparseMatrix.eye(D, value=Ai, device=u0.device, dtype=u0.dtype)
                      if use_sparse
                      else torch.eye(D, device=u0.device, dtype=u0.dtype) * Ai)
            Mi = Mi.type(u0.dtype).to(u0.device)
            Ai = Ai.type(u0.dtype).to(u0.device)
            if isinstance(Bi, torch.Tensor):
                Bi = Bi.type(u0.dtype).to(u0.device)

            # main logic
            for j  in range(self.s):
                lhs[i][j] = -h * a[i,j] * Ai
                if i == j:
                    lhs[i][j] = lhs[i][j] + Mi  
            rhs[i] = Bi + Ai @ u0 


        # pre_solve 
        for i in range(self.s):
            for j in range(self.s):
                lhs[i][j] = self.pre_solve_lhs(lhs[i][j])
            rhs[i] = self.pre_solve_rhs(rhs[i])
        # combine lhs and rhs
        if use_sparse:
            lhs = SparseMatrix.combine(lhs)
        else:
            lhs = torch.cat([torch.cat(lhs[i], 1) for i in range(self.s)], 0)
        rhs = torch.cat(rhs, 0)

        # solve linear system
        if use_sparse:
            k = lhs.solve(rhs)
        else:
            k = torch.linalg.solve(lhs, rhs)

        k = k.reshape(self.s, D)
        u = u0 + h * b @ k

        # post_solve
        u = self.post_solve(u)
    
        return u 
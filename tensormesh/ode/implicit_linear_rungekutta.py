import torch
from tensormesh.sparse import SparseMatrix



class ImplicitLinearRungeKutta:
    r"""Base class for implicit linear Runge-Kutta schemes.

    Integrates the linear (in :math:`u`) system

    .. math::

        M(t) \frac{\partial u}{\partial t} = A(t) u + B(t)

    where :math:`M(t),\,A(t) \in \mathbb{R}^{n \times n}` and
    :math:`B(t) \in \mathbb{R}^{n}`. Each :meth:`step` assembles and
    solves the block system

    .. math::

        \begin{bmatrix}
        M_0 - A_0\tau a_{0,0} & -A_0\tau a_{0,1} & \cdots & -A_0\tau a_{0,s-1} \\
        -A_1\tau a_{1,0} & M_1 - A_1\tau a_{1,1} & \cdots & -A_1\tau a_{1,s-1} \\
        \vdots & \vdots & \ddots & \vdots \\
        -A_{s-1}\tau a_{s-1,0} & -A_{s-1}\tau a_{s-1,1} & \cdots & M_{s-1} - A_{s-1}\tau a_{s-1,s-1}
        \end{bmatrix}
        \begin{bmatrix} k_0 \\ k_1 \\ \vdots \\ k_{s-1} \end{bmatrix}
        =
        \begin{bmatrix} B_0 + A_0 u \\ B_1 + A_1 u \\ \vdots \\ B_{s-1} + A_{s-1} u \end{bmatrix}

    for the stage values :math:`k_i`, then advances
    :math:`u_{n+1} = u_n + \tau \sum_i b_i\,k_i`.

    Parameters
    ----------
    a : torch.Tensor
        2D tensor of shape ``[s, s]``.
    b : torch.Tensor
        1D tensor of shape ``[s]`` with :math:`\sum_i b_i = 1`.

    Examples
    --------
    Solve :math:`\frac{\mathrm{d}u}{\mathrm{d}t} = u` with backward Euler:

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

        a = torch.ones(1, 1)
        b = torch.ones(1)
        u0 = torch.rand(4).double()
        dt = 0.1
        ut = MyImplicitLinearRungeKutta(a, b).step(0, u0, dt)
    """
    def __init__(self, a, b):
        assert a.dim() == 2, f"expected a to be 2D tensor, got {a.dim()}"
        assert b.dim() == 1, f"expected b to be 1D tensor, got {b.dim()}"
        assert a.shape[0] == a.shape[1], f"expected a to be square, got {a.shape}"
        assert a.shape[0] == b.shape[0], f"expected a and b to have same shape, got {a.shape} and {b.shape}"
        assert torch.allclose(b.sum(), torch.tensor(1.0, dtype=b.dtype)), \
            f"expected b to sum to 1, got {b.sum()}"
       
        self.a = a
        self.b = b
        self.c = a.sum(dim=1)
        self.s = b.shape[0]

        self.__post_init__()

    def __post_init__(self):
        """Hook for subclasses to precompute values after ``__init__``.

        Default is a no-op. Subclasses that need to cache derived data
        from ``a`` / ``b`` may override.
        """
        pass

    def forward_M(self, t):
        r"""Mass-like operator :math:`M(t)`.

        .. math::

            M(t) \frac{\partial u}{\partial t} = A(t) u + B(t)

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        :class:`tensormesh.sparse.SparseMatrix` or torch.Tensor or float
            2D operator of shape ``[D, D]``. If returned as ``int`` /
            ``float``, :math:`M` is taken to be that scalar multiple of
            the identity. Default returns ``1.0`` (i.e. :math:`M = I`).
        """
        return 1.0

    def forward_A(self, t):
        r"""Linear operator :math:`A(t)` on the right-hand side.

        .. math::

            M(t) \frac{\partial u}{\partial t} = A(t) u + B(t)

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        :class:`tensormesh.sparse.SparseMatrix` or torch.Tensor or float
            2D operator of shape ``[D, D]``. If returned as ``int`` /
            ``float``, :math:`A` is taken to be that scalar multiple of
            the identity. Default returns ``1.0`` (i.e. :math:`A = I`).
        """
        return 1.0

    def forward_B(self, t):
        r"""Source / forcing term :math:`B(t)`.

        .. math::

            M(t) \frac{\partial u}{\partial t} = A(t) u + B(t)

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        torch.Tensor or float
            1D vector of shape ``[D]``. If ``int`` / ``float``, :math:`B`
            is taken to be that scalar broadcast to all components.
            Default returns ``0.0``.
        """
        return 0.0

    def pre_solve_lhs(self, K):
        r"""Preprocess the assembled block matrix before solving.

        Hook for boundary-condition condensation (or similar). Called
        once per ``[i][j]`` block.

        Parameters
        ----------
        K : torch.Tensor or :class:`tensormesh.sparse.SparseMatrix`
            One block of the left-hand-side matrix.

        Returns
        -------
        torch.Tensor or :class:`tensormesh.sparse.SparseMatrix`
            The (possibly condensed) block. Default returns ``K``
            unchanged.
        """
        return K

    def pre_solve_rhs(self, f):
        r"""Preprocess each stage right-hand side before solving.

        Hook for boundary-condition condensation (or similar). Called
        once per stage.

        Parameters
        ----------
        f : torch.Tensor
            One stage of the right-hand-side vector.

        Returns
        -------
        torch.Tensor
            The (possibly condensed) vector. Default returns ``f``
            unchanged.
        """
        return f

    def recover_stage(self, k_i):
        r"""Lift one stage slope back from the solve space to full DOF.

        Default: identity. Override when :meth:`pre_solve_lhs` /
        :meth:`pre_solve_rhs` *reduce* the system size — e.g. static
        condensation via :class:`~tensormesh.Condenser`. In that case
        each stage value :math:`k_i` is solved in the inner-DOF
        subspace and must be prolonged back to the full DOF layout
        before being combined with :math:`u_0`.

        For Dirichlet boundary conditions the boundary entries of
        :math:`k_i` must be **zero**, regardless of the prescribed value
        of :math:`u` there: a Dirichlet DOF is fixed, so its
        time-derivative is zero. Use
        :meth:`~tensormesh.Condenser.prolong` (or an equivalent
        scatter-with-zero-boundary) here, *not*
        :meth:`~tensormesh.Condenser.recover` (which writes the
        prescribed Dirichlet value into the boundary slots).

        Parameters
        ----------
        k_i : torch.Tensor
            One stage slope from the solve, shape ``[D_solve]``.

        Returns
        -------
        torch.Tensor
            The lifted stage slope of shape ``[D]``. Default returns
            ``k_i`` unchanged.
        """
        return k_i

    def post_solve(self, u):
        r"""Postprocess the combined solution after the linear solve.

        Runs after the stage slopes have already been lifted by
        :meth:`recover_stage` and combined with :math:`u_0`, so ``u``
        is always full-DOF here. Use this hook for things that act on
        the final state — clamping, normalization, projecting onto a
        manifold, etc. For static-condensation BCs, override
        :meth:`recover_stage` instead.

        Parameters
        ----------
        u : torch.Tensor
            Combined solution of shape ``[D]``.

        Returns
        -------
        torch.Tensor
            The (possibly post-processed) solution. Default returns
            ``u`` unchanged.
        """
        return u

    def step(self, t0, u0, dt):
        r"""Advance one implicit-linear Runge-Kutta step from ``t0`` to ``t0 + dt``.

        Builds the block system described in the class docstring, applies
        :meth:`pre_solve_lhs` / :meth:`pre_solve_rhs` to each block,
        solves it (via :meth:`tensormesh.sparse.SparseMatrix.solve` when
        the operators are sparse, otherwise :func:`torch.linalg.solve`),
        applies :meth:`post_solve`, and combines the stage values:

        .. math::

            u_{n+1} = u_0 + \tau \sum_{i=1}^{s} b_i\,k_i

        Parameters
        ----------
        t0 : float
            Initial time.
        u0 : torch.Tensor
            Initial state of shape ``[D]``.
        dt : float
            Time step :math:`\tau`.

        Returns
        -------
        torch.Tensor
            State at time :math:`t_0 + \mathrm{d}t`, same shape as ``u0``.
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
                if use_sparse:
                    assert not (isinstance(Mi, torch.Tensor) or isinstance(Ai, torch.Tensor)), \
                        f"stage 0 mixes SparseMatrix and dense torch.Tensor; pick one (got M={type(Mi)}, A={type(Ai)})"
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
            Mi = Mi.to(device=u0.device, dtype=u0.dtype)
            Ai = Ai.to(device=u0.device, dtype=u0.dtype)
            if isinstance(Bi, torch.Tensor):
                Bi = Bi.to(device=u0.device, dtype=u0.dtype)

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

        # pre_solve_* may reduce the system size (static condensation,
        # row/column elimination, ...) — track the post-reduction stage
        # size so the reshape and per-stage recovery use the correct
        # dimension instead of the original ``D``.
        D_solve = rhs[0].shape[0]

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

        k = k.reshape(self.s, D_solve)

        # Lift each stage back to full DOF. Default ``recover_stage`` is
        # identity, so the no-condensation case is unchanged.
        k = torch.stack(
            [self.recover_stage(k[i]) for i in range(self.s)], dim=0,
        )

        u = u0 + h * b @ k

        # post_solve
        u = self.post_solve(u)

        return u

import torch 
from ..sparse import SparseMatrix

# TODO: add dirichlet_value option for condense_rhs and __call__

class Condenser:
    """Static Condensing Operator for Dirichlet Boundary Condition

    .. math::

        K_{inner} u_{inner} = f_{inner} - K_{ou2in} u_{ou2in}
    
    Parameters
    ----------
    dirichlet_mask: torch.Tensor 
        1D tensor of shape :math:`[n_{\\text{dof}}]`
        the mask of the dirichlet boundary condition
    dirichlet_value: torch.Tensor 
        1D tensor of shape :math:`[n_{\\text{dof}}]` or :math:`[n_{\\text{outer_dof}}]`
        the value of the dirichlet boundary condition

    Attributes
    ----------
    dirichlet_mask: torch.Tensor of shape  :math:`[n_{\\text{dof}}]`
        the mask of the dirichlet boundary condition
    dirichlet_value: torch.Tensor of shape :math:`[n_{\\text{outer_dof}}]`
        the value of the dirichlet boundary condition

    """
    def __init__(self, dirichlet_mask:torch.Tensor, dirichlet_value:torch.Tensor = None):
        assert dirichlet_mask.dtype == torch.bool, "the dtype of dirichlet_mask must be torch.bool"
        assert dirichlet_mask.ndim == 1, "tDirichlet_mask must be 1D tensor"
        assert dirichlet_value is None or dirichlet_value.ndim == 1, "dirichlet_value must be 1D tensor"
        self.dirichlet_mask  = dirichlet_mask
        if dirichlet_value is None:
            self.dirichlet_value = torch.zeros(self.dirichlet_mask.sum())
        elif dirichlet_value.shape[0] == dirichlet_mask.shape[0]:
            self.dirichlet_value = dirichlet_value[self.dirichlet_mask]
        else:
            assert dirichlet_value.shape[0] == dirichlet_mask.sum(), "the shape of dirichlet_value must be [n_dof] or [n_outer_dof]"
            self.dirichlet_value = dirichlet_value
        

        self.inner_row = None
        self.inner_col = None
        self.ou2in_row = None
        self.ou2in_col = None
        self.is_inner_edge = None
        self.is_ou2in_edge = None
        self.layout_hash   = None
        self.K_ou2in       = None

    def _compute_layout(self, matrix:SparseMatrix):
        """
        precompute the condensed components
        Parameters:
        -----------
            matrix: SparseMatrix
                the matrix to be condensed
        Returns:
        --------
            matrix: SparseMatrix
                the condensed matrix
            rhs: torch.Tensor of shape [n_dof]
                the condensed right hand side
        """
        edge_u, edge_v               = matrix.row, matrix.col
        n_dof                        = matrix.shape[0]

        is_inner_dof, is_outer_dof = ~self.dirichlet_mask, self.dirichlet_mask
        
        is_inner_u,    is_inner_v    = is_inner_dof[edge_u], is_inner_dof[edge_v]
        is_outer_u,    is_outer_v    = is_outer_dof[edge_u], is_outer_dof[edge_v]
        is_inner_edge, is_ou2in_edge = is_inner_u & is_inner_v, is_inner_u & is_outer_v
        n_inner_dofs, n_outer_dofs = is_inner_dof.sum(), is_outer_dof.sum()
        local_nids = torch.full((n_dof,), -1, dtype=torch.long)
        local_nids[is_inner_dof] = torch.arange(n_inner_dofs)
        local_nids[is_outer_dof] = torch.arange(n_outer_dofs)

        self.inner_row = local_nids[edge_u[is_inner_edge]]
        self.inner_col = local_nids[edge_v[is_inner_edge]]
        self.ou2in_row = local_nids[edge_u[is_ou2in_edge]]
        self.ou2in_col = local_nids[edge_v[is_ou2in_edge]]
        self.is_inner_edge = is_inner_edge
        self.is_ou2in_edge = is_ou2in_edge
        self.inner_shape = (n_inner_dofs, n_inner_dofs)
        self.ou2in_shape = (n_inner_dofs, n_outer_dofs)
        self.layout_hash = matrix.layout_hash
        self.n_inner_dof = n_inner_dofs
        self.n_outer_dof = n_outer_dofs
        self.n_dof       = n_dof
        self.is_inner_dof = is_inner_dof
        self.is_outer_dof = is_outer_dof

    def __call__(self, matrix:SparseMatrix, rhs:torch.Tensor = None):
        """
        Parameters:
        -----------
        matrix: SparseMatrix
            the matrix to be condensed
        source_value: torch.Tensor 
            1D tensor of shape :math:`[n_{\\text{dof}}]`
            the right hand side of the linear system
        Returns:
        --------
        matrix: SparseMatrix
            the condensed matrix
        rhs: torch.Tensor 
            1D tensor of shape :math:`[n_{\\text{dof}}]`
            the condensed right hand side
        """
        if rhs is None:
            rhs = torch.zeros(matrix.shape[0])

        if self.inner_row is None:
            self._compute_layout(matrix)

        assert matrix.shape[0] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.shape[1] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.has_same_layout(self.layout_hash), "the layout of the matrix is changed, please recompute the condensed matrix"
        assert rhs.ndim == 1, "rhs must be 1D tensor"
        assert rhs.shape[0] == self.n_dof, f"the shape of rhs must be [{self.n_dof}], but got {rhs.shape}"
        
        K_inner = SparseMatrix(
            matrix.edata[self.is_inner_edge], self.inner_row, self.inner_col, self.inner_shape, 
        )
        K_ou2in = SparseMatrix(
            matrix.edata[self.is_ou2in_edge], self.ou2in_row, self.ou2in_col, self.ou2in_shape, 
        )

        self.K_ou2in = K_ou2in

        self.dirichlet_value = self.dirichlet_value.type(K_inner.edata.dtype).to(K_inner.edata.device)
        rhs  = rhs.type(K_inner.edata.dtype).to(K_inner.edata.device)
       
        return K_inner, rhs[self.is_inner_dof] - K_ou2in @ self.dirichlet_value

    def condense_rhs(self, rhs):
        """only condense the right hand side
        
        .. math::

            f_{inner} - K_{ou2in} u_{ou2in}
        
        Parameters
        ----------
        rhs: torch.Tensor
            1D tensor of shape :math:`[n_{\\text{dof}}]`
            the right hand side of the linear system

        Returns
        -------
        torch.Tensor
            1D tensor of shape :math:`[n_{\\text{inner_dof}}]`
            the condensed right hand side
        """
        assert self.K_ou2in is not None, f"please call __call__ first"

        self.dirichlet_value = self.dirichlet_value.type(rhs.dtype).to(rhs.device)
        rhs = rhs.type(self.K_ou2in.edata.dtype).to(self.K_ou2in.edata.device)

        return rhs[self.is_inner_dof] - self.K_ou2in @ self.dirichlet_value
       
    def recover(self, u:torch.Tensor):
        """recovert the solution

        Parameters
        ----------
        u: torch.Tensor 
            1D tensor of shape :math:`[n_{\\text{inner_dof}}]`
            the solution of the condensed linear system

        Returns
        -------
        torch.Tensor
            1D tensor of shape :math:`[n_{\\text{dof}}]`
            the recovered solution of the linear system
        """
        assert u.ndim == 1, "u must be 1D tensor"
        assert u.shape[0] == self.n_inner_dof, f"the shape of u must be [{self.n_inner_dof}], but got {u.shape}"

        u_full = torch.zeros(self.n_dof, dtype=u.dtype, device=u.device)
        u_full[self.is_inner_dof] += u 
        u_full[self.is_outer_dof] += self.dirichlet_value

        return u_full
    

Condenser.__autodoc__ = ["__call__", "condense_rhs", "recover"]
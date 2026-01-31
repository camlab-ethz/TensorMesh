"""
torch-sla based sparse solver for TensorMesh.

Uses torch-sla for differentiable sparse linear algebra with 
better iterative solver support.
"""

import torch
from torch.autograd import Function
import warnings

# Check if torch_sla is available
try:
    import torch_sla
    is_torch_sla_available = True
except ImportError:
    is_torch_sla_available = False
    warnings.warn("torch-sla not available. Install with: pip install torch-sla")


class SparseSolveTorchSLA(Function):
    """
    PyTorch autograd function using torch-sla for sparse solve.
    
    Provides differentiable sparse linear solve with multiple backend options.
    """
    
    @staticmethod
    def forward(ctx, edata, row, col, shape, b, x0=None, tol=1e-5, max_iter=10000,
                backend='scipy', method='cg', preconditioner='jacobi', is_spd=True):
        """
        Forward pass: solve Ax = b using torch-sla.
        
        Parameters
        ----------
        edata : torch.Tensor
            Matrix values (requires_grad=True for differentiation)
        row, col : torch.Tensor
            Row and column indices
        shape : tuple
            Matrix shape (m, n)
        b : torch.Tensor
            Right-hand side vector
        x0 : torch.Tensor, optional
            Initial guess (not currently used by torch-sla)
        tol : float
            Tolerance for iterative solvers
        max_iter : int
            Max iterations
        backend : str
            'scipy', 'pytorch', 'eigen', 'auto'
        method : str
            'cg', 'bicgstab', 'superlu', etc.
        preconditioner : str
            'jacobi', 'ilu', 'none'
        is_spd : bool
            Hint that matrix is symmetric positive definite
            
        Returns
        -------
        u : torch.Tensor
            Solution vector
        """
        if not is_torch_sla_available:
            raise RuntimeError(
                "torch-sla is required.\n"
                "Install with: pip install torch-sla"
            )
        
        # torch_sla.spsolve expects int64 indices
        row_long = row.long()
        col_long = col.long()
        
        # Call torch_sla.spsolve
        u = torch_sla.spsolve(
            val=edata,
            row=row_long,
            col=col_long,
            shape=shape,
            b=b,
            backend=backend,
            method=method,
            atol=tol,
            maxiter=max_iter,
            is_spd=is_spd,
            preconditioner=preconditioner,
        )
        
        # Save for backward
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.backend = backend
        ctx.method = method
        ctx.preconditioner = preconditioner
        ctx.is_spd = is_spd
        
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients.
        
        For Ax = b:
        - grad_b = A^{-T} @ grad_output
        - grad_A = -grad_b @ x^T (sparse)
        """
        edata, row, col, u = ctx.saved_tensors
        
        # For symmetric matrix, A^T = A
        # Solve A^T @ grad_b = grad_output
        row_long = row.long()
        col_long = col.long()
        
        # Transpose indices for A^T
        b_grad = torch_sla.spsolve(
            val=edata,
            row=col_long,  # Transposed
            col=row_long,  # Transposed
            shape=(ctx.A_shape[1], ctx.A_shape[0]),
            b=grad_output,
            backend=ctx.backend,
            method=ctx.method,
            atol=ctx.tol,
            maxiter=ctx.max_iter,
            is_spd=ctx.is_spd,
            preconditioner=ctx.preconditioner,
        )
        
        # Gradient for matrix entries: grad_A[i,j] = -grad_b[i] * u[j]
        edata_grad = -b_grad[row] * u[col]
        
        return edata_grad, None, None, None, b_grad, None, None, None, None, None, None, None


def sparse_solve_torch_sla(edata, row, col, shape, b, x0=None, tol=1e-5, max_iter=10000,
                           backend='scipy', method='cg', preconditioner='jacobi', is_spd=True):
    """
    Convenience function for torch-sla sparse solve with autograd support.
    
    Parameters
    ----------
    edata : torch.Tensor
        Matrix values
    row, col : torch.Tensor
        Row and column indices
    shape : tuple
        Matrix shape
    b : torch.Tensor
        Right-hand side
    x0 : torch.Tensor, optional
        Initial guess (not used)
    tol : float
        Tolerance (default 1e-5)
    max_iter : int
        Max iterations (default 10000)
    backend : str
        'scipy', 'pytorch', 'eigen', 'auto'
    method : str
        'cg', 'bicgstab', 'superlu', etc.
    preconditioner : str
        'jacobi', 'ilu', 'none'
    is_spd : bool
        Hint that matrix is SPD
        
    Returns
    -------
    x : torch.Tensor
        Solution vector
    """
    return SparseSolveTorchSLA.apply(
        edata, row, col, shape, b, x0, tol, max_iter,
        backend, method, preconditioner, is_spd
    )







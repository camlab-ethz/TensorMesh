"""
AMG-preconditioned CG solver for TensorMesh.

Uses pyamg for algebraic multigrid preconditioning, which is essential
for efficiently solving 3D elasticity problems.
"""

import torch
from torch.autograd import Function
import numpy as np
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import minres as scipy_minres
import warnings

# Check if pyamg is available
try:
    import pyamg
    is_pyamg_available = True
except ImportError:
    is_pyamg_available = False
    warnings.warn("pyamg not available. Install with: pip install pyamg")


def cg_amg(indices, values, m, n, b, x0=None, atol=1e-5, max_iter=10000, 
           B=None, method='cg', cached_ml=None):
    """
    Solves Ax = b using CG/MINRES with AMG preconditioning.
    
    This is the recommended solver for 3D elasticity problems.
    
    Parameters
    ----------
    indices : torch.Tensor
        2D tensor of shape [2, nnz], row and column indices
    values : torch.Tensor
        1D tensor of shape [nnz], matrix values
    m, n : int
        Matrix shape
    b : torch.Tensor
        1D tensor of shape [n], right-hand side vector
    x0 : torch.Tensor, optional
        Initial guess for the solution
    atol : float
        Absolute tolerance for convergence
    max_iter : int
        Maximum number of iterations
    B : torch.Tensor, optional
        Null space (rigid body modes) for AMG, shape [n, n_modes]
        For 3D elasticity, typically 6 modes (3 translations + 3 rotations)
    method : str
        'cg' or 'minres'
    cached_ml : pyamg multilevel object, optional
        Cached AMG hierarchy for reuse
        
    Returns
    -------
    x : torch.Tensor
        Solution vector
    ml : pyamg multilevel object
        AMG hierarchy for caching
    """
    if not is_pyamg_available:
        raise RuntimeError(
            "pyamg is required for AMG preconditioning.\n"
            "Install with: pip install pyamg"
        )
    
    assert m == n, f"Matrix must be square, got {m}x{n}"
    
    # Convert to scipy sparse matrix
    device = values.device
    dtype = values.dtype
    
    indices_np = indices.cpu().numpy()
    values_np = values.cpu().detach().numpy()
    b_np = b.cpu().detach().numpy()
    
    A_scipy = scipy_coo_matrix(
        (values_np, (indices_np[0], indices_np[1])), 
        shape=(m, n)
    ).tocsr()
    
    # Initial guess
    if x0 is not None:
        x0_np = x0.cpu().detach().numpy()
    else:
        x0_np = None
    
    # Null space for AMG (rigid body modes)
    if B is not None:
        B_np = B.cpu().detach().numpy()
    else:
        B_np = None
    
    # Build or reuse AMG hierarchy
    if cached_ml is not None:
        ml = cached_ml
    else:
        # Build AMG with smoothed aggregation
        # For elasticity, Jacobi smoother works well
        ml = pyamg.smoothed_aggregation_solver(
            A_scipy, 
            B=B_np,
            smooth='jacobi',
            strength='symmetric',
            max_coarse=500,
        )
    
    # Get preconditioner
    M = ml.aspreconditioner()
    
    # Solve with iterative method
    if method == 'cg':
        x_np, exit_code = scipy_cg(
            A_scipy, b_np, 
            M=M, 
            rtol=atol,  # scipy uses rtol
            x0=x0_np,
            maxiter=max_iter
        )
    elif method == 'minres':
        x_np, exit_code = scipy_minres(
            A_scipy, b_np,
            M=M,
            rtol=atol,
            x0=x0_np,
            maxiter=max_iter
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cg' or 'minres'")
    
    if exit_code != 0:
        residual = np.linalg.norm(A_scipy @ x_np - b_np)
        warnings.warn(
            f"{method.upper()} did not converge (exit={exit_code}), "
            f"residual={residual:.2e}"
        )
    
    # Convert back to torch
    x = torch.tensor(x_np, dtype=dtype, device=device)
    
    return x, ml


class SparseSolveAMG(Function):
    """
    PyTorch autograd function for AMG-preconditioned sparse solve.
    
    Supports automatic differentiation through the linear solve.
    """
    
    @staticmethod
    def forward(ctx, edata, row, col, shape, b, x0=None, tol=1e-5, max_iter=10000,
                B=None, method='cg', cached_ml=None):
        """
        Forward pass: solve Ax = b with AMG preconditioning.
        
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
            Initial guess
        tol : float
            Tolerance
        max_iter : int
            Max iterations
        B : torch.Tensor, optional
            Null space for AMG
        method : str
            'cg' or 'minres'
        cached_ml : object, optional
            Cached AMG hierarchy
            
        Returns
        -------
        u : torch.Tensor
            Solution vector
        """
        indices = torch.stack([row, col], dim=0)
        u, ml = cg_amg(
            indices, edata, shape[0], shape[1], b,
            x0=x0, atol=tol, max_iter=max_iter,
            B=B, method=method, cached_ml=cached_ml
        )
        
        # Save for backward
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.B = B
        ctx.method = method
        ctx.ml = ml  # Cache AMG hierarchy for backward pass
        
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
        
        # Solve A^T @ grad_b = grad_output
        # For symmetric A, A^T = A, so we can reuse the AMG hierarchy
        indices_T = torch.stack([col, row], dim=0)
        
        b_grad, _ = cg_amg(
            indices_T, edata, ctx.A_shape[1], ctx.A_shape[0], grad_output,
            atol=ctx.tol, max_iter=ctx.max_iter,
            B=ctx.B, method=ctx.method, cached_ml=ctx.ml
        )
        
        # Gradient for matrix entries: grad_A[i,j] = -grad_b[i] * u[j]
        edata_grad = -b_grad[row] * u[col]
        
        return edata_grad, None, None, None, b_grad, None, None, None, None, None, None


def sparse_solve_amg(edata, row, col, shape, b, x0=None, tol=1e-5, max_iter=10000,
                     B=None, method='cg', cached_ml=None):
    """
    Convenience function for AMG-preconditioned sparse solve with autograd support.
    
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
        Initial guess
    tol : float
        Tolerance (default 1e-5)
    max_iter : int
        Max iterations (default 10000)
    B : torch.Tensor, optional
        Null space for AMG
    method : str
        'cg' or 'minres' (default 'cg')
    cached_ml : object, optional
        Cached AMG hierarchy
        
    Returns
    -------
    x : torch.Tensor
        Solution vector
    """
    return SparseSolveAMG.apply(
        edata, row, col, shape, b, x0, tol, max_iter, B, method, cached_ml
    )







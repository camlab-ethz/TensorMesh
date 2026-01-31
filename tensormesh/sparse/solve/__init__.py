"""
Sparse linear system solver for TensorMesh.

The default solver is torch-sla, which provides:
- Multiple backends: scipy, pytorch, eigen
- Multiple methods: cg, bicgstab, minres, gmres, superlu
- Preconditioners: jacobi, ilu, none
- Automatic differentiation support
"""

import torch
import warnings

# Check torch-sla availability
try:
    import torch_sla
    is_torch_sla_available = True
except ImportError:
    is_torch_sla_available = False

# Fallback imports for when torch-sla is not available
from ..utils import is_petsc_available, is_cupy_available

# Export for backward compatibility
from .torch_solve import is_cpp_backend_available


def spsolve(edata, row, col, shape, b, 
            backend='auto', method='cg', preconditioner='jacobi',
            tol=1e-5, max_iter=10000, x0=None, is_spd=True,
            verbose=False):
    """Solve sparse linear system Ax = b.
    
    This is the main entry point for solving sparse linear systems in TensorMesh.
    By default, it uses torch-sla which provides differentiable sparse solvers.
    
    Parameters
    ----------
    edata : torch.Tensor
        1D tensor of matrix values, shape [nnz]
    row : torch.Tensor
        1D tensor of row indices, shape [nnz]
    col : torch.Tensor
        1D tensor of column indices, shape [nnz]
    shape : tuple
        Matrix shape (m, n)
    b : torch.Tensor
        Right-hand side vector, shape [n] or [n, batch]
    backend : str, optional
        Solver backend. Default 'auto' (uses torch-sla with best backend).
        Options: 'auto', 'scipy', 'pytorch', 'eigen'
        Legacy options (fallback): 'petsc', 'cupy'
    method : str, optional
        Solver method. Default 'cg' for SPD systems.
        Options: 'cg', 'bicgstab', 'minres', 'gmres', 'lgmres', 'superlu'
    preconditioner : str, optional
        Preconditioner. Default 'jacobi'.
        Options: 'jacobi', 'ilu', 'none'
    tol : float, optional
        Convergence tolerance. Default 1e-5.
    max_iter : int, optional
        Maximum iterations. Default 10000.
    x0 : torch.Tensor, optional
        Initial guess. Default None.
    is_spd : bool, optional
        Hint that matrix is symmetric positive definite. Default True.
    verbose : bool, optional
        Print solver info. Default False.
    
    Returns
    -------
    torch.Tensor
        Solution vector x, same shape as b.
    
    Examples
    --------
    >>> A = SparseMatrix(...)
    >>> x = A.solve(b)  # Uses default torch-sla CG solver
    >>> x = A.solve(b, method='superlu')  # Direct solver
    >>> x = A.solve(b, backend='scipy', method='bicgstab')  # Scipy BiCGSTAB
    
    Notes
    -----
    torch-sla supports automatic differentiation through the solve operation,
    enabling gradient-based optimization of systems involving sparse solves.
    """
    
    # Validate inputs
    assert edata.device == row.device == col.device == b.device, \
        f"All inputs must be on same device, got {edata.device}, {row.device}, {col.device}, {b.device}"
    
    if edata.dtype != torch.float64:
        warnings.warn("float64 recommended for better accuracy in spsolve")
    
    # Handle batched RHS
    is_batched = len(b.shape) == 2
    
    # Use torch-sla if available (preferred)
    if is_torch_sla_available:
        return _solve_torch_sla(
            edata, row, col, shape, b,
            backend=backend, method=method, preconditioner=preconditioner,
            tol=tol, max_iter=max_iter, x0=x0, is_spd=is_spd,
            is_batched=is_batched, verbose=verbose
        )
    
    # Fallback to legacy solvers
    warnings.warn(
        "torch-sla not available, using fallback solver. "
        "Install torch-sla for better performance: pip install torch-sla"
    )
    return _solve_fallback(
        edata, row, col, shape, b,
        backend=backend, tol=tol, max_iter=max_iter, x0=x0,
        is_batched=is_batched, verbose=verbose
    )


def _solve_torch_sla(edata, row, col, shape, b,
                     backend, method, preconditioner,
                     tol, max_iter, x0, is_spd,
                     is_batched, verbose):
    """Solve using torch-sla."""
    from .torch_sla_solve import SparseSolveTorchSLA
    
    # Map 'auto' to appropriate torch-sla backend
    if backend == 'auto':
        if edata.device.type == 'cuda':
            backend = 'pytorch'
        else:
            backend = 'scipy'
    
    # For batched solve, use superlu (direct solver)
    if is_batched and method not in ['superlu', 'lu']:
        if verbose:
            print(f"Using SuperLU for batched solve (batch_size={b.shape[1]})")
        method = 'superlu'
    
    if verbose:
        print(f"Solving with torch-sla: backend={backend}, method={method}, preconditioner={preconditioner}")
    
    return SparseSolveTorchSLA.apply(
        edata, row, col, shape, b,
        x0, tol, max_iter,
        backend, method, preconditioner, is_spd
    )


def _solve_fallback(edata, row, col, shape, b,
                    backend, tol, max_iter, x0,
                    is_batched, verbose):
    """Fallback solver when torch-sla is not available."""
    
    # Import fallback solvers
    from .scipy_solve import SparseSolveScipy, SparseLUSolveScipy
    from .torch_solve import SparseSolveTorch
    
    device = edata.device
    
    if device.type == 'cuda':
        # CUDA fallback
        if is_cupy_available:
            from .cupy_solve import SparseSolveCupy, SparseLUSolveCupy
            if is_batched:
                return SparseLUSolveCupy.apply(edata, row, col, shape, b)
            else:
                return SparseSolveCupy.apply(edata, row, col, shape, b)
        else:
            # Use torch sparse solver
            return SparseSolveTorch.apply(edata, row, col, shape, b, x0, tol, max_iter)
    else:
        # CPU fallback
        if is_batched:
            return SparseLUSolveScipy.apply(edata, row, col, shape, b)
        else:
            if backend == 'petsc' and is_petsc_available:
                from .petsc_solve import SparseSolvePETSc
                return SparseSolvePETSc.apply(edata, row, col, shape, b)
            else:
                return SparseSolveScipy.apply(edata, row, col, shape, b)

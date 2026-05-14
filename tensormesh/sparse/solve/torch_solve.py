import os
import math
import torch
from torch.autograd import Function
import warnings
from ..utils import shapeT
try:
    from ...cpp.spsolve import spsolve_cpp
    is_cpp_backend_available = True
except ImportError:
    is_cpp_backend_available = False

def coo_diagonal(A, at_least=1):
    """
    Returns the diagonal of a CSR matrix.
    The matrix should be symmetric.
    """
    assert A.shape[0] == A.shape[1], f"Matrix is not square. Shape is {A.shape}"
    N = A.shape[0]
    edges = A.indices()
    value = A.values()
    mask  = edges[0] == edges[1]
    cand_value = value[mask]
    cand_index = edges[0][mask]
    # cand_value = cand_value[torch.argsort(cand_index)]
    # diag_mask  = torch.bincount(cand_index, minlength=N).bool()
    diag_value = torch.full(size=(N,), fill_value=at_least, dtype=cand_value.dtype, device=cand_value.device)
    # diag_value = torch.fill(shape=(N,), fill_value=at_least).type(cand_value.dtype).to(cand_value.device)
    # diag_value[diag_mask] = cand_value
    diag_value[cand_index] = cand_value
    return diag_value

def jacobi_precond(A, x = None):
    if x is None:
        return 1.0/coo_diagonal(A, at_least=1.0)
    else:
        return x * (1.0/coo_diagonal(A, at_least=1.0))

def identity_precond(A, x = None):
    if x is None:
        return torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
    else:
        return x

def cg_py(indices, values, m, n, b, x0=None, atol=1e-5, max_iter=10000):
    """
    Solves Ax = b using the Conjugate Gradient method.

    https://en.wikipedia.org/wiki/Conjugate_gradient_method
    
    Parameters
    ----------
    A : torch.sparse_csr_matrix
        2D Sparse tensor of shape [N, N], The matrix A in Ax = b.
    b : torch.Tensor
        1D tensor of shape [N] The right-hand side vector.
    x0 : torch.Tensor, optional
        1D tensor of shape [N] Initial guess for the solution. The default is None.
    tol : float, optional
        Tolerance for convergence. The default is 1e-5.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.
    """
    A = torch.sparse_coo_tensor(indices, values, (m, n), is_coalesced =True)

    if x0 is None:
        x0 = torch.zeros_like(b)
    
    A = A.to_sparse_csr()

    x0 = x0.view(-1)
    b  = b.view(-1)

    r = b - A @ x0
    p = r.clone()
    x = x0
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / torch.dot(p,Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r,r)

        if torch.sqrt(rs_new) < atol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
       
    if torch.norm(A @ x - b) > atol:
        warnings.warn(f"cg did not converge after {i} iterations. with residual {torch.norm(A @ x - b)}")

    return x.view(-1)

def bicgstab_py(indices, values, m, n, b, x0=None, atol=1e-5, max_iter=10000):
    """
    Solves Ax = b using the Preconditioned Bi-Conjugate Gradient Stabilized method.
    Uses Jacobi (diagonal) preconditioning.

    Args:
        indices, values: COO representation of A
        m, n: Shape of A
        b: The right-hand side vector.
        x0: Initial guess for the solution.
        atol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        The approximate solution vector.
    """
    assert m == n, f"Matrix is not square. Shape is {m}x{n}"
    assert b.shape[0] == m, f"Shape mismatch. A is {m}x{n}, b is {b.shape}"
    assert b.dim() == 1, f"b should be a 1D tensor. b is {b.dim()}D"
    assert b.dtype == values.dtype, f"b.dtype {b.dtype} does not match values.dtype {values.dtype}"
    
    # Construct A for matrix multiplication
    A_coo = torch.sparse_coo_tensor(indices, values, (m, n), is_coalesced=True)
    A = A_coo.to_sparse_csr()

    # Construct Jacobi Preconditioner (Inverse Diagonal)
    # Extract diagonal from indices/values
    # Assuming indices are [2, NNZ]
    row_idx, col_idx = indices
    diag_mask = (row_idx == col_idx)
    diag_indices = row_idx[diag_mask]
    diag_values = values[diag_mask]
    
    # Initialize diagonal with ones
    M_diag = torch.ones(m, device=values.device, dtype=values.dtype)
    M_diag.index_put_((diag_indices,), diag_values)
    
    # Avoid division by zero
    M_diag = torch.where(M_diag.abs() < 1e-12, torch.tensor(1.0, device=values.device, dtype=values.dtype), M_diag)
    M_inv = 1.0 / M_diag

    def apply_precond(v):
        return v * M_inv



    if x0 is None:
        x0 = torch.zeros_like(b)
    
    # Initial residual
    r = b - A @ x0
    r0_hat = r.clone()
    
    rho = alpha = omega = 1.0
    v = p = torch.zeros_like(b)
    
    rho = torch.dot(r0_hat, r)
    
    # Reuse p for the first iteration logic to match standard algo structure
    p = r.clone()

    for i in range(max_iter):
        if torch.norm(r) < atol:
            break
            
        p_hat = apply_precond(p)
        v = A @ p_hat
        
        denom = torch.dot(r0_hat, v)
        if denom.abs() < 1e-12:
            break # Breakdown
            
        alpha = rho / denom
        s = r - alpha * v
        
        if torch.norm(s) < atol:
            x0 = x0 + alpha * p_hat
            break
            
        s_hat = apply_precond(s)
        t = A @ s_hat
        
        t_norm2 = torch.dot(t, t)
        if t_norm2.abs() < 1e-12:
            omega = 0.0
        else:
            omega = torch.dot(t, s) / t_norm2
            
        x0 = x0 + alpha * p_hat + omega * s_hat
        r = s - omega * t
        
        if torch.norm(r) < atol:
            break
            
        rho_new = torch.dot(r0_hat, r)
        if omega == 0:
            break
            
        beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new
        p = r + beta * (p - omega * v)

    if torch.norm(A @ x0 - b) > math.sqrt(atol):
        warnings.warn(f"bicgstab did not converge after {max_iter} iterations. with residual {torch.norm(A @ x0 - b)}")

    return x0.view(-1)

if is_cpp_backend_available and (not "TORCH_FEM_USE_CPP" in os.environ or os.environ["TORCH_FEM_USE_CPP"] == "true"):
    def bicgstab_cpp(indices, values, m, n, b, x0=None, atol=1e-5, max_iter=10000):
        indices = indices.long()
        # TODO: pass x0 to cpp solver if supported
        return spsolve_cpp.bicgstab(indices, values, m, n, b, atol, max_iter)
    def cg_cpp(indices, values, m, n, b, x0=None, atol=1e-5, max_iter=10000):
        indices = indices.long()
        # TODO: pass x0 to cpp solver if supported
        return spsolve_cpp.cg(indices, values, m, n, b, atol, max_iter)
    lse_solver = bicgstab_cpp
else:
    lse_solver = bicgstab_py

class SparseSolveTorch(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b, x0=None, tol=1e-5, max_iter=10000):
        u = lse_solver(torch.stack([row, col],0), edata, shape[0], shape[1], b, x0=x0, atol=tol, max_iter=max_iter)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        ctx.tol = tol
        ctx.max_iter = max_iter
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        shape_T = shapeT(ctx.A_shape)
        # Gradient for b: solve A^T * grad_b = grad_output
        # We can also use 'u' (if symmetric) or previous solution as guess? 
        # But backward solve is a new system. x0=None is safe.
        b_grad = lse_solver(torch.stack([col, row],0), edata, shape_T[0], shape_T[1], grad_output, atol=ctx.tol, max_iter=ctx.max_iter)
        edata_grad      = - b_grad[row] * u[col]

        return edata_grad, None, None, None, b_grad, None, None, None
    


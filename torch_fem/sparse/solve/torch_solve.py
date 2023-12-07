
from numpy import diag
import torch 
from torch.autograd import Function
import warnings
from ..utils import tensor2cupy, cupy2tensor, shapeT

import torch


def csr_diagonal(A):
    """
    Returns the diagonal of a CSR matrix.
    The matrix should be symmetric.
    """
    assert A.shape[0] == A.shape[1], f"Matrix is not square. Shape is {A.shape}"
    N = A.shape[0]
    A = A.to_sparse_coo()
    edges = A.indices()
    value = A.values()
    mask  = edges[0] == edges[1]
    cand_value = value[mask]
    cand_index = edges[0][mask]
    cand_value = cand_value[torch.argsort(cand_index)]
    diag_mask  = torch.bincount(cand_index, minlength=N).bool()
    diag_value = torch.zeros(N, dtype=cand_value.dtype, device=cand_value.device)
    diag_value[diag_mask] = cand_value
    return diag_value


def cg(A, b, x0=None, tol=1e-5, max_iter=5000):
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
    if x0 is None:
        # x0 = 1/csr_diagonal(A).view(-1, 1)
        x0 = torch.zeros_like(b)
        # x0 = A.diagonal().clone()

    x0 = x0.view(-1, 1)
    b  = b.view(-1, 1)

    r = b - A @ x0
    p = r.clone()
    x = x0
    rs_old = r.T @ r

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r.T @ r

        if torch.norm(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
       
    if torch.norm(A @ x - b) > tol:
        warnings.warn(f"cg did not converge after {max_iter} iterations. with residual {torch.norm(A @ x - b)}")

    return x.view(-1)

def bicgstab(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solves Ax = b using the Bi-Conjugate Gradient Stabilized method.

    Args:
        A: The matrix A in Ax = b.
        b: The right-hand side vector.
        x0: Initial guess for the solution.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        The approximate solution vector.
    """
    if x0 is None:
        x0 = torch.zeros_like(b)

    x0 = x0.view(-1, 1)
    b  = b.view(-1, 1)

    r0 = b - A @ x0
    r0_hat = r0.clone()
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)
    rho = alpha = omega = 1
    x = x0.clone()

    for i in range(max_iter):
        rho_new = r0_hat.T @ r0
        beta = (rho_new / rho) * (alpha / omega)
        rho = rho_new

        p = r0 + beta * (p - omega * v)
        v = A @ p
        alpha = rho / (r0_hat.T @ v)
        h = x + alpha * p

        if torch.norm(A @ h - b) < tol:
            return h.view(-1)

        s = r0 - alpha * v
        t = A @ s 
        omega = (t.T @ s) / (t.T @ t)
        x = h + omega * s

        r0 = s - omega * t

        if torch.norm(r0) < tol:
            break

    if torch.norm(A @ x - b) > tol:
        warnings.warn(f"bicgstab did not converge after {max_iter} iterations. with residual {torch.norm(A.mv(x) - b)}")
   
    return x.view(-1)

lse_solver = bicgstab

class SparseSolveTorch(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b):
        A = torch.sparse_coo_tensor(torch.stack([row, col]), edata, shape)
        u = lse_solver(A, b)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        A_T           = torch.sparse_coo_tensor(torch.stack([col, row]), edata, shapeT(ctx.A_shape))
        b_grad        = lse_solver(A_T, grad_output)
        edata_grad      = - b_grad[row] * u[col]

        return edata_grad, None, None, None, b_grad
    


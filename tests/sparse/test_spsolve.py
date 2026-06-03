import sys 
sys.path.append("../..")
import torch 
import scipy.sparse 
import numpy as np
import pytest

from tensormesh.sparse import SparseMatrix

"""
    CPU Test
"""

def test_spsolve_forward_cpu(n_times=10):
    
    for _ in range(n_times):
        while True:
            A      = SparseMatrix.random(16, 16, 0.3).double()
            if A.to_dense().det() != 0:
                break
        b      = torch.rand(16).double()
        u      = A.solve(b)
        assert torch.allclose(A @ u - b, torch.zeros_like(b))

def test_splusolve_forward_cpu(n_times = 10):
    for _ in range(n_times):
        while True:
            A      = SparseMatrix.random(16, 16, 0.3).double()
            if A.to_dense().det() != 0:
                break
        b      = torch.rand(16, 8).double()
        u      = A.solve(b)

        assert torch.allclose(A @ u - b, torch.zeros_like(b), rtol=1e-4, atol=1e-4), f"{A @ u - b}, min:{(A @ u - b).min()}, max:{(A @ u - b).max()}"

def test_spsolve_backward_cpu(n_times=10):
    for _ in range(n_times):
        while True:
            A     = SparseMatrix.random(16, 16, 0.3).double().requires_grad_()
            if A.to_dense().det() != 0:
                break
        b     = torch.rand(16).double().requires_grad_()
        u     = A.solve(b)
        u.sum().backward()

        A_dense = A.to_dense().detach().clone()
        b_dense = b.detach().clone()
        A_dense.requires_grad_()
        b_dense.requires_grad_()
        u_dense = torch.linalg.solve(A_dense, b_dense)

        u_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad*A.layout_mask)
        assert torch.allclose(b.grad, b_dense.grad)

def test_splusolve_backward_cpu(n_times = 10):
    for _ in range(n_times):
        while True:
            A     = SparseMatrix.random(16, 16, 0.3).double().requires_grad_()
            if A.to_dense().det() != 0:
                break
        b     = torch.rand(16, 8).double().requires_grad_()
        u     = A.solve(b)
        u.sum().backward()

        A_dense = A.to_dense().detach().clone()
        b_dense = b.detach().clone()
        A_dense.requires_grad_()
        b_dense.requires_grad_()
        u_dense = torch.linalg.solve(A_dense, b_dense)

        u_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad*A.layout_mask)
        assert torch.allclose(b.grad, b_dense.grad)

"""
    GPU Test

The CUDA backend (cuDSS via nvmath) only solves SPD / symmetric systems
correctly; on a general non-symmetric matrix it returns a wrong result, while
scipy's pivoted LU on CPU does not. FEM operators are SPD in practice, so these
GPU checks build an SPD matrix the same way torch-sla's own tests do
(``A = R Rᵀ + n·I``). A single multi-column RHS exercises the multi-RHS path,
so we keep one forward and one backward case rather than separate spsolve /
splusolve variants (both just call ``A.solve``).
"""

def _spd_sparse_cuda(n=16, density=0.3):
    """Sparse SPD matrix on CUDA, mirroring torch-sla's ``create_spd_sparse``."""
    A = torch.rand(n, n, dtype=torch.float64, device="cuda")
    A = A @ A.T + n * torch.eye(n, dtype=torch.float64, device="cuda")
    A[A.abs() < (1 - density)] = 0
    return SparseMatrix.from_dense(A)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_spsolve_forward_gpu(n_times=10):
    for _ in range(n_times):
        A      = _spd_sparse_cuda(16)
        b      = torch.rand(16, 8, device="cuda").double()
        u      = A.solve(b)

        assert torch.allclose(A @ u - b, torch.zeros_like(b), rtol=1e-5, atol=1e-5), f"{A @ u - b}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_spsolve_backward_gpu(n_times=10):
    for _ in range(n_times):
        A     = _spd_sparse_cuda(16).requires_grad_()
        b     = torch.rand(16, 8, device="cuda").double().requires_grad_()
        u     = A.solve(b)
        u.sum().backward()

        A_dense = A.to_dense().detach().clone()
        b_dense = b.detach().clone()
        A_dense.requires_grad_()
        b_dense.requires_grad_()
        u_dense = torch.linalg.solve(A_dense, b_dense)

        u_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad*A.layout_mask)
        assert torch.allclose(b.grad, b_dense.grad)
import sys 
sys.path.append("../..")
import torch 
import scipy.sparse 
import numpy as np

from torch_fem.sparse import SparseMatrix

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
"""

def test_spsolve_forward_gpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 16, 0.3, device="cuda").double()
        if A.to_dense().det() == 0:
            continue
        b      = torch.rand(16, device="cuda").double()
        u      = A.solve(b)

        assert torch.allclose(A @ u - b, torch.zeros_like(b), rtol=1e-5, atol=1e-5),f"{A @ u - b}"

def test_splusolve_forward_gpu(n_times = 10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 16, 0.3, device="cuda").double()
        if A.to_dense().det() == 0:
            continue
        b      = torch.rand(16, 8, device="cuda").double()
        u      = A.solve(b)

        assert torch.allclose(A @ u - b, torch.zeros_like(b), rtol=1e-5, atol=1e-5), f"{A @ u - b}"

def test_spsolve_backward_gpu(n_times = 10):
    for _ in range(n_times):
        A     = SparseMatrix.random(16, 16, 0.3, device="cuda").double().requires_grad_()
        if A.to_dense().det() == 0:
            continue
        b     = torch.rand(16, device="cuda").double().requires_grad_()
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

def test_splusolve_backward_gpu(n_times = 10):
    for _ in range(n_times):
        A     = SparseMatrix.random(16, 16, 0.3, device="cuda").double().requires_grad_()
        if A.to_dense().det() == 0:
            continue
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
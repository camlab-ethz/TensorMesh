import sys 
import torch 
sys.path.append("../..")

from torch_fem.sparse import SparseMatrix

"""
    CPU Test
"""

def test_spmm_forward_cpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3)
        B      = torch.rand(10, 8)
        C      = A @ B # 20 x 8
        
        assert torch.allclose(C, A.to_dense() @ B, rtol=1e-5, atol=1e-5)

def test_spmv_forward_cpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3)
        B      = torch.rand(10)
        C      = A @ B # 20
        
        assert torch.allclose(C, A.to_dense() @ B)
        
def test_spmm_backward_cpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3).requires_grad_()
        B      = torch.rand(10, 8).requires_grad_()
        C      = A @ B # 20 x 8
        C.sum().backward()

        A_dense = A.to_dense().detach().clone()
        B_dense = B.detach().clone()
        A_dense.requires_grad_()
        B_dense.requires_grad_()
        C_dense = A_dense @ B_dense
        C_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad*A.layout_mask)
        assert torch.allclose(B.grad.to_dense(), B_dense.grad)

def test_spmv_backward_cpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3).requires_grad_()
        B      = torch.rand(10).requires_grad_()
        C      = A @ B # 20
        C.sum().backward()
        
        A_dense = A.to_dense().detach().clone()
        B_dense = B.detach().clone()
        A_dense.requires_grad_()
        B_dense.requires_grad_()
        C_dense = A_dense @ B_dense
        C_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad*A.layout_mask)
        assert torch.allclose(B.grad.to_dense(), B_dense.grad)

"""
    GPU Test
"""

def test_spmm_forward_gpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3).cuda()
        B      = torch.rand(10, 8).cuda()
        C      = A @ B # 20 x 8
       
        assert torch.allclose(C, A.to_dense() @ B)

def test_spmv_forward_gpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3).cuda()
        B      = torch.rand(10).cuda()
        C      = A @ B # 20
        
        assert torch.allclose(C, A.to_dense() @ B)

def test_spmm_backward_gpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(16, 10, 0.3).cuda().requires_grad_()
        B      = torch.rand(10, 8).cuda().requires_grad_()
        C      = A @ B # 20 x 8
        C.sum().backward()
        
        A_dense = A.to_dense().detach().clone()
        B_dense = B.detach().clone()
        A_dense.requires_grad_()
        B_dense.requires_grad_()
        C_dense = A_dense @ B_dense
        C_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad * A.layout_mask)
        assert torch.allclose(B.grad.to_dense(), B_dense.grad)

def test_spmv_backward_gpu(n_times=10):
    for _ in range(n_times):
        A      = SparseMatrix.random(20, 10, 0.3).cuda().requires_grad_()
        B      = torch.rand(10).cuda().requires_grad_()
        C      = A @ B # 20
        C.sum().backward()
        
        A_dense = A.to_dense().detach().clone()
        B_dense = B.detach().clone()
        A_dense.requires_grad_()
        B_dense.requires_grad_()
        C_dense = A_dense @ B_dense
        C_dense.sum().backward()

        assert torch.allclose(A.grad.to_dense(), A_dense.grad *  A.layout_mask)
        assert torch.allclose(B.grad.to_dense(), B_dense.grad)
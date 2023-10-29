import sys 
import torch 
sys.path.append("../..")

from torch_fem.sparse import SparseMatrix


def test_add_forward_cpu(n_times=10):
    for _ in range(n_times):
        layout = SparseMatrix.random_layout(10, 10)
        A      = SparseMatrix.random_from_layout(layout)
        B      = SparseMatrix.random_from_layout(layout)
        C      = A + B 
        
        assert torch.allclose(C.to_dense(), A.to_dense() + B.to_dense())
        

def test_add_backward_cpu(n_times=10):
    for _ in range(n_times):
        layout = SparseMatrix.random_layout(10, 10)
        A      = SparseMatrix.random_from_layout(layout).requires_grad_()
        B      = SparseMatrix.random_from_layout(layout).requires_grad_()
        C      = A + B 
        loss   = C.sum()
        loss.backward()
        
        assert torch.allclose(A.edata.grad, torch.ones_like(A.edata.grad))
        assert torch.allclose(B.edata.grad, torch.ones_like(B.edata.grad))

def test_add_forward_gpu(n_times=10):
    for _ in range(n_times):
        layout = SparseMatrix.random_layout(10, 10)
        A      = SparseMatrix.random_from_layout(layout).cuda()
        B      = SparseMatrix.random_from_layout(layout).cuda()
        C      = A + B 
        
        assert torch.allclose(C.to_dense(), A.to_dense() + B.to_dense())

def test_add_backward_gpu(n_times=10):
    for _ in range(n_times):
        layout = SparseMatrix.random_layout(10, 10)
        A      = SparseMatrix.random_from_layout(layout).cuda().requires_grad_()
        B      = SparseMatrix.random_from_layout(layout).cuda().requires_grad_()
        C      = A + B 
        C.sum().backward()
        
        assert torch.allclose(A.edata.grad.to_dense(), torch.ones_like(A.edata.grad.to_dense()))
        assert torch.allclose(B.edata.grad.to_dense(), torch.ones_like(B.edata.grad.to_dense()))

from typing import Any
import torch 
from torch.autograd import Function
import scipy.sparse.linalg
import warnings
from ..utils import tensor2cupy, cupy2tensor, shapeT


class SparseSolveScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        # Ensure inputs are on CPU and converted to numpy
        edata_np = edata.detach().cpu().numpy()
        row_np = row.detach().cpu().numpy()
        col_np = col.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        
        A_scipy = scipy.sparse.coo_matrix((edata_np, (row_np, col_np)), shape=shape).tocsr()
        u = scipy.sparse.linalg.spsolve(A_scipy, b_np)
        
        # Output as tensor on original device
        u = torch.tensor(u, dtype=b.dtype, device=b.device)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        edata_np = edata.detach().cpu().numpy()
        row_np   = row.detach().cpu().numpy()
        col_np   = col.detach().cpu().numpy()
        A_T             = scipy.sparse.coo_matrix((edata_np, (col_np, row_np)), shape=shapeT(ctx.A_shape)).tocsr()
        b_grad          = scipy.sparse.linalg.spsolve(A_T, grad_output.detach().cpu().numpy())
        b_grad          = torch.tensor(b_grad, dtype=u.dtype, device=u.device)

        edata_grad      = - b_grad[row] * u[col]
        return edata_grad, None, None, None, b_grad


class SparseLUSolveScipy(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        edata_np = edata.detach().cpu().numpy()
        row_np   = row.detach().cpu().numpy()
        col_np   = col.detach().cpu().numpy()
        A_scipy = scipy.sparse.coo_matrix((edata_np, (row_np, col_np)), shape=shape).tocsc()
        lu = scipy.sparse.linalg.splu(A_scipy)
        u = lu.solve(b.detach().cpu().numpy())
        u = torch.tensor(u, dtype=b.dtype, device=b.device)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u

    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        edata_np = edata.detach().cpu().numpy()
        row_np   = row.detach().cpu().numpy()
        col_np   = col.detach().cpu().numpy()
        A_T             = scipy.sparse.coo_matrix((edata_np, (col_np, row_np)), shape=[ctx.A_shape[1], ctx.A_shape[0]]).tocsc()
        b_grad          = scipy.sparse.linalg.splu(A_T).solve(grad_output.detach().cpu().numpy())
        b_grad          = torch.tensor(b_grad, dtype=u.dtype, device=u.device)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad
    


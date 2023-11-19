from typing import Any
import torch 
from torch.autograd import Function
import cupy as cp 
import scipy.sparse
import cupyx.scipy.sparse.linalg
import warnings
from .utils import tensor2cupy, cupy2tensor, shapeT


class SparseSolveCPU(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsr()
        u = scipy.sparse.linalg.spsolve(A_scipy, b)
        u = torch.tensor(u, dtype=b.dtype)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        A_T             = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=shapeT(ctx.A_shape)).tocsr()
        b_grad          = scipy.sparse.linalg.spsolve(A_T, grad_output.numpy())
        b_grad          = torch.tensor(b_grad, dtype=u.dtype)

        edata_grad      = - b_grad[row] * u[col] 
        return edata_grad, None, None, None, b_grad
    

class SparseSolveCUDA(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_cupy = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(row), tensor2cupy(col))), 
            shape = shape).tocsr()
        u_cupy = cupyx.scipy.sparse.linalg.spsolve(A_cupy, tensor2cupy(b))
        u = cupy2tensor(u_cupy)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        
        A_T             = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(col), tensor2cupy(row))), 
            shape=shapeT(ctx.A_shape)).tocsr()
        b_grad          = cupyx.scipy.sparse.linalg.spsolve(A_T, tensor2cupy(grad_output))
        b_grad          = cupy2tensor(b_grad)

        edata_grad      = - b_grad[row] * u[col]

        return edata_grad, None, None, None, b_grad
    

class SparseLUSolveCPU(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsc()
        lu = scipy.sparse.linalg.splu(A_scipy)
        u = lu.solve(b.numpy())
        u = torch.tensor(u, dtype=b.dtype)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape

        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        
        A_T             = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]]).tocsc()
        b_grad          = scipy.sparse.linalg.splu(A_T).solve(grad_output.numpy())
        b_grad          = torch.tensor(b_grad)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad
    

class SparseLUSolveCUDA(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_cupy = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(row), tensor2cupy(col))), 
            shape = shape).tocsc()
        lu_cupy = cupyx.scipy.sparse.linalg.splu(A_cupy)
        u_cupy = lu_cupy.solve(tensor2cupy(b))
        u = cupy2tensor(u_cupy)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        
        A_T             = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(col), tensor2cupy(row))), 
            shape=shapeT(ctx.A_shape)).tocsc()
        b_grad          = cupyx.scipy.sparse.linalg.splu(A_T).solve(tensor2cupy(grad_output))
        b_grad          = cupy2tensor(b_grad)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad


def spsolve(edata, row, col, shape, b, verbose=True):
    """solve the sparse linear system Ax = b

    if the b of shape [n_node, n_batch], then a superLU will be used 
    else it will use spsolve 

    Parameters
    ----------
    edata: torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data of the sparse matrix A
    row: torch.Tensor 
        1D tensor of shape [n_edge]
        the row index of the sparse matrix A
    col: torch.Tensor 
        1D tensor of shape [n_edge]
        the col index of the sparse matrix A
    shape: Tuple[int, int]
        the shape of the sparse matrix A
    b: torch.Tensor 
        1D or 2D tensor of shape [n_node] or [n_node,batch]
        the right hand side vector b
    Returns
    -------
    torch.Tensor 
        1D or 2D tensor  of shape [n_node] or [n_node,batch]
        the solution of the linear system
    """
    if edata.dtype != torch.float64:
        warnings.warn("Accuracy insufficient, float64 is recommended for better accuracy in spsolve")
    assert len(b.shape) <= 2, f"b should be of shape [n_node] or [n_node,batch], but got {b.shape}"
    if len(b.shape) == 2:
        if verbose:
            print(f"Use SuperLU to solve the batched linear system")
        if edata.device.type == "cpu":
            return SparseLUSolveCPU.apply(edata, row, col, shape, b)
        elif edata.device.type == "cuda":
            return SparseLUSolveCUDA.apply(edata, row, col, shape, b)
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")
    else:
        if edata.device.type == "cpu":
            return SparseSolveCPU.apply(edata, row, col, shape, b)
        elif edata.device.type == "cuda":
            return SparseSolveCUDA.apply(edata, row, col, shape, b)
        else:
            raise NotImplementedError("Only CPU and CUDA are supported")

import torch
from torch.autograd import Function
import cupy as cp
import cupyx.scipy.sparse 
import scipy.sparse
from .utils import tensor2cupy, cupy2tensor, shapeT


class SparseMMCPU(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        """
        Parameters
        ---------
        edata: torch.Tensor 
            1D tensor of shape [n_edge]
            the edge data
        row  : torch.Tensor 
            1D tensor of shape [n_edge]
            the row indices
        col  : torch.Tensor 
            1D tensor of shape [n_edge]
            the column indices
        shape: Tuple[int, int]
            bi-tuple [M, N]
            the shape of the sparse matrix
        B    : torch.Tensor 
            2D tensor of shape [N, K]
            the dense matrix
        Returns:
        --------
        torch.Tensor 
            2D tensor of shape [M, K]
            the output feature matrix
        """
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Parameters
        -----------
        grad_outputs : torch.Tensor 
            torch.Tensor of shape [M, K]
            the gradient of the output feature matrix
        Returns
        -------
        edata_grad : torch.Tensor 
            1D tensor of shape [n_edge]
            the gradient of the edge data
        row_grad  : None
        col_grad  : None
        shape_grad: None
        B_grad    : torch.Tensor of shape [N, K]
            the gradient of the feature matrix
        """

        edata, row, col, B = ctx.saved_tensors
        edata_grad  = (grad_outputs[row] * B[col]).sum(dim=1)
       
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B

class SparseMVCPU(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        ctx.save_for_backward(edata, row, col, B)
        A_scipy = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape)
        B_numpy = B.numpy()
        C_scipy = A_scipy.dot(B_numpy)
        C = torch.tensor(C_scipy, dtype=B.dtype)
        ctx.A_shape = shape
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):

        edata, row, col, B = ctx.saved_tensors  
        edata_grad  = grad_outputs[row] * B[col]
        A_T    = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=[ctx.A_shape[1], ctx.A_shape[0]])
        grad_B = A_T.dot(grad_outputs.numpy())
        grad_B = torch.tensor(grad_B, dtype=B.dtype)
        return edata_grad, None, None, None, grad_B

class SparseMMCUDA(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        A_cupy = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(row), tensor2cupy(col))), 
            shape = shape)
        C_cupy = A_cupy.dot(tensor2cupy(B))
        C = cupy2tensor(C_cupy)
        ctx.save_for_backward(edata, row, col, B)
        ctx.A_shape = shape
        return C

    @staticmethod
    def backward(ctx, grad_outputs):
        edata, row, col, B = ctx.saved_tensors
        edata_grad  = (grad_outputs[row]* B[col]).sum(dim=1)
        A_T    = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(col),tensor2cupy(row))), 
            shape=shapeT(ctx.A_shape))
        grad_B = A_T.dot(tensor2cupy(grad_outputs))
        grad_B = cupy2tensor(grad_B)
        return edata_grad, None, None, None, grad_B
       
class SparseMVCUDA(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, B):
        A_cupy = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(row), tensor2cupy(col))
            ), shape = shape)
        C_cupy = A_cupy.dot(tensor2cupy(B))
        C = cupy2tensor(C_cupy)
        ctx.save_for_backward(edata, row, col, B)
        ctx.A_shape = shape
        return C

    @staticmethod
    def backward(ctx, grad_outputs):
        edata, row, col, B = ctx.saved_tensors  
        edata_grad  = grad_outputs[row]  * B[col]
        A_T    = cupyx.scipy.sparse.coo_matrix((
            tensor2cupy(edata), (tensor2cupy(col),tensor2cupy(row))), 
            shape=shapeT(ctx.A_shape))
        grad_B = A_T.dot(tensor2cupy(grad_outputs))
        grad_B = cupy2tensor(grad_B)
        return edata_grad, None, None, None, grad_B


def spmv(edata, row, col, shape, B):
    """
    Parameters
    ----------
    edata : torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int,  int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        1D tensor of shape [n_node]
        the dense vector
    Returns
    -------
    torch.Tensor 
        1D tensor of shape [n_node]
        the output vector
    """
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    assert B.dim() == 1
    if edata.device.type == 'cpu':
        return SparseMVCPU.apply(edata, row, col, shape, B)
    elif edata.device.type == 'cuda':
        return SparseMVCUDA.apply(edata, row, col, shape, B)
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")

def spmm(edata, row, col, shape, B):
    """
    Parameters
    ----------
    edata: torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the dense matrix/vector
    Returns:
    --------
    torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the output feature matrix
    """
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    if B.dim() == 1:
        return spmv(edata, row, col, shape, B)
    assert B.dim() == 2
    if edata.device.type == 'cpu':
        return SparseMMCPU.apply(edata, row, col, shape, B)
    elif edata.device.type == 'cuda':
        return SparseMMCUDA.apply(edata, row, col, shape, B)
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")

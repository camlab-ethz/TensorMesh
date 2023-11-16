import torch 
import torch.nn as nn
import scipy.sparse
import hashlib
import inspect
from .mm import spmm 
from .solve import spsolve


class SparseMatrix(nn.Module):
    def __init__(self, edata,  row, col, shape):
        super().__init__()
        assert edata.shape[0] == row.shape[0] == col.shape[0], f"the first dim of edata, row, col should be the same, but got {edata.shape[0]}, {row.shape[0]}, {col.shape[0]}"
        assert edata.device == row.device == col.device, f"edata, row, col should be on the same device, but got {edata.device}, {row.device}, {col.device}"
        self.register_buffer("edata", edata)
        self.register_buffer("row", row)
        self.register_buffer("col", col)
        self.shape = shape

        self.layout_hash = hashlib.sha256(row.cpu().numpy().tobytes() + col.cpu().numpy().tobytes()).hexdigest()

    @property
    def edges(self):
        return torch.stack([self.row, self.col], dim=0)

    def elementwise_operation(self, func, obj):
        if  isinstance(obj, SparseMatrix):
            assert self.shape == obj.shape, f"the shape of the two sparse matrices should be the same, but got {self.shape}, {obj.shape}"
            assert self.has_same_layout(obj), f"the row indices of the two sparse matrices should be the same, but got {self.row}, {obj.row}"
            return SparseMatrix(func(self.edata, obj.edata), self.row, self.col, self.shape)
        elif isinstance(obj, torch.Tensor):
            assert obj.shape == self.shape, f"the shape of the sparse matrix and the tensor should be the same, but got {self.shape}, {obj.shape}"
            return SparseMatrix(func(self.edata, obj), self.row, self.col, self.shape)
        elif isinstance(obj, (int,float)):
            return SparseMatrix(func(self.edata, obj), self.row, self.col, self.shape)
        else:
            raise Exception(f"unsupported type {type(obj)} for SparseMatrix.elementwise_operation {inspect.getsource(func)}")

    def __add__(self, obj):
        return self.elementwise_operation(lambda x,y: x+y, obj)

    def __mul__(self, obj):
        return self.elementwise_operation(lambda x,y: x * y, obj)

    def __rmul__(self, obj):
        return self.elementwise_operation(lambda a,b : torch.mul(b, a), obj)

    def __div__(self, obj):
        return self.elementwise_operation(torch.div, obj)
    
    def __rtruediv__(self, obj):
        return self.elementwise_operation(lambda a,b : torch.div(b, a), obj)
        
    def __pow__(self, obj):
        return self.elementwise_operation(torch.pow, obj)

    def __matmul__(self, x):
        return spmm(self.edata, self.row, self.col, self.shape, x)

    def solve(self, x):
        return spsolve(self.edata, self.row, self.col, self.shape, x)

    def requires_grad_(self, requires_grad: bool = True):
        self.edata.requires_grad_(requires_grad)
        return self
  
    def transpose(self):
        return SparseMatrix(self.edata, self.col, self.row, self.shape[::-1])

    def sqrt(self):
        return SparseMatrix(self.edata.sqrt(), self.row, self.col, self.shape)
    
    def reciprocal(self):
        return SparseMatrix(self.edata.reciprocal(), self.row, self.col, self.shape)

    def sum(self, axis=None):
        if axis is None:
            return self.edata.sum()
        elif axis == 0:
            return self.T @ torch.ones(self.shape[0], device=self.edata.device)
        elif axis == 1:
            return self @ torch.ones(self.shape[1], device=self.edata.device)
        else:
            raise Exception(f"unsupported axis {axis} for SparseMatrix.sum")
        
    def __str__(self):
        return (
            f"SparseMatrix(\n"
            f"    edata: {self.edata}\n"
            f"    row  : {self.row}\n"
            f"    col  : {self.col}\n"
            f"    shape: {self.shape}\n"
            f"{self.edata.grad_fn if self.edata.grad_fn is not None else ''}\n" 
            f")"
        )

    def __repr__(self):
        return str(self)
    
    @property
    def T(self):
        return self.transpose()
    
    @property
    def requires_grad(self):
        return self.edata.requires_grad

    @property
    def dtype(self):
        return self.edata.dtype
    
    @property
    def device(self):
        return self.edata.device

    @property
    def grad(self):
        if self.edata.grad is None:
            return None
        else:
            return SparseMatrix(self.edata.grad, self.row, self.col, self.shape)

    @property
    def grad_fn(self):
        if self.edata.grad_fn is None:
            return None
        else:
            return self.edata.grad_fn

    @property
    def nnz(self):
        return self.edata.shape[0]

    @property
    def layout_mask(self):
        mask = torch.zeros(self.shape, device=self.edata.device)
        mask[self.row, self.col] = 1
        return mask

    def double(self):
        self.edata = self.edata.to(torch.double)
        return self
    
    def float(self):
        self.edata = self.edata.to(torch.float)
        return self
    
    def cuda(self, device=None):
        self.edata.cuda(device=device)
        return self
    
    def cpu(self):
        self.edata.cpu()
        return self

    def type(self, dtype):
        self.edata.to_(dtype)
        return self

    def to(self, arg):
        self.edata.to_(arg)
        return self

    def detach(self):
        return SparseMatrix(self.edata.detach(), self.row, self.col, self.shape).requires_grad_(False)

    def to_scipy_coo(self):
        return scipy.sparse.coo_matrix((
            self.edata.detach().cpu().numpy(),
            (
                self.row.detach().cpu().numpy(),
                self.col.detach().cpu().numpy()
            )), shape=self.shape)
    
    def to_sparse_coo(self):
        return torch.sparse_coo_tensor(
            torch.stack([self.row, self.col]),
            self.edata,
            self.shape
        )

    def to_dense(self):
        matrix = torch.zeros(self.shape, device=self.edata.device, dtype=self.edata.dtype)
        matrix[self.row, self.col] += self.edata
        return matrix

    def has_same_layout(self, obj):
        assert isinstance(obj, (SparseMatrix,str)), f"matrix must be SparseMatrix or str, but got {type(obj)}"
        if  isinstance(obj, str):
            return self.layout_hash == obj
        else:
            return self.layout_hash == obj.layout_hash

    @staticmethod
    def from_scipy_coo(matrix, device="cpu", dtype=torch.float):
        edata = torch.from_numpy(matrix.data).to(device).type(dtype)
        row   = torch.from_numpy(matrix.row).to(device)
        col   = torch.from_numpy(matrix.col).to(device)
        shape = matrix.shape
        return SparseMatrix(edata, row, col, shape)

    @staticmethod
    def from_sparse_coo(matrix):
        edata = matrix.values()
        row   = matrix.indices()[0]
        col   = matrix.indices()[1]
        shape = matrix.shape
        return SparseMatrix(edata, row, col, shape)

    @staticmethod
    def from_block_coo(edata, row, col, shape):
        """
            Parameters:
            -----------
                edata: torch.Tensor of shape [n_edges, block_size, block_size]
                    the block data
                row: torch.Tensor of shape [n_edges]
                    the row indices
                col: torch.Tensor of shape [n_edges]
                    the column indices
                shape: tuple of int
                    the shape of the sparse matrix of the first two dim
        """
        n_edges = edata.shape[0]
        block_size = edata.shape[1]
        assert row.shape == col.shape == (n_edges,), f"the shape of row and col should be {n_edges}, but got {row.shape}, {col.shape}"
        assert edata.shape == (n_edges, block_size, block_size), f"the shape of edata should be {n_edges, block_size, block_size}, but got {edata.shape}"

        edata = edata.flatten()
        row   = row[:, None].repeat(1, block_size * block_size)
        col   = col[:, None].repeat(1, block_size * block_size)

        i,j   = torch.meshgrid(torch.arange(block_size), torch.arange(block_size)) 
       
        row   = row * block_size+ i.flatten()
        col   = col * block_size+ j.flatten()
        
        shape = (shape[0] * block_size, shape[1] * block_size)
        row   = row.flatten()
        col   = col.flatten()

        return SparseMatrix(edata, row, col, shape)

    @staticmethod
    def random(m,n, density=0.1, device="cpu", dtype=torch.float):
        matrix = scipy.sparse.random(m, n, density, format="coo")
        return SparseMatrix.from_scipy_coo(matrix, device=device, dtype=dtype)
    
    @staticmethod
    def random_layout(m, n, density=0.1, device="cpu"):
        matrix = scipy.sparse.random(m, n, density, format="coo")
        row    = torch.from_numpy(matrix.row).to(device)
        col    = torch.from_numpy(matrix.col).to(device)
        shape  = matrix.shape
        return row, col, shape
    
    @staticmethod
    def random_from_layout(layout, device="cpu", dtype=torch.float):
        row, col, shape = layout
        edata = torch.rand(row.shape[0], device=device, dtype=dtype)
        return SparseMatrix(edata, row.to(device), col.to(device), shape)
    

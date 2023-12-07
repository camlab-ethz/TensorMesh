import numpy as np
import torch 
import torch.nn as nn
import scipy.sparse
import hashlib
import inspect
from .mm import spmm 
from .solve import spsolve


class SparseMatrix(nn.Module):
    """coo format sparse matrix

    Parameters
    ----------
    edata: torch.Tensor 
        1D float tensor of shape :math:`[|\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the edge data
    row: torch.Tensor
        1D int tensor of shape :math:`[|\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the row indices
    col: torch.Tensor
        1D int tensor of shape :math:`[|\\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the column indices
    shape: Tuple[int, int]
        the shape of the sparse matrix of the first two dim, e.g. (3, 4)

    Attributes
    ----------
    edata: torch.Tensor 
        1D float tensor of shape :math:`[|\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the edge data
    row: torch.Tensor
        1D int tensor of shape :math:`[|\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the row indices
    col: torch.Tensor
        1D int tensor of shape :math:`[|\\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        the column indices
    shape: Tuple[int, int]
        the shape of the sparse matrix of the first two dim, e.g. (3, 4)
    hash_layout: str
        it will be used to check if two sparse matrices have the same layout,
        the hash of the layout of the sparse matrix
    
    """
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
        """
        Returns
        -------
        torch.Tensor
            the edge indices of shape :math:`[2, |\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
        """
        return torch.stack([self.row, self.col], dim=0)

    def elementwise_operation(self, func, obj):
        """Elementwise operation with another sparse matrix or a tensor or a scalar
        If the object is a sparse matrix, the :attr:`edges` of the two sparse matrices should be the same

        Parameters
        ----------

        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            the elementwise operation
        obj: SparseMatrix or torch.Tensor or int or float
            the object to be elementwise operated with

        Returns
        -------
        result: SparseMatrix
            the result of the elementwise operation
        """
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
        """
        Parameters
        ----------
        x: torch.Tensor
            the dense tensor of shape [b] or [b,h] to be multiplied with the sparse matrix

        Returns
        -------
        torch.Tensor
            the result of the multiplication of shape [a] or [a,h]
        """
        return spmm(self.edata, self.row, self.col, self.shape, x)

    def __neg__(self):
        return SparseMatrix(-self.edata, self.row, self.col, self.shape)

    def solve(self, x, backend=None):
        """
        Parameters
        ----------
        x: torch.Tensor
            the dense tensor of shape [a] or [a,h] to be solved with the sparse matrix
        backend: str, optional
            the backend to solve the sparse matrix, can be :obj:`None`, :obj:`"torch"`, :obj:`"scipy"` 
            or :obj:`"torch_scipy"`, default :obj:`None`
        
        Returns
        -------
        torch.Tensor
            the result of the solution of shape [b] or [b,h]
        """
        assert x.shape[0] == self.shape[1], f"the first dim of x should be the same as the second dim of the sparse matrix, but got {x.shape[0]}, {self.shape[1]}"
        return spsolve(self.edata, self.row, self.col, self.shape, x, backend=backend)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Parameters
        ----------
        requires_grad: bool, optional
            whether the sparse matrix requires gradient, default :obj:`True`
        
        Returns
        -------
        SparseMatrix    
            the sparse matrix with requires_grad set to requires_grad
        """
        self.edata.requires_grad_(requires_grad)
        return self
  
    def transpose(self):
        """tranpose the sparse matrix

        .. math::
            
            A = A^\\top
        
        """
        return SparseMatrix(self.edata, self.col, self.row, self.shape[::-1])

    def sqrt(self):
        """element-wise square root
        
        .. math::

            A_{ij} = \\sqrt{A_{ij}}
        
        """
        return SparseMatrix(self.edata.sqrt(), self.row, self.col, self.shape)
    
    def reciprocal(self):
        """element-wise reciprocal
        
        .. math::
            
            A_{ij} = \\frac{1}{A_{ij}}

        """
        return SparseMatrix(self.edata.reciprocal(), self.row, self.col, self.shape)

    def degree(self, axis=0):
        """how many non-zero element in each row/column
        - axis = :obj:`0`
            
            .. math::
                \\sum_{j}\mathbb{1}_{A_{ij} \\neq 0}    
        
        - axis = :obj:`1`
        
            .. math::
                \\sum_{i}\mathbb{1}_{A_{ij} \\neq 0}     
        
        Parameters
        ----------
        axis: int, optional
            the axis to sum, can be :obj:`0` or :obj:`1`, default :obj:`0`
        
        Returns
        -------
        torch.Tensor
            the degree of shape :math:`[n_{\\text{row}}]` or :math:`[n_{\\text{col}}]`
        """
        nonzero = self.row if axis == 0 else self.col
        return torch.bincount(nonzero, minlength=self.shape[0] if axis == 0 else self.shape[1])

    def sum(self, axis=None):
        """sum of all non-zero elements

        * axis = :obj:`None`

            .. math::
                \sum_{ij}A_{ij}

        * axis = :obj:`0`

            .. math::
                \sum_{j}A_{ij}
        
        * axis = :obj:`1`
        
            .. math::
                \sum_{i}A_{ij}

        Parameters
        ----------
        axis: int, optional
            the axis to sum, can be :obj:`None`, :obj:`0` or :obj:`1`, default :obj:`None`
        
        Returns
        -------
        torch.Tensor
            the sum of shape :math:`[]` or :math:`[n_\\text{row}]` or :math:`[n_\\text{col}]`
        """
        if axis is None:
            return self.edata.sum()
        elif axis == 0:
            return self.T @ torch.ones(self.shape[0], device=self.edata.device)
        elif axis == 1:
            return self @ torch.ones(self.shape[1], device=self.edata.device)
        else:
            raise Exception(f"unsupported axis {axis} for SparseMatrix.sum")

    def clone(self):
        """The cloned sparse matrix will share clone gradient with the original sparse matrix
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the cloned sparse matrix
        """
        return SparseMatrix(self.edata.clone(), self.row.clone(), self.col.clone(), self.shape)
    
    def detach(self):
        """The detached sparse matrix will not share gradient with the original sparse matrix
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the detached sparse matrix
        """
        return SparseMatrix(self.edata.detach(), self.row, self.col, self.shape)

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
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the transpose of the sparse matrix
        """
        return self.transpose()
    
    @property
    def requires_grad(self):
        """
        Returns
        -------
        bool
            whether the sparse matrix requires gradient or not
        """
        return self.edata.requires_grad

    @property
    def dtype(self):
        """
        Returns
        -------
        torch.dtype
            the dtype of the sparse matrix
        """
        return self.edata.dtype
    
    @property
    def device(self):
        """
        Returns
        -------
        torch.device
            the device of the sparse matrix
        """
        return self.edata.device

    @property
    def grad(self):
        """
        Returns
        ------- 
        torch_fem.sparse.SparseMatrix or None
            if the sparse matrix requires gradient, return the grad for each element 
            of the sparse matrix, otherwise return :obj:`None`
        """
        if self.edata.grad is None:
            return None
        else:
            return SparseMatrix(self.edata.grad, self.row, self.col, self.shape)

    @property
    def grad_fn(self):
        """
        Returns
        -------
        torch.autograd.Function or None
            if the sparse matrix requires gradient, return the grad_fn for each element 
            of the sparse matrix, otherwise return :obj:`None`
        """
        if self.edata.grad_fn is None:
            return None
        else:
            return self.edata.grad_fn

    @property
    def nnz(self):
        """
        Returns
        -------
        int
            the number of non-zero elements
        """
        return self.edata.shape[0]

    @property
    def layout_mask(self):
        """
        Returns
        -------
        torch.Tensor
            the mask of the layout, where the non-zero elements are 1, otherwise 0
        """
        mask = torch.zeros(self.shape, device=self.edata.device)
        mask[self.row, self.col] = 1
        return mask

    def type(self, dtype):
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix with dtype set to dtype
        """
        self.edata.to_(dtype)
        return self

    def detach(self):
        """
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix with requires_grad set to False
        """
        return SparseMatrix(self.edata.detach(), self.row, self.col, self.shape).requires_grad_(False)

    def to_scipy_coo(self):
        """
        Returns
        -------
        scipy.sparse.coo_matrix
            the scipy.sparse.coo_matrix of the sparse matrix
        """
        return scipy.sparse.coo_matrix((
            self.edata.detach().cpu().numpy(),
            (
                self.row.detach().cpu().numpy(),
                self.col.detach().cpu().numpy()
            )), shape=self.shape)
    
    def to_sparse_coo(self):
        """Turn the sparse matrix into a torch.sparse_coo_tensor, the gradient will be lost
        Returns
        -------
        torch.sparse_coo_tensor
            the torch.sparse_coo_tensor of the sparse matrix
        """
        return torch.sparse_coo_tensor(
            torch.stack([self.row, self.col]),
            self.edata,
            self.shape
        )

    def to_dense(self):
        """Turn the sparse matrix into a dense matrix, the gradient will be maintained
        Returns
        -------
        torch.Tensor
            the dense tensor of the sparse matrix
        """
        matrix = torch.zeros(self.shape, device=self.edata.device, dtype=self.edata.dtype)
        matrix[self.row, self.col] += self.edata
        return matrix

    def has_same_layout(self, obj):
        """
        Parameters
        ----------
        obj: SparseMatrix or str
            the object to be compared with, if it is a str, it will be compared with the layout_hash of the sparse matrix

        Returns
        -------
        bool
            whether the two sparse matrices have the same layout
        """
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
        """
        Parameters
        ----------
        matrix: torch.sparse_coo_tensor
            the sparse matrix to be converted
        
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix converted from the torch.sparse_coo_tensor
        """
        edata = matrix.values()
        row   = matrix.indices()[0]
        col   = matrix.indices()[1]
        shape = matrix.shape
        return SparseMatrix(edata, row, col, shape)

    @staticmethod
    def from_block_coo(edata, row, col, shape):
        """Each element in a sparse matrix is a block matrix

        Parameters
        ----------
        edata: torch.Tensor 
            3D float tensor of shape :math:`[|\\mathcal E|, C, C]`, where :math:`|\mathcal E|` is the number of edges, :math:`C` is the size of the block data
            the block data
        row: torch.Tensor 
            1D int tensor of shape :math:`[|\\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
            the row indices
        col: torch.Tensor 
            1D int tensor of shape :math:`[|\\mathcal E|]`, where :math:`|\mathcal E|` is the number of edges
            the column indices
        shape: Tuple[int, int]
            the shape of the sparse matrix of the first two dim

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix converted from the block coo format of shape

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
    def from_dense(tensor):
        """
        Parameters
        ----------
        tensor: torch.Tensor
            the dense tensor to be converted

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix converted from the dense tensor

        Examples
        --------
        >>> SparseMatrix.from_dense(torch.eye(3))
        SparseMatrix(
            edata: tensor([1., 1., 1.])
            row  : tensor([0, 1, 2])
            col  : tensor([0, 1, 2])
            shape: (3, 3)
        )
        """
        assert tensor.dim() == 2, f"the tensor should be 2D, but got {tensor.dim()}"
        rows, cols = torch.where(tensor != 0)
        edata = tensor[rows, cols]
        return SparseMatrix(edata, rows, cols, tensor.shape)

    @staticmethod
    def random(m,n, density=0.1, device="cpu", dtype=torch.float):
        """randomly generate a sparse matrix
        Parameters
        ----------
        m: int
            the number of rows
        n: int
            the number of cols
        density: float, optional
            the density of the sparse matrix, default 0.1
        device: str, optional
            the device of the sparse matrix, default cpu
        dtype: torch.dtype, optional
            the dtype of the sparse matrix, default torch.float

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix of shape :math:`[m,n]` with density :obj:`density` and dtype :obj:`dtype`
        """
        matrix = scipy.sparse.random(m, n, density, format="coo")
        return SparseMatrix.from_scipy_coo(matrix, device=device, dtype=dtype)
    
    @staticmethod
    def random_layout(m, n, density=0.1, device="cpu"):
        """randomly generate a sparse matrix layout
        Parameters
        ----------
        m: int
            the number of rows
        n: int
            the number of cols
        density: float, optional
            the density of the sparse matrix, default 0.1
        device: str, optional
            the device of the sparse matrix, default cpu

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
            the layout of the sparse matrix of shape :math:`[m,n]` with density :obj:`density` and dtype :obj:`dtype`
        """
        matrix = scipy.sparse.random(m, n, density, format="coo")
        row    = torch.from_numpy(matrix.row).to(device)
        col    = torch.from_numpy(matrix.col).to(device)
        shape  = matrix.shape
        return row, col, shape
    
    @staticmethod
    def random_from_layout(layout, device="cpu", dtype=torch.float):
        """randomly generate a sparse matrix from a layout
        Parameters
        ----------
        layout: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
            the layout of the sparse matrix
        device: str, optional
            the device of the sparse matrix, default cpu
        dtype: torch.dtype, optional
            the dtype of the sparse matrix, default torch.float
            
        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix of shape :math:`[m,n]` with density :obj:`density` and dtype :obj:`dtype`
        """
        row, col, shape = layout
        edata = torch.rand(row.shape[0], device=device, dtype=dtype)
        return SparseMatrix(edata, row.to(device), col.to(device), shape)
    
    @staticmethod
    def eye(n, value=1., device="cpu", dtype=torch.float):
        """generate a sparse identity matrix
        Parameters
        ----------
        n: int
            the number of rows and columns
        value: float, optional
            the value of the diagonal elements, default 1.
        device: str, optional
            the device of the sparse matrix, default cpu
        dtype: torch.dtype, optional
            the dtype of the sparse matrix, default torch.float

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the sparse matrix of shape :math:`[n,n]` with value :obj:`value` and dtype :obj:`dtype`


        Examples
        --------
        >>> SparseMatrix.eye(3).to_dense()
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
        """
        return SparseMatrix(
            torch.ones(n, device=device, dtype=dtype) * value,
            torch.arange(n, device=device),
            torch.arange(n, device=device),
            (n, n)
        )

    @staticmethod
    def full(m, n, value=1., device="cpu", dtype=torch.float):
        """generate a dense matrix filled with a value
        Parameters
        ----------
        m: int 
            the number of rows
        n: int
            the number of columns
        value: float, optional
            the value of the diagonal elements, default 1.
        device: str, optional
            the device of the sparse matrix, default cpu
        dtype: torch.dtype, optional
            the dtype of the sparse matrix, default torch.float

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the dense tensor of shape :math:`[n,n]` with value :obj:`value` and dtype :obj:`dtype`
        """
        if value == 0:
            return SparseMatrix(
                torch.tensor([], device=device, dtype=dtype),
                torch.tensor([], device=device, dtype=torch.int64),
                torch.tensor([], device=device, dtype=torch.int64),
                (m, n)
            )
        cols = torch.arange(n, device=device)
        rows = torch.arange(m, device=device)
        edata = torch.ones(n*m, device=device, dtype=dtype) * value
        cols, rows = torch.meshgrid(cols, rows)
        return SparseMatrix(edata, rows.flatten(), cols.flatten(), (m, n))

    @staticmethod
    def combine_vector(matrices, axis=0):
        """Combine sparse matrices into a sparse matrix

        Parameters
        ----------
        matrices: List[torch_fem.sparse.SparseMatrix]
            the sparse matrices to be combined
        axis: int, optional
            the axis to combine, can be :obj:`0` or :obj:`1`, default :obj:`0`

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the combined sparse matrix

        Examples
        --------
        >>> SparseMatrix.combine_vector([
        ...     SparseMatrix.eye(3), SparseMatrix.eye(3)    
        ... ]).to_dense()
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])
        """
        row_offset = 0
        col_offset = 0
        cols = []
        rows = []
        edata = []
        for sparse_matrix in matrices:
            assert isinstance(sparse_matrix, (SparseMatrix,torch.Tensor)), f"the sparse matrices should be SparseMatrix, but got {type(sparse_matrix)}"
            assert sparse_matrix.shape[1-axis] == matrices[0].shape[1-axis], f"the shape of the sparse matrices should be the same, but got {sparse_matrix.shape}, {matrices[0].shape}"
            if isinstance(sparse_matrix, torch.Tensor):
                sparse_matrix = SparseMatrix.from_dense(sparse_matrix)
            cols.append(sparse_matrix.col + col_offset)
            rows.append(sparse_matrix.row + row_offset)
            edata.append(sparse_matrix.edata)
            if axis == 0:
                row_offset += sparse_matrix.shape[0]
            elif axis == 1:
                col_offset += sparse_matrix.shape[1]
            else:
                raise Exception(f"unsupported axis {axis} for SparseMatrix.combine_vector, could only be 0 or 1")
        if axis == 0:
            col_offset += matrices[0].shape[1]
        elif axis == 1:
            row_offset += matrices[0].shape[0]
        
        return SparseMatrix(
            torch.cat(edata, dim=0),
            torch.cat(rows, dim=0),
            torch.cat(cols, dim=0),
            (row_offset, col_offset)
        )
    
    @staticmethod
    def combine_matrix(matrices):
        """

        Parameters
        ----------
        matrices: List[List[torch_fem.sparse.SparseMatrix or None or float or int]]
            the sparse matrices to be combined
        axis: int, optional
            the axis to combine, can be :obj:`0` or :obj:`1`, default :obj:`0`

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the combined sparse matrix

        Examples
        --------
        >>> SparseMatrix.combine_matrix([
        ...     [SparseMatrix.eye(3), SparseMatrix.eye(3)],
        ...     [SparseMatrix.eye(3), SparseMatrix.eye(3)]
        ... ]).to_dense()
        tensor([[1., 0., 0., 1., 0., 0.],
                [0., 1., 0., 0., 1., 0.],
                [0., 0., 1., 0., 0., 1.],
                [1., 0., 0., 1., 0., 0.],
                [0., 1., 0., 0., 1., 0.],
                [0., 0., 1., 0., 0., 1.]])

        """
        row_offset = 0 
        col_offset = 0
        rows       = []
        cols       = []
        edata      = []

        n_block_rows = len(matrices)
        n_block_cols = len(matrices[0])
        shape        = np.zeros((n_block_rows, n_block_cols, 2), dtype=np.int64)
        # check numebr of blocks consistency
        for i in range(n_block_rows):
            assert len(matrices[i]) == len(matrices[0]), f"the number of columns of the sparse matrices should be the same, but got {len(matrices[i])}, {len(matrices[0])}"
            for j in range(n_block_cols):
                if isinstance(matrices[i][j],(SparseMatrix,torch.Tensor)):
                    shape[i,j] = np.array(matrices[i][j].shape)
        # check shape inference
        assert (shape[:,:,0] > 0).any(1).all(), f"there should be at least one non-zero sparse matrix in each row, but got dimension collapse at row {np.where(~(shape[:,:,0] > 0).any(1))[0].tolist()}"
        assert (shape[:,:,1] > 0).any(0).all(), f"there should be at least one non-zero sparse matrix in each column, but got dimension collapse at column {np.where(~(shape[:,:,1] > 0).any(0).all())[0].tolist()}"
        # check shape unique
        for i in range(n_block_rows):
            nz_shape = shape[i, :, 0][shape[i, :, 0] > 0]
            uni_shape = np.unique(nz_shape)
            assert len(uni_shape) == 1, f"the number of rows of the sparse matrices should be the same, but got {shape[i, :, 0]} at row {i}"
            shape[i, :, 0][shape[i, :, 0] == 0] = uni_shape
        
        for j in range(n_block_cols):
            nz_shape = shape[:, j, 1][shape[:, j, 1] > 0]
            uni_shape = np.unique(nz_shape)
            assert len(uni_shape) == 1, f"the number of columns of the sparse matrices should be the same, but got {shape[:, j, 1]} at column {j}"
            shape[:, j, 1][shape[:, j, 1] == 0] = uni_shape


        for i in range(len(matrices)):
            col_offset = 0
            for j in range(len(matrices[i])):
                if matrices[i][j] is not None:
                    if isinstance(matrices[i][j], (int, float)):
                        matrices[i][j] = SparseMatrix.full(shape[i,j,0], shape[i,j,1], value=matrices[i][j])
                    elif isinstance(matrices[i][j], torch.Tensor):
                        matrices[i][j] = SparseMatrix.from_dense(matrices[i][j])
                    edata.append(matrices[i][j].edata)
                    rows.append(matrices[i][j].row + row_offset)
                    cols.append(matrices[i][j].col + col_offset)
                col_offset += shape[i, j, 1]
            row_offset += shape[i, 0, 0]
        
        return SparseMatrix(
            torch.cat(edata, dim=0),
            torch.cat(rows, dim=0),
            torch.cat(cols, dim=0),
            (row_offset, col_offset)
        )
    
    @staticmethod
    def combine(matrices):
        """
        the dispatch function for :attr:combine_vector and :attr:combine_matrix

        Parameters
        ----------
        matrices: List[torch_fem.sparse.SparseMatrix or List[torch_fem.sparse.SparseMatrix or None or float or int]]
            the sparse matrices to be combined

        Returns
        -------
        torch_fem.sparse.SparseMatrix
            the combined sparse matrix

        """
        if isinstance(matrices[0], (list, tuple)):
            return SparseMatrix.combine_matrix(matrices)
        else:
            return SparseMatrix.combine_vector(matrices, axis=0)
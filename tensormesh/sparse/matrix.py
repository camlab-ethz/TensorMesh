"""
TensorMesh Sparse Matrix

SparseMatrix extends torch_sla.SparseTensor with FEM-specific utilities.
"""

import numpy as np
import torch
import scipy.sparse
import hashlib
from typing import Callable, List, Optional, Tuple, Union

# Import torch_sla
try:
    from torch_sla import SparseTensor
    HAS_TORCH_SLA = True
except ImportError:
    HAS_TORCH_SLA = False
    raise ImportError(
        "torch-sla is required for TensorMesh sparse operations.\n"
        "Install with: pip install torch-sla>=0.1.4"
    )


class SparseMatrix(SparseTensor):
    """COO format sparse matrix for FEM computations.
    
    Extends torch_sla.SparseTensor with FEM-specific utilities like
    block assembly, layout comparison, and scipy interoperability.

    Examples
    --------
    .. code-block:: python

        import torch
        from tensormesh.sparse import SparseMatrix

        # Create a sparse matrix from COO format
        edata = torch.tensor([1.0, 2.0, 3.0])
        row = torch.tensor([0, 1, 2])
        col = torch.tensor([1, 2, 0]) 
        shape = (3, 3)
        A = SparseMatrix(edata, row, col, shape)

        # Matrix-vector multiplication (inherited from SparseTensor)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = A @ x

        # Solve linear system Ax = b (inherited from SparseTensor)
        b = torch.tensor([1.0, 2.0, 3.0]).double()
        A_double = A.double()
        x = A_double.solve(b)

        # FEM-specific: block COO assembly
        block_data = torch.randn(10, 3, 3)  # 10 element matrices
        elem_row = torch.arange(10)
        elem_col = torch.arange(10)
        K = SparseMatrix.from_block_coo(block_data, elem_row, elem_col, (10, 10))

    Parameters
    ----------
    edata : torch.Tensor 
        1D float tensor of shape [nnz], the non-zero values
    row : torch.Tensor
        1D int tensor of shape [nnz], the row indices
    col : torch.Tensor
        1D int tensor of shape [nnz], the column indices
    shape : Tuple[int, int]
        The shape of the sparse matrix (nrows, ncols)
    """

    def __init__(self, edata: torch.Tensor, row: torch.Tensor, 
                 col: torch.Tensor, shape: Tuple[int, int]):
        # Initialize parent SparseTensor
        super().__init__(edata, row.long(), col.long(), shape)
        
        # Compute layout hash for FEM assembly comparison
        row_cpu = row.detach().cpu()
        col_cpu = col.detach().cpu()
        self._layout_hash = hashlib.sha256(
            row_cpu.numpy().tobytes() + col_cpu.numpy().tobytes()
        ).hexdigest()

    # ==================== Backward Compatibility Properties ====================
    
    @property
    def edata(self) -> torch.Tensor:
        """Alias for values (backward compatibility with old TensorMesh API)."""
        return self.values
    
    @property
    def row(self) -> torch.Tensor:
        """Alias for row_indices."""
        return self.row_indices
    
    @property
    def col(self) -> torch.Tensor:
        """Alias for col_indices."""
        return self.col_indices

    @property
    def edges(self) -> torch.Tensor:
        """Edge indices of shape [2, nnz]."""
        return torch.stack([self.row_indices, self.col_indices], dim=0)

    @property
    def layout_hash(self) -> str:
        """Hash of the sparsity pattern for comparison."""
        return self._layout_hash

    @property
    def layout_mask(self) -> torch.Tensor:
        """Binary mask where non-zero elements are 1."""
        mask = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        mask[self.row_indices, self.col_indices] = 1
        return mask

    @property
    def grad(self) -> Optional['SparseMatrix']:
        """Return gradient as SparseMatrix (if values have gradient)."""
        if self.values.grad is None:
            return None
        return SparseMatrix(
            self.values.grad, self.row_indices, self.col_indices, self.shape
        )

    # ==================== Type-Preserving Helpers ====================

    def _wrap(self, result):
        """Wrap a SparseTensor result back into SparseMatrix."""
        if isinstance(result, SparseTensor):
            return SparseMatrix(result.values, result.row_indices, result.col_indices, result.shape)
        return result

    # ==================== Arithmetic (preserve SparseMatrix type) ====================

    def __add__(self, other):
        return self._wrap(super().__add__(other))

    def __radd__(self, other):
        return self._wrap(super().__radd__(other))

    def __sub__(self, other):
        return self._wrap(super().__sub__(other))

    def __rsub__(self, other):
        return self._wrap(super().__rsub__(other))

    def __mul__(self, other):
        return self._wrap(super().__mul__(other))

    def __rmul__(self, other):
        return self._wrap(super().__rmul__(other))

    def __matmul__(self, other):
        result = super().__matmul__(other)
        if isinstance(result, SparseTensor):
            return self._wrap(result)
        return result

    # ==================== Device/Dtype Methods (preserve type) ====================

    def to(self, *args, **kwargs) -> 'SparseMatrix':
        return self._wrap(super().to(*args, **kwargs))

    def cuda(self, device=None) -> 'SparseMatrix':
        return self._wrap(super().cuda(device))

    def cpu(self) -> 'SparseMatrix':
        return self._wrap(super().cpu())

    def float(self) -> 'SparseMatrix':
        return self._wrap(super().float())

    def double(self) -> 'SparseMatrix':
        return self._wrap(super().double())

    def half(self) -> 'SparseMatrix':
        return self._wrap(super().half())

    def detach(self) -> 'SparseMatrix':
        return self._wrap(super().detach())

    # ==================== FEM-Specific Methods ====================

    def has_same_layout(self, other: Union[str, 'SparseMatrix']) -> bool:
        """Check if two matrices have the same sparsity pattern.
        
        Useful for FEM assembly where multiple matrices share the same mesh topology.
        """
        if isinstance(other, str):
            return self._layout_hash == other
        elif isinstance(other, SparseMatrix):
            return self._layout_hash == other._layout_hash
        else:
            raise TypeError(f"Expected str or SparseMatrix, got {type(other)}")

    def degree(self, axis: int = 0) -> torch.Tensor:
        """Count non-zero elements per row (axis=0) or column (axis=1).
        
        Useful for graph-based operations in FEM.
        """
        indices = self.row_indices if axis == 0 else self.col_indices
        size = self.shape[0] if axis == 0 else self.shape[1]
        return torch.bincount(indices, minlength=size)

    def transpose(self) -> 'SparseMatrix':
        """Transpose the sparse matrix."""
        return SparseMatrix(
            self.values, self.col_indices, self.row_indices, 
            (self.shape[1], self.shape[0])
        )

    @property
    def T(self) -> 'SparseMatrix':
        """Transpose property."""
        return self.transpose()

    # ==================== Scipy Interoperability ====================

    def to_scipy_coo(self) -> scipy.sparse.coo_matrix:
        """Convert to scipy COO matrix."""
        return scipy.sparse.coo_matrix((
            self.values.detach().cpu().numpy(),
            (self.row_indices.detach().cpu().numpy(),
             self.col_indices.detach().cpu().numpy())
        ), shape=self.shape)

    def to_sparse_coo(self) -> torch.Tensor:
        """Convert to torch.sparse_coo_tensor."""
        return torch.sparse_coo_tensor(
            torch.stack([self.row_indices, self.col_indices]),
            self.values,
            self.shape
        )

    # ==================== Static Factory Methods ====================

    @staticmethod
    def from_scipy_coo(matrix: scipy.sparse.coo_matrix, 
                       device: str = "cpu", 
                       dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Create from scipy COO matrix."""
        edata = torch.from_numpy(matrix.data.astype(np.float32)).to(device).type(dtype)
        row = torch.from_numpy(matrix.row.astype(np.int64)).to(device)
        col = torch.from_numpy(matrix.col.astype(np.int64)).to(device)
        return SparseMatrix(edata, row, col, matrix.shape)

    @staticmethod
    def from_sparse_coo(matrix: torch.Tensor) -> 'SparseMatrix':
        """Create from torch.sparse_coo_tensor."""
        matrix = matrix.coalesce()
        return SparseMatrix(
            matrix.values(),
            matrix.indices()[0],
            matrix.indices()[1],
            tuple(matrix.shape)
        )

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> 'SparseMatrix':
        """Create from dense tensor."""
        assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
        rows, cols = torch.where(tensor != 0)
        return SparseMatrix(tensor[rows, cols], rows, cols, tuple(tensor.shape))

    @staticmethod
    def from_block_coo(edata: torch.Tensor, row: torch.Tensor, 
                       col: torch.Tensor, shape: Tuple[int, int]) -> 'SparseMatrix':
        """Create from block COO format (common in FEM element assembly).
        
        Parameters
        ----------
        edata : torch.Tensor
            Block data of shape [n_elements, block_size, block_size]
        row, col : torch.Tensor
            Block indices of shape [n_elements]
        shape : Tuple[int, int]
            Number of block rows and columns
            
        Returns
        -------
        SparseMatrix
            Assembled sparse matrix of shape [shape[0]*block_size, shape[1]*block_size]
        """
        n_elements = edata.shape[0]
        block_size = edata.shape[1]
        
        edata_flat = edata.flatten()
        row_exp = row[:, None].repeat(1, block_size * block_size)
        col_exp = col[:, None].repeat(1, block_size * block_size)
        
        i, j = torch.meshgrid(
            torch.arange(block_size, device=row.device),
            torch.arange(block_size, device=row.device),
            indexing='ij'
        )
        
        row_final = (row_exp * block_size + i.flatten()).flatten()
        col_final = (col_exp * block_size + j.flatten()).flatten()
        
        new_shape = (shape[0] * block_size, shape[1] * block_size)
        return SparseMatrix(edata_flat, row_final, col_final, new_shape)

    @staticmethod
    def random(m: int, n: int, density: float = 0.1, 
               device: str = "cpu", dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Generate a random sparse matrix."""
        matrix = scipy.sparse.random(m, n, density, format="coo")
        return SparseMatrix.from_scipy_coo(matrix, device=device, dtype=dtype)

    @staticmethod
    def random_layout(m: int, n: int, density: float = 0.1, 
                      device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Generate a random sparse layout (row, col, shape)."""
        matrix = scipy.sparse.random(m, n, density, format="coo")
        row = torch.from_numpy(matrix.row.astype(np.int64)).to(device)
        col = torch.from_numpy(matrix.col.astype(np.int64)).to(device)
        return row, col, matrix.shape

    @staticmethod
    def random_from_layout(layout: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]],
                           device: str = "cpu", 
                           dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Generate random values with given layout."""
        row, col, shape = layout
        edata = torch.rand(row.shape[0], device=device, dtype=dtype)
        return SparseMatrix(edata, row.to(device), col.to(device), shape)

    @staticmethod
    def eye(n: int, value: float = 1., 
            device: str = "cpu", dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Generate a sparse identity matrix."""
        indices = torch.arange(n, device=device)
        values = torch.ones(n, device=device, dtype=dtype) * value
        return SparseMatrix(values, indices, indices.clone(), (n, n))

    @staticmethod
    def full(m: int, n: int, value: float = 1., 
             device: str = "cpu", dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Generate a dense matrix filled with a value (stored as sparse)."""
        if value == 0:
            return SparseMatrix(
                torch.tensor([], device=device, dtype=dtype),
                torch.tensor([], device=device, dtype=torch.int64),
                torch.tensor([], device=device, dtype=torch.int64),
                (m, n)
            )
        rows, cols = torch.meshgrid(
            torch.arange(m, device=device),
            torch.arange(n, device=device),
            indexing='ij'
        )
        edata = torch.ones(m * n, device=device, dtype=dtype) * value
        return SparseMatrix(edata, rows.flatten(), cols.flatten(), (m, n))

    # ==================== Block Matrix Operations ====================

    @staticmethod
    def combine_vector(matrices: List['SparseMatrix'], axis: int = 0) -> 'SparseMatrix':
        """Stack sparse matrices along an axis."""
        rows, cols, edatas = [], [], []
        offset = 0
        fixed_dim = matrices[0].shape[1 - axis]
        
        for mat in matrices:
            assert mat.shape[1 - axis] == fixed_dim, "Dimension mismatch"
            if isinstance(mat, torch.Tensor):
                mat = SparseMatrix.from_dense(mat)
            
            if axis == 0:
                rows.append(mat.row_indices + offset)
                cols.append(mat.col_indices)
                offset += mat.shape[0]
            else:
                rows.append(mat.row_indices)
                cols.append(mat.col_indices + offset)
                offset += mat.shape[1]
            edatas.append(mat.values)
        
        if axis == 0:
            shape = (offset, fixed_dim)
        else:
            shape = (fixed_dim, offset)
        
        return SparseMatrix(
            torch.cat(edatas), torch.cat(rows), torch.cat(cols), shape
        )

    @staticmethod
    def combine_matrix(matrices: List[List['SparseMatrix']]) -> 'SparseMatrix':
        """Combine sparse matrices in a 2D block layout."""
        n_rows = len(matrices)
        n_cols = len(matrices[0])
        
        # Infer block shapes
        row_sizes = [0] * n_rows
        col_sizes = [0] * n_cols
        
        for i in range(n_rows):
            for j in range(n_cols):
                mat = matrices[i][j]
                if mat is not None and not isinstance(mat, (int, float)):
                    if row_sizes[i] == 0:
                        row_sizes[i] = mat.shape[0]
                    if col_sizes[j] == 0:
                        col_sizes[j] = mat.shape[1]
        
        # Build combined matrix
        rows, cols, edatas = [], [], []
        row_offset = 0
        
        for i in range(n_rows):
            col_offset = 0
            for j in range(n_cols):
                mat = matrices[i][j]
                if mat is not None:
                    if isinstance(mat, (int, float)):
                        mat = SparseMatrix.full(row_sizes[i], col_sizes[j], value=mat)
                    elif isinstance(mat, torch.Tensor):
                        mat = SparseMatrix.from_dense(mat)
                    
                    rows.append(mat.row_indices + row_offset)
                    cols.append(mat.col_indices + col_offset)
                    edatas.append(mat.values)
                col_offset += col_sizes[j]
            row_offset += row_sizes[i]
        
        return SparseMatrix(
            torch.cat(edatas), torch.cat(rows), torch.cat(cols),
            (row_offset, col_offset)
        )

    @staticmethod
    def combine(matrices) -> 'SparseMatrix':
        """Dispatch to combine_vector or combine_matrix based on input structure."""
        if isinstance(matrices[0], (list, tuple)):
            return SparseMatrix.combine_matrix(matrices)
        else:
            return SparseMatrix.combine_vector(matrices, axis=0)

    # ==================== String Representation ====================

    def __repr__(self) -> str:
        return (
            f"SparseMatrix(\n"
            f"    values: {self.values}\n"
            f"    row   : {self.row_indices}\n"
            f"    col   : {self.col_indices}\n"
            f"    shape : {self.shape}\n"
            f"    nnz   : {self.nnz}\n"
            f")"
        )

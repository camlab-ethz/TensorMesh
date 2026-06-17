"""FEM-flavoured COO sparse matrix.

:class:`SparseMatrix` extends :class:`torch_sla.SparseTensor` with helpers
that come up in finite-element assembly: block-COO construction, sequence-
identity :attr:`layout_signature` for sparsity-pattern caching, scipy
interoperability, and block-stacking. Arithmetic and device-conversion
methods are overridden so that the result type is preserved (``A + B``
of two ``SparseMatrix`` returns ``SparseMatrix``, not the parent type).
"""

import warnings
import numpy as np
import torch
import scipy.sparse
from typing import List, Optional, Tuple, Union

from .mixin import _FEMSparsityMixin

try:
    from torch_sla import SparseTensor
except ImportError as e:
    raise ImportError(
        "torch-sla is required for TensorMesh sparse operations.\n"
        "Install with: pip install torch-sla>=0.2.0"
    ) from e


class SparseMatrix(_FEMSparsityMixin, SparseTensor):
    """COO sparse matrix with FEM-flavoured helpers.

    Subclass of ``torch_sla.SparseTensor``. Inherits ``@`` (spmm),
    ``solve``, autograd-aware values, ``.to_dense()`` and friends from
    the parent; adds layout hashing, block-COO assembly, scipy
    interoperability and type-preserving arithmetic.

    Parameters
    ----------
    edata : torch.Tensor
        1D float tensor of shape ``[nnz]``: the non-zero values.
    row : torch.Tensor
        1D integer tensor of shape ``[nnz]``: row indices.
    col : torch.Tensor
        1D integer tensor of shape ``[nnz]``: column indices.
    shape : Tuple[int, int]
        ``(nrows, ncols)`` of the dense equivalent.

    Attributes
    ----------
    layout_hash : str
        SHA-256 of ``(row, col)`` concatenated bytes. Two matrices with
        the same sparsity pattern share this hash; useful for caching
        any quantity that depends only on the topology (Condenser
        permutations, AMG hierarchies, etc.).

    Examples
    --------
    .. code-block:: python

        import torch
        from tensormesh.sparse import SparseMatrix

        edata = torch.tensor([1.0, 2.0, 3.0])
        row   = torch.tensor([0, 1, 2])
        col   = torch.tensor([1, 2, 0])
        A     = SparseMatrix(edata, row, col, shape=(3, 3))

        # Inherited from SparseTensor:
        y = A @ torch.tensor([1.0, 2.0, 3.0])
        x = A.double().solve(torch.ones(3, dtype=torch.float64))

        # FEM-specific block-COO assembly: 10 element matrices of size 3x3.
        block_data = torch.randn(10, 3, 3)
        elem_row   = torch.arange(10)
        elem_col   = torch.arange(10)
        K = SparseMatrix.from_block_coo(block_data, elem_row, elem_col, (10, 10))
    """

    def __init__(self, edata: torch.Tensor, row: torch.Tensor,
                 col: torch.Tensor, shape: Tuple[int, int]):
        # Initialize parent SparseTensor
        super().__init__(edata, row.long(), col.long(), shape)
        # NB: layout identity now lives in :attr:`layout_signature` (mixin);
        # see ``tensormesh.sparse.mixin`` for the reasoning. Cost of the old
        # SHA-256-at-__init__ is gone -- signature is lazy + zero GPU sync.

    # ==================== Backward Compatibility Properties ====================

    @property
    def edata(self) -> torch.Tensor:
        """Alias for ``values`` (legacy TensorMesh API)."""
        return self.values

    @property
    def row(self) -> torch.Tensor:
        """Alias for ``row_indices``."""
        return self.row_indices

    @property
    def col(self) -> torch.Tensor:
        """Alias for ``col_indices``."""
        return self.col_indices

    @property
    def edges(self) -> torch.Tensor:
        """Stacked ``(row, col)`` indices of shape ``[2, nnz]``."""
        return torch.stack([self.row_indices, self.col_indices], dim=0)

    @property
    def layout_hash(self):
        """**Deprecated.** Use :attr:`layout_signature` instead.

        Returns the new sequence-identity signature wrapped in a small
        compat shell: equality with another deprecated ``layout_hash``
        still works (both routed through :attr:`layout_signature`).
        Kept so existing callers keep functioning during the migration.
        """
        warnings.warn(
            "SparseMatrix.layout_hash is deprecated; use layout_signature "
            "(an opaque hashable tuple) instead. layout_hash now returns "
            "the same signature so old call sites keep working.",
            DeprecationWarning, stacklevel=2,
        )
        return self.layout_signature

    @property
    def layout_mask(self) -> torch.Tensor:
        """Dense ``[nrows, ncols]`` mask with 1.0 on the sparsity pattern."""
        mask = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        mask[self.row_indices, self.col_indices] = 1
        return mask

    @property
    def grad(self) -> Optional['SparseMatrix']:
        """Gradient w.r.t. ``values``, wrapped as a :class:`SparseMatrix`.

        Returns ``None`` when no gradient has accumulated on ``values``.
        """
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

    def has_same_layout(self, other) -> bool:
        """Check whether ``other`` shares this matrix's sparsity pattern.

        Accepts another :class:`SparseMatrix`, an opaque
        :attr:`layout_signature` tuple, or — for legacy callers — the
        previously-deprecated ``layout_hash`` value (which now returns
        the same tuple). Inherits the mixin implementation.
        """
        return super().has_same_layout(other)

    def degree(self, axis: int = 0) -> torch.Tensor:
        """Non-zero count per row (``axis=0``) or per column (``axis=1``).

        Parameters
        ----------
        axis : int, default 0
            Aggregate over ``axis``; ``0`` counts nnz per row, ``1`` per
            column.

        Returns
        -------
        torch.Tensor
            1D int tensor of length ``shape[axis]``.
        """
        indices = self.row_indices if axis == 0 else self.col_indices
        size = self.shape[0] if axis == 0 else self.shape[1]
        return torch.bincount(indices, minlength=size)

    def transpose(self) -> 'SparseMatrix':
        """Return ``A^T`` as a :class:`SparseMatrix` sharing this matrix's values."""
        return SparseMatrix(
            self.values, self.col_indices, self.row_indices,
            (self.shape[1], self.shape[0])
        )

    @property
    def T(self) -> 'SparseMatrix':
        """Shorthand for :meth:`transpose`."""
        return self.transpose()

    # ==================== Scipy Interoperability ====================

    def to_scipy_coo(self) -> scipy.sparse.coo_matrix:
        """Detach and convert to a :class:`scipy.sparse.coo_matrix` on CPU."""
        return scipy.sparse.coo_matrix((
            self.values.detach().cpu().numpy(),
            (self.row_indices.detach().cpu().numpy(),
             self.col_indices.detach().cpu().numpy())
        ), shape=self.shape)

    def to_sparse_coo(self) -> torch.Tensor:
        """Convert to a :func:`torch.sparse_coo_tensor` (autograd-tracked)."""
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
        """Wrap a :class:`scipy.sparse.coo_matrix` as a :class:`SparseMatrix`.

        Parameters
        ----------
        matrix : scipy.sparse.coo_matrix
            Source matrix.
        device : str, default ``"cpu"``
            Target torch device.
        dtype : torch.dtype, default ``torch.float``
            Target value dtype.

        Returns
        -------
        SparseMatrix
        """
        edata = torch.from_numpy(matrix.data.astype(np.float32)).to(device).type(dtype)
        row = torch.from_numpy(matrix.row.astype(np.int64)).to(device)
        col = torch.from_numpy(matrix.col.astype(np.int64)).to(device)
        return SparseMatrix(edata, row, col, matrix.shape)

    @staticmethod
    def from_sparse_coo(matrix: torch.Tensor) -> 'SparseMatrix':
        """Wrap a coalesced :func:`torch.sparse_coo_tensor` as a :class:`SparseMatrix`.

        Parameters
        ----------
        matrix : torch.Tensor
            Sparse COO tensor; will be coalesced in place.

        Returns
        -------
        SparseMatrix
        """
        matrix = matrix.coalesce()
        return SparseMatrix(
            matrix.values(),
            matrix.indices()[0],
            matrix.indices()[1],
            tuple(matrix.shape)
        )

    @staticmethod
    def from_dense(tensor: torch.Tensor) -> 'SparseMatrix':
        """Pull non-zero entries out of a dense 2D tensor into COO.

        Parameters
        ----------
        tensor : torch.Tensor
            2D tensor; entries exactly equal to zero are dropped.

        Returns
        -------
        SparseMatrix
        """
        assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
        rows, cols = torch.where(tensor != 0)
        return SparseMatrix(tensor[rows, cols], rows, cols, tuple(tensor.shape))

    @staticmethod
    def from_block_coo(edata: torch.Tensor, row: torch.Tensor,
                       col: torch.Tensor, shape: Tuple[int, int]) -> 'SparseMatrix':
        """Expand block-COO storage into a flat :class:`SparseMatrix`.

        Each block is a small dense matrix attached to a ``(row, col)``
        pair in a coarser graph. This is the layout produced by FEM
        element assembly when the element data is *vector-valued*
        (e.g. linear elasticity), with one ``[dim, dim]`` block per
        ``(node_i, node_j)`` pair.

        Blocks are assumed **square**; the function uses
        ``edata.shape[1]`` as the block size and ignores ``shape[2]``.

        Parameters
        ----------
        edata : torch.Tensor
            Block data of shape ``[n_elements, block_size, block_size]``.
        row, col : torch.Tensor
            Block indices of shape ``[n_elements]``; entries refer to
            the coarse graph.
        shape : Tuple[int, int]
            ``(block_rows, block_cols)`` of the coarse graph.

        Returns
        -------
        SparseMatrix
            Flat COO matrix of shape
            ``(shape[0] * block_size, shape[1] * block_size)``.
        """
        block_size = edata.shape[1]
        assert edata.shape[2] == block_size, (
            f"from_block_coo expects square blocks, got "
            f"[..., {edata.shape[1]}, {edata.shape[2]}]"
        )

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
        """Random ``m x n`` sparse matrix with the requested density.

        Parameters
        ----------
        m, n : int
            Dense shape.
        density : float, default 0.1
            Fraction of entries that are non-zero. Drawn via
            :func:`scipy.sparse.random`.
        device : str, default ``"cpu"``
        dtype : torch.dtype, default ``torch.float``

        Returns
        -------
        SparseMatrix
        """
        matrix = scipy.sparse.random(m, n, density, format="coo")
        return SparseMatrix.from_scipy_coo(matrix, device=device, dtype=dtype)

    @staticmethod
    def random_layout(m: int, n: int, density: float = 0.1,
                      device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Random ``(row, col, shape)`` triple with no value tensor attached.

        Useful when several matrices need to share the same sparsity
        pattern; combine with :meth:`random_from_layout`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
            ``(row, col, (m, n))``.
        """
        matrix = scipy.sparse.random(m, n, density, format="coo")
        row = torch.from_numpy(matrix.row.astype(np.int64)).to(device)
        col = torch.from_numpy(matrix.col.astype(np.int64)).to(device)
        return row, col, matrix.shape

    @staticmethod
    def random_from_layout(layout: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]],
                           device: str = "cpu",
                           dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Attach random uniform ``[0, 1)`` values to an existing layout.

        Parameters
        ----------
        layout : Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
            ``(row, col, shape)`` triple, e.g. from :meth:`random_layout`.
        device : str, default ``"cpu"``
        dtype : torch.dtype, default ``torch.float``

        Returns
        -------
        SparseMatrix
        """
        row, col, shape = layout
        edata = torch.rand(row.shape[0], device=device, dtype=dtype)
        return SparseMatrix(edata, row.to(device), col.to(device), shape)

    @staticmethod
    def eye(n: int, value: float = 1.,
            device: str = "cpu", dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """``n x n`` sparse identity, optionally scaled by ``value``.

        Parameters
        ----------
        n : int
            Dimension.
        value : float, default 1.0
            Diagonal value; ``value * I_n`` is stored.
        device : str, default ``"cpu"``
        dtype : torch.dtype, default ``torch.float``

        Returns
        -------
        SparseMatrix
        """
        indices = torch.arange(n, device=device)
        values = torch.ones(n, device=device, dtype=dtype) * value
        return SparseMatrix(values, indices, indices.clone(), (n, n))

    @staticmethod
    def full(m: int, n: int, value: float = 1.,
             device: str = "cpu", dtype: torch.dtype = torch.float) -> 'SparseMatrix':
        """Constant ``m x n`` matrix stored densely-as-sparse.

        Mostly a building block for :meth:`combine_matrix` block layouts;
        ``value == 0`` is short-circuited to an empty COO tensor.

        Parameters
        ----------
        m, n : int
            Shape.
        value : float, default 1.0
            Constant entry.
        device : str, default ``"cpu"``
        dtype : torch.dtype, default ``torch.float``

        Returns
        -------
        SparseMatrix
        """
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
        """Stack a list of sparse matrices along ``axis``.

        Equivalent of :func:`torch.cat` for :class:`SparseMatrix`. Dense
        :class:`torch.Tensor` entries are auto-converted via
        :meth:`from_dense`.

        Parameters
        ----------
        matrices : List[SparseMatrix or torch.Tensor]
            Items must agree on ``shape[1 - axis]``.
        axis : int, default 0
            ``0`` stacks vertically (along rows), ``1`` horizontally.

        Returns
        -------
        SparseMatrix
        """
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
        """Assemble a 2D block layout of sparse matrices into one matrix.

        Each entry of ``matrices[i][j]`` may be:

        - ``None`` — treated as a zero block of inferred size;
        - ``int`` or ``float`` — expanded via :meth:`full` to a constant
          block of the inferred size;
        - :class:`torch.Tensor` — auto-converted via :meth:`from_dense`;
        - :class:`SparseMatrix` — used as-is.

        Block sizes are inferred from the first non-scalar entry in each
        row / column; a row or column with only ``None``/scalar entries
        will have inferred size zero.

        Parameters
        ----------
        matrices : List[List[SparseMatrix or torch.Tensor or int or float or None]]
            Rectangular nested list, ``[n_rows][n_cols]``.

        Returns
        -------
        SparseMatrix
            Assembled matrix of shape
            ``(sum(row_sizes), sum(col_sizes))``.
        """
        n_rows = len(matrices)
        n_cols = len(matrices[0])

        # Infer block shapes from the first non-scalar entry in each row/col.
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
        """Dispatch to :meth:`combine_matrix` or :meth:`combine_vector`.

        If the first element is a list or tuple, the input is treated as
        a 2D block layout (calls :meth:`combine_matrix`); otherwise it is
        a 1D stack along axis 0 (calls :meth:`combine_vector`).
        """
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

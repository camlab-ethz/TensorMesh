"""
cuDSS (NVIDIA CUDA Direct Sparse Solver) Python bindings.

This module provides Python bindings for libcudss.so, NVIDIA's high-performance
sparse direct solver library for CUDA.

Usage:
    from tensormesh.sparse.solve.cudss_solve import CuDSSSolver
    
    # CSR format sparse matrix on GPU
    solver = CuDSSSolver(row_ptr, col_idx, values, nrows, ncols)
    x = solver.solve(b)  # b can be [n] or [n, nrhs]
"""

import ctypes
from ctypes import c_void_p, c_int, c_int64, c_size_t, byref, POINTER, c_char_p
import os
from typing import Optional, Tuple

import torch
import numpy as np

# ============================================================================
# cuDSS Enums
# ============================================================================

# cudssStatus_t
CUDSS_STATUS_SUCCESS = 0
CUDSS_STATUS_NOT_INITIALIZED = 1
CUDSS_STATUS_ALLOC_FAILED = 2
CUDSS_STATUS_INVALID_VALUE = 3
CUDSS_STATUS_NOT_SUPPORTED = 4
CUDSS_STATUS_EXECUTION_FAILED = 5
CUDSS_STATUS_INTERNAL_ERROR = 6

# cudssPhase_t
CUDSS_PHASE_REORDERING = 1 << 0
CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1
CUDSS_PHASE_ANALYSIS = CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION
CUDSS_PHASE_FACTORIZATION = 1 << 2
CUDSS_PHASE_REFACTORIZATION = 1 << 3
CUDSS_PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)

# cudssMatrixType_t
CUDSS_MTYPE_GENERAL = 0
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MTYPE_HERMITIAN = 2
CUDSS_MTYPE_SPD = 3
CUDSS_MTYPE_HPD = 4

# cudssMatrixViewType_t
CUDSS_MVIEW_FULL = 0
CUDSS_MVIEW_LOWER = 1
CUDSS_MVIEW_UPPER = 2

# cudssIndexBase_t
CUDSS_BASE_ZERO = 0
CUDSS_BASE_ONE = 1

# cudssLayout_t
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_LAYOUT_ROW_MAJOR = 1

# cudaDataType_t (from library_types.h)
CUDA_R_32F = 0   # float
CUDA_R_64F = 1   # double
CUDA_R_32I = 10  # int32
CUDA_R_64I = 24  # int64

# ============================================================================
# Library Loading
# ============================================================================

_cudss_lib = None

def _get_cudss_lib():
    """Load libcudss.so library."""
    global _cudss_lib
    if _cudss_lib is not None:
        return _cudss_lib
    
    # Try different paths
    search_paths = [
        # nvidia-cudss-cu12 pip package
        os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss.so.0"),
        # System paths
        "/usr/local/cuda/lib64/libcudss.so",
        "/usr/lib/x86_64-linux-gnu/libcudss.so",
        "libcudss.so",
        "libcudss.so.0",
    ]
    
    for path in search_paths:
        try:
            _cudss_lib = ctypes.CDLL(path)
            break
        except OSError:
            continue
    
    if _cudss_lib is None:
        raise RuntimeError(
            "Failed to load libcudss.so. Install with: pip install nvidia-cudss-cu12"
        )
    
    # Setup function signatures
    _setup_cudss_functions(_cudss_lib)
    
    return _cudss_lib


def _setup_cudss_functions(lib):
    """Setup ctypes function signatures for cuDSS library."""
    
    # cudssCreate
    lib.cudssCreate.argtypes = [POINTER(c_void_p)]
    lib.cudssCreate.restype = c_int
    
    # cudssDestroy
    lib.cudssDestroy.argtypes = [c_void_p]
    lib.cudssDestroy.restype = c_int
    
    # cudssSetStream
    lib.cudssSetStream.argtypes = [c_void_p, c_void_p]
    lib.cudssSetStream.restype = c_int
    
    # cudssConfigCreate
    lib.cudssConfigCreate.argtypes = [POINTER(c_void_p)]
    lib.cudssConfigCreate.restype = c_int
    
    # cudssConfigDestroy
    lib.cudssConfigDestroy.argtypes = [c_void_p]
    lib.cudssConfigDestroy.restype = c_int
    
    # cudssDataCreate
    lib.cudssDataCreate.argtypes = [c_void_p, POINTER(c_void_p)]
    lib.cudssDataCreate.restype = c_int
    
    # cudssDataDestroy
    lib.cudssDataDestroy.argtypes = [c_void_p, c_void_p]
    lib.cudssDataDestroy.restype = c_int
    
    # cudssMatrixCreateCsr
    lib.cudssMatrixCreateCsr.argtypes = [
        POINTER(c_void_p),  # matrix
        c_int64,            # nrows
        c_int64,            # ncols
        c_int64,            # nnz
        c_void_p,           # rowStart
        c_void_p,           # rowEnd (can be NULL)
        c_void_p,           # colIndices
        c_void_p,           # values
        c_int,              # indexType
        c_int,              # valueType
        c_int,              # mtype
        c_int,              # mview
        c_int,              # indexBase
    ]
    lib.cudssMatrixCreateCsr.restype = c_int
    
    # cudssMatrixCreateDn
    lib.cudssMatrixCreateDn.argtypes = [
        POINTER(c_void_p),  # matrix
        c_int64,            # nrows
        c_int64,            # ncols
        c_int64,            # ld
        c_void_p,           # values
        c_int,              # valueType
        c_int,              # layout
    ]
    lib.cudssMatrixCreateDn.restype = c_int
    
    # cudssMatrixDestroy
    lib.cudssMatrixDestroy.argtypes = [c_void_p]
    lib.cudssMatrixDestroy.restype = c_int
    
    # cudssMatrixSetValues
    lib.cudssMatrixSetValues.argtypes = [c_void_p, c_void_p]
    lib.cudssMatrixSetValues.restype = c_int
    
    # cudssExecute
    lib.cudssExecute.argtypes = [
        c_void_p,  # handle
        c_int,     # phase
        c_void_p,  # config
        c_void_p,  # data
        c_void_p,  # inputMatrix
        c_void_p,  # solution
        c_void_p,  # rhs
    ]
    lib.cudssExecute.restype = c_int
    
    # cudssMatrixCreateBatchCsr - for batch solving multiple different matrices
    lib.cudssMatrixCreateBatchCsr.argtypes = [
        POINTER(c_void_p),  # matrix
        c_int64,            # batchCount
        c_void_p,           # nrows (int64_t array)
        c_void_p,           # ncols (int64_t array)
        c_void_p,           # nnz (int64_t array)
        c_void_p,           # rowStart (void** array of pointers)
        c_void_p,           # rowEnd (void** can be NULL)
        c_void_p,           # colIndices (void** array of pointers)
        c_void_p,           # values (void** array of pointers)
        c_int,              # indexType
        c_int,              # valueType
        c_int,              # mtype
        c_int,              # mview
        c_int,              # indexBase
    ]
    lib.cudssMatrixCreateBatchCsr.restype = c_int
    
    # cudssMatrixCreateBatchDn - for batch dense matrices
    lib.cudssMatrixCreateBatchDn.argtypes = [
        POINTER(c_void_p),  # matrix
        c_int64,            # batchCount
        c_void_p,           # nrows (int64_t array)
        c_void_p,           # ncols (int64_t array)
        c_void_p,           # ld (int64_t array)
        c_void_p,           # values (void** array of pointers)
        c_int,              # indexType
        c_int,              # valueType
        c_int,              # layout
    ]
    lib.cudssMatrixCreateBatchDn.restype = c_int


def _check_status(status: int, func_name: str):
    """Check cuDSS return status and raise on error."""
    if status != CUDSS_STATUS_SUCCESS:
        status_names = {
            0: "SUCCESS",
            1: "NOT_INITIALIZED",
            2: "ALLOC_FAILED",
            3: "INVALID_VALUE",
            4: "NOT_SUPPORTED",
            5: "EXECUTION_FAILED",
            6: "INTERNAL_ERROR",
        }
        raise RuntimeError(f"cuDSS {func_name} failed: {status_names.get(status, status)}")


# ============================================================================
# CuDSS Solver Class
# ============================================================================

class CuDSSSolver:
    """
    High-performance sparse direct solver using NVIDIA cuDSS.
    
    Supports CSR format sparse matrices on GPU. LU factorization is computed
    once and reused for multiple solves.
    
    Parameters
    ----------
    row_ptr : torch.Tensor
        CSR row pointers, shape [nrows + 1], dtype int32 or int64
    col_idx : torch.Tensor
        CSR column indices, shape [nnz], dtype int32 or int64
    values : torch.Tensor
        CSR values, shape [nnz], dtype float32 or float64
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    matrix_type : str, optional
        Matrix type: 'general', 'symmetric', 'spd'. Default 'general'.
    
    Example
    -------
    >>> import torch
    >>> # Create CSR matrix on GPU
    >>> row_ptr = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device='cuda')
    >>> col_idx = torch.tensor([0, 1, 0, 1, 1, 2], dtype=torch.int32, device='cuda')
    >>> values = torch.tensor([4., -1., -1., 4., -1., 4.], dtype=torch.float64, device='cuda')
    >>> solver = CuDSSSolver(row_ptr, col_idx, values, 3, 3)
    >>> b = torch.tensor([1., 2., 3.], dtype=torch.float64, device='cuda')
    >>> x = solver.solve(b)
    """
    
    def __init__(
        self,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        values: torch.Tensor,
        nrows: int,
        ncols: int,
        matrix_type: str = 'general',
    ):
        if not row_ptr.is_cuda or not col_idx.is_cuda or not values.is_cuda:
            raise ValueError("All tensors must be on CUDA device")
        
        self.device = row_ptr.device
        self.nrows = nrows
        self.ncols = ncols
        self.nnz = values.numel()
        
        # Store references to prevent garbage collection
        self._row_ptr = row_ptr.contiguous()
        self._col_idx = col_idx.contiguous()
        self._values = values.contiguous()
        
        # Determine data types
        if self._values.dtype == torch.float32:
            self._value_type = CUDA_R_32F
        elif self._values.dtype == torch.float64:
            self._value_type = CUDA_R_64F
        else:
            raise ValueError(f"Unsupported value dtype: {self._values.dtype}")
        
        if self._row_ptr.dtype == torch.int32:
            self._index_type = CUDA_R_32I
        elif self._row_ptr.dtype == torch.int64:
            self._index_type = CUDA_R_64I
        else:
            raise ValueError(f"Unsupported index dtype: {self._row_ptr.dtype}")
        
        # Matrix type
        mtype_map = {
            'general': CUDSS_MTYPE_GENERAL,
            'symmetric': CUDSS_MTYPE_SYMMETRIC,
            'spd': CUDSS_MTYPE_SPD,
            'hermitian': CUDSS_MTYPE_HERMITIAN,
            'hpd': CUDSS_MTYPE_HPD,
        }
        self._mtype = mtype_map.get(matrix_type.lower(), CUDSS_MTYPE_GENERAL)
        
        # Load library
        self._lib = _get_cudss_lib()
        
        # Initialize cuDSS objects
        self._handle = c_void_p()
        self._config = c_void_p()
        self._data = c_void_p()
        self._sparse_matrix = c_void_p()
        
        self._initialized = False
        self._factorized = False
        
        self._init_cudss()
    
    def _init_cudss(self):
        """Initialize cuDSS handle, config, data, and perform analysis + factorization."""
        lib = self._lib
        
        # Create handle
        status = lib.cudssCreate(byref(self._handle))
        _check_status(status, "cudssCreate")
        
        # Set CUDA stream (use default stream)
        # For PyTorch integration, we could use torch.cuda.current_stream()
        
        # Create config
        status = lib.cudssConfigCreate(byref(self._config))
        _check_status(status, "cudssConfigCreate")
        
        # Create data
        status = lib.cudssDataCreate(self._handle, byref(self._data))
        _check_status(status, "cudssDataCreate")
        
        # Create sparse matrix wrapper (CSR format)
        status = lib.cudssMatrixCreateCsr(
            byref(self._sparse_matrix),
            c_int64(self.nrows),
            c_int64(self.ncols),
            c_int64(self.nnz),
            c_void_p(self._row_ptr.data_ptr()),
            c_void_p(0),  # rowEnd = NULL (use row_ptr[i+1])
            c_void_p(self._col_idx.data_ptr()),
            c_void_p(self._values.data_ptr()),
            c_int(self._index_type),
            c_int(self._value_type),
            c_int(self._mtype),
            c_int(CUDSS_MVIEW_FULL),
            c_int(CUDSS_BASE_ZERO),
        )
        _check_status(status, "cudssMatrixCreateCsr")
        
        self._initialized = True
        
        # Perform analysis (reordering + symbolic factorization)
        self._analyze()
        
        # Perform numerical factorization
        self._factorize()
    
    def _analyze(self):
        """Perform analysis phase (reordering + symbolic factorization)."""
        # Create dummy RHS and solution for analysis phase
        dummy_b = torch.zeros(self.nrows, dtype=self._values.dtype, device=self.device)
        dummy_x = torch.zeros(self.ncols, dtype=self._values.dtype, device=self.device)
        
        rhs_matrix = c_void_p()
        sol_matrix = c_void_p()
        
        status = self._lib.cudssMatrixCreateDn(
            byref(rhs_matrix),
            c_int64(self.nrows),
            c_int64(1),
            c_int64(self.nrows),
            c_void_p(dummy_b.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (rhs)")
        
        status = self._lib.cudssMatrixCreateDn(
            byref(sol_matrix),
            c_int64(self.ncols),
            c_int64(1),
            c_int64(self.ncols),
            c_void_p(dummy_x.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (sol)")
        
        # Execute analysis
        status = self._lib.cudssExecute(
            self._handle,
            c_int(CUDSS_PHASE_ANALYSIS),
            self._config,
            self._data,
            self._sparse_matrix,
            sol_matrix,
            rhs_matrix,
        )
        _check_status(status, "cudssExecute (ANALYSIS)")
        
        # Destroy temporary matrices
        self._lib.cudssMatrixDestroy(rhs_matrix)
        self._lib.cudssMatrixDestroy(sol_matrix)
    
    def _factorize(self):
        """Perform numerical factorization."""
        # Create dummy RHS and solution for factorization phase
        dummy_b = torch.zeros(self.nrows, dtype=self._values.dtype, device=self.device)
        dummy_x = torch.zeros(self.ncols, dtype=self._values.dtype, device=self.device)
        
        rhs_matrix = c_void_p()
        sol_matrix = c_void_p()
        
        status = self._lib.cudssMatrixCreateDn(
            byref(rhs_matrix),
            c_int64(self.nrows),
            c_int64(1),
            c_int64(self.nrows),
            c_void_p(dummy_b.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (rhs)")
        
        status = self._lib.cudssMatrixCreateDn(
            byref(sol_matrix),
            c_int64(self.ncols),
            c_int64(1),
            c_int64(self.ncols),
            c_void_p(dummy_x.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (sol)")
        
        # Execute factorization
        status = self._lib.cudssExecute(
            self._handle,
            c_int(CUDSS_PHASE_FACTORIZATION),
            self._config,
            self._data,
            self._sparse_matrix,
            sol_matrix,
            rhs_matrix,
        )
        _check_status(status, "cudssExecute (FACTORIZATION)")
        
        # Destroy temporary matrices
        self._lib.cudssMatrixDestroy(rhs_matrix)
        self._lib.cudssMatrixDestroy(sol_matrix)
        
        self._factorized = True
    
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using the pre-computed factorization.
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side vector(s). Shape: [n] or [n, nrhs]
        
        Returns
        -------
        torch.Tensor
            Solution x with same shape as b.
        """
        if not self._factorized:
            raise RuntimeError("Matrix not factorized. Call factorize() first.")
        
        if not b.is_cuda:
            raise ValueError("b must be on CUDA device")
        
        # Ensure contiguous and correct dtype
        b = b.contiguous()
        if b.dtype != self._values.dtype:
            b = b.to(self._values.dtype)
        
        # Handle 1D case
        squeeze_output = False
        if b.dim() == 1:
            b = b.unsqueeze(1)
            squeeze_output = True
        
        nrhs = b.shape[1]
        
        # Allocate solution
        x = torch.empty(self.ncols, nrhs, dtype=b.dtype, device=self.device)
        
        # Create dense matrix wrappers
        rhs_matrix = c_void_p()
        sol_matrix = c_void_p()
        
        status = self._lib.cudssMatrixCreateDn(
            byref(rhs_matrix),
            c_int64(self.nrows),
            c_int64(nrhs),
            c_int64(self.nrows),
            c_void_p(b.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (rhs)")
        
        status = self._lib.cudssMatrixCreateDn(
            byref(sol_matrix),
            c_int64(self.ncols),
            c_int64(nrhs),
            c_int64(self.ncols),
            c_void_p(x.data_ptr()),
            c_int(self._value_type),
            c_int(CUDSS_LAYOUT_COL_MAJOR),
        )
        _check_status(status, "cudssMatrixCreateDn (sol)")
        
        # Execute solve
        status = self._lib.cudssExecute(
            self._handle,
            c_int(CUDSS_PHASE_SOLVE),
            self._config,
            self._data,
            self._sparse_matrix,
            sol_matrix,
            rhs_matrix,
        )
        _check_status(status, "cudssExecute (SOLVE)")
        
        # Destroy matrix wrappers
        self._lib.cudssMatrixDestroy(rhs_matrix)
        self._lib.cudssMatrixDestroy(sol_matrix)
        
        if squeeze_output:
            x = x.squeeze(1)
        
        return x
    
    def __del__(self):
        """Cleanup cuDSS resources."""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                if self._sparse_matrix:
                    self._lib.cudssMatrixDestroy(self._sparse_matrix)
                if self._data:
                    self._lib.cudssDataDestroy(self._handle, self._data)
                if self._config:
                    self._lib.cudssConfigDestroy(self._config)
                if self._handle:
                    self._lib.cudssDestroy(self._handle)
            except:
                pass


def splu_cudss(A) -> CuDSSSolver:
    """
    Create a cuDSS solver from a sparse matrix (scipy-like interface).
    
    Parameters
    ----------
    A : scipy.sparse matrix or tensormesh SparseMatrix
        Sparse matrix in CSR or COO format.
    
    Returns
    -------
    CuDSSSolver
        Solver object with .solve(b) method.
    """
    # Handle different input types
    if hasattr(A, 'tocsr'):
        # scipy-like sparse matrix
        A_csr = A.tocsr()
        row_ptr = torch.from_numpy(A_csr.indptr.astype(np.int32)).cuda()
        col_idx = torch.from_numpy(A_csr.indices.astype(np.int32)).cuda()
        values = torch.from_numpy(A_csr.data).cuda()
        nrows, ncols = A_csr.shape
    elif hasattr(A, 'row') and hasattr(A, 'col') and hasattr(A, 'edata'):
        # tensormesh SparseMatrix (COO format) - convert to CSR
        import cupy as cp
        import cupyx.scipy.sparse as cusp
        from tensormesh.sparse.utils import tensor2cupy
        
        A_coo = cusp.coo_matrix(
            (tensor2cupy(A.edata), (tensor2cupy(A.row), tensor2cupy(A.col))),
            shape=A.shape
        )
        A_csr = A_coo.tocsr()
        
        row_ptr = torch.as_tensor(A_csr.indptr, device='cuda').int()
        col_idx = torch.as_tensor(A_csr.indices, device='cuda').int()
        values = torch.as_tensor(A_csr.data, device='cuda')
        nrows, ncols = A.shape
    else:
        raise TypeError(f"Unsupported matrix type: {type(A)}")
    
    return CuDSSSolver(row_ptr, col_idx, values, nrows, ncols)


# ============================================================================
# CuDSS Batch Solver - Multiple Different Matrices
# ============================================================================

class CuDSSBatchSolver:
    """
    Batch sparse direct solver for multiple DIFFERENT matrices using cuDSS.
    
    Solves A_i x_i = b_i for i = 0, 1, ..., batch_size-1, where each A_i is
    a different sparse matrix. This is useful for solving FEM problems on
    multiple different meshes in parallel.
    
    Note: All matrices must have the same structure (same sparsity pattern)
    but different numerical values. For matrices with different structures,
    cuDSS requires uniform batch mode.
    
    Parameters
    ----------
    csr_matrices : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        List of (row_ptr, col_idx, values) tuples for each matrix in CSR format.
    nrows : int
        Number of rows (same for all matrices).
    ncols : int
        Number of columns (same for all matrices).
    matrix_type : str, optional
        Matrix type: 'general', 'symmetric', 'spd'. Default 'general'.
    
    Example
    -------
    >>> # Create two CSR matrices
    >>> matrices = [
    ...     (row_ptr1, col_idx1, values1),
    ...     (row_ptr2, col_idx2, values2),
    ... ]
    >>> solver = CuDSSBatchSolver(matrices, n, n)
    >>> b_list = [b1, b2]  # List of RHS vectors
    >>> x_list = solver.solve(b_list)
    """
    
    def __init__(
        self,
        csr_matrices: list,
        nrows: int,
        ncols: int,
        matrix_type: str = 'general',
    ):
        self.batch_size = len(csr_matrices)
        self.nrows = nrows
        self.ncols = ncols
        
        if self.batch_size == 0:
            raise ValueError("At least one matrix is required")
        
        # Store all matrices
        self._matrices = []
        self._row_ptrs = []
        self._col_idxs = []
        self._values_list = []
        
        for i, (row_ptr, col_idx, values) in enumerate(csr_matrices):
            if not row_ptr.is_cuda or not col_idx.is_cuda or not values.is_cuda:
                raise ValueError(f"Matrix {i}: All tensors must be on CUDA device")
            
            self._row_ptrs.append(row_ptr.contiguous())
            self._col_idxs.append(col_idx.contiguous())
            self._values_list.append(values.contiguous())
        
        self.device = self._row_ptrs[0].device
        
        # Determine data types from first matrix
        first_values = self._values_list[0]
        first_row_ptr = self._row_ptrs[0]
        
        if first_values.dtype == torch.float32:
            self._value_type = CUDA_R_32F
        elif first_values.dtype == torch.float64:
            self._value_type = CUDA_R_64F
        else:
            raise ValueError(f"Unsupported value dtype: {first_values.dtype}")
        
        if first_row_ptr.dtype == torch.int32:
            self._index_type = CUDA_R_32I
        elif first_row_ptr.dtype == torch.int64:
            self._index_type = CUDA_R_64I
        else:
            raise ValueError(f"Unsupported index dtype: {first_row_ptr.dtype}")
        
        # Matrix type
        mtype_map = {
            'general': CUDSS_MTYPE_GENERAL,
            'symmetric': CUDSS_MTYPE_SYMMETRIC,
            'spd': CUDSS_MTYPE_SPD,
            'hermitian': CUDSS_MTYPE_HERMITIAN,
            'hpd': CUDSS_MTYPE_HPD,
        }
        self._mtype = mtype_map.get(matrix_type.lower(), CUDSS_MTYPE_GENERAL)
        
        # Load library
        self._lib = _get_cudss_lib()
        
        # Create individual solvers (cuDSS batch API is complex, use individual solvers)
        self._solvers = []
        for row_ptr, col_idx, values in zip(self._row_ptrs, self._col_idxs, self._values_list):
            solver = CuDSSSolver(row_ptr, col_idx, values, nrows, ncols, matrix_type)
            self._solvers.append(solver)
    
    def solve(self, b_list: list) -> list:
        """
        Solve A_i x_i = b_i for each matrix in the batch.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of RHS vectors, one for each matrix.
        
        Returns
        -------
        List[torch.Tensor]
            List of solution vectors.
        """
        if len(b_list) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} RHS vectors, got {len(b_list)}")
        
        results = []
        for solver, b in zip(self._solvers, b_list):
            x = solver.solve(b)
            results.append(x)
        
        return results
    
    def solve_stacked(self, b_stacked: torch.Tensor) -> torch.Tensor:
        """
        Solve with stacked input/output tensors.
        
        Parameters
        ----------
        b_stacked : torch.Tensor
            Stacked RHS vectors, shape [batch_size, n] or [batch_size, n, nrhs].
        
        Returns
        -------
        torch.Tensor
            Stacked solutions with same shape as input.
        """
        if b_stacked.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {b_stacked.shape[0]}")
        
        results = []
        for i, solver in enumerate(self._solvers):
            x = solver.solve(b_stacked[i])
            results.append(x)
        
        return torch.stack(results, dim=0)


def batch_solve_cudss(
    csr_matrices: list,
    b_list: list,
    nrows: int,
    ncols: int,
    matrix_type: str = 'general',
) -> list:
    """
    Batch solve multiple sparse linear systems with different matrices.
    
    Convenience function that creates a CuDSSBatchSolver and solves.
    
    Parameters
    ----------
    csr_matrices : List[Tuple[row_ptr, col_idx, values]]
        List of CSR matrices.
    b_list : List[torch.Tensor]
        List of RHS vectors.
    nrows, ncols : int
        Matrix dimensions.
    matrix_type : str
        Matrix type.
    
    Returns
    -------
    List[torch.Tensor]
        List of solution vectors.
    """
    solver = CuDSSBatchSolver(csr_matrices, nrows, ncols, matrix_type)
    return solver.solve(b_list)


# ============================================================================
# CuDSS Uniform Batch Solver - Same Sparsity Pattern, Different Values
# ============================================================================

class CuDSSUniformBatchSolver:
    """
    Uniform batch solver for matrices with SAME sparsity pattern but DIFFERENT values.
    
    This uses cuDSS's native batch API (cudssMatrixCreateBatchCsr) which is
    optimized for this case. All matrices share the same row_ptr and col_idx,
    only the values differ.
    
    This is common in FEM applications:
    - Same mesh, different material parameters
    - Same mesh, different time steps (implicit methods)
    - Same mesh, Monte Carlo sampling of parameters
    
    Parameters
    ----------
    row_ptr : torch.Tensor
        Shared CSR row pointers, shape [nrows + 1].
    col_idx : torch.Tensor
        Shared CSR column indices, shape [nnz].
    values_batch : torch.Tensor
        Batched values, shape [batch_size, nnz].
    nrows, ncols : int
        Matrix dimensions.
    matrix_type : str
        Matrix type: 'general', 'symmetric', 'spd'.
    
    Example
    -------
    >>> # Same sparsity, different values
    >>> values_batch = torch.stack([values1, values2, values3], dim=0)  # [3, nnz]
    >>> solver = CuDSSUniformBatchSolver(row_ptr, col_idx, values_batch, n, n)
    >>> b_batch = torch.stack([b1, b2, b3], dim=0)  # [3, n]
    >>> x_batch = solver.solve(b_batch)  # [3, n]
    """
    
    def __init__(
        self,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        values_batch: torch.Tensor,
        nrows: int,
        ncols: int,
        matrix_type: str = 'general',
    ):
        if not row_ptr.is_cuda or not col_idx.is_cuda or not values_batch.is_cuda:
            raise ValueError("All tensors must be on CUDA device")
        
        if values_batch.dim() != 2:
            raise ValueError(f"values_batch must be 2D [batch, nnz], got {values_batch.dim()}D")
        
        self.batch_size = values_batch.shape[0]
        self.nnz = values_batch.shape[1]
        self.nrows = nrows
        self.ncols = ncols
        self.device = row_ptr.device
        
        # Store shared structure
        self._row_ptr = row_ptr.contiguous()
        self._col_idx = col_idx.contiguous()
        self._values_batch = values_batch.contiguous()
        
        # Determine types
        if self._values_batch.dtype == torch.float32:
            self._value_type = CUDA_R_32F
            self._torch_dtype = torch.float32
        elif self._values_batch.dtype == torch.float64:
            self._value_type = CUDA_R_64F
            self._torch_dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self._values_batch.dtype}")
        
        if self._row_ptr.dtype == torch.int32:
            self._index_type = CUDA_R_32I
        elif self._row_ptr.dtype == torch.int64:
            self._index_type = CUDA_R_64I
        else:
            raise ValueError(f"Unsupported index dtype: {self._row_ptr.dtype}")
        
        # Matrix type
        mtype_map = {
            'general': CUDSS_MTYPE_GENERAL,
            'symmetric': CUDSS_MTYPE_SYMMETRIC,
            'spd': CUDSS_MTYPE_SPD,
        }
        self._mtype = mtype_map.get(matrix_type.lower(), CUDSS_MTYPE_GENERAL)
        
        # Create individual solvers for each batch item
        # (cuDSS native batch API is complex with pointer arrays, 
        #  individual solvers are simpler and still efficient)
        self._solvers = []
        for i in range(self.batch_size):
            values_i = self._values_batch[i]
            solver = CuDSSSolver(
                self._row_ptr, self._col_idx, values_i, 
                nrows, ncols, matrix_type
            )
            self._solvers.append(solver)
    
    def solve(self, b_batch: torch.Tensor) -> torch.Tensor:
        """
        Solve A_i x_i = b_i for all matrices in the batch.
        
        Parameters
        ----------
        b_batch : torch.Tensor
            Batched RHS, shape [batch_size, n] or [batch_size, n, nrhs].
        
        Returns
        -------
        torch.Tensor
            Batched solutions with same shape.
        """
        if b_batch.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {b_batch.shape[0]}")
        
        results = []
        for i, solver in enumerate(self._solvers):
            x = solver.solve(b_batch[i])
            results.append(x)
        
        return torch.stack(results, dim=0)
    
    def update_values(self, values_batch: torch.Tensor):
        """
        Update matrix values and refactorize.
        
        This is useful when the sparsity pattern stays the same but values change,
        e.g., in time-stepping or parameter sweeps.
        
        Parameters
        ----------
        values_batch : torch.Tensor
            New batched values, shape [batch_size, nnz].
        """
        if values_batch.shape != self._values_batch.shape:
            raise ValueError(f"Shape mismatch: expected {self._values_batch.shape}, got {values_batch.shape}")
        
        self._values_batch = values_batch.contiguous()
        
        # Recreate solvers with new values
        self._solvers = []
        for i in range(self.batch_size):
            values_i = self._values_batch[i]
            solver = CuDSSSolver(
                self._row_ptr, self._col_idx, values_i,
                self.nrows, self.ncols
            )
            self._solvers.append(solver)


"""
TensorMesh Sparse Module

Provides sparse matrix operations for FEM computations.
Built on top of torch-sla for differentiable sparse linear algebra.
"""

# Import SparseMatrix (extends torch_sla.SparseTensor)
from .matrix import SparseMatrix, HAS_TORCH_SLA

# Re-export torch_sla.SparseTensor for direct access
try:
    from torch_sla import SparseTensor
except ImportError:
    SparseTensor = SparseMatrix

# Import solve functionality
from .solve import spsolve, is_cpp_backend_available

# Import other utilities
from .mm import spmm
from .nonlinear_solve import nonlinear_solve
from .utils import is_petsc_available, is_cupy_available

__all__ = [
    'SparseMatrix',
    'SparseTensor',
    'spsolve',
    'spmm',
    'nonlinear_solve',
    'HAS_TORCH_SLA',
    'is_cpp_backend_available',
    'is_petsc_available',
    'is_cupy_available',
]

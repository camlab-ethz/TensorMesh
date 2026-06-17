"""Sparse matrix operations for FEM computations.

Built on top of ``torch-sla`` for differentiable sparse linear algebra.
``torch-sla`` is a hard dependency: import of this module fails if it is
not installed (see ``tensormesh.sparse.matrix``).

Solver entry points
-------------------

The canonical path is :class:`SparseMatrix` (a subclass of
``torch_sla.SparseTensor``): assembly returns one, and ``K.solve(b)`` /
``K.nonlinear_solve(residual, u0, *params)`` cover every workflow in
the user guide. Symmetry / positive-definiteness is auto-detected by
``torch-sla``, so no ``is_spd`` hint is needed.

The free functions :func:`spsolve` and :func:`nonlinear_solve`
re-exported here are **legacy entry points scheduled for removal**.
They pre-date the ``torch-sla`` integration and exist only for the
case where a caller holds raw COO arrays / wants to plug in a custom
Jacobian closure. New code should use the :class:`SparseMatrix` methods
instead — see :doc:`/user_guide/linear_solvers`.
"""

from torch_sla import SparseTensor

from .matrix import SparseMatrix
from .solve import spsolve
from .mm import spmm
from .nonlinear_solve import nonlinear_solve
from .utils import is_petsc_available, is_cupy_available

__all__ = [
    'SparseMatrix',
    'SparseTensor',
    'spsolve',
    'spmm',
    'nonlinear_solve',
    'is_petsc_available',
    'is_cupy_available',
]

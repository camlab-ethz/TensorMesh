"""Boundary-condition operators.

Exposes :class:`Condenser`, which applies Dirichlet boundary conditions to an
assembled FEM system via static condensation, and :class:`BlochReducer`, the
periodic counterpart that ties opposite unit-cell faces with a wavevector-
dependent Floquet phase (band structure / Bloch-periodic solves). See the class
docstrings and the User Guide chapter on boundary conditions.
"""

from .condense import Condenser
from .bloch import BlochReducer

__all__ = ["Condenser", "BlochReducer"]

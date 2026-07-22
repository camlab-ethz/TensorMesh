"""Boundary-condition operators.

Exposes :class:`Condenser`, which applies Dirichlet boundary conditions to an
assembled FEM system via static condensation, and :class:`BlochReducer`, the
periodic counterpart that ties opposite unit-cell faces with a wavevector-
dependent Floquet phase (band structure / Bloch-periodic solves); plus
:func:`robin_operator` / :func:`port_source`, the assembled boundary matrix and
load of first-order absorbing / plane-wave-port conditions for wave problems.
See the class docstrings and the User Guide chapter on boundary conditions.
"""

from .condense import Condenser
from .bloch import BlochReducer
from .boundary import robin_operator, port_source

__all__ = ["Condenser", "BlochReducer", "robin_operator", "port_source"]

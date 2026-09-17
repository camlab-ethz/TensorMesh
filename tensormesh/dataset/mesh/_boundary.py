"""Boundary-mask helpers shared by the ``gen_*`` mesh generators.

``is_boundary`` comes from :meth:`tensormesh.Mesh.topological_boundary_mask`
— facet incidence, exact for any geometry and any element order. The
per-side / per-surface masks (``is_left_boundary``, ``is_inner_boundary``,
…) are coordinate predicates **restricted to boundary points**, evaluated
with a tolerance scaled by the mesh size instead of exact float
comparison: an ``x == 0`` or ``r == R`` test misses nodes that sit one ulp
off the boundary curve (gmsh's curved-boundary nodes routinely do), which
silently drops Dirichlet conditions on those nodes.
"""
from typing import Dict

import torch

from ...mesh import Mesh

#: Coordinate tolerance as a fraction of ``chara_length`` — far above
#: floating-point noise, far below any distance between distinct nodes.
TOLERANCE_FACTOR = 1e-6


def boundary_tolerance(chara_length: float) -> float:
    """Absolute coordinate tolerance used by the side predicates."""
    return TOLERANCE_FACTOR * chara_length


def near(x: torch.Tensor, value: float, tol: float) -> torch.Tensor:
    """``|x - value| <= tol`` elementwise."""
    return (x - value).abs() <= tol


def within(x: torch.Tensor, lo: float, hi: float, tol: float) -> torch.Tensor:
    """``lo - tol <= x <= hi + tol`` elementwise."""
    return (x >= lo - tol) & (x <= hi + tol)


def register_boundary_masks(mesh: Mesh, **sides: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Register ``is_boundary`` (topological) plus every ``sides`` mask
    intersected with it, in the given order; return the registered masks."""
    is_boundary = mesh.topological_boundary_mask()
    masks = {"is_boundary": is_boundary}
    mesh.register_point_data("is_boundary", is_boundary)
    for key, predicate in sides.items():
        masks[key] = is_boundary & predicate
        mesh.register_point_data(key, masks[key])
    return masks

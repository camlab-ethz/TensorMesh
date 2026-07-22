r"""Boundary operators for wave problems — Robin / impedance / absorbing ports.

Built on :class:`~tensormesh.FacetBilinearAssembler`, these helpers produce the
boundary **matrix** and **load** of a first-order absorbing / plane-wave-port
condition for the scalar Helmholtz equation.

For an outgoing/absorbing (Sommerfeld) boundary and an incident plane wave
:math:`u_{\text{inc}}` the condition

.. math::

    \frac{\partial u}{\partial n} + i k\, u = 2 i k\, u_{\text{inc}}
    \quad\text{on } \Gamma_{\text{port}}

contributes, in the weak form,

* to the **operator**: :math:`+\,i k \int_\Gamma N_i N_j\,\mathrm dS = i k\,\mathbf B`
  — see :func:`robin_operator`;
* to the **right-hand side**: :math:`+\,2 i k \int_\Gamma u_{\text{inc}} N_i\,\mathrm dS`
  — see :func:`port_source`.

Recipe for a driven, non-reflecting Helmholtz solve::

    A = K - k**2 * M + 1j*k * robin_operator(mesh, port_and_open_boundaries)
    b =               2j*k * port_source(mesh, inlet, incident=1.0)
    u = A.solve(b)

Dropping the incident term (``port_source``) leaves a purely absorbing (open)
boundary; a wall is simply a boundary left out of ``robin_operator`` (natural
Neumann / sound-hard).
"""
from __future__ import annotations

from typing import Optional, Union

import torch

from ..assemble.facet_bilinear import FacetBilinearAssembler
from ..mesh import Mesh
from ..sparse.matrix import SparseMatrix


class _RobinMass(FacetBilinearAssembler):
    """Boundary mass ``\\int_\\Gamma c\\, N_i N_j`` (coefficient ``c`` from point_data)."""
    def forward(self, u, v, c):
        return c * u * v


def robin_operator(mesh: Mesh,
                   boundary_mask: Optional[Union[str, torch.Tensor]] = None,
                   coeff: Union[float, complex, torch.Tensor] = 1.0,
                   *,
                   points: Optional[torch.Tensor] = None,
                   quadrature_order: int = 2,
                   ) -> SparseMatrix:
    r"""Boundary (Robin / impedance) matrix :math:`\int_\Gamma c\, N_i N_j\,\mathrm dS`.

    Parameters
    ----------
    mesh : Mesh
    boundary_mask : str, torch.Tensor, or None
        Which boundary to integrate over (a per-node boolean tensor, a named
        mask, or ``None`` for the full boundary) — passed to
        ``FacetBilinearAssembler.from_mesh``.
    coeff : float, complex, or torch.Tensor
        Impedance/absorption coefficient :math:`c`; a scalar (e.g. ``1j*k`` for a
        first-order absorbing boundary) or a nodal ``[n_points]`` field.
    points : torch.Tensor, optional
        Node coordinates (defaults to ``mesh.points``).
    quadrature_order : int, optional
        Facet quadrature order (default ``2``).

    Returns
    -------
    SparseMatrix
        The ``[n_points, n_points]`` boundary matrix.
    """
    pts = mesh.points if points is None else points
    n = pts.shape[0]
    if not torch.is_tensor(coeff):
        # scalar coefficient: follow the mesh precision (float64 mesh ->
        # complex128 / float64 coefficient) instead of torch's float32 default
        if isinstance(coeff, complex):
            cdtype = torch.complex128 if pts.dtype == torch.float64 else torch.complex64
        else:
            cdtype = pts.dtype
        c = torch.full((n,), coeff, dtype=cdtype, device=pts.device)
    else:
        c = coeff
    asm = _RobinMass.from_mesh(mesh, boundary_mask=boundary_mask,
                               quadrature_order=quadrature_order)
    return asm(pts, point_data={"c": c})


def port_source(mesh: Mesh,
                boundary_mask: Optional[Union[str, torch.Tensor]] = None,
                incident: Union[float, complex, torch.Tensor] = 1.0,
                *,
                points: Optional[torch.Tensor] = None,
                quadrature_order: int = 2,
                ) -> torch.Tensor:
    r"""Port load vector :math:`\int_\Gamma u_{\text{inc}}\, N_i\,\mathrm dS`.

    This is the consistent boundary load of an incident field ``incident`` (a
    scalar amplitude for a uniform plane-wave port, or a nodal ``[n_points]``
    field for a shaped one).  Multiply by ``2j*k`` for the plane-wave-port RHS.

    Returns
    -------
    torch.Tensor
        A ``[n_points]`` load vector (nonzero only on the selected boundary).
    """
    pts = mesh.points if points is None else points
    n = pts.shape[0]
    B = robin_operator(mesh, boundary_mask, 1.0, points=pts,
                       quadrature_order=quadrature_order)
    if not torch.is_tensor(incident):
        inc = torch.full((n,), incident, dtype=B.values.dtype)
    else:
        inc = incident.to(B.values.dtype)
    return B @ inc          # int_Gamma incident * N_i  (Sum_j N_j = 1 recovers int N_i)

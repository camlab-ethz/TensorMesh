r"""Stretched-coordinate Perfectly Matched Layer (PML) coefficients.

A PML makes a truncated FEM domain behave as if it were open: outgoing waves
are absorbed in a thin frame instead of reflecting off the outer boundary.  In
the frequency domain this is realized by an analytic **complex coordinate
stretch** :math:`x \to \tilde x` with :math:`\partial\tilde x/\partial x = s_x`,
which turns the scalar Helmholtz operator into an anisotropic one:

.. math::

    \nabla\cdot(\mathbf\Lambda\,\nabla u) + k_0^2\, \varepsilon\, s_{\mathrm{prod}}\, u = f ,
    \qquad
    \mathbf\Lambda = \operatorname{diag}\!\Big(\tfrac{s_y s_z}{s_x},\,
                                               \tfrac{s_x s_z}{s_y},\,
                                               \tfrac{s_x s_y}{s_z}\Big),
    \quad s_{\mathrm{prod}} = s_x s_y s_z ,

so it plugs straight into :class:`~tensormesh.AnisotropicLaplaceElementAssembler`
(the ``\mathbf\Lambda`` tensor) plus a mass term scaled by
:math:`k_0^2 \varepsilon\, s_{\mathrm{prod}}`.

:func:`cartesian_pml` returns those two nodal fields for a rectangular PML frame.

    Lam, sprod = cartesian_pml(mesh.points, bounds=[(0, L), (0, L)], thickness=d)
    K = AnisotropicLaplaceElementAssembler.from_mesh(mesh)(pts, point_data={"A": Lam})
    #  M scaled by k0**2 * eps_r * sprod  (e.g. ScaledMassElementAssembler)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch


def cartesian_pml(points: torch.Tensor,
                  *,
                  bounds: Sequence[Tuple[float, float]],
                  thickness: Union[float, Sequence[float]],
                  strength: float = 5.0,
                  order: int = 2,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Complex coordinate-stretch tensor field for a Cartesian PML frame.

    The stretch on each axis is unity in the interior and ramps into the PML as
    :math:`s(\xi) = 1 - i\,\sigma\,\xi^{p}`, where :math:`\xi \in [0, 1]` is the
    normalized depth into the PML, :math:`\sigma` is ``strength`` and :math:`p`
    is ``order``.  Larger ``strength`` / thicker layers absorb more strongly;
    too steep a ramp for the mesh re-introduces discretization reflection.

    Parameters
    ----------
    points : torch.Tensor
        Nodal coordinates, shape :math:`[|\mathcal V|, D]` with ``D`` in ``{1,2,3}``.
    bounds : sequence of (lo, hi)
        Outer domain extent per axis; the PML occupies ``[lo, lo+thickness]`` and
        ``[hi-thickness, hi]`` on every axis.
    thickness : float or sequence of float
        PML thickness, scalar (same on all axes) or one value per axis.
    strength : float, optional
        Imaginary stretch amplitude :math:`\sigma`; default ``5.0``.
    order : int, optional
        Polynomial ramp order :math:`p`; default ``2``.

    Returns
    -------
    Lambda : torch.Tensor
        Complex diagonal stretch tensor, shape :math:`[|\mathcal V|, D, D]`, for
        :class:`~tensormesh.AnisotropicLaplaceElementAssembler`.
    s_prod : torch.Tensor
        Complex product :math:`\prod_a s_a`, shape :math:`[|\mathcal V|]`, the
        mass-term scaling.
    """
    if points.dim() != 2:
        raise ValueError(f"points must be [N, D], got {tuple(points.shape)}")
    n, dim = points.shape
    if len(bounds) != dim:
        raise ValueError(f"bounds must have {dim} (lo, hi) pairs, got {len(bounds)}")
    if isinstance(thickness, (int, float)):
        thickness = [float(thickness)] * dim
    dev = points.device

    s_list: List[torch.Tensor] = []
    for a in range(dim):
        lo, hi = bounds[a]
        d = float(thickness[a])
        t = points[:, a].to(torch.float64)
        below = torch.clamp((lo + d - t) / d, min=0.0, max=1.0)   # ramp near lo
        above = torch.clamp((t - (hi - d)) / d, min=0.0, max=1.0)  # ramp near hi
        xi = below + above                                        # 0 in the interior
        s_list.append(1.0 - 1j * strength * xi ** order)
    s = torch.stack(s_list, dim=1).to(torch.complex128).to(dev)   # [N, D]
    s_prod = torch.prod(s, dim=1)                                  # [N]

    Lambda = torch.zeros(n, dim, dim, dtype=torch.complex128, device=dev)
    for a in range(dim):
        # diag_a = (prod_{b != a} s_b) / s_a = s_prod / s_a^2
        Lambda[:, a, a] = s_prod / (s[:, a] * s[:, a])
    return Lambda, s_prod

"""Conforming rational basis of the pyramid (``PyramidBasisFns``).

On TensorMesh's reference pyramid (base [0, 1]^2, apex (0, 0, 1),
cross-section [0, 1 - z]^2) the space is written in the apex-centred
coordinates x_s = 2x + z - 1, y_s = 2y + z - 1. At order 1 it is the classical
five-node pyramid, phi_3 = xy / (1 - z) and so on. Under the collapse
(x, y, z) = ((1 - t) u, (1 - t) v, t) every function of the space, and every
reference gradient, is a polynomial in (u, v, t), which is why the pyramid
quadrature (``pyr_quadrature``) collapses the unit cube.
"""
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn


def pyramid_exponents(order: int) -> torch.Tensor:
    r"""Exponents of the generators of :math:`B_r`.

    Row :math:`(p, q, c, m)` stands for
    :math:`x_s^p\, y_s^q\, z^c\, (1 - z)^{-m}`: first the monomials of
    :math:`P_r` (:math:`m = 0`), then
    :math:`(x_s y_s / (1 - z))^m\, x_s^a y_s^b` for :math:`1 \le m \le r`,
    :math:`a + b \le r - m`.

    Parameters
    ----------
    order : int
        The order :math:`r \ge 1`.

    Returns
    -------
    torch.Tensor
        float64 tensor of shape :math:`[N_b, 4]` with
        :math:`N_b = (r + 1)(r + 2)(2r + 3) / 6`.
    """
    assert order >= 1, f"order should be at least 1, but got {order}"
    exps = [(a, b, c, 0) for c in range(order + 1)
            for b in range(order + 1 - c) for a in range(order + 1 - c - b)]
    exps += [(a + m, b + m, 0, m) for m in range(1, order + 1)
             for b in range(order + 1 - m) for a in range(order + 1 - m - b)]
    return torch.tensor(exps, dtype=torch.float64)


class PyramidBasisFns(nn.Module):
    r"""Nodal basis of the conforming pyramid space :math:`B_r` (or its gradient).

    Returned by :meth:`tensormesh.Pyramid.get_basis_fns` (and so by
    :meth:`~tensormesh.Element.get_basis_grad_fns` for a pyramid) in place
    of a :class:`~tensormesh.element.Polynomials`, with the same evaluation
    interface: :meth:`map` evaluates every function at every point,
    :meth:`forward` pairs function :math:`i` with point :math:`i`, and
    :meth:`grad` gives the reference gradient.

    The space is the rational space of Bergot, Cohen and Duruflé
    (*Higher-order finite elements for hybrid meshes using new nodal
    pyramidal elements*, J. Sci. Comput., 2010),

    .. math::

        B_r = P_r \oplus \operatorname{span}\Big\{
              \Big(\frac{x_s\, y_s}{1 - z}\Big)^m x_s^a\, y_s^b
              \;:\; 1 \le m \le r,\ a + b \le r - m \Big\},

    in the apex-centred coordinates :math:`x_s = 2x + z - 1`,
    :math:`y_s = 2y + z - 1`. It contains :math:`P_r`, its traces are
    :math:`P_r` on the triangular faces and :math:`Q_r` on the base, so
    pyramids conform to neighbouring tetrahedra and hexahedra, which no
    polynomial space allows.

    Each function is :math:`\phi_j = \sum_k g_k C_{kj}`, where :math:`g_k` are
    the generators listed by ``pyramid_exponents`` and :math:`C` inverts their
    values at the nodes. The generators with :math:`m \ge 1` vanish at the
    apex, where they are evaluated as their limit :math:`0`. Their gradients
    have no limit there, so they are evaluated as :math:`0` too; quadrature
    points never sit on the apex.

    Parameters
    ----------
    exponents : torch.Tensor
        :math:`[N_g, 4]` generator exponents, see ``pyramid_exponents``.
    coef : torch.Tensor
        :math:`[N_g, N_b]` coefficients of the functions in the generators.
    is_grad : bool, optional
        Evaluate the reference gradients instead of the values.

    Examples
    --------
    .. code-block:: python

        import torch
        from tensormesh import Pyramid

        phi = Pyramid.get_basis_fns(order=1, dtype=torch.float64)
        phi.map(torch.tensor([[0.25, 0.25, 0.5]]))   # [1, 5], sums to 1
        phi.grad().map(torch.tensor([[0.25, 0.25, 0.5]])).shape
        # torch.Size([1, 3, 5])
    """

    def __init__(self, exponents: torch.Tensor, coef: torch.Tensor, is_grad: bool = False):
        super().__init__()
        self.register_buffer("exponents", exponents)
        self.register_buffer("coef", coef)
        self.is_grad = is_grad

    @classmethod
    def from_nodes(cls, nodes: torch.Tensor, order: int) -> "PyramidBasisFns":
        r"""The nodal basis of :math:`B_r` on ``nodes``.

        Parameters
        ----------
        nodes : torch.Tensor
            :math:`[N_b, 3]` reference coordinates of the nodes, e.g.
            :meth:`tensormesh.Element.get_basis` of the pyramid.
        order : int
            The order :math:`r`.

        Returns
        -------
        PyramidBasisFns
            float64 basis whose function :math:`j` is 1 at node :math:`j`
            and 0 at the others.
        """
        exponents = pyramid_exponents(order)
        assert exponents.shape[0] == nodes.shape[0], \
            f"B_{order} has {exponents.shape[0]} functions but {nodes.shape[0]} nodes were given"
        generators = cls(exponents, torch.eye(exponents.shape[0], dtype=torch.float64))
        vandermonde = generators.map(nodes.to(torch.float64))   # [n_node, n_generator]
        return cls(exponents, torch.linalg.inv(vandermonde))

    @property
    def shape(self) -> Tuple[int, ...]:
        n_basis = self.coef.shape[1]
        return (3, n_basis) if self.is_grad else (n_basis,)

    @property
    def n_vars(self) -> int:
        return 3

    def numel(self) -> int:
        return math.prod(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        # one callable per function, as for Polynomials (used by plot_3d)
        assert not self.is_grad, "iterate over the basis, not its gradient"
        return iter([partial(self._component, j) for j in range(len(self))])

    def _component(self, j: int, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)[..., j]

    def extra_repr(self) -> str:
        return f"shape={self.shape}, is_grad={self.is_grad}"

    def _apply(self, fn, recurse: bool = True):
        # keep the buffers real when the enclosing module is cast to a
        # complex dtype, like Polynomials
        def _real_only(t):
            out = fn(t)
            if out.is_complex():
                out = out.real.to(torch.float32 if out.dtype == torch.complex64 else torch.float64)
            return out
        return super()._apply(_real_only, recurse=recurse)

    def grad(self) -> "PyramidBasisFns":
        """The reference gradient, of shape :math:`[3, N_b]`."""
        assert not self.is_grad, "second derivatives are not implemented"
        return PyramidBasisFns(self.exponents, self.coef, is_grad=True)

    def _generators(self, x: torch.Tensor) -> torch.Tensor:
        """Generators (or their gradients) at ``x`` [N, 3]: [N, N_g] (or [N, 3, N_g])."""
        p, q, c, m = self.exponents.to(x.dtype).unbind(-1)                 # [N_g] each
        xs = (2 * x[:, 0] + x[:, 2] - 1)[:, None]                          # [N, 1]
        ys = (2 * x[:, 1] + x[:, 2] - 1)[:, None]
        z = x[:, 2:3]
        inv_s = 1 / (1 - z).clamp(min=torch.finfo(x.dtype).eps)            # 1 / (1 - z), finite at the apex
        if not self.is_grad:
            return xs**p * ys**q * z**c * inv_s**m
        # k * t**(k - 1), with the k = 0 term read as 0
        d_xs = p * xs**(p - 1).clamp(min=0) * ys**q * z**c * inv_s**m
        d_ys = q * xs**p * ys**(q - 1).clamp(min=0) * z**c * inv_s**m
        d_z = (c * z**(c - 1).clamp(min=0) + m * z**c * inv_s) * xs**p * ys**q * inv_s**m
        # chain rule through x_s = 2x + z - 1, y_s = 2y + z - 1
        return torch.stack([2 * d_xs, 2 * d_ys, d_xs + d_ys + d_z], 1)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Every function at every point.

        Parameters
        ----------
        x : torch.Tensor
            Reference points of shape :math:`[\ldots, 3]`.

        Returns
        -------
        torch.Tensor
            Shape :math:`[\ldots, N_b]`, or :math:`[\ldots, 3, N_b]` for the
            gradient.
        """
        lead = x.shape[:-1]
        g = self._generators(x.reshape(-1, 3))
        return (g @ self.coef.to(g.dtype)).reshape(*lead, *self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Each function at its own point, broadcasting like :class:`~tensormesh.element.Polynomials`.

        Parameters
        ----------
        x : torch.Tensor
            Points of shape :math:`[\ldots, *\mathrm{shape}, 3]`, or a single
            point of shape :math:`[3]`, which is used for every function.

        Returns
        -------
        torch.Tensor
            Shape :math:`[\ldots, *\mathrm{shape}]`.
        """
        shape = torch.broadcast_shapes(x.shape[:-1], self.shape)
        values = self.map(x.expand(*shape, 3))                             # [*shape, *self.shape]
        n = self.numel()
        return values.reshape(-1, n, n).diagonal(dim1=1, dim2=2).reshape(shape)

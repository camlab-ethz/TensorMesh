r"""Facet **bilinear** assembler — boundary matrices for Robin/impedance/port BCs.

:class:`~tensormesh.FacetAssembler` integrates a *linear* form over the boundary
and returns a node **vector** (tractions, loads, energy).  Robin / impedance /
absorbing / plane-wave-port conditions instead need a boundary *bilinear* form

.. math::

    B_{ij} = \int_{\Gamma} c(\mathbf x)\, N_i N_j \, \mathrm dS

i.e. a boundary **mass matrix** (optionally with a coefficient), which adds to
the volume operator.  :class:`FacetBilinearAssembler` fills exactly that gap: its
``forward(u, v, ...)`` is evaluated per basis pair (like
:class:`~tensormesh.ElementAssembler`, but over facets) and the result is
scattered into a :class:`~tensormesh.sparse.matrix.SparseMatrix`.

It shares all of :class:`FacetAssembler`'s topology construction
(``from_mesh`` / ``from_elements``); only the assembly returns a matrix.
"""
from __future__ import annotations

import inspect
from typing import Callable, Mapping, Optional

import torch

from .facet_assembler import FacetAssembler
from ..sparse.matrix import SparseMatrix
from ..vmap import vmap


class FacetBilinearAssembler(FacetAssembler):
    r"""Assemble a bilinear form over boundary facets into a sparse matrix.

    Override ``forward`` with the integrand of a boundary bilinear form at a
    single facet-quadrature point and basis pair — e.g. ``u * v`` for a boundary
    mass matrix, or ``c * u * v`` with a nodal coefficient ``c`` (impedance,
    radiation).  ``gradu`` / ``gradv`` and any ``point_data`` key are available
    just as in :class:`~tensormesh.ElementAssembler`.

    Calling the assembler returns a :class:`~tensormesh.sparse.matrix.SparseMatrix`
    of shape ``[n_points, n_points]``.

    Examples
    --------
    Boundary mass matrix on the selected boundary:

    .. code-block:: python

        class BoundaryMass(FacetBilinearAssembler):
            def forward(self, u, v):
                return u * v

        B = BoundaryMass.from_mesh(mesh, boundary_mask=inlet)(mesh.points)
    """

    __autodoc__ = ["__call__", "forward"]

    def __call__(self, points: Optional[torch.Tensor] = None,
                 func: Optional[Callable] = None,
                 point_data: Optional[Mapping[str, torch.Tensor]] = None,
                 ) -> SparseMatrix:
        r"""Integrate the facet bilinear form; return a ``[N, N]`` SparseMatrix."""
        if point_data is None:
            point_data = {}
        point_data = dict(point_data)

        if points is not None:
            self = self.type(points.dtype).to(points.device)
            for et in self.element_types:
                self.transformation[et].update_points(points)
        else:
            points = next(iter(self.transformation.values())).points  # type: ignore
        point_data["x"] = points  # type: ignore
        for key, value in point_data.items():
            assert value.shape[0] == self.n_points, \
                f"point_data['{key}'] must be [n_points, ...], got {tuple(value.shape)}"

        fn = self.forward if func is None else func
        signature = inspect.signature(fn)

        # Every facet argument carries a leading (facet, quadrature) pair, so both
        # of those vmap levels map dim 0 for *all* args; only the two basis levels
        # differ.  in_dims per arg at (facet, quad, u_basis, v_basis):
        def _dims(key):
            if key in ("u", "gradu"):
                return (0, 0, 0, None)
            if key in ("v", "gradv"):
                return (0, 0, None, 0)
            if key in point_data or (key.startswith("grad") and key[4:] in point_data):
                return (0, 0, None, None)             # coefficient: facet+quad only
            raise ValueError(f"key {key} is not supported (use u/v/gradu/gradv "
                             f"or a point_data key)")
        dims = [_dims(k) for k in signature.parameters]
        facet_d = tuple(d[0] for d in dims)
        quad_d = tuple(d[1] for d in dims)
        u_d = tuple(d[2] for d in dims)
        v_d = tuple(d[3] for d in dims)
        parallel_fn = vmap(vmap(vmap(vmap(fn, in_dims=v_d), in_dims=u_d),
                                in_dims=quad_d), in_dims=facet_d)

        N = self.n_points
        rows_all, cols_all, vals_all = [], [], []
        for et in self.element_types:
            trans = self.transformation[et]   # type: ignore
            proj = self.projector[et]         # type: ignore
            m: torch.Tensor = self.facet_mask[et].item()   # type: ignore [n_elem, n_facet]
            n_basis = trans.n_basis

            # selected-facet shape values / gradients / area-weights
            sv = trans.facet_shape_val.repeat(trans.n_elements, 1, 1, 1)[m]  # [F, Q, B]
            sg = trans.facet_shape_grad[m]                                   # [F, Q, B, D]
            fxw = trans.FxW[m]                                               # [F, Q]
            n_facet = sv.shape[0]
            ele_pd = {k: v[trans.elements] for k, v in point_data.items()}

            args = []
            for key in signature.parameters:
                if key in ("u", "v"):
                    args.append(sv)
                elif key in ("gradu", "gradv"):
                    args.append(sg)
                elif key in ele_pd:
                    fsv = trans.facet_shape_val
                    if ele_pd[key].is_complex() and not fsv.is_complex():
                        fsv = fsv.to(ele_pd[key].dtype)                     # promote real shapes
                    pd = torch.einsum("eb...,fqb->efq...", ele_pd[key], fsv)[m]
                    args.append(pd)                                         # [F, Q, ...]
                elif key.startswith("grad") and key[4:] in ele_pd:
                    coeff = ele_pd[key[4:]]
                    fsg = trans.facet_shape_grad
                    if coeff.is_complex() and not fsg.is_complex():
                        fsg = fsg.to(coeff.dtype)
                    gd = torch.einsum("eb...,efqbd->efq...d", coeff, fsg)[m]
                    args.append(gd)
                else:
                    raise NotImplementedError(f"key {key} not implemented")

            integ = parallel_fn(*args)            # [F, Q, B, B]
            local = torch.einsum("fqab,fq->fab", integ, fxw.to(integ.dtype))  # [F, B, B]

            # scatter: entry (a, b) of facet f -> global (dof[f,a], dof[f,b])
            cell_dofs = proj.indices.reshape(n_facet, n_basis).to(local.device)  # [F, B]
            rows = cell_dofs[:, :, None].expand(n_facet, n_basis, n_basis).reshape(-1)
            cols = cell_dofs[:, None, :].expand(n_facet, n_basis, n_basis).reshape(-1)
            rows_all.append(rows); cols_all.append(cols); vals_all.append(local.reshape(-1))

        row = torch.cat(rows_all); col = torch.cat(cols_all); val = torch.cat(vals_all)
        # coalesce duplicate (row, col) entries (nodes shared by several facets)
        coo = torch.sparse_coo_tensor(torch.stack([row, col]), val, (N, N)).coalesce()
        idx = coo.indices()
        return SparseMatrix(coo.values(), idx[0], idx[1], (N, N))

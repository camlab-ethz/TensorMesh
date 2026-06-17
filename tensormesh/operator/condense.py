"""Static condensation for applying Dirichlet boundary conditions.

This module provides :class:`Condenser`, which splits a global FEM system

.. math::

    K\\, u = f

into an *inner* (free-DOF) sub-system

.. math::

    K_{ii}\\, u_i = f_i - K_{io}\\, u_o,

solves it, then re-assembles the full solution from the inner solution plus
the prescribed boundary values :math:`u_o`.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..sparse import SparseMatrix


class Condenser(nn.Module):
    """Static-condensation operator for Dirichlet boundary conditions.

    Partitions a global system :math:`K u = f` into inner (free) DOFs and
    outer (Dirichlet) DOFs and condenses the prescribed values into the
    right-hand side:

    .. math::

        K_{ii}\\, u_i = f_i - K_{io}\\, u_o.

    Parameters
    ----------
    dirichlet_mask : torch.Tensor
        1D boolean tensor of shape :math:`[n_{\\text{dof}}]`. ``True`` marks
        DOFs whose value is prescribed.
    dirichlet_value : torch.Tensor, optional
        1D tensor of shape :math:`[n_{\\text{dof}}]` (a full vector — only
        the entries where ``dirichlet_mask`` is ``True`` are read) **or**
        :math:`[n_{\\text{outer\\_dof}}]` (already restricted to the
        boundary). Defaults to all zeros.

    Attributes
    ----------
    dirichlet_mask : torch.Tensor
        Boolean mask of shape :math:`[n_{\\text{dof}}]`.
    dirichlet_value : torch.Tensor
        Prescribed values restricted to the boundary, shape
        :math:`[n_{\\text{outer\\_dof}}]`.
    inner_row, inner_col : torch.Tensor or None
        Row/column indices of the inner block :math:`K_{ii}` in *local*
        inner-DOF numbering. Populated lazily on the first call.
    ou2in_row, ou2in_col : torch.Tensor or None
        Row/column indices of the coupling block :math:`K_{io}` in local
        numbering. Populated lazily.
    is_inner_edge, is_ou2in_edge : torch.Tensor or None
        Boolean masks over the matrix's COO edge list selecting the
        :math:`K_{ii}` / :math:`K_{io}` entries.
    is_inner_dof, is_outer_dof : torch.Tensor or None
        Boolean masks over the global DOFs.
    inner_shape, ou2in_shape : tuple of int or None
        Shapes of :math:`K_{ii}` and :math:`K_{io}`.
    n_inner_dof, n_outer_dof, n_dof : int or None
        DOF counts.
    layout_hash : int or None
        Sparsity-pattern hash cached from the first input matrix; used to
        detect a pattern change on subsequent calls.
    K_ou2in : SparseMatrix or None
        Cached :math:`K_{io}` block; reused by :meth:`condense_rhs`.

    Notes
    -----
    :class:`Condenser` is a :class:`torch.nn.Module`. All tensor-valued
    attributes (``dirichlet_mask``, ``dirichlet_value``, and the lazily
    computed index buffers) are registered as PyTorch buffers, so
    ``condenser.to(device)`` / ``condenser.cuda()`` / ``condenser.cpu()``
    move them together with the input system.

    The first call to ``__call__`` lazily computes the inner / outer edge
    masks and caches them on the instance. Subsequent calls reuse the
    cached layout as long as the input
    :class:`~tensormesh.sparse.SparseMatrix` has the same sparsity pattern
    (checked via ``matrix.layout_hash``). The lazy buffers are registered
    with ``persistent=False`` so they are not saved into ``state_dict``.

    Examples
    --------
    .. code-block:: python

        import torch
        from tensormesh import Mesh, Condenser
        from tensormesh.assemble import LaplaceElementAssembler

        mesh = Mesh.gen_rectangle(chara_length=0.2)
        K    = LaplaceElementAssembler.from_mesh(mesh)()
        f    = torch.ones(mesh.n_points, dtype=mesh.dtype)

        # Homogeneous Dirichlet on the whole boundary
        condenser = Condenser(mesh.boundary_mask)

        # Condense: returns (K_inner, f_inner) — note this is __call__,
        # NOT a separate "condense_matrix" method.
        K_inner, f_inner = condenser(K, f)

        # Solve the inner system and recover the full solution
        u_inner = K_inner.solve(f_inner)
        u       = condenser.recover(u_inner)

    For time-dependent boundary data, update the prescribed values between
    solves via :meth:`update_dirichlet`, then re-condense only the
    right-hand side with :meth:`condense_rhs` (cheaper than rebuilding
    ``K_inner``).
    """
    dirichlet_mask:torch.Tensor
    dirichlet_value:torch.Tensor

    inner_row:Optional[torch.Tensor]
    inner_col:Optional[torch.Tensor]
    ou2in_row:Optional[torch.Tensor]
    ou2in_col:Optional[torch.Tensor]
    is_inner_edge:Optional[torch.Tensor]
    is_ou2in_edge:Optional[torch.Tensor]
    is_inner_dof:Optional[torch.Tensor]
    is_outer_dof:Optional[torch.Tensor]
    inner_shape:Optional[Tuple[int, int]]
    ou2in_shape:Optional[Tuple[int, int]]
    n_inner_dof:Optional[int]
    n_outer_dof:Optional[int]
    n_dof:Optional[int]
    layout_signature:Optional[Tuple]
    K_ou2in:Optional[SparseMatrix]

    _LAZY_BUFFERS = (
        "inner_row", "inner_col", "ou2in_row", "ou2in_col",
        "is_inner_edge", "is_ou2in_edge",
        "is_inner_dof", "is_outer_dof",
    )

    def __init__(self,
                 dirichlet_mask:torch.Tensor,
                 dirichlet_value:Optional[torch.Tensor] = None):
        super().__init__()
        assert dirichlet_mask.dtype == torch.bool, \
            f"dirichlet_mask must be a bool tensor, got {dirichlet_mask.dtype}"
        assert dirichlet_mask.ndim == 1, \
            f"dirichlet_mask must be 1D, got shape {tuple(dirichlet_mask.shape)}"

        self.register_buffer("dirichlet_mask",  dirichlet_mask)
        self.register_buffer("dirichlet_value", self._normalize_value(dirichlet_value))

        for name in self._LAZY_BUFFERS:
            self.register_buffer(name, None, persistent=False)

        self.inner_shape   = None
        self.ou2in_shape   = None
        self.n_inner_dof   = None
        self.n_outer_dof   = None
        self.n_dof          = None
        self.layout_signature = None
        self.K_ou2in       = None

    def _normalize_value(self, value:Optional[torch.Tensor])->torch.Tensor:
        """Restrict a full or boundary-only prescribed-value vector to the boundary."""
        mask    = self.dirichlet_mask
        n_outer = int(mask.sum())
        if value is None:
            return torch.zeros(n_outer, device=mask.device)
        assert value.ndim == 1, \
            f"dirichlet_value must be 1D, got shape {tuple(value.shape)}"
        if value.shape[0] == mask.shape[0]:
            return value[mask]
        assert value.shape[0] == n_outer, \
            f"dirichlet_value must have length n_dof ({mask.shape[0]}) " \
            f"or n_outer_dof ({n_outer}), got {value.shape[0]}"
        return value

    def _compute_layout(self, matrix:SparseMatrix):
        """Precompute and cache inner / outer edge masks and local DOF indices.

        Called lazily on the first :meth:`__call__` and refreshed whenever
        ``matrix.layout_hash`` changes between calls.

        Parameters
        ----------
        matrix : SparseMatrix
            Reference matrix whose sparsity pattern determines the inner /
            outer edge partition.
        """
        edge_u, edge_v               = matrix.row, matrix.col
        n_dof                        = matrix.shape[0]

        is_inner_dof, is_outer_dof   = ~self.dirichlet_mask, self.dirichlet_mask

        is_inner_u,    is_inner_v    = is_inner_dof[edge_u], is_inner_dof[edge_v]
        is_outer_u,    is_outer_v    = is_outer_dof[edge_u], is_outer_dof[edge_v]
        is_inner_edge, is_ou2in_edge = is_inner_u & is_inner_v, is_inner_u & is_outer_v
        n_inner_dofs, n_outer_dofs   = is_inner_dof.sum().item(), is_outer_dof.sum().item()
        local_nids                   = torch.full((n_dof,), -1, dtype=torch.long, device=matrix.device)
        local_nids[is_inner_dof]     = torch.arange(n_inner_dofs, device=matrix.device)
        local_nids[is_outer_dof]     = torch.arange(n_outer_dofs, device=matrix.device)

        self.inner_row     = local_nids[edge_u[is_inner_edge]]
        self.inner_col     = local_nids[edge_v[is_inner_edge]]
        self.ou2in_row     = local_nids[edge_u[is_ou2in_edge]]
        self.ou2in_col     = local_nids[edge_v[is_ou2in_edge]]
        self.is_inner_edge = is_inner_edge
        self.is_ou2in_edge = is_ou2in_edge
        self.is_inner_dof  = is_inner_dof
        self.is_outer_dof  = is_outer_dof
        self.inner_shape   = (n_inner_dofs, n_inner_dofs)
        self.ou2in_shape   = (n_inner_dofs, n_outer_dofs)
        self.n_inner_dof   = n_inner_dofs
        self.n_outer_dof   = n_outer_dofs
        self.n_dof         = n_dof
        self.layout_signature = matrix.layout_signature

    def update_dirichlet(self, dirichlet_value:torch.Tensor):
        """Replace the cached prescribed boundary values.

        Useful for time-dependent or parameter-swept problems where only
        the right-hand side changes between solves; the cached
        :math:`K_{io}` block (populated by ``__call__``) is preserved.

        Parameters
        ----------
        dirichlet_value : torch.Tensor
            1D tensor of shape :math:`[n_{\\text{dof}}]` or
            :math:`[n_{\\text{outer\\_dof}}]`, with the same conventions
            as the ``dirichlet_value`` argument to :meth:`__init__`.
        """
        self.dirichlet_value = self._normalize_value(dirichlet_value)

    def __call__(self,
                 matrix,
                 rhs:Optional[torch.Tensor] = None,
                 ):
        """Condense both the matrix and the right-hand side.

        Parameters
        ----------
        matrix : SparseMatrix
            Global stiffness/mass matrix of shape
            :math:`[n_{\\text{dof}}, n_{\\text{dof}}]`.
        rhs : torch.Tensor, optional
            Global right-hand side of shape
            :math:`[n_{\\text{dof}}, \\ldots]`. Trailing dimensions are
            preserved (broadcasting is applied to the Dirichlet correction
            term). Defaults to a zero vector.

        Returns
        -------
        K_inner : SparseMatrix
            Condensed matrix of shape
            :math:`[n_{\\text{inner\\_dof}}, n_{\\text{inner\\_dof}}]`.
        f_inner : torch.Tensor
            Condensed right-hand side of shape
            :math:`[n_{\\text{inner\\_dof}}, \\ldots]`.

        Notes
        -----
        Caches the matrix sparsity layout on the first call. If you later
        condense a matrix with a different sparsity pattern, instantiate a
        new :class:`Condenser`.
        """
        # ---- DSparseMatrix dispatch ---------------------------------- #
        # The single-device condensation logic indexes ``matrix.edata``
        # by cached position-aware boolean masks (``is_inner_edge`` /
        # ``is_ou2in_edge``). Adapting that to a row-sharded
        # DSparseMatrix requires a from-scratch implementation (boundary
        # mask broadcast + per-rank inner/outer partition + repartition
        # of the condensed system). Until that lands, fall back through
        # ``.to_single()`` so the API contract is honoured and meshes
        # below the cluster threshold keep working.
        from ..distributed import DSparseMatrix
        if isinstance(matrix, DSparseMatrix):
            return self._call_distributed(matrix, rhs)

        if rhs is None:
            rhs = torch.zeros(matrix.shape[0])

        if self.inner_row is None:
            self._compute_layout(matrix)

        assert matrix.shape[0] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.shape[1] == self.n_dof, f"the shape of matrix must be [{self.n_dof}, {self.n_dof}], but got {matrix.shape}"
        assert matrix.has_same_layout(self.layout_signature), "the layout of the matrix is changed, please recompute the condensed matrix"
        assert rhs.shape[0] == self.n_dof, f"the shape of rhs must be [{self.n_dof}, ...], but got {rhs.shape}"

        K_inner = SparseMatrix(
            matrix.edata[self.is_inner_edge], self.inner_row, self.inner_col, self.inner_shape,
        )
        K_ou2in = SparseMatrix(
            matrix.edata[self.is_ou2in_edge], self.ou2in_row, self.ou2in_col, self.ou2in_shape,
        )
        self.K_ou2in = K_ou2in

        self.dirichlet_value = self.dirichlet_value.type(K_inner.edata.dtype).to(K_inner.edata.device)
        rhs = rhs.type(K_inner.edata.dtype).to(K_inner.edata.device)

        minus_term = K_ou2in @ self.dirichlet_value
        for _ in range(rhs.dim() - 1):
            minus_term = minus_term.unsqueeze(-1)

        return K_inner, rhs[self.is_inner_dof] - minus_term

    def _call_distributed(self, dmatrix, rhs):
        """Condense a :class:`DSparseMatrix` by round-tripping through
        the single-device path (``.to_single()`` allgather, condense,
        re-partition).

        Correct but inefficient: cost is one global allgather of the
        COO triples plus a re-partition. Intended as the interim
        contract while the true per-rank Condenser is designed.

        TODO(future PR): native distributed condensation that keeps the
        matrix sharded throughout (boundary mask broadcast + per-rank
        inner/outer split + automatic owned/halo rewrite).
        """
        import warnings
        warnings.warn(
            "Condenser on DSparseMatrix currently routes through .to_single() "
            "(allgather + condense + re-partition). The per-rank distributed "
            "Condenser is a follow-up PR; this path keeps the API working.",
            stacklevel=2,
        )
        # Round-trip to single-device. Force CPU because the cached
        # dirichlet_mask in this Condenser lives on the device chosen
        # at construction time (typically CPU); avoid a cross-device
        # index in _compute_layout. Result is moved back to dmatrix's
        # device before re-partitioning below.
        target_device = dmatrix.device
        K_global = dmatrix.to_single().cpu()
        if rhs is None:
            f_global = None
        else:
            # rhs comes in as either DTensor (Shard(0)) or owned slice;
            # for the round-trip we need the full vector. DTensor moved
            # from torch.distributed._tensor (torch 2.0-2.1) to
            # torch.distributed.tensor (torch >= 2.2); accept both.
            DTensor = None
            try:
                from torch.distributed.tensor import DTensor as _DT
                DTensor = _DT
            except ImportError:
                try:
                    from torch.distributed._tensor import DTensor as _DT
                    DTensor = _DT
                except ImportError:
                    pass
            if DTensor is not None and isinstance(rhs, DTensor):
                f_global = rhs.full_tensor()
            else:
                f_global = rhs
            if f_global is not None:
                f_global = f_global.cpu()
        K_inner, f_inner = self.__call__(K_global, f_global)
        K_inner = K_inner.to(target_device)
        f_inner = f_inner.to(target_device)
        # Re-partition the condensed result so downstream stays distributed.
        # Use the same partition_method the caller's DSparseMatrix used.
        from ..distributed import DSparseMatrix as _DM
        from torch_sla.distributed import DSparseTensor
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            from torch.distributed._tensor.device_mesh import init_device_mesh
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mesh = init_device_mesh(device, (world,))
            K_inner_d_tensor = DSparseTensor.partition(
                K_inner, mesh, partition_method="metis",
            )
        else:
            K_inner_d_tensor = DSparseTensor.partition(K_inner, mesh=None,
                                                       partition_method="metis")
        # Fresh UUID -- this is a new partition build that won't share
        # caches with the input's partition.
        K_inner_d = _DM(K_inner_d_tensor)
        return K_inner_d, f_inner

    def condense_rhs(self, rhs:torch.Tensor)->torch.Tensor:
        """Condense the right-hand side only, reusing the cached matrix layout.

        .. math::

            f_i \\leftarrow f_i - K_{io}\\, u_o.

        Use this after a first ``__call__`` to re-condense ``f`` when the
        matrix is unchanged but the load vector changes (e.g. between
        time steps).

        Parameters
        ----------
        rhs : torch.Tensor
            Global right-hand side of shape
            :math:`[n_{\\text{dof}}, \\ldots]`.

        Returns
        -------
        torch.Tensor
            Condensed right-hand side of shape
            :math:`[n_{\\text{inner\\_dof}}, \\ldots]`.

        Raises
        ------
        AssertionError
            If ``__call__`` has not been invoked yet: the operator has no
            cached :math:`K_{io}` block to apply.
        """
        assert self.K_ou2in is not None, \
            "call the Condenser on a matrix first; condense_rhs reuses the cached K_io block"

        self.dirichlet_value = self.dirichlet_value.type(rhs.dtype).to(rhs.device)
        rhs = rhs.type(self.K_ou2in.edata.dtype).to(self.K_ou2in.edata.device)

        minus_term = self.K_ou2in @ self.dirichlet_value
        for _ in range(rhs.dim() - 1):
            minus_term = minus_term.unsqueeze(-1)

        return rhs[self.is_inner_dof] - minus_term

    def recover(self, u:torch.Tensor)->torch.Tensor:
        """Recover the full-DOF solution from an inner-DOF solution.

        Scatters the condensed solution ``u`` back into the free-DOF slots
        and writes the prescribed boundary values into the constrained
        slots.

        Parameters
        ----------
        u : torch.Tensor
            Inner-system solution of shape
            :math:`[n_{\\text{inner\\_dof}}, \\ldots]`.

        Returns
        -------
        torch.Tensor
            Full-system solution of shape
            :math:`[n_{\\text{dof}}, \\ldots]`.
        """
        assert u.shape[0] == self.n_inner_dof, f"the shape of u must be [{self.n_inner_dof}, ...], but got {u.shape}"
        shape    = list(u.shape)
        shape[0] = self.n_dof
        u_full   = torch.zeros(shape, dtype=u.dtype, device=u.device)
        u_full[self.is_inner_dof] += u
        boundary_value = self.dirichlet_value
        for _ in range(u.dim() - 1):
            boundary_value = boundary_value.unsqueeze(-1)
        u_full[self.is_outer_dof] += boundary_value

        return u_full

    def restrict(self, f:torch.Tensor)->torch.Tensor:
        """Project a full-DOF vector down to inner DOFs.

        Pure linear restriction :math:`f_i \\leftarrow f|_{\\text{inner}}`,
        with **no** Dirichlet-value correction. Use this when the
        right-hand side has no implicit Dirichlet contribution to
        subtract — for example, the per-stage right-hand side of a
        time-integration scheme such as
        :class:`tensormesh.ode.ImplicitLinearRungeKutta`, where the
        time-derivative at a Dirichlet DOF is zero by construction
        and so the :math:`-K_{io}\\,u_o` term in :meth:`condense_rhs`
        would over-apply the boundary correction.

        Unlike ``Condenser.__call__`` / :meth:`condense_rhs`,
        ``restrict`` does not require the matrix layout to be cached
        first: it only needs ``dirichlet_mask``.

        Parameters
        ----------
        f : torch.Tensor
            Full-DOF vector of shape
            :math:`[n_{\\text{dof}}, \\ldots]`.

        Returns
        -------
        torch.Tensor
            Inner-DOF vector of shape
            :math:`[n_{\\text{inner\\_dof}}, \\ldots]`.
        """
        assert f.shape[0] == self.dirichlet_mask.shape[0], \
            f"the shape of f must be [{self.dirichlet_mask.shape[0]}, ...], but got {f.shape}"
        return f[~self.dirichlet_mask]

    def prolong(self, f_inner:torch.Tensor)->torch.Tensor:
        """Lift an inner-DOF vector up to full DOF with zeros on the boundary.

        Pure linear prolongation: inner entries are scattered into the
        free-DOF slots, constrained slots are filled with **zero** —
        not with ``dirichlet_value``. Use this when the quantity being
        lifted should vanish on the boundary regardless of the
        prescribed Dirichlet value, e.g. the per-stage slope of a
        time integrator (since a fixed-value DOF has zero
        time-derivative).

        Like :meth:`restrict`, ``prolong`` only needs
        ``dirichlet_mask`` and does not require the matrix layout to
        be cached first.

        Parameters
        ----------
        f_inner : torch.Tensor
            Inner-DOF vector of shape
            :math:`[n_{\\text{inner\\_dof}}, \\ldots]`.

        Returns
        -------
        torch.Tensor
            Full-DOF vector of shape
            :math:`[n_{\\text{dof}}, \\ldots]` with zeros in the
            constrained slots.
        """
        n_inner = int((~self.dirichlet_mask).sum())
        assert f_inner.shape[0] == n_inner, \
            f"the shape of f_inner must be [{n_inner}, ...], but got {f_inner.shape}"
        shape    = list(f_inner.shape)
        shape[0] = self.dirichlet_mask.shape[0]
        f_full   = torch.zeros(shape, dtype=f_inner.dtype, device=f_inner.device)
        f_full[~self.dirichlet_mask] = f_inner
        return f_full


Condenser.__autodoc__ = [
    "__call__", "condense_rhs", "recover", "restrict", "prolong", "update_dirichlet",
]

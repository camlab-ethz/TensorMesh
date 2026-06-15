"""Distributed FEM sparse matrix.

:class:`DSparseMatrix` wraps :class:`torch_sla.DSparseTensor` with the
same FEM-flavoured surface as the single-device :class:`SparseMatrix`
(layout signature, type-preserving arithmetic, scipy interop bridge via
:meth:`to_single`). Unlike :class:`SparseMatrix`, this uses
**composition** over the torch-sla primitive rather than inheritance --
the boundary between the FEM layer and the distributed sparse algebra
layer stays explicit and the brittleness of subclassing a third-party
distributed class is avoided.

Partition-broadcasted UUID
--------------------------

Each rank's local ``(row, col)`` is itself a
:class:`torch_sla.SparseTensor`, so the mixin's sequence-identity
signature would normally only tag the local layout. That is unsafe for
cross-rank caching: two distinct partitions can produce identical local
``(row, col)`` shapes by coincidence (e.g. METIS rerun with a different
seed) and ranks would silently share caches. To prevent that, every
``DSparseMatrix`` carries a 63-bit **partition UUID**, generated on rank
0 with :func:`uuid.uuid4` and broadcast through the active process
group at construction time. The UUID is included in
:attr:`layout_signature`, so two ``DSparseMatrix`` instances with the
same partition (e.g. ``A + B`` of matrices built against the same
partition) share signatures, while two independently-partitioned matrices
do not -- even if local layouts coincidentally match.

Arithmetic that derives from an existing ``DSparseMatrix``
(:meth:`__add__`, :meth:`__sub__`, etc.) propagates the parent's UUID
instead of generating a fresh one: the resulting matrix shares the
same partition and therefore the same caches.
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple, Union

import torch

try:
    from torch_sla.distributed import DSparseTensor
except ImportError as e:
    raise ImportError(
        "torch-sla with DSparseTensor support is required.\n"
        "Install with: pip install torch-sla>=0.2.0"
    ) from e

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

from .mixin import _FEMSparsityMixin


def _generate_partition_uuid() -> int:
    """Generate a 63-bit UUID identifying *this partition build*.

    Distributed: rank 0 draws a fresh :func:`uuid.uuid4`, the lower 63
    bits are broadcast across the active process group on the matching
    device, every rank returns the broadcast value.

    Single-process: just a fresh ``uuid4``.
    """
    if _DIST_AVAILABLE and dist.is_initialized():
        rank = dist.get_rank()
        # Stay inside int63 so the result is a positive Python int that
        # round-trips through CPU int64 tensors regardless of backend.
        if rank == 0:
            u = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
        else:
            u = 0
        backend = dist.get_backend()
        device = torch.device("cuda", torch.cuda.current_device()) \
            if backend == "nccl" else torch.device("cpu")
        t = torch.tensor([u], dtype=torch.long, device=device)
        dist.broadcast(t, src=0)
        return int(t.item())
    return uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF


class DSparseMatrix(_FEMSparsityMixin):
    """Distributed FEM sparse matrix.

    Composition over :class:`torch_sla.distributed.DSparseTensor` --
    delegates primitives (shape, device, ``@``, ``solve``) and adds the
    FEM-flavoured surface (layout signature with broadcast UUID,
    type-preserving arithmetic, ``to_single()`` allgather bridge).

    Parameters
    ----------
    dsparse_tensor : torch_sla.distributed.DSparseTensor
        The underlying distributed sparse tensor (carries the
        :class:`~torch_sla.partition.Partition` and the local
        :class:`~torch_sla.SparseTensor` shard).
    partition_uuid : int, optional
        63-bit UUID identifying the partition build. Pass this when
        deriving a new ``DSparseMatrix`` from an existing one (so the
        cache caches share); leave as ``None`` to freshly generate +
        broadcast (default).
    """

    def __init__(
        self,
        dsparse_tensor: "DSparseTensor",
        partition_uuid: Optional[int] = None,
    ):
        self._t = dsparse_tensor
        self._partition_uuid = (
            partition_uuid if partition_uuid is not None
            else _generate_partition_uuid()
        )

    # ==================== Delegation to DSparseTensor ====================

    @property
    def shape(self) -> Tuple[int, int]:
        return self._t.shape

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def device(self):
        return self._t.device

    @property
    def values(self) -> torch.Tensor:
        """Local owned-row values (the rank's slice of the global value array)."""
        return self._t._local_tensor.values

    @property
    def row_indices(self) -> torch.Tensor:
        """Local row indices (already in local coordinates)."""
        return self._t._local_tensor.row_indices

    @property
    def col_indices(self) -> torch.Tensor:
        """Local column indices (in local coordinates; halo columns
        point to halo positions in ``local_to_global``)."""
        return self._t._local_tensor.col_indices

    @property
    def partition(self):
        return self._t._spec.placement.partition

    @property
    def partition_uuid(self) -> int:
        return self._partition_uuid

    # ==================== Layout signature (overrides mixin) ====================

    @property
    def layout_signature(self) -> Tuple:
        """Sequence-identity signature scoped to the partition build.

        Combines the mixin's local sequence identity with the
        partition UUID so that two matrices on the same partition share
        signatures (and therefore caches) while two matrices on
        independently-built partitions do not, even if local layouts
        coincidentally match.
        """
        base = super().layout_signature   # uses self.row_indices / col_indices
        return base + (self._partition_uuid,)

    # ==================== matvec / solve ====================

    def __matmul__(self, x):
        """``D @ x``: returns a DTensor or owned-slice tensor depending
        on ``x``'s placement, exactly like the underlying DSparseTensor."""
        return self._t @ x

    def solve(self, b, **kw):
        """Distributed solve. Delegates to ``torch_sla.solve``; the
        SolverConfig context applies as usual."""
        from torch_sla import solve as _solve
        return _solve(self._t, b, **kw)

    # ==================== Type-preserving arithmetic ====================

    def _wrap(self, result):
        """Wrap a DSparseTensor result back into DSparseMatrix while
        preserving the partition UUID (the result lives on the same
        partition as ``self``, so it should share caches)."""
        if isinstance(result, DSparseTensor):
            return DSparseMatrix(result, partition_uuid=self._partition_uuid)
        return result

    def __add__(self, other):
        rhs = other._t if isinstance(other, DSparseMatrix) else other
        return self._wrap(self._t + rhs)

    def __sub__(self, other):
        rhs = other._t if isinstance(other, DSparseMatrix) else other
        return self._wrap(self._t - rhs)

    def __mul__(self, scalar):
        return self._wrap(self._t * scalar)

    def __rmul__(self, scalar):
        return self._wrap(scalar * self._t)

    def __neg__(self):
        return self._wrap(-self._t)

    # ==================== Device / dtype ====================

    def to(self, *args, **kw) -> "DSparseMatrix":
        return self._wrap(self._t.to(*args, **kw))

    def cuda(self, device=None) -> "DSparseMatrix":
        return self._wrap(self._t.cuda(device))

    def cpu(self) -> "DSparseMatrix":
        return self._wrap(self._t.cpu())

    def double(self) -> "DSparseMatrix":
        return self._wrap(self._t.double())

    def float(self) -> "DSparseMatrix":
        return self._wrap(self._t.float())

    # ==================== Bridges ====================

    def to_single(self) -> "SparseMatrix":
        """Allgather the distributed matrix into a global single-device
        :class:`SparseMatrix` materialised on every rank.

        Used as a temporary bridge to FEM operators that have not yet
        learned to consume :class:`DSparseMatrix` directly (notably
        :class:`Condenser`). Costs an all-gather of the COO triples;
        avoid in hot paths.
        """
        from .matrix import SparseMatrix
        st_global = self._t.full_tensor()
        return SparseMatrix(
            st_global.values, st_global.row_indices, st_global.col_indices,
            tuple(st_global.shape),
        )

    # ==================== Constructors ====================

    @classmethod
    def from_dsparse_tensor(
        cls,
        dst: "DSparseTensor",
        partition_uuid: Optional[int] = None,
    ) -> "DSparseMatrix":
        """Wrap an existing :class:`~torch_sla.distributed.DSparseTensor`.

        If ``partition_uuid`` is not given, a fresh broadcast UUID is
        drawn -- only safe when this is the first DSparseMatrix derived
        from that partition. Otherwise pass the UUID of an existing
        sibling so they share caches.
        """
        return cls(dst, partition_uuid=partition_uuid)

    # ==================== repr ====================

    def __repr__(self) -> str:
        try:
            world = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world, rank = 1, 0
        return (
            f"DSparseMatrix(shape={self.shape}, "
            f"local_nnz={self.row_indices.numel()}, "
            f"rank={rank}/{world}, "
            f"partition_uuid={self._partition_uuid:#x})"
        )

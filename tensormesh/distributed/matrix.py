"""Distributed FEM sparse matrix.

:class:`DSparseMatrix` is a subclass of
:class:`torch_sla.distributed.DSparseTensor` that adds the FEM-flavoured
surface (layout signature with broadcast UUID, type-preserving
arithmetic, ``to_single()`` allgather bridge).

Subclass rather than composition
--------------------------------

The earlier draft of this class used composition (``self._t:
DSparseTensor``) for cleaner separation between the FEM layer and the
distributed sparse algebra layer. That design failed the Liskov
substitution test: torch-sla's free functions (``torch_sla.solve``,
``torch_sla.io.save``, ``DSparseTensor.__add__``) gate on
``isinstance(x, DSparseTensor)`` to dispatch into the distributed
path; a composition-based ``DSparseMatrix`` silently misses the gate
and routes to the single-device or duck-typed fallback. Subclassing
makes ``isinstance(D, DSparseTensor)`` true and every torch-sla API
accepts ``DSparseMatrix`` directly.

The cost is the same boilerplate ``_wrap`` pattern as
:class:`SparseMatrix`: arithmetic and device-conversion methods inherit
from the parent but return parent-typed ``DSparseTensor`` instances, so
each one is overridden to re-wrap into ``DSparseMatrix`` and preserve
the FEM identity (partition UUID + mixin signature).

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
same partition share signatures while two independently-partitioned
matrices do not -- even if local layouts coincidentally match.

Arithmetic / device methods propagate the parent's UUID (the result
lives on the same partition); fresh constructions (``partition``,
``from_sparse_local``) draw a new UUID.
"""
from __future__ import annotations

import uuid
from typing import Optional, Tuple

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

from ..sparse.mixin import _FEMSparsityMixin


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


class DSparseMatrix(_FEMSparsityMixin, DSparseTensor):
    """Distributed FEM sparse matrix.

    Subclass of :class:`torch_sla.distributed.DSparseTensor` so
    ``isinstance(D, DSparseTensor)`` holds and torch-sla's free
    functions (solve / io / arithmetic dispatch) accept it.

    Constructed by wrapping an existing :class:`DSparseTensor`
    (typically returned by :meth:`DSparseTensor.partition` or
    :func:`~tensormesh.distributed.distributed_element_assemble`).

    Parameters
    ----------
    dsparse_tensor
        The underlying distributed sparse tensor. ``DSparseMatrix``
        copies its state and adds the FEM identity layer on top.
    partition_uuid
        63-bit UUID identifying the partition build. Pass when
        deriving a new ``DSparseMatrix`` from an existing one (so
        caches are shared); leave as ``None`` to freshly generate +
        broadcast.
    """

    def __init__(
        self,
        dsparse_tensor: "DSparseTensor",
        partition_uuid: Optional[int] = None,
    ):
        # DSparseTensor.__init__ raises TypeError to discourage direct
        # instantiation. We bypass it because we are not constructing
        # from scratch -- we are wrapping an already-built instance.
        # ``__dict__.update`` is a shallow copy of every backing
        # attribute (``_local_tensor``, ``_spec``, halo buffers, etc.),
        # which is exactly what subclassing-by-promotion needs.
        self.__dict__.update(dsparse_tensor.__dict__)
        self._partition_uuid = (
            partition_uuid if partition_uuid is not None
            else _generate_partition_uuid()
        )

    # ==================== Identity ====================

    @property
    def partition_uuid(self) -> int:
        return self._partition_uuid

    @property
    def partition(self):
        """The :class:`~torch_sla.partition.Partition` from the spec."""
        return self._spec.placement.partition

    @property
    def row_indices(self) -> torch.Tensor:
        """Local row indices (in local coordinates)."""
        return self._local_tensor.row_indices

    @property
    def col_indices(self) -> torch.Tensor:
        """Local column indices (in local coordinates)."""
        return self._local_tensor.col_indices

    # ==================== Layout signature (overrides mixin) ====================

    @property
    def layout_signature(self) -> Tuple:
        """Sequence-identity signature scoped to the partition build.

        Combines the mixin's local sequence identity with the
        partition UUID so two matrices on the same partition share
        signatures (and therefore caches) while two matrices on
        independently-built partitions do not, even if local layouts
        coincidentally match.
        """
        base = super().layout_signature   # uses self.row_indices / col_indices
        return base + (self._partition_uuid,)

    # ==================== Type-preserving wrappers ====================

    def _wrap(self, result):
        """Re-wrap a DSparseTensor result back into a DSparseMatrix on
        the same partition.

        Three cases to handle:

        1. Non-DSparseTensor (e.g. DTensor from matvec) → pass through.
        2. Plain DSparseTensor → wrap into DSparseMatrix with our UUID.
        3. DSparseMatrix that lacks ``_partition_uuid``. This happens
           because ``DSparseTensor._wrap_local`` uses
           ``type(self).__new__(type(self))`` to build the result --
           that picks up our subclass type but skips
           ``__init__``, so the FEM identity attributes were never
           set. Stamp the UUID retroactively.
        """
        if not isinstance(result, DSparseTensor):
            return result
        if not isinstance(result, DSparseMatrix):
            return DSparseMatrix(result, partition_uuid=self._partition_uuid)
        # Already a DSparseMatrix (from parent's _wrap_local) but
        # potentially missing our FEM identity layer.
        result._partition_uuid = self._partition_uuid
        return result

    def __add__(self, other):
        return self._wrap(super().__add__(other))

    def __sub__(self, other):
        return self._wrap(super().__sub__(other))

    def __mul__(self, scalar):
        return self._wrap(super().__mul__(scalar))

    def __rmul__(self, scalar):
        return self._wrap(super().__rmul__(scalar))

    def __neg__(self):
        return self._wrap(super().__neg__())

    # Device / dtype methods on DSparseTensor return DSparseTensor; rewrap.

    def to(self, *args, **kw) -> "DSparseMatrix":
        return self._wrap(super().to(*args, **kw))

    def cuda(self, device=None) -> "DSparseMatrix":
        return self._wrap(super().cuda(device))

    def cpu(self) -> "DSparseMatrix":
        return self._wrap(super().cpu())

    def double(self) -> "DSparseMatrix":
        return self._wrap(super().double())

    def float(self) -> "DSparseMatrix":
        return self._wrap(super().float())

    # ==================== Bridges ====================

    def to_single(self) -> "SparseMatrix":
        """Allgather the distributed matrix into a global single-device
        :class:`SparseMatrix` materialised on every rank.

        Used as a temporary bridge to FEM operators that have not yet
        learned to consume :class:`DSparseMatrix` directly (notably
        :class:`Condenser`). Costs an all-gather of the COO triples;
        avoid in hot paths.
        """
        from ..sparse.matrix import SparseMatrix
        st_global = self.full_tensor()
        return SparseMatrix(
            st_global.values, st_global.row_indices, st_global.col_indices,
            tuple(st_global.shape),
        )

    # ==================== repr ====================

    def __repr__(self) -> str:
        try:
            world = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            world, rank = 1, 0
        return (
            f"DSparseMatrix(shape={tuple(self.shape)}, "
            f"local_nnz={self.row_indices.numel()}, "
            f"rank={rank}/{world}, "
            f"partition_uuid={self._partition_uuid:#x})"
        )

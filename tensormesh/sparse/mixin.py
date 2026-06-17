"""Shared FEM sparsity behaviour for single-device and distributed
sparse matrix classes.

The mixin currently provides one thing — :attr:`layout_signature` — a
hashable opaque tuple that downstream FEM caches (Condenser, AMG
hierarchy, matrix coloring) use as a dict key to skip re-computing
quantities that depend only on the sparsity pattern.

Design notes
------------

``layout_signature`` is **sequence-identity** rather than content hash:

* it uses ``(row.data_ptr, row._version, col.data_ptr, col._version,
  numel, shape)``;
* it is zero-cost — no GPU sync, no data movement, no autograd graph
  break (so it can sit on a differentiable assembly path);
* it correctly handles in-place modification of ``row_indices`` /
  ``col_indices`` via PyTorch's tensor version counter;
* it intentionally does NOT identify two semantically-equal matrices in
  different memory regions as equal — that is the correct invariant for
  FEM caches whose cached state (Condenser's ``is_inner_edge`` boolean
  mask, AMG hierarchies) is indexed by COO position. Two "same set of
  edges but reordered" matrices look different and *should* miss the
  cache; treating them as equal would silently corrupt the cached
  position-indexed buffers.

For distributed sparse matrices the signature must additionally tag the
:class:`~torch_sla.partition.Partition` identity so that two
``DSparseMatrix`` instances that share local ``(row, col)`` storage but
were built against different partitions do not alias each other's
caches. The distributed subclass overrides the mixin's signature with a
broadcasted partition UUID; see :mod:`tensormesh.distributed.matrix`.
"""
from __future__ import annotations

from typing import Tuple, Union


class _FEMSparsityMixin:
    """Mixin providing :attr:`layout_signature` for FEM sparse matrices.

    Expected attributes on the concrete class: ``row_indices``,
    ``col_indices`` (both ``torch.Tensor`` with ``data_ptr`` and
    ``_version``) and ``shape`` (a 2-tuple).
    """

    @property
    def layout_signature(self) -> Tuple:
        """Opaque hashable sequence-identity signature.

        Use as a ``dict`` key when caching anything that depends only on
        the sparsity pattern. Zero-cost; no GPU sync; no graph break.

        Properties:

        * Same tensor reference (or in-place untouched alias) → same
          signature → cache hit.
        * In-place modification of ``row_indices`` / ``col_indices``
          (``tensor.copy_(...)`` etc.) → ``_version`` increment → cache
          miss (correct).
        * Re-wrap with new tensors of identical content (``clone``,
          ``row[perm]``, fresh allocation) → different ``data_ptr`` →
          cache miss (correct — see module-level "Design notes").
        """
        r = self.row_indices
        c = self.col_indices
        return (
            r.data_ptr(), r._version,
            c.data_ptr(), c._version,
            r.numel(), self.shape,
        )

    def has_same_layout(self, other: Union["_FEMSparsityMixin", Tuple]) -> bool:
        """Compare :attr:`layout_signature` against another matrix or a
        previously-captured signature tuple.

        Parameters
        ----------
        other
            Either another sparse matrix with :attr:`layout_signature`,
            or a tuple previously obtained from :attr:`layout_signature`.
        """
        if isinstance(other, tuple):
            return self.layout_signature == other
        return self.layout_signature == other.layout_signature

"""``@distributed`` class decorator for Assembler classes.

Turns any single-device ``ElementAssembler`` / ``NodeAssembler``
subclass into its distributed counterpart by overriding two methods:

* ``from_mesh(dmesh)``  -- accepts a :class:`DistributedMesh` instead
  of a single :class:`Mesh`;
* ``__call__()``        -- runs per-rank submesh assembly via the
  existing :func:`distributed_element_assemble_per_rank` /
  :func:`distributed_node_assemble` infrastructure and wraps the
  result in :class:`~tensormesh.sparse.DSparseMatrix` (element) or
  returns a per-rank tensor (node).

Usage::

    from tensormesh.assemble import LaplaceElementAssembler
    from tensormesh.distributed import distributed, DistributedMesh

    DistLaplace = distributed(LaplaceElementAssembler)
    K_dist = DistLaplace.from_mesh(dmesh)()    # -> DSparseMatrix

This avoids the alternative of defining a parallel ``DAssembler`` class
hierarchy alongside every existing ``Assembler`` -- one decorator covers
the whole set, including third-party assemblers the user writes.
"""
from __future__ import annotations

from typing import Optional, Type, TypeVar


T = TypeVar("T")


def distributed(asm_cls: Type[T]) -> Type[T]:
    """Class decorator: turn an Assembler class into a distributed one.

    The wrapped class shares the original's weak form, quadrature setup
    and element kernel -- only the entry points (``from_mesh`` +
    ``__call__``) are swapped to run a distributed assembly path.

    Parameters
    ----------
    asm_cls
        An ``ElementAssembler`` or ``NodeAssembler`` subclass. The
        decision between matrix- and vector-flavoured assembly is taken
        at the wrapped ``__call__`` based on the original class type.

    Returns
    -------
    A new subclass of ``asm_cls`` with overridden entry points; the
    name is prefixed with ``Distributed`` for repr / debugging clarity.
    """
    # Import lazily to avoid a circular import with the assembly module
    # (which itself imports from tensormesh.assemble for the base classes).
    from .assemble import (
        distributed_element_assemble,
        distributed_node_assemble,
    )

    # Element vs Node detection: inspect MRO names to avoid heavy imports.
    is_element = any(
        c.__name__ == "ElementAssembler" for c in asm_cls.__mro__
    )
    is_node = any(
        c.__name__ == "NodeAssembler" for c in asm_cls.__mro__
    )
    if not (is_element or is_node):
        raise TypeError(
            f"{asm_cls.__name__} is neither an ElementAssembler nor a "
            "NodeAssembler subclass; @distributed cannot wrap it."
        )

    class _DistributedWrapped(asm_cls):  # type: ignore[misc, valid-type]
        # The wrapper stores enough state to defer assembly until
        # ``__call__``; the parent __init__ is bypassed because we don't
        # have a Mesh, only a DistributedMesh.

        @classmethod
        def from_mesh(cls, dmesh, **kw):  # type: ignore[override]
            """Build a distributed assembler bound to ``dmesh``.

            Mirrors :meth:`ElementAssembler.from_mesh` but accepts a
            :class:`DistributedMesh`. The actual assembly is deferred to
            :meth:`__call__`.
            """
            self = cls.__new__(cls)
            self._dmesh = dmesh
            self._from_mesh_kw = kw
            return self

        def __call__(self, **call_kw):  # type: ignore[override]
            """Trigger the distributed assembly.

            Returns
            -------
            :class:`~tensormesh.sparse.DSparseMatrix` (element flavour)
            or a per-rank :class:`torch.Tensor` (node flavour).
            """
            if is_element:
                from . import DSparseMatrix
                dst = distributed_element_assemble(
                    asm_cls, self._dmesh,
                    **self._from_mesh_kw,
                    call_kwargs=call_kw or None,
                )
                # dst is a DSparseTensor; wrap with a fresh UUID since
                # this is the first DSparseMatrix derived from this
                # partition build.
                return DSparseMatrix(dst)
            # Node assembler
            return distributed_node_assemble(
                asm_cls, self._dmesh,
                **self._from_mesh_kw,
                call_kwargs=call_kw or None,
            )

    _DistributedWrapped.__name__ = f"Distributed{asm_cls.__name__}"
    _DistributedWrapped.__qualname__ = _DistributedWrapped.__name__
    _DistributedWrapped.__doc__ = (
        f"Distributed wrapper of :class:`{asm_cls.__name__}` produced by "
        ":func:`tensormesh.distributed.distributed`. See the decorator's "
        "docstring for the new entry-point contract."
    )
    return _DistributedWrapped

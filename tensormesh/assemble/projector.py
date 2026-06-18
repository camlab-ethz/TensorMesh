from typing import Union, Sequence

import numpy as np
import torch
import torch.nn as nn

Tensor = Union[torch.Tensor, np.ndarray]
Shape = Union[Sequence[int], int, np.ndarray, torch.Size]


class Projector(nn.Module):
    """Abstract base for the element-to-global scatter operators.

    A :class:`Projector` consumes a tensor with leading shape ``from_shape``
    (per-element / per-facet quantities) and returns a tensor with leading
    shape ``to_shape`` (global edge / node indexing), summing duplicates.
    The two concrete implementations are ``ReduceProjector`` (uses
    :meth:`torch.Tensor.index_add_`) and ``SparseProjector`` (uses a sparse
    mat-vec product).
    """
    pass


class ReduceProjector(Projector):
    """Element-to-global scatter backed by :meth:`torch.Tensor.index_add_`.

    More widely compatible than ``SparseProjector`` because it only relies
    on the dense ``index_add_`` kernel that PyTorch ships for every backend.

    Attributes
    ----------
    indices : torch.Tensor
        Long tensor of shape :math:`[\prod \text{from\_shape}]` mapping each
        flat-from index to its flat-to slot.
    from_shape : tuple
        Leading shape of accepted inputs (``input.shape[:len(from_shape)]``).
    to_shape : tuple
        Leading shape of returned outputs.
    use_fp64 : bool
        If ``True``, the accumulation runs in ``float64`` and is cast back
        to the input dtype on return — useful for deterministic accumulation
        of many small contributions.
    """
    indices:torch.Tensor
    from_shape:Shape 
    to_shape:Shape
    use_fp64:bool

    def __init__(self,
                 indices:torch.Tensor,
                 from_shape:Shape,
                 to_shape:Shape,
                 use_fp64:bool = False):
        """Wire up the scatter indices and the input/output shapes.

        Parameters
        ----------
        indices : torch.Tensor or np.ndarray
            1D index tensor of length :math:`\prod \text{from\_shape}`.
        from_shape : tuple, int, np.ndarray, or torch.Size
            Leading shape of accepted inputs.
        to_shape : tuple, int, np.ndarray, or torch.Size
            Leading shape of returned outputs.
        use_fp64 : bool, optional
            Accumulate in ``float64`` (default ``False``).
        """
        super().__init__()

        if isinstance(indices, np.ndarray):
            indices   = torch.from_numpy(indices)
        assert indices.dim() == 1, f"indices must be 1D, but got {indices.dim()}"
        
        if isinstance(from_shape, int):
            from_shape = (from_shape,)
        elif isinstance(from_shape, np.ndarray):
            assert from_shape.ndim == 1, f"from_shape must be 1D, but got {from_shape.ndim}"

        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        elif isinstance(to_shape, np.ndarray):
            assert to_shape.ndim == 1, f"to_shape must be 1D, but got {to_shape.ndim}"

        self.register_buffer("indices", indices)
        self.from_shape = from_shape
        self.to_shape   = to_shape
        self.use_fp64   = use_fp64
        # Pre-compute for torch.compile compatibility
        self._from_size = int(np.prod(from_shape))
        self._to_size = int(np.prod(to_shape))

    @property
    def device(self):
        return self.indices.device

    def __call__(self, x:torch.Tensor)->torch.Tensor:
        """Scatter ``x`` from ``from_shape`` to ``to_shape``, summing duplicates.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[*from_shape, ...]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[*to_shape, ...]``.
        """
        assert self.device == x.device, f"the device of x must be {self.device}, but got {x.device}"
        assert x.shape[:len(self.from_shape)] == self.from_shape, f"the shape of x must be [{self.from_shape}, ...], but got {x.shape}"

        dim_shape = x.shape[len(self.from_shape):]
        x = x.reshape(self._from_size, *dim_shape)

        if self.use_fp64:
            original_dtype = x.dtype
            # Pick the right wider precision: complex128 for complex inputs,
            # float64 otherwise. ``.double()`` would silently drop the
            # imaginary part on a complex tensor.
            wide_dtype = torch.complex128 if x.is_complex() else torch.float64
            x = x.to(wide_dtype)
            o = torch.zeros(self._to_size, *dim_shape, device=x.device, dtype=wide_dtype)
        else:
            o = torch.zeros(self._to_size, *dim_shape, device=x.device, dtype=x.dtype)

        o = o.index_add_(0, self.indices, x)
        if self.use_fp64:
            o = o.to(original_dtype)

        o = o.reshape(*self.to_shape, *dim_shape)
        return o

    def __str__(self):
        return f"{type(self).__name__}({self.from_shape} -> {self.to_shape}, device={self.device})"

    def __repr__(self):
        return str(self)

class SparseProjector(Projector):
    """Element-to-global scatter backed by a CSR sparse mat-vec product.

    Faster than ``ReduceProjector`` for large meshes on backends that
    optimize sparse mat-vec, at the cost of materializing the projection
    matrix.

    Attributes
    ----------
    projection : torch.Tensor
        CSR sparse tensor of shape :math:`(\prod \text{to\_shape}, \prod \text{from\_shape})`.
    from_shape : tuple
        Leading shape of accepted inputs.
    to_shape : tuple
        Leading shape of returned outputs.
    """
    projection:torch.sparse_csr_tensor
    from_shape:Shape
    to_shape:Shape

    def __init__(self, from_:Tensor,
                        to_:Tensor,
                        from_shape:Shape,
                        to_shape:Shape, dtype = None):
        """Wire up the scatter index pairs and the input/output shapes.

        Parameters
        ----------
        from_ : torch.Tensor or np.ndarray
            1D source index tensor.
        to_ : torch.Tensor or np.ndarray
            1D destination index tensor (same length as ``from_``).
        from_shape : tuple, int, np.ndarray, or torch.Size
            Leading shape of accepted inputs.
        to_shape : tuple, int, np.ndarray, or torch.Size
            Leading shape of returned outputs.

        Examples
        --------
        .. code-block:: python

            import scipy.sparse
            m = scipy.sparse.rand(3, 4, 0.5, format="coo")
            p = SparseProjector(m.col, m.row, 4, 3)
        """
        super().__init__()
        if isinstance(from_shape, int):
            from_shape = (from_shape,)
        elif isinstance(from_shape, np.ndarray):
            assert from_shape.ndim == 1, f"from_shape must be 1D, but got {from_shape.ndim}"

        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        elif isinstance(to_shape, np.ndarray):
            assert to_shape.ndim == 1, f"to_shape must be 1D, but got {to_shape.ndim}"

        if isinstance(from_, np.ndarray):
            from_ = torch.from_numpy(from_)
        if isinstance(to_, np.ndarray):
            to_   = torch.from_numpy(to_)
        assert from_.shape == to_.shape, f"from_ and to_ must have the same shape, but got {from_.shape}, {to_.shape}"
        assert len(from_.shape) == 1, f"from_ and to_ must be 1D, but got {from_.shape}, {to_.shape}"
        # ``from_`` is a long index tensor — only use it as the dtype hint
        # if the caller didn't override. Default to ``float32`` for the
        # projection-matrix data so existing call sites keep their behaviour;
        # callers needing complex / float64 pass ``dtype`` explicitly or
        # cast later via ``.type(...)``.
        if dtype is None or not (dtype.is_floating_point or dtype.is_complex):
            dtype = torch.float32
        projection = torch.sparse_coo_tensor(
            torch.stack([to_,from_],0),
            torch.ones_like(from_, dtype=dtype),
            size = (np.prod(to_shape), np.prod(from_shape))
        ).to_sparse_csr()
        self.register_buffer("projection", projection)
        self.from_shape = from_shape
        self.to_shape   = to_shape

    def type(self, dtype:torch.dtype):
        if dtype != self.dtype:
            self.projection = self.projection.type(dtype)
        return self

    @property
    def device(self):
        return self.projection.device

    @property
    def dtype(self):
        return self.projection.dtype

    def __call__(self, x:torch.Tensor)->torch.Tensor:
        """Scatter ``x`` via the cached CSR projection matrix.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[*from_shape, ...]``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[*to_shape, ...]``.
        """
        assert self.dtype == x.dtype, f"the dtype of x must be {self.dtype}, but got {x.dtype}"
        assert self.device == x.device, f"the device of x must be {self.device}, but got {x.device}"
        assert x.shape[:len(self.from_shape)] == self.from_shape, f"the shape of x must be [{self.from_shape}, ...], but got {x.shape}"

        dim_shape = x.shape[len(self.from_shape):]
        x = x.reshape(np.prod(self.from_shape), -1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            x = (self.projection @ x).squeeze(-1)
        else:
            x = self.projection @ x
        x = x.reshape(*self.to_shape, *dim_shape)
        return x

    def __str__(self):
        return f"{type(self).__name__}({self.from_shape} -> {self.to_shape}, device={self.device})"

    def __repr__(self):
        return str(self)

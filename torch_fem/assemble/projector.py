import torch
import torch.nn as nn
import numpy as np 

class Projector(nn.Module):
    """
    
    Attributes
    ----------
    projection: torch.sparse_csr_matrix
        the projection matrix
    from_shape: tuple or int or np.ndarray or torch.Size
        the input tensor should be of shape [*from_shape, ...]
    to_shape: tuple or int or np.ndarray or torch.Size
        the output tensor should be of shape [*to_shape, ...]
    
    
    """
    def __init__(self, from_, to_, from_shape, to_shape, dtype = None):
        """
        Parameters
        ----------
        from_: torch.Tensor or np.ndarray
            1D torch.Tensor or np.ndarray of shape [n_edges]
        to_: torch.Tensor or np.ndarray
            1D torch.Tensor or np.ndarray of shape [n_edges]
        from_shape: tuple or int or np.ndarray or torch.Size
            the input tensor should be of shape [*from_shape, ...]
        to_shape: tuple or int or np.ndarray or torch.Size
            the output tensor should be of shape [*to_shape, ...]
        Examples
        --------
        the basic usage of projector is like a sparse matrix
        >>> m = scipy.sparse.rand(3, 4, 0.5, format="coo")
        >>> p = Projector(m.col, m.row, 3, 4)
        """
        super().__init__()
        if isinstance(from_shape, int):
            from_shape = (from_shape,)
        elif isinstance(from_shape, np.ndarray):
            assert from_shape.ndim == 1, f"from_shape must be 1D, but got {from_shape.ndim}"

        if isinstance(to_shape, int):
            to_shape = (to_,)
        elif isinstance(to_shape, np.ndarray):
            assert to_shape.ndim == 1, f"to_shape must be 1D, but got {to_shape.ndim}"

        if isinstance(from_, np.ndarray):
            from_ = torch.from_numpy(from_)
        if isinstance(to_, np.ndarray):
            to_   = torch.from_numpy(to_)
        assert from_.shape == to_.shape, f"from_ and to_ must have the same shape, but got {from_.shape}, {to_.shape}"
        assert len(from_.shape) == 1, f"from_ and to_ must be 1D, but got {from_.shape}, {to_.shape}"
        if dtype is None:
            dtype = from_.dtype
        projection = torch.sparse_coo_tensor(
            torch.stack([to_,from_],0),
            torch.ones_like(from_,dtype=torch.float32),
            size = (np.prod(to_shape), np.prod(from_shape))
        ).to_sparse_csr()
        self.register_buffer("projection", projection)
        self.from_shape = from_shape
        self.to_shape   = to_shape

    def type(self, dtype):
        if dtype != self.dtype:
            self.projection = self.projection.type(dtype)
        return self

    @property
    def device(self):
        return self.projection.device

    @property
    def dtype(self):
        return self.projection.dtype

    def __call__(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            the input tensor of shape [*from_shape, ...]
        Returns
        -------
        torch.Tensor
            the output tensor of shape [*to_shape, ...]
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


Projector.type.__doc__ = nn.Module.type.__doc__
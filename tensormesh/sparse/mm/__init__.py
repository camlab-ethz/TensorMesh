from .cupy_mm  import SparseMMCupy, SparseMVCupy
from .scipy_mm import SparseMMScipy, SparseMVScipy
from .torch_mm import SparseMVTorch, SparseMMTorch
from ..utils import is_cupy_available

def spmv(edata, row, col, shape, B, backend=None):
    """
    Parameters
    ----------
    edata : torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int,  int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        1D tensor of shape [n_node]
        the dense vector
    backend: Optional[str]
        the backend to use, one of [None, 'scipy', 'torch', 'cupy']
        the backend should be correlated with the device of edata
        if device is CPU, then backend can be None, 'scipy', 'torch'
        if device is CUDA, then backend can be None, 'cupy'(you should install cupy first), 'torch'
    Returns
    -------
    torch.Tensor 
        1D tensor of shape [n_node]
        the output vector
    """
    assert backend in [None, 'scipy', 'torch', 'cupy']
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    assert B.dim() == 1
    if edata.device.type == 'cpu':
        if backend is None or backend == 'scipy':
            return SparseMVScipy.apply(edata, row, col, shape, B)
        elif backend == 'torch':
            return SparseMVTorch.apply(edata, row, col, shape, B)
        else:
            raise NotImplementedError(f"backend {backend} not supported for CPU")
    elif edata.device.type == 'cuda':
        if backend is None:
            if is_cupy_available:
                return SparseMVCupy.apply(edata, row, col, shape, B)
            else:
                # Use torch backend when cupy is not available (scipy doesn't support CUDA)
                return SparseMVTorch.apply(edata, row, col, shape, B)
        elif backend == 'cupy':
            assert is_cupy_available, f"cupy is not available"
            return SparseMVCupy.apply(edata, row, col, shape, B)
        elif backend == 'torch':
            return SparseMVTorch.apply(edata, row, col, shape, B)
        else:
            raise NotImplementedError(f"backend {backend} not supported for CUDA")
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")

def spmm(edata, row, col, shape, B, backend=None):
    """
    Parameters
    ----------
    edata: torch.Tensor 
        1D tensor of shape [n_edge]
        the edge data
    row  : torch.Tensor 
        1D tensor of shape [n_edge]
        the row indices
    col  : torch.Tensor 
        1D tensor of shape [n_edge]
        the column indices
    shape: Tuple[int int]
        the shape of the sparse matrix
    B    : torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the dense matrix/vector
    Returns:
    --------
    torch.Tensor 
        2D or 1D torch.Tensor of shape [n_node, n_feature] or [n_node]
        the output feature matrix
    """
    assert edata.dtype == B.dtype, f"A.dtype {edata.dtype} != B.dtype {B.dtype}"
    if B.dim() == 1:
        return spmv(edata, row, col, shape, B)
    assert B.dim() == 2
    if edata.device.type == 'cpu':
        if backend is None or backend == 'scipy':
            return SparseMMScipy.apply(edata, row, col, shape, B)
        elif backend == 'torch':
            return SparseMMTorch.apply(edata, row, col, shape, B)
        else:
            raise NotImplementedError(f"backend {backend} not supported for CPU")
    elif edata.device.type == 'cuda':
        if backend is None:
            if is_cupy_available:
                return SparseMMCupy.apply(edata, row, col, shape, B)
            else:
                # Use torch backend when cupy is not available (scipy doesn't support CUDA)
                return SparseMMTorch.apply(edata, row, col, shape, B)
        elif backend == 'cupy':
            assert is_cupy_available, f"cupy is not available"
            return SparseMMCupy.apply(edata, row, col, shape, B)
        elif backend == 'torch':
            return SparseMMTorch.apply(edata, row, col, shape, B)
        else:
            raise NotImplementedError(f"backend {backend} not supported for CUDA")
    else:
        raise NotImplementedError(f"device {edata.device.type} not supported")

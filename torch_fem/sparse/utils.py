import torch 

try:
    import petsc4py
    is_petsc_available = True
except ImportError:
    is_petsc_available = False
    
try:
    import cupy 
    is_cupy_available = True
except ImportError:
    is_cupy_available = False


def tensor2cupy(tensor):
    """turn torch.Tensor to cupy.ndarray
    Parameters
    ----------
    tensor : torch.Tensor
        the input tensor
    Returns
    -------
    cupy.ndarray
        the output cupy array
    """
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
def cupy2tensor(cupy):
    """turn cupy.ndarray to torch.Tensor
    
    Parameters
    ----------
    cupy : cupy.ndarray
        the input cupy array
    Returns
    -------
    torch.Tensor
        the output tensor
    
    """
    return torch.utils.dlpack.from_dlpack(cupy.toDlpack())
def shapeT(shape):
    """transpose the shape
    
    Parameters
    ----------
    shape : Tuple[int, int]
        the input shape
    Returns
    -------
    Tuple[int, int]
        the output shape
    """
    return (shape[1], shape[0])
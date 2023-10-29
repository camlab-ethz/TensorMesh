import torch 
import cupy as cp

def tensor2cupy(tensor):
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
def cupy2tensor(cupy):
    return torch.utils.dlpack.from_dlpack(cupy.toDlpack())
def shapeT(shape):
    return (shape[1], shape[0])
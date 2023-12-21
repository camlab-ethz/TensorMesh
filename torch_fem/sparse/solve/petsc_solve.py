from typing import Any
import torch 
from torch.autograd import Function
import scipy.sparse
import warnings
import importlib
from ..utils import tensor2cupy, cupy2tensor, shapeT, is_petsc_available



if is_petsc_available:
    PETSc = importlib.import_module('petsc4py.PETSc')
def petscvec2tensor(petscvec):
        """turn PETSc vector to torch.Tensor
        
        Parameters
        ----------
        petscvec : PETSc.Vec
            the input PETSc vector
        Returns
        -------
        torch.Tensor
            the output tensor
        """
        return torch.from_numpy(petscvec.getArray())


class SparseSolvePETSc(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_csr   = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsr()
        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        b_petsc = PETSc.Vec().createWithArray(b.numpy())
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setFromOptions()
        ksp.setType('cg')
        pc = ksp.getPC() # preconditioner
        pc.setType('ilu')
        x_petsc = b_petsc.duplicate()
        ksp.solve(b_petsc, x_petsc)
        u = petscvec2tensor(x_petsc)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        A_T_csr         = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=shapeT(ctx.A_shape)).tocsr()
        A_T_petsc       = PETSc.Mat().createAIJ(size=A_T_csr.shape, csr=(A_T_csr.indptr, A_T_csr.indices, A_T_csr.data))
        b_grad_petsc    = PETSc.Vec().createWithArray(grad_output.numpy())
        ksp             = PETSc.KSP().create()
        ksp.setOperators(A_T_petsc)
        ksp.setFromOptions()
        x_petsc         = b_grad_petsc.duplicate()
        ksp.solve(b_grad_petsc, x_petsc)
        b_grad          = petscvec2tensor(x_petsc)

        edata_grad      = - b_grad[row] * u[col]

        return edata_grad, None, None, None, b_grad


class SparseLUSolvePETSc(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        A_csc   = scipy.sparse.coo_matrix((edata.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsc()
        A_petsc = PETSc.Mat().createAIJ(size=A_csc.shape, csr=(A_csc.indptr, A_csc.indices, A_csc.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setFromOptions()    
        u = torch.zeros_like(b)
        for i in range(b.shape[1]):
            b_petsc = PETSc.Vec().createWithArray(b[:,i].numpy())
            x_petsc = b_petsc.duplicate()
            ksp.solve(b_petsc[i], x_petsc[i])
            u[:,i] = petscvec2tensor(x_petsc)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u
    
    @staticmethod
    def backward(ctx, grad_output):
        edata, row, col, u = ctx.saved_tensors
        
        A_T_csc         = scipy.sparse.coo_matrix((edata.numpy(), (col.numpy(), row.numpy())), shape=shapeT(ctx.A_shape)).tocsc()
        A_T_petsc       = PETSc.Mat().createAIJ(size=A_T_csc.shape, csr=(A_T_csc.indptr, A_T_csc.indices, A_T_csc.data))
        ksp             = PETSc.KSP().create()
        ksp.setOperators(A_T_petsc)
        ksp.setFromOptions()
        b_grad          = torch.zeros_like(grad_output)
        for i in range(b_grad.shape[1]):
            b_grad_petsc    = PETSc.Vec().createWithArray(grad_output[:,i].numpy())
            x_petsc         = b_grad_petsc.duplicate()
            ksp.solve(b_grad_petsc[i], x_petsc[i])
            b_grad[:,i]     = petscvec2tensor(x_petsc)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad
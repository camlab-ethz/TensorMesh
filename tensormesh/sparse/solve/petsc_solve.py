from typing import Any
import torch 
from torch.autograd import Function
import scipy.sparse
import warnings
from ..utils import shapeT, is_petsc_available

# Lazy import PETSc only when needed
PETSc = None

def _get_petsc():
    global PETSc
    if PETSc is None:
        if not is_petsc_available:
            raise ImportError(
                "petsc4py is not available. Install it with: pip install petsc4py "
                "or use a different backend (scipy, torch, cupy)"
            )
        from petsc4py import PETSc as _PETSc
        PETSc = _PETSc
    return PETSc
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


def _coo_arrays(edata, row, col):
    """Detach + move to CPU + numpy. Required because saved tensors keep requires_grad."""
    return (
        edata.detach().cpu().numpy(),
        row.detach().cpu().numpy(),
        col.detach().cpu().numpy(),
    )


class SparseSolvePETSc(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        PETSc = _get_petsc()
        edata_np, row_np, col_np = _coo_arrays(edata, row, col)
        A_csr   = scipy.sparse.coo_matrix((edata_np, (row_np, col_np)), shape=shape).tocsr()
        A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        b_petsc = PETSc.Vec().createWithArray(b.detach().cpu().numpy())
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setFromOptions()
        ksp.setType('bcgs')
        pc = ksp.getPC() # preconditioner
        pc.setType('ilu')
        x_petsc = b_petsc.duplicate()
        ksp.solve(b_petsc, x_petsc)
        u = petscvec2tensor(x_petsc).to(dtype=b.dtype, device=b.device)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u

    @staticmethod
    def backward(ctx, grad_output):
        PETSc = _get_petsc()
        edata, row, col, u = ctx.saved_tensors
        edata_np, row_np, col_np = _coo_arrays(edata, row, col)
        A_T_csr         = scipy.sparse.coo_matrix((edata_np, (col_np, row_np)), shape=shapeT(ctx.A_shape)).tocsr()
        A_T_petsc       = PETSc.Mat().createAIJ(size=A_T_csr.shape, csr=(A_T_csr.indptr, A_T_csr.indices, A_T_csr.data))
        b_grad_petsc    = PETSc.Vec().createWithArray(grad_output.detach().cpu().numpy())
        ksp             = PETSc.KSP().create()
        ksp.setOperators(A_T_petsc)
        ksp.setFromOptions()
        x_petsc         = b_grad_petsc.duplicate()
        ksp.solve(b_grad_petsc, x_petsc)
        b_grad          = petscvec2tensor(x_petsc).to(dtype=u.dtype, device=u.device)

        edata_grad      = - b_grad[row] * u[col]

        return edata_grad, None, None, None, b_grad


class SparseLUSolvePETSc(Function):
    @staticmethod
    def forward(ctx, edata, row, col, shape, b) -> Any:
        PETSc = _get_petsc()
        edata_np, row_np, col_np = _coo_arrays(edata, row, col)
        b_np = b.detach().cpu().numpy()
        A_csc   = scipy.sparse.coo_matrix((edata_np, (row_np, col_np)), shape=shape).tocsc()
        A_petsc = PETSc.Mat().createAIJ(size=A_csc.shape, csr=(A_csc.indptr, A_csc.indices, A_csc.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setFromOptions()
        u = torch.zeros_like(b)
        b_petsc = PETSc.Vec().createWithArray(b_np[:, 0])
        for i in range(b.shape[1]):
            b_petsc.setArray(b_np[:, i])
            x_petsc = b_petsc.duplicate()
            ksp.solve(b_petsc, x_petsc)
            u[:, i] = petscvec2tensor(x_petsc).to(dtype=b.dtype, device=b.device)
        ctx.save_for_backward(edata, row, col, u)
        ctx.A_shape = shape
        return u

    @staticmethod
    def backward(ctx, grad_output):
        PETSc = _get_petsc()
        edata, row, col, u = ctx.saved_tensors
        edata_np, row_np, col_np = _coo_arrays(edata, row, col)
        grad_np = grad_output.detach().cpu().numpy()

        A_T_csc         = scipy.sparse.coo_matrix((edata_np, (col_np, row_np)), shape=shapeT(ctx.A_shape)).tocsc()
        A_T_petsc       = PETSc.Mat().createAIJ(size=A_T_csc.shape, csr=(A_T_csc.indptr, A_T_csc.indices, A_T_csc.data))
        ksp             = PETSc.KSP().create()
        ksp.setOperators(A_T_petsc)
        ksp.setFromOptions()
        b_grad          = torch.zeros_like(grad_output)
        for i in range(b_grad.shape[1]):
            b_grad_petsc    = PETSc.Vec().createWithArray(grad_np[:, i])
            x_petsc         = b_grad_petsc.duplicate()
            ksp.solve(b_grad_petsc, x_petsc)
            b_grad[:, i]    = petscvec2tensor(x_petsc).to(dtype=u.dtype, device=u.device)

        edata_grad      = - (b_grad[row] * u[col]).sum(-1)
        return edata_grad, None, None, None, b_grad

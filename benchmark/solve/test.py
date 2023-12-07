import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
sys.path.append("../..")
import torch 
import skfem
import matplotlib
import matplotlib.pyplot as plt
from torch_fem import ElementAssembler, LaplaceElementAssembler, func_node_assembler, Mesh, Condenser, dot
from torch_fem.dataset import PoissonMultiFrequency
from torch_fem.sparse import SparseMatrix
matplotlib.use('Agg')

def skfem_asm(mesh):
    mesh_skfem = skfem.MeshTri(mesh.points.T.cpu().numpy(), mesh.elements().T.numpy())
    basis = skfem.InteriorBasis(mesh_skfem, skfem.ElementTriP1())

    @skfem.BilinearForm
    def laplace(u, v, w):
        from skfem.helpers import dot,grad 
        return dot(grad(u), grad(v))
    
    K = laplace.assemble(basis)
    
    return K


dataset = PoissonMultiFrequency(K=2)
mesh = Mesh.gen_rectangle(chara_length=0.05, element_type="tri") 


class PoissonElementAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
       
        return dot(gradu, gradv)

K_asm = PoissonElementAssembler.from_mesh(mesh)
F     = dataset.initial_condition(mesh.points).flatten()
U     = dataset.solution(mesh.points).flatten()


condenser = Condenser(mesh.boundary_mask)

K =  K_asm(mesh.points)

K_skfem =  skfem_asm(mesh)

K_, F_ = condenser(K, F)
# breakpoint()

u_scipy_ = K_.solve(F_,  backend="scipy")
u_scipy  = condenser.recover(u_scipy_)

# breakpoint()
print(f"scipy error: {np.linalg.norm(u_scipy - u_scipy)}")

u_petsc_ = K_.solve(F_, backend="petsc")
u_petsc  = condenser.recover(u_petsc_)

print(f"petsc error: {np.linalg.norm(u_petsc - u_scipy)}")


u_torch_ = K_.solve(F_, backend="torch")
u_torch  = condenser.recover(u_torch_)


print(f"torch error: {np.linalg.norm(u_torch - u_scipy)}")

# breakpoint()

# mesh.plot(values={
#     "initial" : F,
#     "solution" : U,
#     "prediction" : u_scipy
# })

# plt.show()

fig, axes = plt.subplots(1,3)
axes[0].hist(u_scipy-u_scipy, bins=200)
axes[1].hist(u_petsc-u_scipy, bins=200)
axes[2].hist(u_torch-u_scipy, bins=200)

plt.savefig("hist.png")

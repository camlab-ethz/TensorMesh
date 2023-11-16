import sys 
sys.path.append("../..")

import torch
from torch_fem import ElementAssembler, Mesh,Condenser
from torch_fem.dataset import WaveMultiSinCos
from torch_fem.utils import mul, dot
import skfem
import meshio
import numpy as np

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return dot(gradu, gradv)
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return mul(u, v)

def wave_torchfem(dt=0.001, c=4.0, n=100):
    torch.random.manual_seed(123456)
    
    mesh = Mesh.gen_rectangle(chara_length=0.05)
    # mesh = Mesh.gen_rectangle(chara_length=0.01, element_type="quad")
   
    dataset = WaveMultiSinCos(K=4, c=c)

    u0 = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    
    M = M_asm(mesh.points)
    A = A_asm(mesh.points)

    condenser = Condenser(mesh.boundary_mask)

    Us  = [u0] 
    v0 = torch.zeros_like(u0) 
    A = c*c*A
    K = 2 * M
    F = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
    K_, F_ = condenser(K, F)
    U_     = K_.solve(F_)
    U      = condenser.recover(U_)
    M_     = condenser(M)[0]
    Us.append(U)
    for _ in range(n-2):
        U1, U2 = Us[-2:]

        F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
        
        F_ = condenser.condense_rhs(F)

        U_ = M_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in range(n)]
    mesh.plot({"prediction":Us, "ground truth":Us_gt},save_path="wave_torchfem.mp4", backend="matplotlib", dt=dt, show_mesh=True )
        

    return Us

def wave_skfem(dt=0.001, c=4.0, n=100):
    torch.random.manual_seed(123456)
    
    # mesh = skfem.MeshTri().refined(3).with_boundaries(
    #     {
    #         "left": lambda x: x[0] == 0,
    #         "right": lambda x: x[0] == 1,
    #         "top": lambda x: x[1] == 1,
    #         "bottom": lambda x: x[1] == 0
    #     }
    # )
    mesh = Mesh.gen_rectangle(chara_length=0.1)

    mesh_skfem = skfem.MeshTri(mesh.points.T.detach().numpy(), mesh.elements().T.numpy())

    basis = skfem.InteriorBasis(mesh_skfem, skfem.ElementTriP1())

    dataset = WaveMultiSinCos(K=4, c=c)

    u0 = dataset.initial_condition(mesh.points.detach())
    u0 = u0.numpy()

    @skfem.BilinearForm
    def laplace(u, v, w):
        from skfem.helpers import dot,grad 
        return dot(grad(u), grad(v))

    @skfem.BilinearForm
    def mass(u, v, w):
        from skfem.helpers import dot,grad 
        return u * v
    
    M = skfem.asm(mass, basis)
    A = skfem.asm(laplace, basis)

    boundary_dofs = basis.get_dofs(torch.where(mesh.boundary_mask)[0].detach().numpy()).all()


    Us  = [u0] 
    v0 = np.zeros_like(u0) 
    A = c*c*A
    K = 2 * M
    F = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
    F  = dt * dt * A @ u0 + 2 * M @ u0 - 2 * dt * M @ v0
    U  = skfem.solve(*skfem.condense(K, F, D=boundary_dofs))
    
    Us.append(U)
    for _ in range(n-2):
        U1, U2 = Us[-2:]

        F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
        # F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2

        U = skfem.solve(*skfem.condense(M, F, D=boundary_dofs))

        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in range(n)]

    mesh.plot({"prediction":Us, "ground truth":Us_gt},save_path="wave_skfem.mp4", backend="matplotlib", dt=dt, show_mesh=True )
    
    return Us


if __name__ == '__main__':

    # wave_skfem()
    wave_torchfem()

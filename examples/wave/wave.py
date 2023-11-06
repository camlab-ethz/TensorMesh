import sys 
sys.path.append("../..")

import torch
from torch_fem import ElementAssembler, Mesh,Condenser
from torch_fem.dataset import WaveMultiSinCos
from torch_fem.utils import mul, dot

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

if __name__ == '__main__':

    dt = 0.001 
    c  = 4.0
    n  = 100

    torch.random.manual_seed(123456)
    mesh = Mesh.gen_rectangle(chara_length=0.02)
   
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

    mesh.plot({"prediction":Us, "ground truth":Us_gt},save_path="wave.gif", backend="matplotlib", dt=dt)
    

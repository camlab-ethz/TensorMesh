import sys 
sys.path.append("../..")

import torch
from torch_fem import ElementAssembler, Mesh,Condenser
from torch_fem.dataset import HeatMultiSinCos

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
       
        return gradu @ gradv.T
    
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
        return torch.outer(u, v)

if __name__ == '__main__':
    torch.random.manual_seed(42)
    mesh = Mesh.gen_rectangle(chara_length=0.02, element_type="tri")

    dataset = HeatMultiSinCos(d=8)

    u0 = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    M = M_asm(mesh.points)
    A = A_asm(mesh.points)
    condenser = Condenser(mesh.boundary_mask)

    U = u0 
    dt = 0.001 
    D  = 1
    n  = 50
    K  = M + dt * D * D * A 
    K_ = condenser(K)[0]

    Us = [U]
    for _ in range(n):
        F = M @ U # [num_node]

        F_ = condenser.condense_rhs(F)

        U_ = K_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in range(n)]

    mesh.plot({"prediction":Us, "ground truth":Us_gt},save_path="heat.gif", backend="matplotlib", dt=dt)
    

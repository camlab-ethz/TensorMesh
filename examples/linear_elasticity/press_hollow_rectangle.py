import sys 
sys.path.append("../..")
import torch 

from torch_fem import ElementAssembler, NodeAssembler, Mesh,Condenser
from torch_fem.dataset import WaveMultiSinCos
from torch_fem.utils import trace, eye, sym, dot, matrix, transpose, matmul, vector


class KAssembler(ElementAssembler):
    def precompute(self):
        E = 1
        nu = 0.3

        D = E/(1 - nu*nu) * torch.tensor([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, 0.5*(1-nu)]
        ])
        self.D = D


    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                K: torch.Tensor[n_basis, n_basis, n_dim, n_dim]
        """

        D = self.D.type(gradu.dtype).to(gradu.device)

        zeros = torch.zeros_like(gradu[:, 0]) # [n_basis]
        B = matrix([
            [gradu[:, 0], zeros],
            [zeros, gradu[:, 1]],
            [gradu[:, 1], gradu[:, 0]]
        ]) # [n_basis, 3, 2]
     
        C  = matmul(D, B) # [3, 3] @ [n_basis, 3, 2] = [n_basis, 3, 2]

        K  = dot(C, B, reduce_dim=-2) # [n_basis, n_basis, 2, 2]
       
        return K
    

    

if __name__ == '__main__':
    mesh = Mesh.gen_hollow_rectangle(chara_length=0.02, element_type="quad").double()

    K_asm = KAssembler.from_mesh(mesh, quadrature_order=3)

    K = K_asm(mesh.points)
    F = torch.zeros_like(mesh.points)
    F[mesh.point_data['is_outer_top_boundary'], 1] = 0.05
    F = F.flatten()

    dirichlet_mask = torch.zeros_like(mesh.points, dtype=torch.bool)
    dirichlet_mask[mesh.point_data['is_outer_bottom_boundary'], :] = True

    condenser = Condenser(dirichlet_mask.flatten())

    K_, f_ = condenser(K, F)

    u_ = K_.solve(f_)

    u = condenser.recover(u_)

    # u = u.reshape(-1, 2)

    mesh.plot(
        {"ux":u[0::2],
         "uy":u[1::2]},
         backend="matplotlib",
        save_path="press_hollow_rectangle.png",
        show_mesh=True
    )

import sys 
sys.path.append("../..")
import torch 

from torch_fem import ElementAssembler, NodeAssembler, Mesh,Condenser
from torch_fem.dataset import WaveMultiSinCos
from torch_fem.utils import trace, eye, sym, ddot


class KAssembler(ElementAssembler):
    def forward(self, u, v, gradu, gradv):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                K: torch.Tensor[n_basis, n_basis, n_dim, n_dim]
        """
        d = 0.1
        E = 200e9
        nu = 0.3
        n_basis, n_dim = gradu.shape
        strain_u = sym(gradu) # [n_basis, n_dim, n_dim]
        stress_u = E / (1 + nu) * (strain_u + nu / (1 - nu) * eye(trace(strain_u), n_dim))   # [n_basis, n_dim, n_dim]
        strain_v = sym(gradv) # [n_basis, n_dim, n_dim]

        return d**3 / 12.0 * ddot(stress_u, strain_v)
    
class FAssembler(NodeAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                F: torch.Tensor[n_basis]
        """
        return v
       
    

if __name__ == '__main__':
    mesh = Mesh.gen_rectangle(chara_length=0.02, element_type="quad").double()

    K_asm = KAssembler.from_mesh(mesh)
    F_asm = FAssembler.from_mesh(mesh)

    K = K_asm(mesh.points)
    F = F_asm(mesh.points) * 1e6

    boundary_mask = mesh.point_data['is_left_boundary'] | mesh.point_data['is_top_boundary'] | mesh.point_data['is_right_boundary']

    condenser = Condenser(boundary_mask)

    K_, f_ = condenser(K, F)

    u_ = K_.solve(f_)

    u = condenser.recover(u_)

    mesh.plot(
        {"u":u},
         backend="matplotlib",
        save_path="kirchhoff_plate_bending.png",
        show_mesh=True
    )

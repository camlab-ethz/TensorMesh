import sys 
sys.path.append("../../..")

import torch
from tqdm import tqdm
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.dataset import HeatMultiFrequency

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """gradu, gradv: [n_dim] → scalar"""
        return gradu @ gradv

class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """u, v: scalar → scalar"""
        return u * v

if __name__ == '__main__':
    torch.random.manual_seed(3)
    mesh = Mesh.gen_rectangle(chara_length=0.02,order=2, element_type="tri")
    #mesh = Mesh.gen_L(chara_length=0.008, element_type="tri")
    dataset = HeatMultiFrequency(d=16)

    u0 = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    # M = M_asm(mesh.points)
    # A = A_asm(mesh.points)
    M = M_asm() 
    A = A_asm()
    
    # new_boundary_mask = torch.zeros_like(mesh.boundary_mask, dtype=torch.bool)
    # mesh.boundary_mask = new_boundary_mask
    condenser = Condenser(mesh.boundary_mask)

    U = u0 
    dt = 0.00005
    D  = 1
    n  = 100
    K  = M + dt * D * D * A
    K_ = condenser(K)[0]

    Us = [U]
    for _ in tqdm(range(n-1), desc="Time stepping"):
        F = M @ U # [num_node]

        F_ = condenser.condense_rhs(F)

        U_ = K_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in tqdm(range(n), desc="Ground truth")]

    mesh.plot(
        {"prediction":Us, "ground truth":Us_gt},
        save_path="heat.mp4",
        dt=dt,
        show_mesh=False,
        fix_clim=False)
    

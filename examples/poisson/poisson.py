import sys
sys.path.append("../..")

import torch
import numpy as np
from tqdm import tqdm
from tensormesh import LaplaceElementAssembler, Mesh, Condenser
from tensormesh.dataset import PoissonMultiFrequency
from tensormesh.visualization import StreamPlotter
from tensormesh import NodeAssembler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.random.manual_seed(1)
    #mesh      = Mesh.gen_circle(chara_length=0.015)
    #mesh    = Mesh.gen_L(chara_length=0.02)
    mesh    = Mesh.gen_rectangle(chara_length=0.02)
    # mesh = Mesh.gen_rectangle(chara_length=0.02, order=2, element_type="tri")
    
    assembler = LaplaceElementAssembler.from_mesh(mesh)
    equation  = PoissonMultiFrequency(K=16)
    dirchlet_value = torch.zeros(mesh.boundary_mask.shape)
    condenser = Condenser(mesh.boundary_mask, dirchlet_value)
    f = equation.source_term(mesh.points,domain="rectangle")
    K = assembler(mesh.points)
    class FAssembler(NodeAssembler):
        def forward(self, v, f):
            return v * f
    F_asm = FAssembler.from_mesh(mesh)
    f = F_asm(mesh.points, point_data={"f":f})
    K, f = condenser(K, f)
    
    u_analytical = equation.solution(mesh.points)
    u = K.solve(f)
    u = condenser.recover(u)
    f = condenser.recover(f)
    mesh.plot(
        {
            "f":f,
            "u_fem":u,
            "u_analytical":u_analytical
        },
        save_path="poisson.png"
    )



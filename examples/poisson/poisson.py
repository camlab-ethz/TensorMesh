import sys
sys.path.append("../..")

import torch
from tensormesh import (LaplaceElementAssembler, Mesh,
                        Condenser, NodeAssembler)
from tensormesh.dataset import PoissonMultiFrequency

device = "cuda" if torch.cuda.is_available() else "cpu"
mesh      = Mesh.gen_rectangle(chara_length=0.02).to(device=device)
assembler = LaplaceElementAssembler.from_mesh(mesh)
equation  = PoissonMultiFrequency(K=16)
boundary_value = torch.zeros(mesh.boundary_mask.shape).to(device=device)
condenser = Condenser(mesh.boundary_mask,
                      boundary_value)

f = equation.source_term(mesh.points, domain="rectangle")
K = assembler(mesh.points)

class FAssembler(NodeAssembler):
    def forward(self, v, f):
        return v * f

F_asm = FAssembler.from_mesh(mesh)
b     = F_asm(mesh.points, point_data={"f": f})

K_, b_       = condenser(K, b)
u_           = K_.solve(b_, verbose=True)
u            = condenser.recover(u_)
u_analytical = equation.solution(mesh.points)

mesh.plot({"f": f, "u_fem": u, "u_analytical": u_analytical},
          save_path="poisson.png")



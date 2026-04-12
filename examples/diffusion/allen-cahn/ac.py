import sys 
sys.path.append("../../..")
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from tensormesh import ElementAssembler, NodeAssembler, Condenser, Mesh
from tensormesh.dataset import WaveMultiFrequency, PoissonMultiFrequency


dt = 1e-6


class KAssembler(ElementAssembler):
    def __post_init__(self):
        epsilon = 220
        self.dt = dt
        self.dcdotdc = 1.0/dt
        self.D  = lambda x: 1.0e0 
        self.dD = lambda x: 0.0e0
        self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
        self.df = lambda x: -epsilon**2*(3*x**2 - 1)
    def forward(self, u, v, gradu, gradv, c, gradc, cold):
       
        return -1.0 * (self.dcdotdc * (u * v) +
                self.dD(c) * u * (gradv  @ gradc) +
                self.D(c) * (gradu @ gradv) -
                self.df(c) * (u * v))
   

class RAssembler(NodeAssembler):
    def __post_init__(self):
        self.dt = dt
        epsilon = 220
        self.dcdotdc = 1.0/dt
        self.D  = lambda x: 1.0e0 
        self.dD = lambda x: 0.0e0
        self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
        self.df = lambda x: -epsilon**2*(3*x**2 - 1)

    def forward(self, v, gradv, c, gradc, cold):
        cdot = (c - cold) / self.dt
        
        R =  cdot * v + self.D(c) * (gradv @ gradc) - self.f(c) * v
      
        return R
    

if __name__ == '__main__':
    # mesh = Mesh.gen_rectangle(chara_length=0.02, element_type="quad")
    mesh = Mesh.gen_rectangle(chara_length=0.02, element_type="tri")
    # dataset = WaveMultiFrequency(K=24, r=1)
    dataset = PoissonMultiFrequency(K=24, r=1)

    cold = dataset.source_term(mesh.points)

    cs = []
    cs.append(cold)

    K_asm = KAssembler.from_mesh(mesh)
    R_asm = RAssembler.from_mesh(mesh)
    new_boundary_mask = torch.zeros_like(mesh.boundary_mask, dtype=torch.bool)
    #mesh.boundary_mask = new_boundary_mask
    condenser = Condenser(new_boundary_mask)

    max_iter = 50
    steps    = 200
    pbar = tqdm(total=steps)
    for step in range(steps):
        c = cold
        n_iter, converged = 0, False
        while n_iter < max_iter and (not converged):  
            point_data = {"c":c,"cold":cold}  
          
            K = K_asm(mesh.points, point_data=point_data)
            R = R_asm(mesh.points, point_data=point_data)
            
            K_, R_ = condenser(K, R)
            dC_ = K_.solve(R_)
            dC  = condenser.recover(dC_)

            c  = c + dC
            n_iter += 1

            rnorm = torch.linalg.norm(R_)
            if rnorm < 1e-10:
                converged = True

        pbar.update(1)
        pbar.set_postfix({"rnorm":rnorm})

        cold = c 
        cs.append(cold)

    mesh.plot(values={
        "cs":cs
    },show_mesh=False, dt=1e-6, save_path="Allen-Cahn.mp4")      


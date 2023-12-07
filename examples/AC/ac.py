import sys 
sys.path.append("../..")
import torch
from tqdm import tqdm
from torch_fem import ElementAssembler, NodeAssembler, Condenser, Mesh, dot, mul
from torch_fem.dataset import PoissonMultiFrequency




class KAssembler(ElementAssembler):
    def __post_init__(self):
        dt = 4e-6
        epsilon = 220
        self.dt = dt
        self.dcdotdc = 1.0/dt
        self.D  = lambda x: 1.0e0 
        self.dD = lambda x: 0.0e0
        self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
        self.df = lambda x: -epsilon**2*(3*x**2 - 1)
    def forward(self, u, v, gradu, gradv, c, gradc, cold):
        breakpoint()
        return -1.0 * (self.dcdotdc * mul(u, v) +
                self.dD(c) * u * dot(gradc, gradv) + 
                self.D(c) * dot(gradu, gradv) -
                self.df(c) * mul(u, v))
   

class RAssembler(NodeAssembler):
    def __post_init__(self):
        self.dt = 4e-6
        epsilon = 220
        self.dcdotdc = 1.0/self.dt
        self.D  = lambda x: 1.0e0 
        self.dD = lambda x: 0.0e0
        self.f  = lambda x: -epsilon**2*x*(x**2 - 1)
        self.df = lambda x: -epsilon**2*(3*x**2 - 1)

    def forward(self, v, gradv, c, gradc, cold):
        cdot = (c - cold) / self.dt

        R =  cdot * v + self.D(c) * (gradc * gradv).sum(-1) - self.f(c) * v
      
        return R
    

if __name__ == '__main__':
    mesh = Mesh.gen_rectangle(chara_length=0.05, element_type="quad")
    dataset = PoissonMultiFrequency(K=24, r=1)
    
    cold = dataset.initial_condition(mesh.points)
    cs = []
    cs.append(cold)

    K_asm = KAssembler.from_mesh(mesh)
    R_asm = RAssembler.from_mesh(mesh)

    condenser = Condenser(mesh.boundary_mask)

    max_iter = 50
    pbar = tqdm(total=50)
    for step in range(50):
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
            if rnorm < 1e-5:
                converged = True

        pbar.update(1)
        pbar.set_postfix({"rnorm":rnorm})

        cold = c 
        cs.append(cold)

    mesh.plot(values={
        "cs":cs
    },show_mesh=True, dt=4e-6, save_path="cs.mp4")      


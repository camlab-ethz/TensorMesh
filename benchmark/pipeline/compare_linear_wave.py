import sys 
import os
import torch
sys.path.append("../..")
import torch_fem as thfem
import skfem
import skfem.helpers
import pandas as pd
import time
import cupy as cp
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import fenics
from torch_fem.profile import TimeProfiler, CPUProfiler, CUDAProfiler, get_max_memory_for_index, get_memory_for_index
import argparse
import numpy as np

class SkFEM:
    def __init__(self, mesh, n, c, dt, u0, element="tri"):
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        skfem_mesh = skfem.Mesh.load("tmp.msh")
    
        self.mesh  = skfem_mesh
        self.element = element
        self.c = c 
        self.dt = dt
        self.n = n
        self.u0 = u0.numpy()
        self.boundary = mesh.boundary_mask.numpy()

    def __call__(self):
        element = skfem.ElementTriP1() if self.element == "tri" else skfem.ElementTetP1()
        @skfem.BilinearForm
        def A_asm(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v)) 
        
        @skfem.BilinearForm
        def M_asm(u, v, w):
            return u*v
        
        basis = skfem.Basis(self.mesh, element)
        A     = skfem.asm(A_asm, basis)
        M     = skfem.asm(M_asm, basis)

        v0    = np.zeros(basis.N)
        A     = self.c * self.c * A 
        K     = 2 * M 
        F     = -self.dt * self.dt * A @ self.u0 + 2 * M @ self.u0 + 2 * self.dt * M @ v0

        U     = skfem.solve(*skfem.condense(K, F, D=self.boundary))

        U1, U2 = self.u0, U
        for _ in range(self.n-2):
            F = 2 * M @ U2 - M @ U1 - self.dt * self.dt * A @ U2
            
            U  = skfem.solve(*skfem.condense(M, F, D=self.boundary))
         
            U1, U2 = U2, U

      
class ThFEM:
    def __init__(self, mesh, n, c, dt,u0):
        self.mesh = mesh
        self.A_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.M_asm  = thfem.MassElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)

        self.n = n
        self.c = c
        self.dt = dt
        self.u0 = u0.type(mesh.dtype).to(mesh.device)

    def __call__(self):
        v0 = torch.zeros_like(self.u0)        
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        condenser = thfem.Condenser(self.mesh.boundary_mask)        
        A     = self.A_asm(self.mesh.points)
        M     = self.M_asm(self.mesh.points)

        A     = self.c * self.c * A
        K     = 2 * M
        F     = -self.dt * self.dt * A @ self.u0 + 2 * M @ self.u0 + 2 * self.dt * M @ v0
        K_, F_ = condenser(K, F)
        U_     = K_.solve(F_, backend=backend)
        U      = condenser.recover(U_)
        M_     = condenser(M)[0]

        U1, U2 = self.u0, U
        for _ in range(self.n-2):
            F = 2 * M @ U2 - M @ U1 - self.dt * self.dt * A @ U2
            
            F_ = condenser.condense_rhs(F)

            U_ = M_.solve(F_, backend=backend)

            U  = condenser.recover(U_)

            U1, U2 = U2, U


class feFEM:
    def __init__(self, mesh, n, c, dt,u0):
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        self.fenics_mesh = fenics_mesh

        boundary_facets = fenics.MeshFunction('size_t', fenics_mesh, fenics_mesh.topology().dim()-1)
        boundary_facets.set_all(0)

        # Mark facets as boundary if any of its nodes are in the boundary mask
        for facet in fenics.SubsetIterator(boundary_facets, 0):
            for vertex in facet.entities(0):  # Checking the vertices of each facet
                if mesh.boundary_mask[vertex]:
                    boundary_facets[facet] = 1
                    break

        self.n = n
        self.c = c
        self.dt = dt
        self.u0 = u0.numpy()
        self.boundary_facets = boundary_facets
        

    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        bc = fenics.DirichletBC(V, fenics.Constant(0), self.boundary_facets, 1)

        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        m = fenics.inner(u, v) * fenics.dx

        A = fenics.assemble(self.c * self.c * a)
        M = fenics.assemble(2 * m)

        # Assuming self.u0 is a FEniCS Function or a similar compatible object
        u0_fenics = fenics.Function(V)
        u0_fenics.vector()[:] = self.u0  
        u0_vec = u0_fenics.vector()[:]  # Get the numpy representation if it's a FEniCS Function
        v0 = np.zeros_like(u0_vec)

        F_vec = -self.dt * self.dt * A * u0_vec + 2 * M * u0_vec + 2 * self.dt * M * v0
        F = fenics.Function(V)
        F.vector()[:] = F_vec

        U = fenics.Function(V)
        fenics.solve(M, U.vector(), F.vector(), "cg", "petsc_amg")

        U1, U2 = u0_fenics, U
        for _ in range(self.n-2):
            F_vec = 2 * M * U2.vector()[:] - M * U1.vector()[:] - self.dt * self.dt * A * U2.vector()[:]
            F.vector()[:] = F_vec
            fenics.solve(M, U.vector(), F.vector(), "cg", "petsc_amg")
            U1, U2 = U2, U

def plot_comparison(element_type, chara_lengths, n_times, csv_path, 
                    c, dt, n,
                    device="cuda:0"):
    data = {
        "degree of freedom":[],
        "backend":[],
        "time in s":[],
        "chara length":[],
        "CPU peak mem in GB":[],
        "CPU mean mem in GB":[],
        "GPU peak mem in GB":[],
        "GPU mean mem in GB":[],
    }
    pbar = tqdm(total=len(chara_lengths)*n_times*4)

    for chara_length in chara_lengths:
        if element_type == "tri":
            mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
        elif element_type == "tetra":
            mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
        else:
            raise NotImplementedError(f"element_type={element_type} is not supported")
        u0 = torch.rand(mesh.points.shape[0])
        th_fem_cpu_1 = ThFEM(mesh.clone(), n=n, c=c, dt=dt, u0=u0)
        th_fem_gpu_1 = ThFEM(mesh.clone().to(device), n=n, c=c, dt=dt, u0=u0)
        sk_fem       = SkFEM(mesh, element=element_type, n=n, c=c, dt=dt, u0=u0)
        fe_fem       = feFEM(mesh,  n=n, c=c, dt=dt, u0=u0)
        
        for name, fem in zip([
                                "torch_fem cpu", 
                                "torch_fem cuda",
                                "scikit-fem", 
                                "fenics"], [
                                            th_fem_cpu_1,
                                            th_fem_gpu_1,
                                            sk_fem, 
                                            fe_fem]):
            for _ in range(n_times):
                
                with CPUProfiler() as cpu_profiler:
                    fem()
                data["CPU peak mem in GB"].append(cpu_profiler.max())
                data["CPU mean mem in GB"].append(cpu_profiler.mean())
                with CUDAProfiler() as cuda_profiler:
                    fem()
                data["GPU peak mem in GB"].append(cuda_profiler.max())
                data["GPU mean mem in GB"].append(cuda_profiler.mean())
                with TimeProfiler() as time_profiler:
                    fem()
                data["time in s"].append(time_profiler.time)
                data["chara length"].append(chara_length)
                data["degree of freedom"].append(mesh.points.shape[0])
                data["backend"].append(name)
                pbar.update(1)
                pbar.set_postfix({
                    "dofs":mesh.points.shape[0],
                    "chara_length":chara_length,
                    "backend":name,
                })
    

    df = pd.DataFrame(data)
    df.to_csv(csv_path)
   
    fig, ax = plt.subplots(ncols=3, figsize=(15, 4))

    markers = ["o", "s", "p", "^"]
    linestyles = ["--", "--", "--", "--"]
    sns.pointplot(x="degree of freedom", y="time in s", hue="backend", markers=markers, linestyles=linestyles,  data=df, ax=ax[0])
    sns.pointplot(x="degree of freedom", y="CPU peak mem in GB", hue="backend",markers=markers, linestyles=linestyles,  data=df, ax=ax[1])
    sns.pointplot(x="degree of freedom", y="GPU peak mem in GB", hue="backend",markers=markers, linestyles=linestyles, data=df, ax=ax[2])
    # for i in range(3):
    #     ax[i].set_xscale("log")
    #     ax[i].set_yscale("log")

    fig.tight_layout()

    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_index", type=int, default=0)
    parser.add_argument("-n", "--num_steps", type=int ,  default=20)
    parser.add_argument("-c", "--c", type=float ,  default=1.0)
    parser.add_argument("-dt", "--chara_length", type=float ,  default=0.001)
    parser.add_argument("-t", "--times", type=int ,  default=5)
    args = parser.parse_args()

    mem = get_max_memory_for_index(args.device_index) - get_memory_for_index(args.device_index)

    class Rectangle:
        @staticmethod
        def mem2dof(mem):
            return (50 + mem) / 0.02
        @staticmethod
        def dof2mem(dof):
            return dof * 0.02 - 50
        @staticmethod
        def dof2chara_length(dof):
            return 1/np.sqrt(dof) * 1.2
        @staticmethod
        def mem2chara_length(mem):
            return Rectangle.dof2chara_length(Rectangle.mem2dof(mem))
    
    # max_mem  = 0.9 * mem 
    # max_dof= Rectangle.mem2dof(max_mem)
    # dofs = np.linspace(100, max_dof, args.num_dofs)
    # print(f"mems: {Rectangle.dof2mem(dofs)}")
    # chara_lengths = Rectangle.dof2chara_length(dofs)
    # chara_lengths = [0.2, 0.1, 0.05, 0.01, 0.005, 0.004,0.002, 0.0015,0.00145]
    chara_lengths = [0.2, 0.1, 0.05, 0.01, 0.005, 0.004]
    fig = plot_comparison(
        element_type="tri",
        chara_lengths=chara_lengths,
        n_times=args.times,
        csv_path="compare_2d.csv",
        c=args.c,
        dt=args.chara_length,
        n=args.num_steps,
        device=f"cuda:{args.device_index}"
    )
    fig.savefig("compare_linear_wave_2d.png")

    class Cube:
        @staticmethod
        def mem2dof(mem):
            return (50 + mem) / 0.04
        @staticmethod
        def dof2chara_length(dof):
            return 1/(dof)**(1/3) * 1.33
        @staticmethod
        def mem2chara_length(mem):
            return Cube.dof2chara_length(Cube.mem2dof(mem))


    # max_mem  = 0.9 * mem 
    # max_dof= Cube.mem2dof(max_mem)
    # print(f"dofs: {dofs}")
    # dofs = np.linspace(100, max_dof, args.num_dofs)
    # chara_lengths = Cube.dof2chara_length(dofs)

    # chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.03]
    chara_lengths = [0.2, 0.1, 0.05]
    fig = plot_comparison(
        element_type="tetra",
        chara_lengths=chara_lengths,
        n_times=args.times,
        csv_path="compare_3d.csv",
        c=args.c,
        dt=args.chara_length,
        n=args.num_steps,
        device=f"cuda:{args.device_index}"
    )

    fig.savefig("compare_3d.png")

    # test()

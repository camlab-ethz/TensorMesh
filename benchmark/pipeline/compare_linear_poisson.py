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
    def __init__(self, mesh, element="tri"):
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")

        self.mesh  = mesh
        self.element = element

    def __call__(self):
        element = skfem.ElementTriP1() if self.element == "tri" else skfem.ElementTetP1()
        @skfem.BilinearForm
        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v)) 
        
        @skfem.LinearForm
        def load(v, w):
            return 1.0*v
        basis = skfem.Basis(self.mesh, element)
        K     = skfem.asm(laplace, basis)

        f     = skfem.asm(load, basis)
        u     = skfem.solve(K,f)
        return  u 
    
class ThFEM:
    def __init__(self, mesh, batch_size=None):
        self.mesh = mesh
        self.batch_size = batch_size
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)

    def __call__(self):
        K     = self.K_asm(self.mesh.points, batch_size=self.batch_size)
        f     = self.f_asm(self.mesh.points, batch_size=self.batch_size)
       
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        u     = K.solve(f, backend=backend)
        return u 

class feFEM:
    def __init__(self, mesh):
        mesh.save("tmp.xdmf")
        fenics_mesh = fenics.Mesh()
        with fenics.XDMFFile("tmp.xdmf") as infile:
            infile.read(fenics_mesh)

        # Create a FunctionSpace on the fenics_mesh, not the original mesh
        self.fenics_mesh = fenics_mesh

    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        # Use fenics, not fem, for the methods inner, grad, and dx
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = fenics.inner(fenics.Constant(1), v) * fenics.dx
        u = fenics.Function(a.arguments()[0].function_space())

        fenics.solve(a == L, u)
        return u

def plot_comparison(element_type, chara_lengths, n_times, csv_path, device="cuda:0"):
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
        # th_fem_cpu = ThFEM(mesh.clone())
        th_fem_cpu_1 = ThFEM(mesh.clone(), batch_size=1)
        # th_fem_gpu   = ThFEM(mesh.clone().to("cuda:0"))
        th_fem_gpu_1 = ThFEM(mesh.clone().to(device), batch_size=1)
        sk_fem       = SkFEM(mesh, element=element_type)
        fe_fem       = feFEM(mesh)
        
        for name, fem in zip([
                                # "torch_fem cpu(None)", 
                                "torch_fem cpu",
                                # "torch_fem cuda(None)", 
                                "torch_fem cuda",
                                "scikit-fem", 
                                "fenics"], [
                                            # th_fem_cpu, 
                                            th_fem_cpu_1,
                                            # th_fem_gpu, 
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

def test():
    # from torch.profiler import profile, record_function, ProfilerActivity
    mesh = thfem.Mesh.gen_rectangle(chara_length=0.005, element_type="tri")
    skfem_ = SkFEM(mesh)
    start = time.perf_counter()
    skfem_()
    end = time.perf_counter()
    print(f"skfem: {end-start}s")
    fefem = feFEM(mesh)
    start = time.perf_counter()
    fefem()
    end = time.perf_counter()
    print(f"fenics: {end-start}s")
    fem = ThFEM(mesh.cuda(), batch_size=1)
    start = time.perf_counter()
    fem()
    end = time.perf_counter()
    print(f"torch_fem: {end-start}s")
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              record_shapes=True, 
    #              profile_memory=True, 
    #              with_stack=True,
    #              with_modules = True
    #              ) as prof:
    #     with record_function("fem"):
    #         fem()
    # with SimpleMemoryHook() as hook:
    #     fem()
    # cpu_peak = cpu_mem_peak(fem)
    # print(f"cpu peak: {cpu_peak}")
    # prof.export_chrome_trace("trace.json")
    # prof.export_memory_timeline("mem.json")
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_memory_usage", row_limit=10))

    import tracemalloc

    tracemalloc.start()
    
    fem()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    peak_memory = max(stat.size for stat in top_stats) / (1024 * 1024)

    print(f"trace_malloc peak memory: {peak_memory} MB")

    @profile
    def wrapper():
        K     = fem.K_asm(fem.mesh.points, batch_size=1)
        f     = fem.f_asm(fem.mesh.points, batch_size=1)
        u     = K.solve(f, backend="cupy")
    wrapper()

    def dummy():
        pass

    cpid_mem = memory_usage(dummy, max_usage=True)
    spid_mem = memory_usage(fem, max_usage=True)
    print(f"cpu_mem_peak: {spid_mem}/{cpid_mem} MB")
    # print(f"memory_usage peak memory: {cpu_mem_peak(fem)} MB")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device_index", type=int, default=0)
    parser.add_argument("-n", "--num_dofs", type=int ,  default=3)
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
    chara_lengths = [0.2, 0.1, 0.05, 0.01, 0.005, 0.004,0.002, 0.0015,0.00145]

    fig = plot_comparison(
        element_type="tri",
        chara_lengths=chara_lengths,
        n_times=args.times,
        csv_path="compare_2d.csv",
        device=f"cuda:{args.device_index}"
    )
    fig.savefig("compare_linear_poisson_2d.png")

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

    chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.03]

    fig = plot_comparison(
        element_type="tetra",
        chara_lengths=chara_lengths,
        n_times=args.times,
        csv_path="compare_3d.csv",
        device=f"cuda:{args.device_index}"
    )

    fig.savefig("compare_linear_poisson_3d.png")

    # test()

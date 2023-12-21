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
import gc

class SkFEM:
    def __init__(self, mesh, element="tri"):
        assert element in ["tri", "tetra"]
        mesh.save("tmp.msh", file_format='gmsh')

        skfem_mesh = skfem.Mesh.load("tmp.msh")

        self.mesh  = skfem_mesh
        self.element = element
        self.boundary = mesh.boundary_mask.numpy()

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
        u     = skfem.solve(*skfem.condense(K,f, D=self.boundary))
        return  u 
    
class ThFEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)
        self.condenser = thfem.Condenser(self.mesh.boundary_mask)

    def __call__(self):
        K     = self.K_asm(self.mesh.points)
        f     = self.f_asm(self.mesh.points)
        K_, f_ = self.condenser(K, f)
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "torch"
        u_    = K_.solve(f_, backend=backend)
        u     = self.condenser.recover(u_)
        return u 

class feFEM:
    def __init__(self, mesh):
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
        
        self.boundary_facets = boundary_facets
        


    def __call__(self):
        V = fenics.FunctionSpace(self.fenics_mesh, "Lagrange", 1)
        u = fenics.TrialFunction(V)
        v = fenics.TestFunction(V)
        bc = fenics.DirichletBC(V, fenics.Constant(0), self.boundary_facets, 1)
        # Use fenics, not fem, for the methods inner, grad, and dx
        a = fenics.inner(fenics.grad(u), fenics.grad(v)) * fenics.dx
        L = fenics.inner(fenics.Constant(1), v) * fenics.dx
        u = fenics.Function(a.arguments()[0].function_space())

        fenics.solve(a == L, u, bc)
        return u


def benchmark_fem(data, element_type, chara_length, pbar, ntimes=5, target="torch_fem cpu", device="cpu"):
    
    if element_type == "tri":
        mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length, element_type=element_type)
    elif element_type == "tetra":
        mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
    else:
        raise NotImplementedError(f"element_type={element_type} is not supported")
    
    if target == "torch_fem cpu":
        fem = ThFEM(mesh)
    elif target == "torch_fem cuda":
        fem = ThFEM(mesh.to(device))
    elif target == "scikit-fem":
        fem       = SkFEM(mesh, element=element_type)
    elif target == "fenics":
        fem       = feFEM(mesh)
    else:
        raise NotImplementedError(f"target={target} is not supported")

    fem() # heat up
    gpu_peak_mems = []
    gpu_mean_mems = []
    cpu_peak_mems = []
    cpu_mean_mems = []
    times = []

    for _ in range(ntimes):
        with CPUProfiler() as cpu_profiler:
            fem()
        cpu_peak_mems.append(cpu_profiler.max())
        cpu_mean_mems.append(cpu_profiler.mean())
       
        with CUDAProfiler() as cuda_profiler:
            fem()

        gpu_peak_mems.append(cuda_profiler.max())
        gpu_mean_mems.append(cuda_profiler.mean())
        with TimeProfiler() as time_profiler:
            fem()
        times.append(time_profiler.time)
       
        pbar.update(1)
        pbar.set_postfix({
            "dofs":(~mesh.boundary_mask).sum().item(),
            "chara_length":chara_length,
            "backend":target,
        })
    gpu_peak_mems = np.array(gpu_peak_mems)
    gpu_mean_mems = np.array(gpu_mean_mems)
    cpu_peak_mems = np.array(cpu_peak_mems)
    cpu_mean_mems = np.array(cpu_mean_mems)
    times = np.array(times)

    # remove the outlier by 3 sigma
    values = cpu_peak_mems[(cpu_peak_mems != cpu_peak_mems.max()) & (cpu_peak_mems != cpu_peak_mems.min())]
    valid_mask = np.abs( cpu_peak_mems - values.mean() ) < 3*values.std()

    cpu_peak_mems = cpu_peak_mems[valid_mask]
    cpu_mean_mems = cpu_mean_mems[valid_mask]
    gpu_peak_mems = gpu_peak_mems[valid_mask]
    gpu_mean_mems = gpu_mean_mems[valid_mask]
    times = times[valid_mask]


    data["CPU peak mem in MB"].extend(cpu_peak_mems.tolist())
    data["CPU mean mem in MB"].extend(cpu_mean_mems.tolist())
    data["GPU peak mem in MB"].extend(gpu_peak_mems.tolist())
    data["GPU mean mem in MB"].extend(gpu_mean_mems.tolist())
    data["time in s"].extend(times.tolist())
    data["chara length"].extend([chara_length]*len(times))
    data["degree of freedom"].extend([(~mesh.boundary_mask).sum().item()]*len(times))
    data["backend"].extend([target]*len(times))


def draw_error_bar(data, x,  y, hue, ax):
    _line_styles = ["-", "--", "-.", ":"]
    _markers = ["o", "s", "p", "^"]
    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    groups = data.groupby(hue)
    for i, (_hue, subdf) in enumerate(groups):
        subgroups = subdf.groupby(x)
        xs, means, stds = [], [], []
        for _x, subsubdf in subgroups:
            mean = subsubdf[y].mean()
            std  = subsubdf[y].std()
            xs.append(_x)
            means.append(mean)
            stds.append(std)
        ax.errorbar(xs, means, yerr=np.array(stds)*3, 
                    color=_colors[i], marker=_markers[i], linestyle=_line_styles[i], 
                    capsize=1.0, 
                    alpha=0.5,
                    label=_hue)
    ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim(left=np.array(xs).min()/2, right=np.array(xs).max()*2)
    ax.legend()

def plot_comparison(element_type, 
                    chara_lengths,
                    backends,
                    n_times, csv_path, force=False, device="cuda:0"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0).to_dict()
        for key, value in df.items():
            df[key] = list(value.values())
    else:
        df = {
            "degree of freedom":[],
            "backend":[],
            "time in s":[],
            "chara length":[],
            "CPU peak mem in MB":[],
            "CPU mean mem in MB":[],
            "GPU peak mem in MB":[],
            "GPU mean mem in MB":[],
        }
    pbar = tqdm(total=len(chara_lengths)*n_times*len(backends))
    for chara_length in chara_lengths:
        chara_indexes = set(np.where(np.array(df["chara length"]) == chara_length)[0].tolist())
        for backend in backends:
            backend_indexes = set(np.where(np.array(df["backend"]) == backend)[0].tolist())
            if not force and len(chara_indexes.intersection(backend_indexes)) > 0:
                for _ in range(n_times):
                    pbar.update(1)
                continue
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            benchmark_fem(df, element_type, chara_length, pbar, n_times, backend, device)

    
    df = pd.DataFrame.from_dict(df)

    df.to_csv(csv_path)
   
    fig, ax = plt.subplots(ncols=3, figsize=(15, 4))

    draw_error_bar(df, "degree of freedom", "time in s", "backend", ax[0])
    draw_error_bar(df, "degree of freedom", "CPU peak mem in MB", "backend", ax[1])
    draw_error_bar(df, "degree of freedom", "GPU peak mem in MB", "backend", ax[2])

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
    parser.add_argument("--only_2d", action="store_true")
    parser.add_argument("--only_3d", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backend", type=str,  default="torchfem_cpu",  choices=[
                                "torchfem_cpu", 
                                "torchfem_cuda",
                                "fenics",
                                "skfem"])   
    parser.add_argument("--server", action="store_true", help="run on server")
    args = parser.parse_args()

    mem = get_max_memory_for_index(args.device_index) - get_memory_for_index(args.device_index)

    
    if not args.only_3d:
        if args.backend == "torchfem_cpu":
            if args.server:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015, 0.0005, 0.00025]
            else:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015, 0.0012]
            backends = ["torch_fem cpu"]
        elif args.backend == "torchfem_cuda":
            if args.server:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015, 0.0005, 0.00025]
            else:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015, 0.0012]
            backends = ["torch_fem cuda"]
        elif args.backend == "fenics":
            if args.server:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015]
            else:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015]
            backends = ["fenics"]
        elif args.backend == "skfem":
            if args.server:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015]
            else:
                chara_lengths = [0.05, 0.01, 0.005, 0.002, 0.0015]
            backends = ["scikit-fem"]
        else:
            raise NotImplementedError(f"mode={args.mode} is not supported")
           
        fig = plot_comparison(
            element_type="tri",
            chara_lengths=chara_lengths,
            backends=backends,
            n_times=args.times,
            csv_path="compare_linear_poisson_2d.csv",
            force = args.force,
            device=f"cuda:{args.device_index}"
        )
        fig.savefig("compare_linear_poisson_2d.png")
        fig.savefig("compare_linear_poisson_2d.pdf")

    if not args.only_2d:

        if args.backend == "torchfem_cpu":
            if args.server:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02,0.015]
            else:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02, 0.01, 0.008]
            backends = ["torch_fem cpu"]

        elif args.backend == "torchfem_cuda":
            if args.server:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02,0.015]
            else:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02, 0.01, 0.008]
            backends = ["torch_fem cuda"]

        elif args.backend == "fenics":
            if args.server:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02]
            else:
                chara_lengths = [0.2, 0.1, 0.05, 0.04, 0.02]
            backends = ["fenics"]

        elif args.backend == "skfem":
            if args.server:
                chara_lengths = [0.2, 0.1, 0.05, 0.04]
            else:
                chara_lengths = [0.2, 0.1, 0.05, 0.04]
            backends = ["scikit-fem"]

        else:
            raise NotImplementedError(f"mode={args.mode} is not supported")

        fig = plot_comparison(
            element_type="tetra",
            chara_lengths=chara_lengths,
            backends=backends,
            n_times=args.times,
            csv_path="compare_linear_poisson_3d.csv",
            force = args.force,
            device=f"cuda:{args.device_index}"
        )

        fig.savefig("compare_linear_poisson_3d.png")
        fig.savefig("compare_linear_poisson_3d.pdf")


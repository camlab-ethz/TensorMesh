import sys 
import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
sys.path.append("../..")
import torch_fem as thfem
from torch_fem.profile import TimeProfiler, CPUProfiler, CUDAProfiler

class ThFEM:
    def __init__(self, mesh):
        self.mesh = mesh
        self.K_asm  = thfem.LaplaceElementAssembler.from_mesh(self.mesh)
        self.f_asm  = thfem.const_node_assembler(c=1).from_mesh(self.mesh)
        # self.f_asm  = thfem.ConstNodeAssembler.from_mesh(self.mesh)

    def __call__(self):
        K     = self.K_asm(self.mesh.points, batch_size=1)
        f     = self.f_asm(self.mesh.points, batch_size=1)
       
        backend = "petsc" if self.mesh.points.device.type == "cpu" else "cupy"
        u     = K.solve(f, backend=backend)
        return u 
    

def relation_2d():
    chara_lengths = []
    dofs          = []
    mems          = []  
    for chara_length in tqdm.tqdm([0.2, 0.1, 0.05, 0.01, 0.005, 0.004]):
        with CUDAProfiler() as cuda_profiler:
            mesh = thfem.Mesh.gen_rectangle(chara_length=chara_length)
            th_fem = ThFEM(mesh.cuda())
            th_fem()
        mems.append(cuda_profiler.max())
        chara_lengths.append(chara_length)
        dofs.append( mesh.points.shape[0])

    fig, ax = plt.subplots()
    line = scipy.stats.linregress(dofs, mems)
    ax.plot(dofs, mems)
    ax.set_xlabel("DOFs")
    ax.set_ylabel("GPU memory usage (MB)")
    ax.set_title(f"Linear regression: {line.slope:.2f}x + {line.intercept:.2f}")
    fig.savefig("2D-mem-dofs.png")

    fig, ax = plt.subplots()
    ax.plot(chara_lengths, mems)
    ax.set_xlabel("Characteristic length")
    ax.set_ylabel("GPU memory usage (MB)")
    fig.savefig("2D-mem-chara-length.png")
    plt.close()

    fig, ax = plt.subplots()
    line = scipy.stats.linregress(np.sqrt(1/np.array(dofs)), chara_lengths)
    ax.plot(np.sqrt(1/np.array(dofs)), chara_lengths)
    ax.set_xlabel("1/sqrt(DOFs)")
    ax.set_ylabel("Characteristic length")
    ax.set_title(f"Linear regression: {line.slope:.2f}x + {line.intercept:.2f}")
    fig.savefig("2D-dofs-chara-length.png")

def relation_3d():

    chara_lengths = []
    dofs          = []
    mems          = []  
    for chara_length in tqdm.tqdm([0.2, 0.1, 0.05, 0.04]):
        with CUDAProfiler() as cuda_profiler:
            mesh = thfem.Mesh.gen_cube(chara_length=chara_length)
            th_fem = ThFEM(mesh.cuda())
            th_fem()
        mems.append(cuda_profiler.max())
        chara_lengths.append(chara_length)
        dofs.append( mesh.points.shape[0])

    fig, ax = plt.subplots()
    line = scipy.stats.linregress(dofs, mems)
    ax.plot(dofs, mems)
    ax.set_xlabel("DOFs")
    ax.set_ylabel("GPU memory usage (MB)")
    ax.set_title(f"Linear regression: {line.slope:.2f}x + {line.intercept:.2f}")
    fig.savefig("3D-mem-dofs.png")

    fig, ax = plt.subplots()
    ax.plot(chara_lengths, mems)
    ax.set_xlabel("Characteristic length")
    ax.set_ylabel("GPU memory usage (MB)")
    fig.savefig("3D-mem-chara-length.png")
    plt.close()

    fig, ax = plt.subplots()
    line = scipy.stats.linregress((1/np.array(dofs))**(1/3), chara_lengths)
    ax.plot((1/np.array(dofs))*(1/3), chara_lengths)
    ax.set_xlabel("1/(DOFs^(1/3))")
    ax.set_ylabel("Characteristic length")
    ax.set_title(f"Linear regression: {line.slope:.2f}x + {line.intercept:.2f}")
    fig.savefig("3D-dofs-chara-length.png")


if __name__ == '__main__':
    # relation_2d()
    # relation_3d()
    relation_cusolve()
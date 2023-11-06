import sys 
import os
import torch
sys.path.append("../..")
import torch_fem as fem
import skfem
import skfem.helpers
import pandas as pd
import time
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

class SkfEMAsmTri:
    def __init__(self, mesh):

        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v))

        self.bilinear = skfem.BilinearForm(laplace)

        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")
       
        self.basis = skfem.Basis(mesh, skfem.ElementTriP1())

        os.remove("tmp.msh")

    def __call__(self):
        return skfem.asm(self.bilinear, self.basis)
    
class SkfEMAsmTet:
    def __init__(self, mesh):

        def laplace(u, v, w):
            dot, grad = skfem.helpers.dot, skfem.helpers.grad
            return dot(grad(u), grad(v))

        self.bilinear = skfem.BilinearForm(laplace)

        mesh.save("tmp.msh", file_format='gmsh')

        mesh = skfem.Mesh.load("tmp.msh")

        self.basis = skfem.Basis(mesh, skfem.ElementTetP1())

        os.remove("tmp.msh")

    def __call__(self):
        return skfem.asm(self.bilinear, self.basis)

class ThFEMAsmCPU:
    def __init__(self, mesh):

        class KAsm(fem.ElementAssembler):
            def forward(self, gradu, gradv):
                return fem.dot(gradu, gradv)

        self.asm = KAsm.from_mesh(mesh, quadrature_order=2)
        self.points = mesh.points

    def __call__(self):
        with torch.no_grad():
            return self.asm(self.points)

class ThFEMAsmCUDA:
    def __init__(self, mesh):

        class KAsm(fem.ElementAssembler):
            def forward(self,gradu, gradv):
                return fem.dot(gradu, gradv)

        mesh = mesh.cuda()

        self.asm = KAsm.from_mesh(mesh, quadrature_order=2)
        self.points = mesh.points

    def __call__(self):
        with torch.no_grad():
            result = self.asm(self.points)
            torch.cuda.synchronize()
            return result
   


def plot_comparison(cell_type, chara_lengths, n_times, csv_path, ax):

    data = {
        "chara_length":[],
        "assembler":[],
        "time":[]
    }
    pbar = tqdm(total=len(chara_lengths)*n_times*3)

    for chara_length in chara_lengths:
        if cell_type == "tri":
            mesh = fem.Mesh.gen_rectangle(chara_length=chara_length, cell_type=cell_type)
        elif cell_type == "tetra":
            mesh = fem.Mesh.gen_cube(chara_length=chara_length)
        th_asm_cpu = ThFEMAsmCPU(mesh)
        th_asm_gpu = ThFEMAsmCUDA(mesh)
        sk_asm = {
            "tri":SkfEMAsmTri,
            "tetra":SkfEMAsmTet
        }[cell_type](mesh)
        for _ in range(n_times):
            for name, assembler in zip(["torch_fem cpu", "torch_fem cuda", "scikit-fem"], [th_asm_cpu, th_asm_gpu, sk_asm]):
                start = time.time()
                assembler()
                end = time.time()
                data["chara_length"].append(chara_length)
                data["assembler"].append(name)
                data["time"].append(end-start)
                pbar.update(1)
                pbar.set_postfix({
                    "chara_length":chara_length,
                    "assembler":name,
                    "time":f"{end-start:7.5f}s"
                })
    df = pd.DataFrame(data)
    df.to_csv(csv_path)
    sns.lineplot(x="chara_length", y="time",
             hue="assembler",data=df,ax=ax)
    ax.set_yscale("log")
    ax.set_xscale("log")

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(12,  8))

    plot_comparison(
        cell_type="tri",
        chara_lengths=[0.05, 0.01, 0.005, 0.002],
        n_times=5,
        csv_path="skfem_compare_2d.csv",
        ax=ax
    )

    fig.savefig("skfem_compare_2d.png")

    fig, ax = plt.subplots(figsize=(12,  8))

    plot_comparison(
        cell_type="tetra",
        chara_lengths=[0.1,  0.05, 0.02],
        n_times=5,
        csv_path="skfem_compare_3d.csv",
        ax=ax
    )

    fig.savefig("skfem_compare_3d.png")

    




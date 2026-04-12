import sys
sys.path.append("../..")
import os
import torch
import matplotlib.pyplot as plt
from tensormesh.element import Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism
from tensormesh.element.plot import plot_1d, plot_2d, plot_3d

# Plot 1D basis functions for Line element

def plot_line_basis_fns():

    fig, axes = plt.subplots(ncols=4, figsize=(16,4))

    for i, order in enumerate(range(1,5)):
        basis = Line.get_basis(order)
        basis_fns = Line.get_basis_fns(order)
        plot_1d(basis, basis_fns, ax=axes[i], legend=False)
        axes[i].set_title(f"Order:{order}")

    plt.title("Line Basis Functions")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/linear.png")

def plot_triangle_basis_fns():
    fig = plt.figure(figsize=(16,4))
    axes = [fig.add_subplot(1,4,i+1, projection='3d') for i in range(4)]

    for i, order in enumerate(range(1,5)):
        basis = Triangle.get_basis(order)
        basis_fns = Triangle.get_basis_fns(order)
        axes[i] = plot_2d(Triangle, basis, basis_fns, ax=axes[i], legend=False)
        axes[i].set_title(f"Order:{order}")

    plt.title("Triangle Basis Functions") 
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/triangle.png")

def plot_quadrilateral_basis_fns():
    fig = plt.figure(figsize=(16,4))
    axes = [fig.add_subplot(1,4,i+1, projection='3d') for i in range(4)]

    for i, order in enumerate(range(1,5)):
        basis = Quadrilateral.get_basis(order)
        basis_fns = Quadrilateral.get_basis_fns(order)
        axes[i] = plot_2d(Quadrilateral, basis, basis_fns, ax=axes[i], legend=False)
        axes[i].set_title(f"Order:{order}")

    plt.title("Quadrilateral Basis Functions")
    os.makedirs("output", exist_ok=True) 
    plt.savefig("output/quadrilateral.png")

def plot_tetrahedron_basis_fns():
    for order in range(1,5):
        basis = Tetrahedron.get_basis(order)
        basis_fns = Tetrahedron.get_basis_fns(order)
        plot_3d(Tetrahedron, basis, basis_fns)
        plt.suptitle(f"Tetrahedron Basis Functions - Order {order}")
        plt.tight_layout()
        
        output_dir = f"output/tetrahedron"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{order}.png")
        plt.close()

def plot_hexahedron_basis_fns():
    for order in range(1,5):
        basis = Hexahedron.get_basis(order)
        basis_fns = Hexahedron.get_basis_fns(order)
        plot_3d(Hexahedron, basis, basis_fns)
        plt.suptitle(f"Hexahedron Basis Functions - Order {order}")
        plt.tight_layout()
        
        output_dir = f"output/hexahedron" 
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{order}.png")
        plt.close()

def plot_prism_basis_fns():
    for order in range(1,5):
        basis = Prism.get_basis(order)
        basis_fns = Prism.get_basis_fns(order)
        plot_3d(Prism, basis, basis_fns)
        plt.suptitle(f"Prism Basis Functions - Order {order}")
        plt.tight_layout()
        
        output_dir = f"output/prism"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{order}.png")
        plt.close()

def plot_pyramid_basis_fns():
    for order in range(1,5):
        basis = Pyramid.get_basis(order)
        basis_fns = Pyramid.get_basis_fns(order)
        plot_3d(Pyramid, basis, basis_fns)
        plt.suptitle(f"Pyramid Basis Functions - Order {order}")
        plt.tight_layout()
        
        output_dir = f"output/pyramid"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{order}.png")
        plt.close()



if __name__ == "__main__":
    plot_line_basis_fns()
    plot_triangle_basis_fns()
    plot_quadrilateral_basis_fns()
    plot_tetrahedron_basis_fns()
    plot_hexahedron_basis_fns()
    plot_prism_basis_fns()
    plot_pyramid_basis_fns()

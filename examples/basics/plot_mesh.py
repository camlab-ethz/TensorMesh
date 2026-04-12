import os 
import sys 
sys.path.append("../..")
import numpy as np
import torch
import matplotlib.pyplot as plt
from tensormesh import mesh,MeshGen
import tensormesh as tm
import tensormesh.visualization as V
import tensormesh.element as E

GRAPH_ALPHA = 0.5
GRAPH_COLOR = "lightblue"
GRAPH_LINEWIDTH = 1 
MESH_COLOR = "black"
MESH_LINEWIDTH = 2

def plot_rectangle_mesh(return_fig=False):
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for ele, ax in zip(["tri","quad"], axes):
        mesh_gen = tm.MeshGen(element_type=None, chara_length=0.5, order=4)
        mesh_gen.add_rectangle(0, 0, 1, 1, element=ele)
        mesh_gen.gen().plot(ax=ax)
    axes[0].set_title("Triangle Mesh")
    axes[1].set_title("Quadrilateral Mesh")
    if return_fig:
        return fig
    fig.savefig("output/rectangle_mesh.png")
    plt.close()

def plot_cube_mesh(return_fig=False):
    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for i, (ele, order, ax) in enumerate(zip(["tet", "hex"], [3,2], axes)):
        mesh_gen = tm.MeshGen(element_type=None, dimension=3, chara_length=0.9, order=order)
        mesh_gen.add_cube(0, 0, 0, 1, 1, 1, element=ele)
        ax = mesh_gen.gen().plot(ax=ax, draw_basis=False)
        axes[i] = ax
    axes[0].set_title("Tetrahedral Mesh")
    axes[1].set_title("Hexahedral Mesh")
    if return_fig:
        return fig
    fig.savefig("output/cube_mesh.png")
    plt.close()

def plot_circle_mesh(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=2)
    mesh_gen.add_circle(0.5, 0.5, 0.4, element="tri")
    ax = mesh_gen.gen().plot(save_path="output/circle_mesh.png")
    fig = ax.get_figure()
    if return_fig:
        return fig
    plt.close()

# def plot_sphere_mesh():
#     os.makedirs("output", exist_ok=True) 
#     mesh_gen = tm.MeshGen(element_type=None, dimension=3, chara_length=0.1, order=1)
#     mesh_gen.add_sphere(0.5, 0.5, 0.5, 0.5)
#     ax = mesh_gen.gen().plot(save_path="output/sphere_mesh.png", show=True)
#     fig = ax.get_figure()
#     plt.close()

def plot_mix_mesh_2d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen:tm.Mesh = tm.MeshGen(element_type=None, chara_length=0.1, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    ax = mesh_gen.gen().plot(save_path="output/hybrid_mesh2d.png")
    fig = ax.get_figure()
    if return_fig:
        return fig
    plt.close()

def plot_node_adj_2d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency(), mesh.points, 
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    fig = ax.get_figure()
    if return_fig:
        return fig
    fig.savefig("output/node_adj_2d.png")
    plt.close()

def plot_ele_adj_2d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri") 
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    points = []
    for key , elements in mesh.elements().items():
        points.append(mesh.points[elements].mean(-2))
    points = torch.cat(points, 0)

    ax = V.draw_graph(mesh.element_adjacency(), points,
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    fig = ax.get_figure()
    if return_fig:
        return fig
    fig.savefig("output/ele_adj_2d.png")
    plt.close()

def plot_node_adj_3d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    # mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency(), mesh.points,
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    fig = ax.get_figure()
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    if return_fig:
        return fig
    fig.savefig("output/node_adj_3d.png")
    plt.close()

def plot_ele_adj_3d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.2, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    # mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.element_adjacency(), mesh.points[mesh.elements()].mean(-2),
                      color=GRAPH_COLOR,
                      linewidth=GRAPH_LINEWIDTH,
                      alpha=GRAPH_ALPHA)
    fig = ax.get_figure()
    mesh.plot(ax=ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    if return_fig:
        return fig
    fig.savefig("output/ele_adj_3d.png")
    plt.close()


def plot_point_value_2d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.05, order=1)
    mesh_gen.add_rectangle(0,0,1,1, element="tri")
    mesh:tm.Mesh = mesh_gen.gen()
    
    # Create some sample point values
    points = mesh.points
    point_values = torch.sin(2*np.pi*points[:,0]) * torch.cos(2*np.pi*points[:,1])
    
    # Plot using interpolation
    img, ax = V.draw_point_value(mesh, point_values)
    fig = ax.get_figure()
    fig.colorbar(img)
    mesh.plot(ax = ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    if return_fig:
        return fig
    fig.savefig("output/point_value_2d.png")
    plt.close()

def plot_point_value_3d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1, element="tet")
    mesh:tm.Mesh = mesh_gen.gen()
    # Create some sample point values
    points = mesh.points
    point_values = torch.sin(2*np.pi*points[:,0]) * torch.cos(2*np.pi*points[:,1]) * torch.sin(2*np.pi*points[:,2])
    
    # Plot using interpolation
    img, ax = V.draw_point_value(mesh, point_values)
    fig = ax.get_figure()
    fig.colorbar(img)
    mesh.plot(ax = ax, edgecolor=MESH_COLOR, linewidth=MESH_LINEWIDTH)
    if return_fig:
        return fig
    fig.savefig("output/point_value_3d.png")
    plt.close()

def plot_ele_value_2d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.05, order=1)
    mesh_gen.add_rectangle(0,0,1,1, element="tri")
    mesh:tm.Mesh = mesh_gen.gen()
    
    # Create some sample element values
    elements = mesh.elements(mesh.dim)
    element_values = {"triangle": torch.sin(2*np.pi*mesh.points[elements["triangle"]].mean(1)[:,0]) * 
                           torch.cos(2*np.pi*mesh.points[elements["triangle"]].mean(1)[:,1])}
    
    # Plot using interpolation
    collections, ax = V.draw_element_value(mesh, element_values)
    fig = ax.get_figure()
    fig.colorbar(collections["triangle"])
    if return_fig:
        return fig
    fig.savefig("output/element_value_2d.png")
    plt.close()

def plot_ele_value_3d(return_fig=False):
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1, element="tet")
    mesh:tm.Mesh = mesh_gen.gen()
    
    # Create some sample element values
    elements = mesh.elements(mesh.dim)
    centroids = mesh.points[elements["tetra"]].mean(1)
    element_values = {"tetra": torch.sin(2*np.pi*centroids[:,0]) * 
                           torch.cos(2*np.pi*centroids[:,1]) *
                           torch.sin(2*np.pi*centroids[:,2])}
    
    # Plot using interpolation
    collections, ax = V.draw_element_value(mesh, element_values)
    fig = ax.get_figure()
    fig.colorbar(collections["tetra"])
    if return_fig:
        return fig
    fig.savefig("output/element_value_3d.png")
    plt.close()


if __name__ == "__main__":
    plot_rectangle_mesh()
    plot_cube_mesh()
    plot_circle_mesh()
    plot_mix_mesh_2d()
    plot_node_adj_2d()
    plot_ele_adj_2d()
    plot_node_adj_3d()
    plot_ele_adj_3d()
    plot_point_value_2d()
    plot_point_value_3d()
    plot_ele_value_2d()
    plot_ele_value_3d()
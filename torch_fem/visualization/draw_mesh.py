import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection, LineCollection

def draw_mesh(mesh, edgecolor="blue", linewidth=3, alpha=0.3, ax=None):
    """
    Parameters
    ----------
    mesh: torch_fem.mesh.Mesh
        the mesh
    edgecolor: str, optional
        the color of the edge, default is "blue"
    linewidth: float, optional
        the width of the edge, default is 3
    alpha: float, optional
        the transparency of the edge, default is 0.3
    ax: matplotlib.axes.Axes, optional
        the axis, default is None

    Returns
    -------
    ax: matplotlib.axes.Axes
        the axis
    """
    if isinstance(pos, torch.Tensor):
        pos = mesh.vertices().numpy()
    for value in mesh.elements().values():
        edgecolor = 'blue'
        linewidth = 3
        alpha     = 0.3
        if value.shape[1] == 3: # tri
            ax.triplot(pos[:,0], pos[:,1], value, color=edgecolor, linewidth=linewidth)
        elif value.shape[1] == 4: # quad
            polygons = [patches.Polygon(pos[element], closed=True, fill=False, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth) for element in value]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        elif value.shape[1] == 6: # tri6
            order = np.array([0, 3, 1, 4, 2, 5])
            polygons = [patches.Polygon(pos[element[order]], closed=True, fill=False, edgecolor=edgecolor, alpha=alpha,  linewidth=linewidth) for element in value]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        elif value.shape[1] == 9: # quad9 
            order = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            polygons = [patches.Polygon(pos[element[order]], closed=True, fill=False, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth) for element in value]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
        else:
            raise NotImplementedError(f"element type {value.shape[1]} is not supported")
    return ax
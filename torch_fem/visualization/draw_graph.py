import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon, Arc
from matplotlib import patches

def draw_graph(sparse_matrix, points, ax=None):
    """
    Parameters
    ----------
    sparse_matrix: torch_fem.sparse.SparseTensor
        the sparse matrix
    points: torch.Tensor
        2D tensor of shape :math:`[|\mathcal V|, 2]`, where  :math:`|\mathcal V|` is the number of vertices

    Returns
    -------
    ax: matplotlib.axes.Axes
        the axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    if isinstance(points, torch.Tensor):
        points = points.numpy()
    pos = points.numpy()
    row, col = sparse_matrix.row, sparse_matrix.col
    if isinstance(row, torch.Tensor):
        row, col = row.numpy(), col.numpy()
    fig, ax = plt.subplots(figsize=(10,10))
    lines = []
    selfloops = []
    diameter = 0.02
    for (u, v) in zip(row, col):
        if u == v:
            selfloops.append(Arc((pos[u,0],pos[u,1]+diameter/2), diameter, diameter, 0, 0, 360, color="black", linewidth=0.5))
        else:
            line = (pos[u], pos[v])
            lines.append(line)
    lc = LineCollection(lines, color="black", linewidth=0.5)
    loops = PatchCollection(selfloops, match_original=True)
    ax.add_collection(lc)
    ax.add_collection(loops)

    return ax
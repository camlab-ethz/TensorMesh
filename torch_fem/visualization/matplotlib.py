
import torch 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
import matplotlib.tri as tri
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def plot(kwargs, mesh,  save_path=None, dt=None,show_mesh=False):
    """
        Parameters:
        -----------
            kwargs: dict
                the key is the name of the variable, the value is the value of the variable
            mesh: torch_fem.mesh.mesh.Mesh
    """
    points = mesh.points
    elements = mesh.elements(mesh.default_cell_type)
   
    ncols = len(kwargs.keys())
    fig, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    key, value = next(iter(kwargs.items()))
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(elements, torch.Tensor):
        elements = elements.detach().cpu().numpy()
    if not isinstance(ax,  np.ndarray):
        ax = [ax]
    if isinstance(value,(torch.Tensor, np.ndarray)):      
        if save_path is None:
            save_path = 'mesh.png'
        for i, (key, value) in enumerate(kwargs.items()):
            img, cb = draw_mesh(points, elements, value, ax=ax[i], show_colorbar=True, show_mesh=show_mesh)
            ax[i].set_title(key)
        fig.savefig(save_path, dpi=400)
    elif isinstance(value, (list, tuple)):
        if save_path is None:
            save_path = 'mesh.gif'
        cbs = []
        imgs = []
        for i, (key, value) in enumerate(kwargs.items()):
            img,cb = draw_mesh(points, elements, value[0], ax=ax[i], show_colorbar=True,show_mesh=show_mesh)
            ax[i].set_title(key)
            cbs.append(cb)
            cbs.append(fig.colorbar(img, ax=ax[i]))
            imgs.append(img)
        if dt is not None:
            fig.suptitle(f"t={0*dt:7.5f}")
        else:
            fig.suptitle(f"Frame:{0:5d}")
        def update(frame):
            for i, (key, value) in enumerate(kwargs.items()):
                v   = value[frame].detach().cpu().numpy() if isinstance(value[frame], torch.Tensor) else value[frame]
                imgs[i].set_clim(v.min(), v.max())
                imgs[i].set_array(v)
                cbs[i].update_normal(imgs[i])
            if dt is not None:
                fig.suptitle(f"t={frame*dt:7.5f}")
            else:
                fig.suptitle(f"Frame:{frame:5d}")
        anim = FuncAnimation(fig, update, frames=len(value), interval=100)
        anim.save(save_path, fps=10,  dpi=400)

def draw_mesh(points, elements, value, ax=None, show_colorbar=True,show_mesh=False):
    """
        Parameters:
        -----------
            points: torch.Tensor [n_point, n_dim]
                the coordinates of the points
            elements: torch.Tensor [n_element, n_basis]
                the indices of the element corners
            value: torch.Tensor [n_point]
                the value of the points
            ax: matplotlib.axes.Axes
                the axes to plot on
                if None, then use plt.gca()
            show_colorbar: bool
                whether to show the colorbar
                default is True
    """
    assert points.shape[1] == 2, f"points must be 2D, but got {points.shape}"

    if ax is None:
        ax = plt.gca()

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(elements, torch.Tensor):
        elements = elements.detach().cpu().numpy()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if elements.shape[1] == 3: # tri
        triang = tri.Triangulation(points[:, 0], points[:, 1], elements)
        img = ax.tripcolor(triang, value, cmap=plt.cm.jet, shading='gouraud')
        if show_mesh:
            ax.triplot(triang, color='k', linewidth=0.5)
    elif elements.shape[1] == 4: # quad 
        triang = tri.Triangulation(points[:, 0], points[:, 1], np.concatenate([elements[:,(0,1,2)],elements[:,(0,2,3)]],0))
        img = ax.tripcolor(triang, value, cmap=plt.cm.jet, shading='gouraud')
        if show_mesh:
            polygons = [patches.Polygon(points[element], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)
    else:
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        z_grid = griddata(points, value, (x_grid, y_grid), method='linear')
        img = ax.imshow(z_grid.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=plt.cm.jet, aspect='auto')


        if show_mesh:
            polygons = [patches.Polygon(points[element], closed=True, fill=False, edgecolor='k', linewidth=0.5) for element in elements]
            polygons = PatchCollection(polygons, match_original=True)
            ax.add_collection(polygons)


    ax.axis("equal")
    ax.axis("off")

    if show_colorbar:
        cb = plt.colorbar(img, ax=ax)
        return img, cb
    
    return img
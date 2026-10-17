import torch
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from typing import Optional, TYPE_CHECKING, Union, Type


from .polynomial import Polynomials
from .element import Element, Triangle

def plot_1d(basis:torch.Tensor, 
            basis_fns:Polynomials, 
            resolution:int=100,
            ax:Optional[Axes] = None,
            legend:bool = True,
            show:bool = False
        )->Axes:
    """Plot 1D basis functions.

    Examples
    --------
    >>> import torch
    >>> from tensormesh.element import Triangle
    >>> # Create linear basis functions for a triangle element
    >>> basis = Triangle.get_basis(1)  # [n_basis, n_dim]
    >>> basis_fns = Triangle.get_basis_fns(1)  # Polynomials object
    >>> # Plot the basis functions
    >>> plot_1d(basis, basis_fns)

    Parameters
    ----------
    basis : torch.Tensor
        Tensor of shape [n_basis, 1] containing the basis points
    basis_fns : list
        List of basis functions to plot
    resolution : int, optional
        Number of points to evaluate functions at, by default 100
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes, by default None
    legend : bool, optional
        Whether to display the legend, by default True
    show : bool, optional
        Whether to call plt.show() after plotting, by default False

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """

    assert basis_fns.n_vars == 1, f"basis_fns must be 1D, got {basis_fns.n_vars}D"
    assert basis_fns.numel() == basis.shape[0], f"Number of basis functions ({basis_fns.numel()}) must match number of basis points ({basis.shape[0]})"
    assert basis.dim() == 2, f"basis must be 2D tensor, got {basis.dim()}D"
    assert basis.shape[1] == 1, f"basis must have shape [n_basis, 1], got {basis.shape}"
    if ax is None:
        fig, ax = plt.subplots()

    x = torch.linspace(0, 1, resolution)[:, None]

    lines = basis_fns.map(x) # [resolution, n_basis]

    shape_val = basis_fns.map(basis) # [n_basis, n_basis]

    # Create a colormap for the lines
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, lines.shape[1]))

    # Plot each line with a different color from the colormap
    for i in range(lines.shape[1]):
        ax.plot(x, lines[:, i], color=colors[i], label=f'Basis {i+1}  {basis_fns[i]}')
        ax.scatter(basis[i, 0], shape_val[i,i], color=colors[i], marker='*', s=150, edgecolor='black', linewidth=1, alpha=0.8, label=f'Basis Fn {i+1}')

    if legend:
        ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('value')
    ax.grid(True)

    if show:
        plt.show()
    
    return ax


def plot_2d(element:Union[Type[Element],Element],
            basis: torch.Tensor, 
            basis_fns: Polynomials, 
            resolution: int = 100, 
            ax: Optional[Axes3D] = None, 
            legend: bool = True,
            show: bool = False) -> plt.Axes:
    """Plot 2D basis functions.

    Parameters
    ----------
    element : Element
        Element object containing the geometry information
    basis : torch.Tensor
        Tensor of shape [n_basis, 2] containing the basis points
    basis_fns : Polynomials
        Polynomials object containing the basis functions to plot
    resolution : int, optional
        Number of points to evaluate functions at, by default 100
    ax : Optional[plt.Axes], optional
        Axes to plot on. If None, creates new figure and axes, by default None
    legend : bool 
        Whether to show the legend, by default True
    show : bool, optional
        Whether to call plt.show() after plotting, by default False

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    assert basis_fns.n_vars == 2, f"basis_fns must be 2D, got {basis_fns.n_vars}D"
    assert basis_fns.numel() == basis.shape[0], f"Number of basis functions ({basis_fns.numel()}) must match number of basis points ({basis.shape[0]})"
    assert basis.dim() == 2, f"basis must be 2D tensor, got {basis.dim()}D"
    assert basis.shape[1] == 2, f"basis must have shape [n_basis, 2], got {basis.shape}"



    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if ax is not None and not isinstance(ax, Axes3D):
        # Get the figure and position from existing axes
        fig = ax.figure
        pos = ax.get_position()
        # Remove the old 2D axes
        fig.delaxes(ax)
        # Create new 3D axes in same position
        ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')
        ax.set_position(pos)

    x, y = torch.meshgrid(torch.linspace(0, 1, resolution), 
                         torch.linspace(0, 1, resolution), indexing='ij')
    xy = torch.stack([x.flatten(), y.flatten()], -1)

    # Create a colormap for the surfaces
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, basis_fns.numel()))


    # Plot each surface with a different color from the colormap
    for i in range(basis_fns.numel()):
        basis_z = basis_fns.map(basis[i:i+1])[0,i]
        z = basis_fns.map(xy)[:,i]
        # For triangle elements, mask the z values outside the triangle
        if element == Triangle or isinstance(element, Triangle):
            z = z.reshape(resolution, resolution)
            mask = (x + y > 1)
            z[mask] = float('nan')
        else:
            z = z.reshape(resolution, resolution)
            
        surf = ax.plot_surface(x.numpy(), y.numpy(), z.numpy(),
                             color=colors[i], alpha=0.2, label=f'Basis {i+1}')
        ax.scatter(basis[i,0].numpy(), basis[i,1].numpy(), basis_z.numpy(), 
                  color=colors[i], label=f'Basis Fn {i+1}')

    if legend:
        ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    ax.set_zlabel('value')
    ax.grid(True)

    if show:
        plt.show()

    return ax

def _inside_reference(element:Union[Type[Element],Element], points:torch.Tensor, tol:float = 1e-6)->torch.Tensor:
    """Mask of the ``points`` [n, 3] that lie in the (convex) 3D reference element."""
    vertices = element.points.to(points.dtype)
    centre = vertices.mean(0)
    mask = torch.ones(points.shape[0], dtype=torch.bool)
    for face in element.face:
        a, b, c = vertices[list(face[:3])]
        n = torch.linalg.cross(b - a, c - a)
        if torch.dot(n, centre - a) > 0:  # make the normal point outwards
            n = -n
        mask &= (points - a) @ n <= tol
    return mask

def plot_3d(element:Union[Type[Element],Element],
            basis:torch.Tensor,
            basis_fns:Polynomials, 
            resolution:int=15,
            legend:bool = True,
            show:bool = False):
    """Plot 3D basis functions for an element.

    Parameters
    ----------
    element : Element
        Element to plot basis functions for
    basis : torch.Tensor
        Basis points, shape [n_basis, 3]
    basis_fns : List[Polynomial]
        List of basis functions to plot
    resolution : int, optional
        Number of points to evaluate functions at, by default 15
    legend : bool 
        Whether to show the legend, by default True
    show : bool, optional
        Whether to call plt.show() after plotting, by default False

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    assert basis_fns.n_vars == 3, f"basis_fns must be 3D, got {basis_fns.n_vars}D"
    assert basis_fns.numel() == basis.shape[0], f"Number of basis functions ({basis_fns.numel()}) must match number of basis points ({basis.shape[0]})"
    assert basis.dim() == 2, f"basis must be 2D tensor, got {basis.dim()}D"
    assert basis.shape[1] == 3, f"basis must have shape [n_basis, 3], got {basis.shape}"

    n_basis = len(basis_fns)
    n_cols = 4
    n_rows = (n_basis + 3) // 4

    fig = plt.figure(figsize=(4*n_cols,4*n_rows))
    axes = [fig.add_subplot(n_rows, n_cols, i+1, projection='3d') for i in range(n_basis)]
    ax = None


    x, y, z = torch.meshgrid(torch.linspace(0, 1, resolution),
                            torch.linspace(0, 1, resolution),
                            torch.linspace(0, 1, resolution), indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    xyz = torch.stack([x, y, z], -1)
    # the basis functions live on the element only; the rational pyramid
    # basis is singular outside it, near z = 1
    xyz = xyz[_inside_reference(element, xyz)]
    x, y, z = xyz.unbind(-1)

    cmap = plt.get_cmap('Spectral_r')

    for i, basis_fn in enumerate(basis_fns):
        ax = axes[i]
        basis_w = basis_fn(basis[i])
        w = basis_fn(xyz)

        norm = mcolors.Normalize(vmin=w.min(), vmax=w.max())
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.scatter(x, y, z, c=scalar_map.to_rgba(w), alpha=0.1)
        ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], color=scalar_map.to_rgba(basis_w))
        ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f"{basis_w}")

        cbar = fig.colorbar(scalar_map, ax=ax)

        edges = element.points[element.edge]
        for edge in edges:
            ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), color='black')
        ax.set_title(f'Basis {i+1}')

        if legend:
            ax.legend()

    if show:
        plt.show()
    return fig
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from typing import Union, Optional, Dict, Sequence
from matplotlib import tri
from scipy.interpolate import griddata
import torch
from .utils import dim, as_ndarray
from .draw_facet import draw_facet_2d
from .draw_point_value import draw_point_value_2d, update_point_value_2d


class StreamPlotter:
    def __init__(self, 
                 nrows:int=1, 
                 ncols:int=1, 
                 width:int=5, 
                 height:int=5, 
                 filename:str="stream_plotter.mp4"):

        fig, axes = plt.subplots(nrows, ncols, figsize=(width*ncols, height)) 
        self.ncols = ncols 
        self.nrows = nrows
        self.fig  = fig 
        self.axes = axes
        self.filename = filename
        self.ax2img = {}
        self.ax2cb  = {}

    def __enter__(self):
        # Set up the writer
        self.writer = animation.writers['ffmpeg'](fps=10, metadata=dict(artist='Me'), bitrate=1800)
        self.writer.setup(self.fig, self.filename, dpi=100)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Finish the animation
        self.writer.finish()
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
            return False

    def grab_frame(self, **savefig_kwargs):
        # Grab the current frame
        self.writer.grab_frame(**savefig_kwargs)

    def update(self):
        self.grab_frame()

    def draw_mesh_2d(self, 
                  points:torch.Tensor|np.ndarray,
                  elements:Dict[str,torch.Tensor|np.ndarray],
                  point_values:torch.Tensor|np.ndarray, 
                  ax:Optional[plt.Axes] = None, 
                  cmap:str = "jet",
                  density:int = 100,
                  use_scatter:bool=False,
                  show_colorbar:bool=True, 
                  title:str="", 
                  update:bool=True, 
                  show_mesh:bool=True, 
                  show_basis:bool=True,
                  umin:Optional[float]=None, 
                  umax:Optional[float]=None,
                  linewidth:int = 1, 
                  basiscolor:str= 'orange',
                  linecolor:str = 'blue'):
        """
        Parameters
        ----------
        mesh: torch_fem.mesh.Mesh
            the mesh
        point_values: torch.Tensor [n_point]
            the value of the points
        ax: matplotlib.axes.Axes, optional
            the axis, default is None
        cmap:str 
        density:int 
        use_scatter:bool
        show_colorbar: bool, optional   
            whether to show the colorbar, default is True
        title: str, optional
            the title of the plot, default is ""
        update: bool, optional
            whether to update the plot, default is True
        show_mesh: bool, optional
            whether to show the mesh, default is True
        """

        # assertion
        assert dim(points) == 2 
        assert points.shape[1] == 2 
        n_point = points.shape[0]
         
        assert n_point == point_values.shape[0], f"points.shape[0] must be {point_values.shape[0]}, but got {n_point}"
        if ax is None: 
            assert not isinstance(self.axes, np.ndarray), "ax must be specified when there are multiple axes"
            ax = self.axes 
      
        # draw the main figure
        if ax in self.ax2img: # update not draw from scratch
            img = self.ax2img[ax]
            update_point_value_2d(img, points, point_values)
            if umin is not None and umax is not None:
                img.set_clim(umin, umax)
            if show_colorbar:
                assert ax in self.ax2cb, f"You should draw the point value with colorbar first before updating the colorbar"
                self.ax2cb[ax].update_normal(img)
        else: # draw from scratch
            img,_ = draw_point_value_2d(
                    points,  # 
                    point_values, 
                    elements, # type:ignore
                    density = density,
                    cmap = cmap,
                    use_scatter=use_scatter,
                    ax   = ax 
                    )
            if show_mesh:
                draw_facet_2d(points, 
                              elements, 
                              ax=ax,
                              point_color=basiscolor,
                              draw_basis=show_basis,
                              color=linecolor,
                              linewidth=linewidth)
            ax.axis("equal")
            ax.axis("off")
            if show_colorbar:
                cb = plt.colorbar(img, ax=ax)
                self.ax2img[ax] = img
                self.ax2cb[ax] = cb
                if umin is not None and umax is not None:
                    img.set_clim(umin, umax)
                    cb.update_normal(img)
            else:
                self.ax2img[ax] = img
                if umin is not None and umax is not None:
                    img.set_clim(umin, umax)

        # draw title
        if title is not None:
            ax.set_title(title)

        # refresh the frame
        if update:
            self.grab_frame()


Tensor = np.ndarray|torch.Tensor 
SequenceTensor = Sequence[Tensor]|Tensor

def draw_mesh_2d_static(points:torch.Tensor|np.ndarray,
                        elements:Dict[str,torch.Tensor|np.ndarray], 
                        point_values:Dict[str, Tensor]|Tensor,
                        show_colorbar:bool = True,
                        show_mesh:bool     = False,
                        filename:str="mesh_2d_stream.jpq",
                        umin:Optional[float] = None,
                        umax:Optional[float] = None,
                        **kwargs
                          ):
    
    assert dim(points) == 2 
    assert points.shape[1] == 2
    n_point = points.shape[0]

    if isinstance(point_values,(torch.Tensor, np.ndarray)):
        assert dim(point_values) == 1
        assert point_values.shape[0] == n_point 
        point_values = as_ndarray(point_values)

        fig, ax = plt.subplots(figsize=(10, 10))
        img,_ = draw_point_value_2d(
                        points,  # 
                        point_values, 
                        elements, # type:ignore
                        ax   = ax,
                        **kwargs
                        )
        if show_mesh:
            draw_facet_2d(points, elements, ax=ax)
        ax.axis("equal")
        ax.axis("off")
        if show_colorbar:
            cb = plt.colorbar(img, ax=ax)
            if umin is not None and umax is not None:
                img.set_clim(umin, umax)
                cb.update_normal(img)

    elif isinstance(point_values, dict):
        for k,v in point_values.items():
            point_values[k] = as_ndarray(v)
            assert dim(v) == 1
            assert v.shape[0] == n_point 

        ncols = len(point_values)
        fig, ax = plt.subplots(ncols=ncols, figsize=(ncols*5, 5))
        for i,(k,v) in enumerate(point_values.items()):
            img,_ = draw_point_value_2d(
                        points,  # 
                        v, 
                        elements, # type:ignore
                        ax   = ax[i],
                        **kwargs
                        )
            if show_mesh:
                draw_facet_2d(points, elements, ax=ax[i])
            ax[i].axis("equal")
            ax[i].axis("off")
            ax[i].set_title(k)
            if show_colorbar:
                cb = plt.colorbar(img, ax=ax[i])
                if umin is not None and umax is not None:
                    img.set_clim(umin, umax)
                    cb.update_normal(img)
    else:
        raise Exception(f"{type(point_values)} is not supported for draw_mesh_2d_static")
    
    fig.savefig(filename)

def draw_mesh_2d_stream(
                        points:torch.Tensor,
                        elements:Dict[str,torch.Tensor], 
                        point_values:Dict[str, SequenceTensor]|SequenceTensor,
                        dt:Optional[float] = None,
                        show_colorbar:bool = True,
                        fix_colorbar:bool  = False,
                        show_mesh:bool     = False,
                        filename:str="mesh_2d_stream.mp4",
                        **kwargs
                          ):
    r"""
    Parameters
    ----------
        mesh:Mesh 
        point_values:Dict[str,np.ndarray|torch.Tensor]|torch.Tensor|np.ndarray
            if more than one value is supposed to be drawn, 
            then one could use {value_name:value} format 
            each tensor or ndarray should be of shape [time_seq, n_points]

    """
    assert points.dim() == 2 
    assert points.shape[1] == 2 
    n_point = points.shape[0]

    if isinstance(point_values, (torch.Tensor,np.ndarray,list,tuple)):
        if isinstance(point_values, (list,  tuple)):
            point_values = [as_ndarray(x) for x in point_values]
            point_values = np.stack(point_values, 0)
        else:
            point_values = as_ndarray(point_values)
        assert dim(point_values) == 2 
        assert point_values.shape[1] == n_point
        time_seq = point_values.shape[0]

        ncols = len(point_values)

        if fix_colorbar:
            umin, umax = point_values.min(), point_values.max()
            kwargs['umin'] = umin
            kwargs['umax'] = umax 
        
        with StreamPlotter(filename=filename) as plotter:
            for t in range(time_seq):
                if dt is None:
                    plotter.axes.set_title(f"Frame:{t:4}")
                else:
                    plotter.axes.set_title(f"t={t*dt:7.5f}s")
                plotter.draw_mesh_2d(points,
                                     elements, 
                                     point_values[t], 
                                     show_colorbar=show_colorbar,
                                     show_mesh  = show_mesh,
                                     **kwargs)
                

    elif isinstance(point_values, dict):
        
        for k, v in point_values.items():
            if isinstance(v, (list,  tuple)):
                v = [as_ndarray(x) for x in v]
                v = np.stack(v, 0)
            assert dim(v) == 2
            assert v.shape[1] == n_point 
            point_values[k] = as_ndarray(v)
        time_seq = next(iter(point_values.values())).shape[0]
        for k, v in point_values.items():
            assert v.shape[0] == time_seq,f"{k} is of shape {v.shape} but expected {(time_seq, n_point)}"

        ncols = len(point_values)

        _kwargs = {k:{} for k in point_values.keys()}
        if fix_colorbar:
            for k,v in point_values.items():
                umin, umax = v.min(), v.max()
                _kwargs[k]['umin'] = umin
                _kwargs[k]['umax'] = umax

        # TODO: Try to use FuncAnimation to make it faster 

        with StreamPlotter(ncols=ncols, filename=filename) as plotter:
            if len(point_values) == 1:
                plotter.axes.set_title(k)
            else: # axes is a sequence
                for i,(k,v) in enumerate(point_values.items()):
                    plotter.axes[i].set_title(k)

            for t in range(time_seq):
                if dt is None:
                    plotter.fig.suptitle(f"Frame:{t:4}")
                else:
                    plotter.fig.suptitle(f"t={t*dt:7.5f}s")
                for i, (k,v) in enumerate(point_values.items()):
                    if len(point_values) == 1:
                        ax = plotter.axes
                    else: # axes is a sequence
                        ax = plotter.axes[i]
                    plotter.draw_mesh_2d(points,
                                         elements,
                                         v[t],
                                         ax=ax,
                                         show_colorbar=show_colorbar,
                                         show_mesh  = show_mesh,
                                         update=False,
                                         **_kwargs[k],
                                         **kwargs)
                    ax.set_title(k)
                plotter.update()

    else:
        raise TypeError(f"{type(point_values)} is not supported for draw_mesh_2d_stream")


    
    



 
 
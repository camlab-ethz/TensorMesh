import os
import torch
import pyvista as pv
import numpy as np

# Make sure to import torch if you're using it
# import torch

def plot_value(kwargs, mesh, save_path=None, dt=None, show_mesh=False):
    is_2d = mesh.points.shape[1] == 2
    mesh.write("tmp.vtk", file_format="vtk")
    mesh = pv.read("tmp.vtk",)
    os.remove("tmp.vtk")

    key, value = next(iter(kwargs.items()))
    ncols = len(kwargs.keys())

    if isinstance(value, (np.ndarray, torch.Tensor)):
        if save_path is None:
            save_path = "mesh.png"
        plotter = pv.Plotter(shape=(1, ncols),off_screen=True)
        for i, (key, value) in enumerate(kwargs.items()):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            plotter.subplot(0, i)
            plotter.add_mesh(mesh, scalars=value, show_edges=show_mesh, cmap='jet', show_scalar_bar=False)
            plotter.add_scalar_bar(title=key, vertical=True if is_2d else False)
            plotter.view_xy() if is_2d else plotter.view_isometric()
            plotter.add_text(key, font_size=10, position='upper_edge')
        plotter.screenshot(save_path)
        plotter.close()
    elif isinstance(value, (list, tuple)):
        if save_path is None:
            save_path = "mesh.gif"
        plotter = pv.Plotter(shape=(1, ncols))
        plotter.open_gif(save_path)
        for i, (key, value) in enumerate(kwargs.items()):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            mesh.point_data[key] = value[0]
            plotter.subplot(0, i)
            plotter.add_mesh(mesh, scalars=key, show_edges=show_mesh, cmap='jet')
            plotter.add_scalar_bar(title=key, vertical=True if is_2d else False)
            plotter.view_xy() if is_2d else plotter.view_isometric()
            plotter.add_text(key, font_size=10, position='upper_edge')
        for i in range(len(value)): # for each frame
            for j, (key, values) in enumerate(kwargs.items()):
                value = values[i]
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                mesh.point_data[key] = value
                plotter.subplot(0, j)
                plotter.update_scalars(value)
            plotter.write_frame()
            plotter.set_title(f"Frame: {i:5d}" if dt is None else f"t={i*dt:7.5f}")
        plotter.close()
    else:
        raise ValueError(f"Unsupported type of value: {type(value)}")

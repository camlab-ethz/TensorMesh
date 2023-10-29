import os
import numpy as np
import torch 
import torch.nn as nn
import meshio
import pyvista as pv
import warnings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import re

from .dimension import topological_dimension


def highest_dimension_cell_type(cells):
    """
        Parameters:
        -----------
            cells: dict
                the cells of the mesh
        Returns:
        --------
            cell_type: str
                the type of the highest order cell
    """
    return max(cells.keys(), key=lambda x: topological_dimension[x])

def is_int(tensor):
    if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return True
    else:
        return False

def is_float(tensor):
    if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
        return True
    else:
        return False


class Mesh(nn.Module):
    def __init__(self, mesh):
        """
            mesh: meshio.Mesh
                a meshio mesh object
        """
        super().__init__()

        # remove invalid keys
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        for key in list(mesh.cell_data.keys()):
            if not pattern.match(key):
                mesh.cell_data.pop(key)
                warnings.warn(f"remove invalid key {key} in cell_data")

        for key in list(mesh.point_data.keys()):
            if not pattern.match(key):
                mesh.point_data.pop(key)
                warnings.warn(f"remove invalid key {key} in point_data")

        for key in  list(mesh.field_data.keys()):
            if not pattern.match(key):
                mesh.field_data.pop(key)
                warnings.warn(f"remove invalid key {key} in field_data")
        
        # cells
        self.register_buffer
        self.cells  = mesh.cells_dict
        for k, v in self.cells.items():
            self.cells[k] = torch.from_numpy(v)
            self.register_buffer(f"cells_{k}", self.cells[k])
        
        # point data
        self.point_data = mesh.point_data
        for k, v in self.point_data.items():
            self.point_data[k] = torch.from_numpy(v)
            if is_float(self.point_data[k]):
                self.register_parameter(f"point_data_{k}", self.point_data[k])
            else:
                self.register_buffer(f"point_data_{k}", self.point_data[k])
        
        # cell data
        self.cell_data  = mesh.cell_data
        for k, v in self.cell_data.items():
            for i,_v in enumerate(v):
                v[i] = torch.from_numpy(_v)
                if is_float(_v):
                    self.register_parameter(f"cell_data_{k}_{i}", v[i])
                else:
                    self.register_parameter(f"cell_data_{k}_{i}", v[i])
            self.cell_data[k] = v
   
        # field data
        self.field_data = mesh.field_data
        for k, v in self.field_data.items():
            self.field_data[k] = torch.from_numpy(v)
            if is_float(self.field_data[k]):
                self.register_parameter(f"field_data_{k}", self.field_data[k])
            else:
                self.register_buffer(f"field_data_{k}", self.field_data[k])

        self.default_cell_type = highest_dimension_cell_type(self.cells)

        self.points = nn.Parameter(torch.from_numpy(mesh.points[:, :topological_dimension[self.default_cell_type]]))

    def register_point_data(self, key, value):
        assert key not in self.point_data, f"the key {key} already exists in point_data"
        assert value.shape[0] == self.points.shape[0], f"the first dimension of value should be {self.points.shape[0]}, but got {value.shape[0]}"
        self.point_data[key] = value
        self.register_buffer(f"point_data_{key}", self.point_data[key])
      
    def __str__(self):
        return self.__repr__()
        # return f"Mesh(n_points={self.points.shape[0]}, cells=({','.join(f'{k}:{v.shape}' for k,v in self.cells.items())}))"

    def __repr__(self):
        return (
            f"Mesh(\n"
            f"    points: {self.points.shape}\n"
            f"    cells: {','.join(f'{k}:{v.shape}' for k,v in self.cells.items())}\n"
            f"    point_data: {','.join(f'{k}({v.dtype}):{v.shape[-1]}' for k,v in self.point_data.items())}\n"
            f"    cell_data: {','.join(f'{k}({next(iter(v.values())).dtype}):{next(iter(v.values())).shape[-1]}' for k,v in self.cell_data.items())}\n"
            f"    field_data: {','.join(f'{k}({v.dtype}):{v.shape[-1]}' for k,v in self.field_data.items())}\n"
            f")"
        )

    def to_meshio(self):
        
        mesh = meshio.Mesh(
            points = self.points.detach().numpy(),
            cells  = {k:v.detach().cpu().numpy() for k,v in self.cells.items()},
            point_data = {k:v.detach().cpu().numpy() for k,v in self.point_data.items()},
            cell_data  = {k:{_v.detach().cpu().numpy() for _v in v} for k,v in self.cell_data.items()},
            field_data = {k:v.detach().cpu().numpy() for k,v in self.field_data.items()},
        )  
        return mesh

    def save(self, file_name:str, file_format:str=None):
        """
            Parameters:
            -----------
                file_name: str
                    the name of the file
                file_format: str
                    the format of the file, e.g., 'msh', 'vtk', 'obj'
                    default is the file extension
            Returns:
            --------
                Mesh
        """
        mesh = self.to_meshio()
        # turn bool to int
        for key in list(mesh.point_data.keys()):
            if mesh.point_data[key].dtype == bool:
                mesh.point_data[key] = mesh.point_data[key].astype(int)
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                if _v.dtype == bool:
                    mesh.cell_data[key][i] = _v.astype(int)
        for key in list(mesh.field_data.keys()):
            if mesh.field_data[key].dtype == bool:
                mesh.field_data[key] = mesh.field_data[key].astype(int)    
        
        
        if file_name.endswith(".vtk") or file_name.endswith(".vtu"):
            # if vtk/vtu turn 2d to 3d 
            if mesh.points.shape[1] == 2:
                mesh.points = np.concatenate([mesh.points, torch.zeros(mesh.points.shape[0], 1)], -1)
            if "u" not in mesh.point_data.keys():
                mesh.point_data["u"] = np.zeros((mesh.points.shape[0], )) 
         
        meshio.write(file_name, mesh, file_format)

    def to_file(self, file_name:str, file_format:str=None):
        return self.save(file_name, file_format)

    def elements(self, cell_type=None):
        if cell_type is None:
            cell_type = self.default_cell_type
        return self.cells[cell_type]
    
    def plot(self, kwargs, save_path=None, backend="pyvista", dt=None):
        """
            Parameters:
            -----------
                kwargs: dict
                    
        """
    
        if backend == "pyvista":
            assert len(kwargs.keys()) == 1, "only one keyword argument is allowed"
            file_path = 'tmp.vtu'
            self.save(file_path)        
            pv_mesh = pv.read(file_path)
            os.remove(file_path)
            key, value = next(iter(kwargs.items()))
            if isinstance(value, (torch.Tensor,np.ndarray)):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                if save_path is None:
                    save_path = 'mesh.png'
                pv_mesh.point_data[key] = value
                pv_mesh.plot(scalars=key, show_edges=True,screenshot=save_path)
                
            elif isinstance(value, (list,  tuple)):
                if save_path is None:
                    save_path = 'mesh.gif'
                plotter = pv.Plotter()
                plotter.add_mesh(pv_mesh, scalars=key)
                plotter.show(auto_close=False, interactive_update=True)
                plotter.open_gif(save_path)
                for v in value:
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    plotter.update_scalars(v)
                    plotter.render()
                plotter.close()

            else:
                raise NotImplementedError(f"the type of value {type(value)} is not supported")
        elif backend == "matplotlib":
            assert self.default_cell_type == "triangle", "only triangle mesh is supported"
            ncols = len(kwargs.keys())
            fig, ax = plt.subplots(1, ncols, figsize=(5*ncols, 5))
            points = self.points.detach().cpu().numpy()
            x,  y  = points[:, 0], points[:, 1]
            triang = tri.Triangulation(x, y, self.elements().detach().cpu().numpy())
            key, value = next(iter(kwargs.items()))
            if not isinstance(ax,  np.ndarray):
                ax = [ax]
            if isinstance(value,(torch.Tensor, np.ndarray)):      
                if save_path is None:
                    save_path = 'mesh.png'
                for i, (key, value) in enumerate(kwargs.keys()):
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    ax[i].tripcolor(triang, value, shading="gouraud", cmap="jet")
                    ax[i].set_title(key)
                    ax[i].axis("off")
                    ax[i].set_aspect("equal")
                fig.savefig(save_path, dpi=400)
            elif isinstance(value, (list, tuple)):
                if save_path is None:
                    save_path = 'mesh.gif'
                cbs = []
                imgs = []
                for i, (key, value) in enumerate(kwargs.items()):
                    v   = value[0].detach().cpu().numpy() if isinstance(value[0], torch.Tensor) else value[0]
                    img = ax[i].tripcolor(triang, v, shading="gouraud", cmap="jet")
                    ax[i].set_title(key)
                    ax[i].axis("off")
                    ax[i].set_aspect("equal")
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
        else:
            raise Exception(f"backend {backend} is not supported")
            
    @property
    def n_point(self):
        return self.points.shape[0]

    @property
    def boundary_mask(self):
        if "is_boundary" in self.point_data.keys():
            return self.point_data["is_boundary"]
        elif "boundary_mask" in self.point_data.keys():
            return self.point_data["boundary_mask"]
        else:
            raise Exception("'boundary_mask' or 'is_boundary' is not found in point_data")

    @property
    def dtype(self):
        return self.points.dtype
    
    @property 
    def device(self):
        return self.points.device

    def type(self, dtype):
        for k, v in self.named_buffers():
            self.register_buffer(k, v.type(dtype))
        self.points = self.points.type(dtype)
        return self
    
    def to(self, device):
        for k, v in self.named_buffers():
            self.register_buffer(k, v.to(device))
        for k, v in self.cells.items():
            self.cells[k] = self.cells[k].to(device)
        self.points = self.points.to(device)
        return self
    
    def float(self):
        return self.type(torch.float)

    def double(self):
        return self.type(torch.double)
    
    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self, device=None):
        return self.to(torch.device("cuda", device))

    @classmethod
    def from_meshio(cls,mesh):
        """
            Parameters:
            -----------
                mesh: meshio.Mesh
                    a meshio mesh object
            Returns:
            --------
                Mesh
        """
        return cls(mesh)
    
    @classmethod
    def read_file(cls, file_name:str, file_format:str=None):
        """
            Parameters:
            -----------
                file_name: str
                    the name of the file
                file_format: str
                    the format of the file, e.g., 'msh', 'vtk', 'obj'
                    default is the file extension
            Returns:
            --------
                Mesh
        """
        return cls(meshio.read(file_name, file_format))
    
    @classmethod
    def from_file(cls, file_name:str, file_format:str=None):
        return cls.read_file(file_name, file_format)

    @staticmethod
    def rectangle(chara_length=0.1,
             order=1,
             element_type="tri",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
             visualize=False,
             cache_path=".gmsh_cache/tmp.msh"):
        from .gen import rectangle
        return rectangle(chara_length, order, element_type, left, right, bottom, top, visualize, cache_path)
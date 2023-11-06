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
from itertools import chain

from ..utils import is_float
from .dimension import topological_dimension
# from .vis import plot as plot_matplotlib


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

class BufferDict(nn.Module):
    def __init__(self, data):
        super().__init__()
        self._data = {}
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        for key in list(data.keys()):
            if not pattern.match(key):
                self._data[key] = data.pop(key)
        for key, value in data.items():
            self.register_buffer(key, value)
    
    def as_parameter(self, key):
        buffer = self._buffers.pop(key)
        self.register_parameter(key, buffer)
        
    def as_buffer(self, key):
        parameter = self._parameters.pop(key)
        self.register_buffer(key, parameter)
        
    def keys(self):
        return chain(self._buffers.keys(), self._parameters.keys(), self._data.keys())
    
    def items(self):
        return chain(self._buffers.items(), self._parameters.items(), self._data.items())
    
    def values(self):
        return chain(self._buffers.values(), self._parameters.values(), self._data.values())
    
    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(f"{key} is not found in the BufferDict")
        return self._buffers[key] if key in self._buffers else self._parameters[key] if key in self._parameters else self._data[key]

    @property
    def dtype(self):
        return next(iter(self.buffers().values())).dtype
    
    @property
    def device(self):
        return next(iter(self.buffers().values())).device
    

class Mesh(nn.Module):
    def __init__(self, mesh):
        """
            mesh: meshio.Mesh
                a meshio mesh object
        """
        super().__init__()
        # turn is_... or ..._mask to bool
        for key in list(mesh.point_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.point_data[key] = mesh.point_data[key].astype(bool)
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                if key.startswith("is_") or key.endswith("_mask"):
                    mesh.cell_data[key][i] = _v.astype(bool)
        for key in list(mesh.field_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.field_data[key] = mesh.field_data[key].astype(bool)
        
        # cells
        self.cells  = BufferDict({k:torch.from_numpy(v) for k,v in mesh.cells_dict.items()})
        
        # point data
        self.point_data = BufferDict({k:torch.from_numpy(v) for k,v in mesh.point_data.items()})

        # cell data
        self.cell_data  = BufferDict({
            k:BufferDict({i:torch.from_numpy(_v) for i,_v in v.items()}) for k,v in mesh.cell_data_dict.items()
        })
   
        # field data
        self.field_data = BufferDict({k:torch.from_numpy(v) for k,v in mesh.field_data.items()})

        # cell setes useless
        self.cell_sets = mesh.cell_sets

        self.default_cell_type = highest_dimension_cell_type(self.cells)

        self.points = nn.Parameter(torch.from_numpy(mesh.points[:, :topological_dimension[self.default_cell_type]]))

    def register_point_data(self, key, value):
        assert key not in self.point_data.keys(), f"the key {key} already exists in point_data"
        assert value.shape[0] == self.points.shape[0], f"the first dimension of value should be {self.points.shape[0]}, but got {value.shape[0]}"
        self.point_data.register_buffer(key, value)
      
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
            points = self.points.detach().cpu().numpy(),
            cells  = {k:v.detach().cpu().numpy() for k,v in self.cells.items()},
            point_data = {k:v.detach().cpu().numpy() for k,v in self.point_data.items()},
            cell_data  = {k:[_v.detach().cpu().numpy() for _v in v.values()] for k,v in self.cell_data.items()},
            field_data = {k:v.detach().cpu().numpy() for k,v in self.field_data.items()},
            cell_sets = self.cell_sets
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
        # turn is_... or ..._mask to float
        for key in list(mesh.point_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.point_data[key] = mesh.point_data[key].astype(float)
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                if key.startswith("is_") or key.endswith("_mask"):
                    mesh.cell_data[key][i] = _v.astype(float)
        for key in list(mesh.field_data.keys()):
            if key.startswith("is_") or key.endswith("_mask"):
                mesh.field_data[key] = mesh.field_data[key].astype(float)
        
        # assert no bool variables, since file cannot save bool
        for key in list(mesh.point_data.keys()):
            assert mesh.point_data[key].dtype != bool, f"PointData: bool is not supported in meshio, but got {key}"
        for key in list(mesh.cell_data.keys()):
            for i, _v in enumerate(mesh.cell_data[key]):
                assert _v.dtype != bool, f"CellData: bool is not supported in meshio, but got {key}"
        for key in list(mesh.field_data.keys()):
            assert mesh.field_data[key].dtype != bool, f"FieldData: bool is not supported in meshio, but got {key}"
        
        if file_name.endswith(".vtk") or file_name.endswith(".vtu"):
            # if vtk/vtu turn 2d to 3d 
            if mesh.points.shape[1] == 2:
                mesh.points = np.concatenate([mesh.points, torch.zeros(mesh.points.shape[0], 1)], -1)
            if "u" not in mesh.point_data.keys():
                mesh.point_data["u"] = np.zeros((mesh.points.shape[0], )) 
         
        meshio.write(file_name, mesh, file_format)
        return self

    def to_file(self, file_name:str, file_format:str=None):
        return self.save(file_name, file_format)
    
    def write(self, file_name:str,file_format:str=None):
        return self.save(file_name, file_format)

    def elements(self, cell_type=None):
        if cell_type is None:
            cell_type = self.default_cell_type
        return self.cells[cell_type]
    
    def plot(self, kwargs, save_path=None, backend="pyvista", dt=None, show_mesh=False):
        """
            Parameters:
            -----------
                kwargs: dict
                    
        """
        from ..visualization import plot_matplotlib, plot_pyvista

        plot_fns = {
            "pyvista":plot_pyvista,
            "matplotlib":plot_matplotlib,
        }
        assert  backend in plot_fns.keys(), f"backend must be one of {list(plot_fns.keys())}, but got {backend}"

        return plot_fns[backend](kwargs, self,  save_path, dt, show_mesh)
             
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
    def gen_rectangle(chara_length=0.1,
             order=1,
             cell_type="tri",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_rectangle
        return gen_rectangle(chara_length, order, cell_type, left, right, bottom, top, visualize, cache_path)

    @staticmethod
    def gen_hollow_rectangle(
        chara_length=0.1,
        order=1,
        cell_type="quad",
        outer_left=0.0, outer_right=1.0, outer_bottom=0.0, outer_top=1.0,
        inner_left = 0.25,  inner_right=0.75,
        inner_bottom =0.25, inner_top=0.75,
        visualize=False,
        cache_path=None
    ):
        from ..dataset import gen_hollow_rectangle
        return gen_hollow_rectangle(chara_length,
             order,
             cell_type,
             outer_left, outer_right, outer_bottom, outer_top,
             inner_left,  inner_right,
             inner_bottom, inner_top,
             visualize,
             cache_path)

    @staticmethod
    def gen_circle(chara_length=0.1,
            order=1,
            cell_type="tri",
            cx = 0.0, cy = 0.0, r = 1.0,
            visualize=False,
            cache_path=None):
        from ..dataset import gen_circle
        return gen_circle(chara_length, order, cell_type, cx, cy, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_circle(chara_length=0.1,
             order=1,
             cell_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_hollow_circle
        return gen_hollow_circle(chara_length,
             order,
             cell_type,
             cx, cy, r_inner, r_outer,
             visualize,
             cache_path)

    @staticmethod
    def gen_L(chara_length=0.1,
             order=1,
             cell_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0, 
             top_inner=0.5,
             right_inner=0.5,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_L
        return gen_L(chara_length, order, cell_type, left, right, bottom, top, top_inner, right_inner, visualize, cache_path)

    @staticmethod
    def gen_cube(chara_length=0.1, 
             order=1,
             left=0.0, right=1.0,
             bottom=0.0, top=1.0,
             front=0.0, back=1.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_cube
        return gen_cube(chara_length, order, left, right, bottom, top, front, back, visualize, cache_path)
    
    @staticmethod
    def gen_hollow_cube(chara_length=0.1,
             order=1,
             outer_left=0.0, outer_right=1.0, 
             outer_bottom=0.0, outer_top=1.0,
             outer_front=0.0, outer_back=1.0,
             inner_left=0.25, inner_right=0.75,
             inner_bottom=0.25, inner_top=0.75,
             inner_front=0.25, inner_back=0.75,
             visualize=False,
             cache_path=".gmsh_cache/tmp.msh"):
        from ..dataset import gen_hollow_cube
        return gen_hollow_cube(chara_length,
             order,
             outer_left, outer_right, 
             outer_bottom, outer_top,
             outer_front, outer_back,
             inner_left, inner_right,
             inner_bottom, inner_top,
             inner_front, inner_back,
             visualize,
             cache_path)
    
    @staticmethod
    def gen_sphere(chara_length=0.1,
                order=1,
                cx = 0.0, cy = 0.0, cz=0.0, r = 1.0,
                visualize=False,
                cache_path=None):
        from ..dataset import gen_sphere
        return gen_sphere(chara_length, order, cx, cy, cz, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_sphere(chara_length=0.1,
             order=1,
              cx = 0.0, cy = 0.0, cz=0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_hollow_sphere
        return gen_hollow_sphere(chara_length, order, cx, cy, cz, r_inner, r_outer, visualize, cache_path)
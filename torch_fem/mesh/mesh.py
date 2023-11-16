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
import scipy.spatial
from itertools import chain
from functools  import reduce
from operator import eq
from collections import defaultdict
from typing import Iterable


from ..shape import get_basis, get_boundary
from .dimension import topological_dimension
from ..sparse import SparseMatrix
from ..nn import BufferDict





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
        self.cells  = BufferDict({k:torch.from_numpy(v).long() for k,v in mesh.cells_dict.items()})
        
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

        self.dim2eletyp = defaultdict(list) # Dict[int, List[str]]
        for element_type in self.cells.keys():
            self.dim2eletyp[topological_dimension[element_type]].append(element_type)
        self.default_eletyp = self.dim2eletyp[max(self.dim2eletyp.keys())] 
        if len(self.default_eletyp) == 1: # if only one element type, use it as default
            self.default_eletyp = self.default_eletyp[0]
        self.default_element_type = self.default_eletyp


        dimension = max(self.dim2eletyp.keys())

        self.register_buffer(
            "points",
            torch.from_numpy(mesh.points[:, :dimension])
        )

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

    def node_adjacency(self, element_type=None):
        """
            Parameters:
            -----------
                element_type: str or Iterable[str] or None
                    the type of the elements
                    if None is the default_eletyp
                    default : None
            Returns:
            --------
                SparseMatrix [n_point, n_point]
        """
        elements = self.elements(element_type)
        if isinstance(elements, torch.Tensor):
            edges = torch.vmap(lambda x:torch.stack(torch.meshgrid(x,x),-1))(elements).reshape(-1,2).T
        elif isinstance(elements, dict):
            edges = []
            for k,v in elements.items():
                edges.append(torch.vmap(lambda x:torch.stack(torch.meshgrid(x,x),-1))(v).reshape(-1,2).T)
            edges = torch.cat(edges, -1)
      
        connections = torch.sparse_coo_tensor(
            edges, torch.ones(edges.shape[1]), size=(self.n_point, self.n_point)
        ).coalesce()
        edges = connections.indices()
        return SparseMatrix(torch.ones(edges.shape[1]), edges[0], edges[1], (self.n_point, self.n_point))

    def element_adjacency(self, element_type=None):
        """
            Parameters:
            -----------
                element_type: str or Iterable[str] or None
                    the type of the elements, should be of same dimension
                    if None is the default_eletyp
                    default : None
            Returns:
            --------
                SparseMatrix [n_element, n_element]
        """

        def get_element_adjacency(ele_ids, boundaries):
            """
                Parameters:
                -----------
                    ele_ids: torch.Tensor [\int n_element*n_boundary_per_element]
                    boundaries: torch.Tensor [\int n_element*n_boundary_per_element, n_boundary_basis]
            """
            assert ele_ids.dim() ==  1, f"ele_ids should be 1d, but got {ele_ids.dim()}"	
            assert boundaries.dim() == 2, f"boundaries should be 2d, but got {boundaries.dim()}"
            assert boundaries.shape[0] == ele_ids.shape[0], f"the first dimension of boundaries should be {ele_ids.shape[0]}, but got {boundaries.shape[0]}"
            boundaries= boundaries.reshape(-1, boundaries.shape[-1]) # [n_element * n_boundary_per_element, n_boundary_basis]
            # make sure the index is ascending, so it's unique
            boundaries= boundaries.sort(dim=-1).values # [n_element * n_boundary_per_element, n_boundary_basis] 
            # the count = 2 means the boundary is shared by two elements, otherwise the boundary is on the boundary of the domain
            unique_boundaries, inverse_indices, counts = boundaries.unique(dim=0, return_counts=True,  return_inverse=True) # [n_boundary_element, n_boundary_basis]
            assert counts.max() == 2, f"the maximum number of elements sharing a boundary is 2, but got {counts.max()}"
            valid_mask = counts == 2 # [n_boundary_element]
            # for the each element, which boundary is shared by two elements
            valid_mask = valid_mask[inverse_indices] # [n_element * n_boundary_per_element]
            ele_ids_bd = ele_ids    # [n_element * n_boundary_per_element]
            # only keep the shared boundary elements, but now it's shuffled
            ele_ids_bd = ele_ids_bd[valid_mask] # [n_shared_boundary * 2]
            # by sorting the inverse_indices, we can get the order like [0,0,1,1,2,2,3,3,...]
            sort_index=torch.argsort(inverse_indices[valid_mask]) # [n_shared_boundary * 2]
            # and then we can get the shared boundary elements in order 
            ele_ids_bd = ele_ids_bd[sort_index] # [n_shared_boundary * 2]
            edges     = ele_ids_bd.reshape(-1, 2).T # [2, n_shared_boundary]
            # add the reverse direction
            edges = torch.cat([edges, torch.stack([edges[1], edges[0]])], -1)
          
            return edges

        elements = self.elements(element_type)
        if isinstance(elements, torch.Tensor):
            n_element = elements.shape[0]
            boundaries= get_boundary(self.default_eletyp) # [n_boundary_per_element, n_boundary_basis]
            
            if isinstance(boundaries, torch.Tensor):
                boundaries= elements[:, boundaries] # [n_element, n_boundary_per_element, n_boundary_basis]
                n_boundary_per_element = boundaries.shape[1]
                n_boundary_basis = boundaries.shape[-1]
                edges = get_element_adjacency(torch.arange(n_element).repeat_interleave(n_boundary_per_element), boundaries.reshape(-1, n_boundary_basis))
            elif isinstance(boundaries, dict):
                edges = []
                for k,v in boundaries.items():
                    n_boundary_per_element = v.shape[1]
                    n_boundary_basis = v.shape[-1]
                    edges.append(get_element_adjacency(torch.arange(n_element).repeat_interleave(n_boundary_per_element), v.reshape(-1, n_boundary_basis)))
                edges = torch.cat(edges, -1)
            return SparseMatrix(torch.ones(edges.shape[1]), edges[0], edges[1], (n_element, n_element))
            
            
        else: # mix of different element types
            assert reduce(eq, [topological_dimension[k] for k in elements.keys()]), f"all elements should be of same dimension, but got {elements.keys()}"
            n_element = sum([v.shape[0] for v in elements.values()])
            ele_ids   = torch.arange(n_element)
            boundaries = {}
            ele_ids    = {}
            ele_ptr    = 0
            for element_type, element in elements.items():
                # breakpoint()
                partial_boundaries = get_boundary(element_type) 
                partial_boundaries = element[:, partial_boundaries] # [n_element, n_boundary_per_element, n_boundary_basis]
                partial_ele_ids = torch.arange(ele_ptr, ele_ptr + element.shape[0])
                if isinstance(partial_boundaries, torch.Tensor):
                    n_boundary_per_element = partial_boundaries.shape[1]
                    n_boundary_basis = partial_boundaries.shape[-1]
                    if n_boundary_basis in boundaries:
                        boundaries[n_boundary_basis].append(partial_boundaries)
                        ele_ids[n_boundary_basis].append(partial_ele_ids.repeat_interleave(n_boundary_per_element))
                    else:
                        boundaries[n_boundary_basis] = [partial_boundaries]
                        ele_ids[n_boundary_basis] = [partial_ele_ids.repeat_interleave(n_boundary_per_element)]
                elif isinstance(partial_boundaries, dict):
                    for k,v in partial_boundaries.items():
                        n_boundary_per_element = v.shape[1]
                        n_boundary_basis = v.shape[-1]
                        if n_boundary_basis in boundaries:
                            boundaries[n_boundary_basis].append(v)
                            ele_ids[n_boundary_basis].append(partial_ele_ids.repeat_interleave(n_boundary_per_element))
                        else:
                            boundaries[n_boundary_basis] = [v]
                            ele_ids[n_boundary_basis] = [partial_ele_ids.repeat_interleave(n_boundary_per_element)]
                ele_ptr += element.shape[0]
           
            for k, v in boundaries.items():
                boundaries[k] = torch.cat([i.reshape(-1,i.shape[-1]) for i in v], 0)
                ele_ids[k]    = torch.cat(ele_ids[k], 0)
            edges = []
            for k in boundaries.keys():
                edges.append(get_element_adjacency(ele_ids[k], boundaries[k]))
            edges = torch.cat(edges, -1)
            return SparseMatrix(torch.ones(edges.shape[1]), edges[0], edges[1], (n_element, n_element))

    def elements(self, element_type=None):
        """
            Parameters:
            -----------
                element_type: str or Iterable[str] or None
                    the type of the elements
                    if None is the default_eletyp
                    default : None 
            Returns:
            --------
                torch.Tensor [n_element, n_basis] if element_type is str
                Dict[str, torch.Tensor [n_element, n_basis]] if element_type is Iterable[str]
        """
        if element_type is None:
            element_type = self.default_eletyp
        if isinstance(element_type, str):
            return self.cells[element_type]
        elif isinstance(element_type, Iterable):
            return {k:self.cells[k] for k in element_type}
        else:
            raise Exception(f"element_type must be str or Iterable[str], but got {element_type}")
    
    def plot(self, values= None, save_path=None, backend="matplotlib", dt=None, show_mesh=False):
        """
            Parameters:
            -----------
                values: dict
                    
        """
        # from ..visualization import plot_value_matplotlib, plot_pyvista

        # plot_fns = {
        #     "pyvista":plot_pyvista,
        #     "matplotlib":plot_matplotlib,
        # }
        # assert  backend in plot_fns.keys(), f"backend must be one of {list(plot_fns.keys())}, but got {backend}"

        # return plot_fns[backend](kwargs, self,  save_path, dt, show_mesh)
             
        from ..visualization import plot_value_matplotlib, plot_mesh_matplotlib

        if values is None:
            return plot_mesh_matplotlib(self, save_path)
        else:
            return plot_value_matplotlib(values, self, save_path, dt, show_mesh)

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
             element_type="tri",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_rectangle
        return gen_rectangle(chara_length, order, element_type, left, right, bottom, top, visualize, cache_path)

    @staticmethod
    def gen_hollow_rectangle(
        chara_length=0.1,
        order=1,
        element_type="quad",
        outer_left=0.0, outer_right=1.0, outer_bottom=0.0, outer_top=1.0,
        inner_left = 0.25,  inner_right=0.75,
        inner_bottom =0.25, inner_top=0.75,
        visualize=False,
        cache_path=None
    ):
        from ..dataset import gen_hollow_rectangle
        return gen_hollow_rectangle(chara_length,
             order,
             element_type,
             outer_left, outer_right, outer_bottom, outer_top,
             inner_left,  inner_right,
             inner_bottom, inner_top,
             visualize,
             cache_path)

    @staticmethod
    def gen_circle(chara_length=0.1,
            order=1,
            element_type="tri",
            cx = 0.0, cy = 0.0, r = 1.0,
            visualize=False,
            cache_path=None):
        from ..dataset import gen_circle
        return gen_circle(chara_length, order, element_type, cx, cy, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_hollow_circle
        return gen_hollow_circle(chara_length,
             order,
             element_type,
             cx, cy, r_inner, r_outer,
             visualize,
             cache_path)

    @staticmethod
    def gen_L(chara_length=0.1,
             order=1,
             element_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0, 
             top_inner=0.5,
             right_inner=0.5,
             visualize=False,
             cache_path=None):
        from ..dataset import gen_L
        return gen_L(chara_length, order, element_type, left, right, bottom, top, top_inner, right_inner, visualize, cache_path)

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
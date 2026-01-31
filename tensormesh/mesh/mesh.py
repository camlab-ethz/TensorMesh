from fileinput import filename
import os
from typing import Optional, Union, Sequence
import numpy as np
import torch 
import torch.nn as nn
import meshio
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Iterable, Dict, Optional,List
from .adjacency import node_adjacency, element_adjacency
from .coloring import graph_coloring
from .partition import graph_partition
from .. import element as E
from .. import sparse
from ..nn import BufferDict

# Lazy import visualization to avoid matplotlib dependency issues
V = None

def _get_visualization():
    global V
    if V is None:
        from .. import visualization as _V
        V = _V
    return V


def tri_reorder(elements: torch.Tensor) -> torch.Tensor:
    """
    (Deprecated) Reorder triangle connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.

    This is kept for backward compatibility. The canonical implementation lives in
    :meth:`tensormesh.element.Element.reorder`.
    """
    return E.Triangle.reorder(elements, to_gmsh=False)

def quad_reorder(elements: torch.Tensor) -> torch.Tensor:
    """
    (Deprecated) Reorder quadrilateral connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.

    This is kept for backward compatibility. The canonical implementation lives in
    :meth:`tensormesh.element.Element.reorder`.
    """
    return E.Quadrilateral.reorder(elements, to_gmsh=False)
    

def tet_reorder(elements: torch.Tensor) -> torch.Tensor:
    """
    (Deprecated) Reorder tetrahedron connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.

    This is kept for backward compatibility. The canonical implementation lives in
    :meth:`tensormesh.element.Element.reorder`.
    """
    return E.Tetrahedron.reorder(elements, to_gmsh=False)
    
def hex_reorder(elements: torch.Tensor) -> torch.Tensor:
    """
    (Deprecated) Reorder hexahedron connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.

    Note: Hexahedron gmsh permutation is currently identity unless implemented in
    :class:`tensormesh.element.Hexahedron`. This wrapper removes the old breakpoint-based code.
    """
    return E.Hexahedron.reorder(elements, to_gmsh=False)


def pyr_reorder(elements: torch.Tensor) -> torch.Tensor:
    """
    (Deprecated) Reorder pyramid connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.
    """
    return E.Pyramid.reorder(elements, to_gmsh=False)
        


def pri_reorder(elements:torch.Tensor)->torch.Tensor:
    """
    (Deprecated) Reorder prism connectivity from Gmsh/VTK ordering to TensorMesh internal ordering.
    """
    return E.Prism.reorder(elements, to_gmsh=False)


class Mesh(nn.Module):
    r"""
    Parameters
    ----------
    mesh: :meth:`meshio.Mesh`
        a meshio mesh object
    

    Attributes
    ----------
    points: torch.Tensor 
        2D tensor of shape :math:`[|\mathcal V|, D]`, where :math:`|\mathcal V|` is the number of points and :math:`D` is the dimension of the space
        the coordinates of the points
    cells: BufferDict[str, torch.Tensor]
        Each key is a :meth:`tensormesh.shape.element_type`, 
        and for each :obj:`element_type`, there is a corresponding 2D tensor of shape :math:`[|\mathcal V, B]`, where :math:`B` is the number of basis functions
        the cells of the mesh
    point_data: BufferDict[str, torch.Tensor], optional
        Each key is a :meth:`tensormesh.shape.element_type`, 
        the point data
    cell_data: BufferDict[str, BufferDict[int, torch.Tensor]], optional
        Each key is a :meth:`tensormesh.shape.element_type`, 
        the cell data
    field_data: BufferDict[str, torch.Tensor], optional
        Each key is a :meth:`tensormesh.shape.element_type`, 
        the field data
    cell_sets: dict, optional
        Each key is a :meth:`tensormesh.shape.element_type`, 
        the cell sets
    dim2eletyp: Dict[int, List[str]]
        Each key is a dimension, and the value is a list of element types of the dimension
    default_eletyp: str
        the default element type
    default_element_type: str
        the default element type

    """

    cells:BufferDict # str->torch.Tensor[n_element,n_basis]
    point_data:BufferDict # str->torch.Tensor[n_point,...]
    cell_data:nn.ModuleDict # str->Dict[str,torch.Tensor[n_element,...]]
    field_data:BufferDict # str->torch.Tensor[n_field,...]
    cell_sets:Dict
    points:torch.Tensor # [n_point, n_dim]
    dim2eletyp:Dict[int, List[str]] 
    default_eletyp:Union[str,List[str]]

    def __init__(self, mesh:meshio.Mesh, reorder:bool=False):
    
        super().__init__()

        
    
        for i, cell in enumerate(mesh.cells):
            if reorder:
                # Centralized reorder implementation lives in tensormesh.element.Element.reorder
                # Convert from Gmsh/VTK ordering -> TensorMesh internal ordering.
                elem_cls = E.element_type2element(cell.type)
                data_t = torch.from_numpy(cell.data)
                data_t = elem_cls.reorder(data_t, to_gmsh=False)
                cell.data = data_t.numpy()

            
           

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
        self.cell_data  = nn.ModuleDict({
            k:BufferDict({i:torch.from_numpy(_v) for i,_v in v.items()}) for k,v in mesh.cell_data_dict.items()
        })
   
        # field data
        self.field_data = BufferDict({k:torch.from_numpy(v) for k,v in mesh.field_data.items()})

        # cell setes useless
        self.cell_sets = mesh.cell_sets

        self.dim2eletyp = defaultdict(list) # Dict[int, List[str]]
        for element_type in self.cells.keys():
            self.dim2eletyp[E.element_type2dimension[element_type]].append(element_type)
        self.default_eletyp = self.dim2eletyp[max(self.dim2eletyp.keys())] 
        if len(self.default_eletyp) == 1: # if only one element type, use it as default
            self.default_eletyp = self.default_eletyp[0]

        dimension = max(self.dim2eletyp.keys())

        self.register_buffer(
            "points",
            torch.from_numpy(mesh.points[:, :dimension])
        )

    def register_point_data(self, key:str, value:torch.Tensor):
        """Add key-value pair to :attr:`point_data` buffer,
        since the :attr:`point_data` is a :class:`tensormesh.BufferDict`, you are recommended to use this method instead of :obj:`__setitem__`
        
        Parameters
        ----------
        key: str
            the key of the value
        value: torch.Tensor
            1D tensor of shape :math:`[|\\mathcal V|,...]`, where :math:`\\mathcal V` is the number of nodes/vertices/points, the value to be registered

        Returns
        -------
        tensormesh.mesh.Mesh
            self will be returned
        """
        assert key not in self.point_data.keys(), f"the key {key} already exists in point_data"
        assert value.shape[0] == self.points.shape[0], f"the first dimension of value should be {self.points.shape[0]}, but got {value.shape[0]}"
        self.point_data.register_buffer(key, value)

        return self
      
    def register_element_data(self, key:str, value:Union[Dict[str,torch.Tensor],torch.Tensor]):
        """Add key-value pair to :attr:`cell_data`
        """
        if isinstance(value, torch.Tensor):
            assert isinstance(self.elements(), torch.Tensor), f"Only for homogenous elements, value can be passed as torch.Tensor, else it should be Dict[element_type,torch.Tensor]"
            assert value.shape[0] == self.elements().shape[0], f"the first dimension of value should be {self.elements().shape[0]}, but got {value.shape[0]}"
            self.cell_data[self.default_element_type][key] =  value    
        else:
            assert isinstance(value, dict), f"The value should be either torch.Tensor or Dict[element_type, torch.Tensor]"
            assert len(set(self.default_element_type).difference(value.keys())) == 0, f"The keys of value should be exactly the same as default_element_type, but got {value.keys()}"

            for element_type, _value in value.items():
                assert _value.shape[0] == self.elements(element_type).shape[0], f"the first dimension of value should be {self.elements(element_type).shape[0]}, but got {_value.shape[0]}"
                self.cell_data[element_type][key] = _value  

        return self

    def __str__(self):
        return self.__repr__()
        # return f"Mesh(n_points={self.points.shape[0]}, cells=({','.join(f'{k}:{v.shape}' for k,v in self.cells.items())}))"

    def __repr__(self):
        # Build cell_data string safely (handle empty nested dicts)
        cell_data_strs = []
        for k, v in self.cell_data.items():
            if len(v) > 0:
                first_val = next(iter(v.values()))
                cell_data_strs.append(f'{k}({first_val.dtype}):{first_val.shape[-1] if first_val.dim() > 0 else 1}')
        cell_data_str = ','.join(cell_data_strs) if cell_data_strs else ''
        
        # Build point_data and field_data strings safely
        point_data_str = ','.join(
            f'{k}({v.dtype}):{v.shape[-1] if v.dim() > 0 else 1}' 
            for k, v in self.point_data.items()
        ) if self.point_data else ''
        
        field_data_str = ','.join(
            f'{k}({v.dtype}):{v.shape[-1] if v.dim() > 0 else 1}' 
            for k, v in self.field_data.items()
        ) if self.field_data else ''
        
        return (
            f"Mesh(\n"
            f"    points: {self.points.shape}\n"
            f"    cells: {','.join(f'{k}:{v.shape}' for k,v in self.cells.items())}\n"
            f"    point_data: {point_data_str}\n"
            f"    cell_data: {cell_data_str}\n"
            f"    field_data: {field_data_str}\n"
            f")"
        )

    def to_meshio(self, reorder: bool = False)->meshio.Mesh:
        """
        Returns
        -------
        meshio.Mesh
            the meshio mesh object
        """
        
        # Build cells (optionally reorder to Gmsh/VTK for export)
        cells_out = {}
        for k, v in self.cells.items():
            conn = v.detach().cpu()
            if reorder:
                elem_cls = E.element_type2element(k)
                conn = elem_cls.reorder(conn, to_gmsh=True)
            cells_out[k] = conn.numpy()

        mesh = meshio.Mesh(
            points = self.points.detach().cpu().numpy(),
            cells  = cells_out,
            point_data = {k:v.detach().cpu().numpy() for k,v in self.point_data.items()},
            cell_data  = {k:[_v.detach().cpu().numpy() for _v in v.values()] for k,v in self.cell_data.items()},
            field_data = {k:v.detach().cpu().numpy() for k,v in self.field_data.items()},
            cell_sets = self.cell_sets
        )  
        return mesh

    @property
    def dim(self)->int:
        return self.points.shape[1]

    def save(self, file_name:str, file_format:Optional[str]=None):
        """
        Parameters
        ----------
        file_name: str
            the name of the file
        file_format: str
            the format of the file, e.g., 'msh', 'vtk', 'obj'
            default is the file extension
        Returns
        -------
        tensormesh.mesh.Mesh
            self will be returned
        """
        do_reorder = file_name.endswith((".vtk", ".vtu"))
        mesh = self.to_meshio(reorder=do_reorder)
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
                mesh.points = np.concatenate([mesh.points, np.zeros((mesh.points.shape[0], 1))], -1)
            if "u" not in mesh.point_data.keys():
                mesh.point_data["u"] = np.zeros((mesh.points.shape[0], )) 

            # they don't support cell_sets either
            for key in mesh.cell_sets.copy().keys():
                mesh.cell_sets.pop(key)
         
        meshio.write(file_name, mesh, file_format)
        return self

    to_file = save
    
    def node_adjacency(self, element_type:Optional[Union[str, Iterable[str]]]=None)->sparse.SparseMatrix:
        """get the node adjacency matrix, inside each element, the nodes are considered fully connected

        Parameters
        ----------
        element_type : str or Iterable[str] or None
            the type of the elements
            if :obj:`None` is the :obj:`default_element_type`
            default : :obj:`None`

        Returns
        -------
        SparseMatrix 
            the adjacency matrix of nodes :math:`[|\\mathcal V|,|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of nodes
        """
        elements = self.elements(element_type)
        if isinstance(elements, dict):
            elements = elements.values()
        return node_adjacency(elements, self.n_points) # type:ignore

    def element_adjacency(self, element_type:Optional[str]=None)->sparse.SparseMatrix:
        """get the element adjacency matrix, the element are considered connected only if they share a boundary/facet
        
        Parameters
        ----------
        element_type : str or Iterable[str] or None
            the type of the elements, should be of same dimension
            if :obj:`None` is the :obj:`default_element_type`
            default : :obj:`None`

        Returns
        -------
        SparseMatrix 
            the adjacency matrix of elements :math:`[|\\mathcal C|,|\\mathcal C|]`, where :math:`|\\mathcal C|` is the number of elements
        """
        elements = self.elements(element_type)
        if isinstance(elements, torch.Tensor):
            elements = {self.default_element_type:elements}
        return element_adjacency(elements) # type:ignore

    def partition(self, n_parts:int, method:str="spectral", element_type:Optional[str]=None)->torch.Tensor:
        """Partition the mesh into n_parts
        
        Parameters
        ----------
        n_parts : int
            Number of partitions
        method : str, optional
            Partition method: 'spectral' or 'metis'. Default is 'spectral'.
        element_type : str or Iterable[str] or None
            the type of the elements to partition based on connectivity.
            
        Returns
        -------
        torch.Tensor
            IntTensor of shape [n_elements] containing partition ID
        """
        adj = self.element_adjacency(element_type)
        return graph_partition(adj, n_parts, method=method)

    def color(self, element_type:Optional[str]=None)->torch.Tensor:
        """Color the mesh elements such that no adjacent elements share the same color.
        
        Parameters
        ----------
        element_type : str or Iterable[str] or None
            the type of the elements.
            
        Returns
        -------
        torch.Tensor
            IntTensor of shape [n_elements] containing color ID
        """
        adj = self.element_adjacency(element_type)
        return graph_coloring(adj)

    def elements(self, element_type:Optional[Union[int, str, Iterable[str]]]=None
                 )->Union[torch.Tensor,Dict[str,torch.Tensor]]:
        """Get the element connectivity for specified element types.

        Examples
        --------
        1. Get elements of default type:

        .. code-block:: python

            import tensormesh
            mesh = tensormesh.Mesh.gen_rectangle()
            elements = mesh.elements() # Returns tensor of shape [n_elements, n_basis]

        2. Get elements of specific type:

        .. code-block:: python

            elements = mesh.elements("tri6") # Returns tensor for triangle elements
            
        3. Get elements of multiple types:

        .. code-block:: python

            elements = mesh.elements(["tri6", "quad9"]) # Returns dict of tensors
            
        4. Get all element types:

        .. code-block:: python

            elements = mesh.elements("all") # Returns dict of all element tensors

        5. Get elements of specific dimension:

        .. code-block:: python

            # Get all 2D elements (triangles, quads)
            elements = mesh.elements(2) # Returns dict of 2D element tensors
            
            # Get all elements matching mesh dimension
            elements = mesh.elements(-1) # Same as mesh.elements(mesh.dim)
            
        Parameters
        ----------
        element_type: str or Iterable[str] or int or None 
            the type of the elements:

            - if :obj:`all`, return dict of all elements
            - if :obj:`int`, return dict of elements of that dimension
            - if :obj:`str`, return elements of that type
            - if :obj:`Iterable[str]`, return elements of those types
            - if :obj:`None`, use :obj:`default_eletyp` (default)

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor] 

            - if :obj:`element_type` is :obj:`str`, return the corresponding elements connections of shape :math:`[|\\mathcal C|, B]`, where :math:`|\\mathcal C|` is the number of elements and :math:`B` is the number of basis functions
            - if :obj:`element_type` is :obj:`int`, return dict of elements of that dimension
            - if :obj:`element_typs` is :obj:`Iterable[str]`, return the mapping of corresponding elements connections of shape :math:`[|\\mathcal C|, B]`, where :math:`|\\mathcal C|` is the number of elements and :math:`B` is the number of basis functions 
            - if :obj:`element_type` is :obj:`None`, the :obj:`element_type` will be the :obj:`default_element_type` and do as above
            - if :obj:`element_type` is :obj:`"all"`, return all elements as a dictionary

        """

        if element_type == "all":
            return self.cells
        elif element_type is None:
            if isinstance(self.default_eletyp, str):
                return self.cells[self.default_eletyp]
            elif isinstance(self.default_eletyp, list):
                return {k:self.cells[k] for k in self.default_element_type}
            else:
                raise Exception(f"default_eletyp must be str or list, but got {type(self.default_eletyp)}")
        elif isinstance(element_type, int):
            if element_type == -1:
                element_type = self.dim
            return {k:self.cells[k] for k in self.dim2eletyp[element_type]}
        elif isinstance(element_type, str):
            return self.cells[element_type]
        elif isinstance(element_type, Iterable):
            return {k:self.cells[k] for k in element_type}
        else:
            raise Exception(f"element_type must be str or Iterable[str], but got {element_type}")
    
    def clone(self)->'Mesh':
        """The gradient will vanish if you use :obj:`torch.Tensor.clone` to clone the mesh, so we provide this method to clone the mesh
        Returns
        -------
        tensormesh.mesh.Mesh
            the cloned mesh
        """
        return Mesh(self.to_meshio())

    def plot(self, values:Optional[Dict[str,torch.Tensor] | Dict[str,Iterable[torch.Tensor]]]= None, 
                   save_path:Optional[str] = None, 
                   dt:Optional[float] = None, 
                   show_mesh:bool = False, 
                   fix_clim:bool =False,
                   show:bool = False,
                   **kwargs):
        """
        Parameters
        ----------
        values: None or Dict[str, torch.Tensor] or Dict[str, List[torch.Tensor]]
            the values to plot, if None, only plot the mesh
            if :obj:`Dict[str, torch.Tensor]`, a static subplots will be plotted, the key is the name of the subplot, the value is of shape :math:`[|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of points
            if :obj:`Dict[str, List[torch.Tensor]]`, a mp4/gif will be plotted, the key is the name of the subplot, each item in the list is of shape :math:`[|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of points
            default: None
        save_path: str or None
            the path to save the plot, if None, it will not be saved
            if the :obj:`values` is passed in as :obj:`Dict[str, List[torch.Tensor]]`, the :obj:`save_path` must endswith '.mp4' or '.gif'
            default: None
        dt: float or None
            the time interval between each frame, only used when :obj:`values` is passed in as :obj:`Dict[str, List[torch.Tensor]]`
            default: None
        show_mesh: bool
            whether to show the mesh, when :obj:`values` is passed in as :obj:`Dict[str, List[torch.Tensor]]` or :obj:`Dict[str, torch.Tensor]`
            default: False
                    
        """
        points:torch.Tensor = self.points # type:ignore
        elements = self.elements()
        if isinstance(elements,torch.Tensor):
            elements = {self.default_element_type:elements}
        assert isinstance(elements, dict)

        if values is None:
            ax = _get_visualization().draw_mesh(self, **kwargs)
            save_path = "tmp.jpg" if save_path is None else save_path
            if "ax" not in kwargs:
                plt.savefig(save_path)

            if show:
                plt.show()
            return ax

        elif isinstance(values, (tuple,list,torch.Tensor,np.ndarray)):
            if isinstance(values,(tuple,list)) or (isinstance(values, (torch.Tensor,np.ndarray))  and len(values.shape) == 2):
                save_path = "tmp.mp4"  if save_path is None else save_path 
                _get_visualization().draw_mesh_2d_stream(points, elements, values, dt,  # type:ignore
                                    fix_colorbar=fix_clim,
                                    show_mesh   =show_mesh,
                                    filename =  save_path,
                                    **kwargs)
            elif len(values.shape) == 1:
                save_path = "tmp.jpg" if save_path is None else save_path
                _get_visualization().draw_mesh_2d_static(points, elements, values, # type:ignore
                                    show_mesh = show_mesh,
                                    filename=save_path,
                                    **kwargs)
            else:
                raise NotImplementedError(f"{type(values)} is not implemented for plot")
        
        elif isinstance(values,dict):
            v = next(iter(values.values()))
            if isinstance(v,(tuple,list)) or (isinstance(v, (torch.Tensor,np.ndarray))  and len(v.shape) == 2):
                save_path = "tmp.mp4"  if save_path is None else save_path 
                _get_visualization().draw_mesh_2d_stream(points, elements, values, dt,  # type:ignore
                                    fix_colorbar=fix_clim,
                                    show_mesh   =show_mesh,
                                    filename =  save_path,
                                    **kwargs)
            elif isinstance(v, (torch.Tensor,np.ndarray)) and len(v.shape) == 1:
                save_path = "tmp.jpg" if save_path is None else save_path
                _get_visualization().draw_mesh(points, elements, values, # type:ignore
                                    show_mesh = show_mesh,
                                    filename=save_path,
                                    **kwargs)
            else:
                raise NotImplementedError(f"{type(values)} is not implemented for plot")
        else:   
            raise NotImplementedError(f"{type(values)} is not implemented for plot")
        
        # from ..visualization import plot_value_matplotlib, plot_pyvista

        # plot_fns = {
        #     "pyvista":plot_pyvista,
        #     "matplotlib":plot_matplotlib,
        # }
        # assert  backend in plot_fns.keys(), f"backend must be one of {list(plot_fns.keys())}, but got {backend}"

        # return plot_fns[backend](kwargs, self,  save_path, dt, show_mesh)
             
        # from ..visualization import plot_value_matplotlib, plot_mesh_matplotlib

        # if values is None:
        #     return plot_mesh_matplotlib(self, save_path)
        # else:
        #     return plot_value_matplotlib(values, self, save_path, dt, show_mesh, fix_clim)

    @property
    def n_points(self)->int:
        """
        Returns
        -------
        int
            the number of nodes/vertices/points :math:`|\\mathcal V|`
        """
        return self.points.shape[0]

    @property 
    def n_elements(self)->int:
        """
        Returns
        -------
        int
            the number of elements :math:`|\\mathcal C|`
        """
        if isinstance(self.default_element_type, str):
            return self.cells[self.default_element_type].shape[0]
        else:
            return sum([self.cells[k].shape[0] for k in self.default_element_type])

    @property
    def boundary_mask(self)->torch.Tensor:
        r"""
        Returns
        -------
        torch.Tensor 
            1D tensor of shape :math:`[|\mathcal V|]`, where  :math:`|\mathcal V|` is the number of points
            the mask of the boundary points, :obj:`"is_boundary"` key or :obj:`"boundary_mask"` key is required in :attr:`point_data`
        """
        if "is_boundary" in self.point_data.keys():
            return self.point_data["is_boundary"]
        elif "boundary_mask" in self.point_data.keys():
            return self.point_data["boundary_mask"]
        else:
            raise Exception("'boundary_mask' or 'is_boundary' is not found in point_data")

    @property
    def default_element_type(self)->str:
        """
        Returns
        -------
        str or List[str]
            the default element type, if the mesh is composed of mixed elements, it will return List[str],  
            otherwise it will return str

        :noindex:
        """
        return self.default_eletyp

    @property
    def dtype(self)->torch.dtype:
        """
        Returns
        -------
        torch.dtype
            the data type of the points, e.g., torch.float32, torch.float64
        """
        return self.points.dtype
    
    @property 
    def device(self)->torch.device:
        """
        Returns
        -------
        torch.device
            the device of the points, e.g., torch.device("cpu"), torch.device("cuda:0")
        """
        return self.points.device

    @classmethod
    def from_meshio(cls,
                    mesh:meshio.Mesh, 
                    reorder:bool=False):
        """
        Parameters
        ----------
        mesh: meshio.Mesh
            a meshio mesh object
        reorder: bool
            whether to turn [0,1,2,3] -> [0,1,3,2]
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
        return cls(mesh, reorder)
    
    @classmethod
    def read(cls, file_name:str, 
             file_format:Optional[str] = None, 
             reorder:bool = False):
        """
        Parameters
        ----------
            file_name: str
                the name of the file
            file_format: str
                the format of the file, e.g., 'msh', 'vtk', 'obj'
                default is the file extension
            reorder: bool
                whether to turn [0,1,2,3] -> [0,1,3,2]
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
        return cls(meshio.read(file_name, file_format), reorder)
    
    from_file = read

    @staticmethod
    def gen_rectangle(
             chara_length:float=0.1,
             order:int         = 1,
             element_type:str  ="tri",
             left:float        = 0.0, right:float  =  1.0, 
             bottom:float      = 0.0, top:float    =  1.0,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        element_type: str, optional
            the type of the element,
            default: :obj:`"tri"`
        left: float, optional
            the left boundary of the rectangle,
            default: :obj:`0.0`
        right: float, optional
            the right boundary of the rectangle,
            default: :obj:`1.0`
        bottom: float, optional
            the bottom boundary of the rectangle,
            default: :obj:`0.0`
        top: float, optional
            the top boundary of the rectangle,
            default: :obj:`1.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_rectangle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1` 
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        element_type: str, optional
            the type of the element,
            default: :obj:`"quad"`
        outer_left: float, optional
            the left boundary of the outer rectangle,
            default: :obj:`0.0`
        outer_right: float, optional
            the right boundary of the outer rectangle,
            default: :obj:`1.0`
        outer_bottom: float, optional
            the bottom boundary of the outer rectangle,
            default: :obj:`0.0`
        outer_top: float, optional
            the top boundary of the outer rectangle,
            default: :obj:`1.0`
        inner_left: float, optional
            the left boundary of the inner rectangle,
            default: :obj:`0.25`
        inner_right: float, optional
            the right boundary of the inner rectangle,
            default: :obj:`0.75`
        inner_bottom: float, optional
            the bottom boundary of the inner rectangle,
            default: :obj:`0.25`
        inner_top: float, optional
            the top boundary of the inner rectangle,
            default: :obj:`0.75`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_hollow_rectangle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """

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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        element_type: str, optional
            the type of the element,
            default: :obj:`"tri"`
        cx: float, optional
            the x coordinate of the center of the circle,
            default: :obj:`0.0`
        cy: float, optional
            the y coordinate of the center of the circle,
            default: :obj:`0.0`
        r: float, optional
            the radius of the circle,
            default: :obj:`1.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_circle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
        from ..dataset import gen_circle
        return gen_circle(chara_length, order, element_type, cx, cy, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        element_type: str, optional
            the type of the element,
            default: :obj:`"quad"`
        cx: float, optional
            the x coordinate of the center of the circle,
            default: :obj:`0.0`
        cy: float, optional
            the y coordinate of the center of the circle,
            default: :obj:`0.0`
        r_inner: float, optional
            the inner radius of the circle,
            default: :obj:`1.0`
        r_outer: float, optional
            the outer radius of the circle,
            default: :obj:`2.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_hollow_circle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        element_type: str, optional
            the type of the element,
            default: :obj:`"quad"`
        left: float, optional
            the left boundary of the rectangle,
            default: :obj:`0.0`
        right: float, optional
            the right boundary of the rectangle,
            default: :obj:`1.0`
        bottom: float, optional
            the bottom boundary of the rectangle,
            default: :obj:`0.0`
        top: float, optional
            the top boundary of the rectangle,
            default: :obj:`1.0`
        top_inner: float, optional
            the top inner boundary of the rectangle,
            default: :obj:`0.5`
        right_inner: float, optional
            the right inner boundary of the rectangle,
            default: :obj:`0.5`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_L`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object 
        """
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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        left: float, optional
            the left boundary of the cube,
            default: :obj:`0.0`
        right: float, optional
            the right boundary of the cube,
            default: :obj:`1.0`
        bottom: float, optional
            the bottom boundary of the cube,
            default: :obj:`0.0`
        top: float, optional
            the top boundary of the cube,
            default: :obj:`1.0`
        front: float, optional
            the front boundary of the cube,
            default: :obj:`0.0`
        back: float, optional
            the back boundary of the cube,
            default: :obj:`1.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_cube`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        outer_left: float, optional
            the left boundary of the outer cube,
            default: :obj:`0.0`
        outer_right: float, optional
            the right boundary of the outer cube,
            default: :obj:`1.0`
        outer_bottom: float, optional
            the bottom boundary of the outer cube,
            default: :obj:`0.0`
        outer_top: float, optional
            the top boundary of the outer cube,
            default: :obj:`1.0`
        outer_front: float, optional
            the front boundary of the outer cube,
            default: :obj:`0.0`
        outer_back: float, optional
            the back boundary of the outer cube,
            default: :obj:`1.0`
        inner_left: float, optional
            the left boundary of the inner cube,
            default: :obj:`0.25`
        inner_right: float, optional
            the right boundary of the inner cube,
            default: :obj:`0.75`
        inner_bottom: float, optional
            the bottom boundary of the inner cube,
            default: :obj:`0.25`
        inner_top: float, optional
            the top boundary of the inner cube,
            default: :obj:`0.75`
        inner_front: float, optional
            the front boundary of the inner cube,
            default: :obj:`0.25`
        inner_back: float, optional
            the back boundary of the inner cube,
            default: :obj:`0.75`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_hollow_cube`,
            default: :obj:`None`

        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object 
        """
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
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        cx: float, optional
            the x coordinate of the center of the sphere,
            default: :obj:`0.0`
        cy: float, optional
            the y coordinate of the center of the sphere,
            default: :obj:`0.0`
        cz: float, optional
            the z coordinate of the center of the sphere,
            default: :obj:`0.0`
        r: float, optional
            the radius of the sphere,
            default: :obj:`1.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_sphere`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
        from ..dataset import gen_sphere
        return gen_sphere(chara_length, order, cx, cy, cz, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_sphere(chara_length=0.1,
             order=1,
              cx = 0.0, cy = 0.0, cz=0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
        """
        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: :obj:`0.1`
        order: int, optional
            the order of the basis function,
            default: :obj:`1`
        cx: float, optional
            the x coordinate of the center of the sphere,
            default: :obj:`0.0`
        cy: float, optional
            the y coordinate of the center of the sphere,
            default: :obj:`0.0`
        cz: float, optional
            the z coordinate of the center of the sphere,
            default: :obj:`0.0`
        r_inner: float, optional
            the inner radius of the sphere,
            default: :obj:`1.0`
        r_outer: float, optional
            the outer radius of the sphere,
            default: :obj:`2.0`
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :meth:`tensormesh.dataset.mesh.gen_hollow_sphere`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.mesh.Mesh
            the mesh object
        """
        from ..dataset import gen_hollow_sphere
        return gen_hollow_sphere(chara_length, order, cx, cy, cz, r_inner, r_outer, visualize, cache_path)

Mesh.__autodoc__ = [i for i in dir(Mesh) if not i.startswith("_")]

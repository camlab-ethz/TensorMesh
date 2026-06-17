from typing import Optional, Union, Iterable, Dict, List
import numpy as np
import torch
import torch.nn as nn
import meshio
from collections import defaultdict
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


class Mesh(nn.Module):
    r"""FEM mesh — interpolation-node coordinates, per-element-type
    connectivity, and point/cell/field data attached to either.
    Mixed-element meshes are supported via :attr:`cells` being a
    :class:`~tensormesh.nn.BufferDict` keyed by element type string
    (e.g. ``"triangle"``, ``"quad"``, ``"tetra"``).

    A "point" throughout the API means an interpolation node / degree of
    freedom — for ``order=1`` this is the corner vertex of an element,
    for ``order>=2`` it also includes mid-edge, mid-face, and interior
    nodes.

    Parameters
    ----------
    mesh : meshio.Mesh
        A meshio mesh object to wrap.
    reorder : bool, default=False
        Whether to convert connectivity from Gmsh/VTK ordering to TensorMesh
        internal ordering (delegates to
        :meth:`tensormesh.Element.reorder`).

    Attributes
    ----------
    points: torch.Tensor
        2D tensor of shape :math:`[|\mathcal V|, D]`, where :math:`|\mathcal V|`
        is the number of interpolation nodes and :math:`D` is the spatial
        dimension. Includes high-order nodes (mid-edge / mid-face /
        interior) when the mesh uses ``order >= 2`` elements.
    cells: BufferDict[str, torch.Tensor]
        Each key is an ``element_type`` string (see
        :mod:`tensormesh.element`); the value is a 2D tensor of shape
        :math:`[|\mathcal C|, B]`, where :math:`|\mathcal C|` is the number
        of elements and :math:`B` is the number of basis functions.
    point_data: BufferDict[str, torch.Tensor], optional
        Per-point fields, keyed by name.
    cell_data: ModuleDict[str, BufferDict[str, torch.Tensor]], optional
        Per-element fields. The outer key is an ``element_type``; the
        inner key is the field name.
    field_data: BufferDict[str, torch.Tensor], optional
        Global named fields.
    cell_sets: dict, optional
        Named subsets of cells, kept in meshio's native format.
    dim2eletyp: Dict[int, List[str]]
        Each key is a spatial dimension, and the value is a list of element
        types of that dimension present in the mesh.
    default_eletyp: str or List[str]
        The default element type — a single string for homogeneous meshes,
        a list of strings for mixed-element meshes. Exposed publicly via the
        :attr:`default_element_type` property.

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
                # Centralized reorder implementation lives in tensormesh.Element.reorder
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
        """Register a per-point field on :attr:`point_data`.

        :attr:`point_data` is a :class:`tensormesh.nn.BufferDict`, so prefer
        this method over ``__setitem__`` to make sure the tensor is tracked
        as a buffer of the underlying :class:`torch.nn.Module`.

        Parameters
        ----------
        key: str
            the key of the value
        value: torch.Tensor
            tensor of shape :math:`[|\\mathcal V|, ...]`, where :math:`|\\mathcal V|` is the number of interpolation nodes (``mesh.n_points``)

        Returns
        -------
        tensormesh.Mesh
            self will be returned
        """
        assert key not in self.point_data.keys(), f"the key {key} already exists in point_data"
        assert value.shape[0] == self.points.shape[0], f"the first dimension of value should be {self.points.shape[0]}, but got {value.shape[0]}"
        self.point_data.register_buffer(key, value)

        return self
      
    def register_element_data(self, key:str, value:Union[Dict[str,torch.Tensor],torch.Tensor]):
        """Register a per-element field on :attr:`cell_data`.

        For homogeneous meshes ``value`` may be a single tensor; for
        mixed-element meshes pass a dict keyed by element type with one
        tensor per type.
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
        """Export this mesh as an in-memory ``meshio.Mesh``.

        Parameters
        ----------
        reorder : bool, default=False
            If :obj:`True`, convert connectivity from the internal ordering
            back to Gmsh/VTK ordering before returning (delegates to
            :meth:`tensormesh.Element.reorder`).

        Returns
        -------
        meshio.Mesh
            The meshio mesh object.
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
        """Spatial dimension of the mesh (``2`` for surface meshes, ``3`` for volume meshes)."""
        return self.points.shape[1]

    def save(self, file_name:str, file_format:Optional[str]=None):
        """Write this mesh to disk via ``meshio.write``.

        Boolean point/cell/field arrays are cast to ``float`` before writing
        (meshio does not support ``bool``). For ``.vtk`` / ``.vtu`` outputs
        2-D meshes are padded to 3-D and connectivity is reordered to the
        Gmsh/VTK convention.

        Parameters
        ----------
        file_name: str
            the name of the file
        file_format: str
            the format of the file, e.g., 'msh', 'vtk', 'obj'
            default is the file extension

        Returns
        -------
        tensormesh.Mesh
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
            if :obj:`None` is the ``default_element_type``
            default : :obj:`None`

        Returns
        -------
        SparseMatrix 
            the adjacency matrix of points :math:`[|\\mathcal V|,|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of interpolation nodes
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
            if :obj:`None` is the ``default_element_type``
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
            - if ``Iterable[str]``, return elements of those types
            - if :obj:`None`, use :obj:`default_eletyp` (default)

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor] 

            - if ``element_type`` is :obj:`str`, return the corresponding elements connections of shape :math:`[|\\mathcal C|, B]`, where :math:`|\\mathcal C|` is the number of elements and :math:`B` is the number of basis functions
            - if ``element_type`` is :obj:`int`, return dict of elements of that dimension
            - if ``element_type`` is ``Iterable[str]``, return the mapping of corresponding elements connections of shape :math:`[|\\mathcal C|, B]`, where :math:`|\\mathcal C|` is the number of elements and :math:`B` is the number of basis functions
            - if ``element_type`` is :obj:`None`, the ``element_type`` will be the ``default_element_type`` and do as above
            - if ``element_type`` is ``"all"``, return all elements as a dictionary

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
        """Return a deep copy of the mesh that preserves the autograd graph.

        Calling :obj:`torch.Tensor.clone` on the underlying buffers detaches
        them from the computation graph, so gradients flowing through
        ``points`` / ``cell_data`` would vanish. This method round-trips
        through ``meshio`` to reconstruct the mesh while keeping the
        connectivity and metadata intact.

        Returns
        -------
        tensormesh.Mesh
            The cloned mesh.
        """
        return Mesh(self.to_meshio())

    def plot(self, values:Optional[Dict[str,torch.Tensor] | Dict[str,Iterable[torch.Tensor]]]= None, 
                   save_path:Optional[str] = None, 
                   dt:Optional[float] = None, 
                   show_mesh:bool = False, 
                   fix_clim:bool =False,
                   show:bool = False,
                   **kwargs):
        """Plot the mesh, optionally overlaying scalar fields or animations.

        With no ``values`` only the mesh wireframe is drawn. Passing
        ``Dict[str, torch.Tensor]`` produces a static multi-panel figure;
        passing ``Dict[str, List[torch.Tensor]]`` produces an mp4/gif
        animation (one frame per list element).

        Parameters
        ----------
        values: None or Dict[str, torch.Tensor] or Dict[str, List[torch.Tensor]]
            the values to plot, if None, only plot the mesh
            if ``Dict[str, torch.Tensor]``, a static subplots will be plotted, the key is the name of the subplot, the value is of shape :math:`[|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of interpolation nodes
            if ``Dict[str, List[torch.Tensor]]``, a mp4/gif will be plotted, the key is the name of the subplot, each item in the list is of shape :math:`[|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of interpolation nodes
            default: None
        save_path: str or None
            the path to save the plot, if None, it will not be saved
            if the ``values`` is passed in as ``Dict[str, List[torch.Tensor]]``, the ``save_path`` must endswith '.mp4' or '.gif'
            default: None
        dt: float or None
            the time interval between each frame, only used when ``values`` is passed in as ``Dict[str, List[torch.Tensor]]``
            default: None
        show_mesh: bool
            whether to overlay the mesh wireframe (and, at ``order >= 2``,
            the interpolation nodes) on top of the colour-filled field.
            Only takes effect when ``values`` is given.
            default: False
        fix_clim: bool
            whether to fix the color limits across all frames, only used when ``values`` is passed in as ``Dict[str, List[torch.Tensor]]``.
            If True, the color limits are determined by the global min and max across all frames, ensuring a consistent colorbar throughout the animation.
            default: False
        show: bool
            whether to display the plot interactively (e.g., via :func:`matplotlib.pyplot.show`)
            default: False
        **kwargs
            additional keyword arguments passed to the underlying visualization functions
        """
        points:torch.Tensor = self.points # type:ignore
        elements = self.elements()
        if isinstance(elements,torch.Tensor):
            elements = {self.default_element_type:elements}
        assert isinstance(elements, dict)

        if values is None:
            import matplotlib.pyplot as plt
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
                _get_visualization().draw_mesh_2d_static(points, elements, values, # type:ignore
                                    show_mesh = show_mesh,
                                    filename=save_path,
                                    **kwargs)
            else:
                raise NotImplementedError(f"{type(values)} is not implemented for plot")
        else:
            raise NotImplementedError(f"{type(values)} is not implemented for plot")

    @property
    def n_points(self)->int:
        """Number of interpolation nodes :math:`|\\mathcal V|` in the mesh.

        Equals ``mesh.points.shape[0]``. For ``order >= 2`` this counts
        high-order nodes (mid-edge, mid-face, interior) as well, not
        only corner vertices.

        Returns
        -------
        int
            the number of interpolation nodes :math:`|\\mathcal V|`
        """
        return self.points.shape[0]

    @property
    def n_elements(self)->int:
        """Number of elements :math:`|\\mathcal C|` of the :attr:`default_element_type`.

        For mixed-element meshes this sums element counts across every
        type in ``default_element_type``.

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
        r"""Boolean mask flagging boundary points.

        Looked up from :attr:`point_data` under the key ``"is_boundary"``
        (preferred) or ``"boundary_mask"``. Mesh generators in
        :mod:`tensormesh.dataset` populate this automatically.

        Returns
        -------
        torch.Tensor
            1D bool tensor of shape :math:`[|\mathcal V|]`, where  :math:`|\mathcal V|` is the number of interpolation nodes;
            requires ``"is_boundary"`` or ``"boundary_mask"`` to live in :attr:`point_data`
        """
        if "is_boundary" in self.point_data.keys():
            return self.point_data["is_boundary"]
        elif "boundary_mask" in self.point_data.keys():
            return self.point_data["boundary_mask"]
        else:
            raise Exception("'boundary_mask' or 'is_boundary' is not found in point_data")

    @property
    def default_element_type(self)->str:
        """Element type(s) of the highest-dimensional cells in the mesh.

        Methods like :meth:`elements`, :meth:`n_elements`, and the FEM
        assemblers fall back to this when no ``element_type`` is given.

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
        """Floating-point dtype of :attr:`points` (and, by convention, of every buffer in the mesh).

        Returns
        -------
        torch.dtype
            the data type of the points, e.g., torch.float32, torch.float64
        """
        return self.points.dtype

    @property
    def device(self)->torch.device:
        """Device on which the mesh tensors live.

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
        """Build a :class:`Mesh` from an in-memory ``meshio.Mesh``.

        Parameters
        ----------
        mesh: meshio.Mesh
            a meshio mesh object
        reorder: bool
            whether to convert connectivity from Gmsh/VTK ordering to
            TensorMesh internal ordering (delegates to
            :meth:`tensormesh.Element.reorder`).

        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        return cls(mesh, reorder)
    
    @classmethod
    def read(cls, file_name:str,
             file_format:Optional[str] = None,
             reorder:bool = False):
        """Read a mesh from disk via ``meshio.read``.

        Parameters
        ----------
        file_name: str
            the name of the file
        file_format: str
            the format of the file, e.g., 'msh', 'vtk', 'obj'
            default is the file extension
        reorder: bool
            whether to convert connectivity from Gmsh/VTK ordering to
            TensorMesh internal ordering (delegates to
            :meth:`tensormesh.Element.reorder`).

        Returns
        -------
        tensormesh.Mesh
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
        """Generate a 2-D mesh of an axis-aligned rectangle.

        Delegates to :func:`~tensormesh.dataset.gen_rectangle`, which calls
        Gmsh under the hood and caches the result if ``cache_path`` is
        given.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        element_type: str, optional
            the type of the element,
            default: ``"tri"``
        left: float, optional
            the left boundary of the rectangle,
            default: ``0.0``
        right: float, optional
            the right boundary of the rectangle,
            default: ``1.0``
        bottom: float, optional
            the bottom boundary of the rectangle,
            default: ``0.0``
        top: float, optional
            the top boundary of the rectangle,
            default: ``1.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_rectangle`,
            default: :obj:`None`

        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        from ..dataset import gen_rectangle
        return gen_rectangle(chara_length, order, element_type, left, right, bottom, top, visualize, cache_path)

    @staticmethod
    def gen_hollow_rectangle(
        chara_length:float=0.1,
        order:int=1,
        element_type:str="quad",
        outer_left:float=0.0, outer_right:float=1.0,
        outer_bottom:float=0.0, outer_top:float=1.0,
        inner_left:float=0.25,  inner_right:float=0.75,
        inner_bottom:float=0.25, inner_top:float=0.75,
        visualize:bool=False,
        cache_path:Optional[str]=None,
    )->'Mesh':
        """Generate a 2-D mesh of a rectangle with a rectangular hole cut out.

        Delegates to :func:`~tensormesh.dataset.gen_hollow_rectangle`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        element_type: str, optional
            the type of the element,
            default: ``"quad"``
        outer_left: float, optional
            the left boundary of the outer rectangle,
            default: ``0.0``
        outer_right: float, optional
            the right boundary of the outer rectangle,
            default: ``1.0``
        outer_bottom: float, optional
            the bottom boundary of the outer rectangle,
            default: ``0.0``
        outer_top: float, optional
            the top boundary of the outer rectangle,
            default: ``1.0``
        inner_left: float, optional
            the left boundary of the inner rectangle,
            default: ``0.25``
        inner_right: float, optional
            the right boundary of the inner rectangle,
            default: ``0.75``
        inner_bottom: float, optional
            the bottom boundary of the inner rectangle,
            default: ``0.25``
        inner_top: float, optional
            the top boundary of the inner rectangle,
            default: ``0.75``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_hollow_rectangle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
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
    def gen_circle(chara_length:float=0.1,
            order:int=1,
            element_type:str="tri",
            cx:float=0.0, cy:float=0.0, r:float=1.0,
            visualize:bool=False,
            cache_path:Optional[str]=None)->'Mesh':
        """Generate a 2-D mesh of a disk (filled circle).

        Delegates to :func:`~tensormesh.dataset.gen_circle`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        element_type: str, optional
            the type of the element,
            default: ``"tri"``
        cx: float, optional
            the x coordinate of the center of the circle,
            default: ``0.0``
        cy: float, optional
            the y coordinate of the center of the circle,
            default: ``0.0``
        r: float, optional
            the radius of the circle,
            default: ``1.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_circle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        from ..dataset import gen_circle
        return gen_circle(chara_length, order, element_type, cx, cy, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_circle(chara_length:float=0.1,
             order:int=1,
             element_type:str="quad",
             cx:float=0.0, cy:float=0.0,
             r_inner:float=1.0, r_outer:float=2.0,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """Generate a 2-D mesh of an annulus (disk with a circular hole).

        Delegates to :func:`~tensormesh.dataset.gen_hollow_circle`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        element_type: str, optional
            the type of the element,
            default: ``"quad"``
        cx: float, optional
            the x coordinate of the center of the circle,
            default: ``0.0``
        cy: float, optional
            the y coordinate of the center of the circle,
            default: ``0.0``
        r_inner: float, optional
            the inner radius of the circle,
            default: ``1.0``
        r_outer: float, optional
            the outer radius of the circle,
            default: ``2.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_hollow_circle`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
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
    def gen_L(chara_length:float=0.1,
             order:int=1,
             element_type:str="quad",
             left:float=0.0, right:float=1.0,
             bottom:float=0.0, top:float=1.0,
             top_inner:float=0.5,
             right_inner:float=0.5,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """Generate a 2-D mesh of an L-shaped domain (re-entrant corner benchmark).

        Delegates to :func:`~tensormesh.dataset.gen_L`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        element_type: str, optional
            the type of the element,
            default: ``"quad"``
        left: float, optional
            the left boundary of the rectangle,
            default: ``0.0``
        right: float, optional
            the right boundary of the rectangle,
            default: ``1.0``
        bottom: float, optional
            the bottom boundary of the rectangle,
            default: ``0.0``
        top: float, optional
            the top boundary of the rectangle,
            default: ``1.0``
        top_inner: float, optional
            the top inner boundary of the rectangle,
            default: ``0.5``
        right_inner: float, optional
            the right inner boundary of the rectangle,
            default: ``0.5``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_L`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
            the mesh object 
        """
        from ..dataset import gen_L
        return gen_L(chara_length, order, element_type, left, right, bottom, top, top_inner, right_inner, visualize, cache_path)

    @staticmethod
    def gen_cube(chara_length:float=0.1,
             order:int=1,
             left:float=0.0, right:float=1.0,
             bottom:float=0.0, top:float=1.0,
             front:float=0.0, back:float=1.0,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """Generate a 3-D mesh of an axis-aligned cuboid.

        Delegates to :func:`~tensormesh.dataset.gen_cube`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        left: float, optional
            the left boundary of the cube,
            default: ``0.0``
        right: float, optional
            the right boundary of the cube,
            default: ``1.0``
        bottom: float, optional
            the bottom boundary of the cube,
            default: ``0.0``
        top: float, optional
            the top boundary of the cube,
            default: ``1.0``
        front: float, optional
            the front boundary of the cube,
            default: ``0.0``
        back: float, optional
            the back boundary of the cube,
            default: ``1.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_cube`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        from ..dataset import gen_cube
        return gen_cube(chara_length, order, left, right, bottom, top, front, back, visualize, cache_path)
    
    @staticmethod
    def gen_hollow_cube(chara_length:float=0.1,
             order:int=1,
             outer_left:float=0.0, outer_right:float=1.0,
             outer_bottom:float=0.0, outer_top:float=1.0,
             outer_front:float=0.0, outer_back:float=1.0,
             inner_left:float=0.25, inner_right:float=0.75,
             inner_bottom:float=0.25, inner_top:float=0.75,
             inner_front:float=0.25, inner_back:float=0.75,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """Generate a 3-D mesh of a cuboid with a cuboidal hole.

        Delegates to :func:`~tensormesh.dataset.gen_hollow_cube`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        outer_left: float, optional
            the left boundary of the outer cube,
            default: ``0.0``
        outer_right: float, optional
            the right boundary of the outer cube,
            default: ``1.0``
        outer_bottom: float, optional
            the bottom boundary of the outer cube,
            default: ``0.0``
        outer_top: float, optional
            the top boundary of the outer cube,
            default: ``1.0``
        outer_front: float, optional
            the front boundary of the outer cube,
            default: ``0.0``
        outer_back: float, optional
            the back boundary of the outer cube,
            default: ``1.0``
        inner_left: float, optional
            the left boundary of the inner cube,
            default: ``0.25``
        inner_right: float, optional
            the right boundary of the inner cube,
            default: ``0.75``
        inner_bottom: float, optional
            the bottom boundary of the inner cube,
            default: ``0.25``
        inner_top: float, optional
            the top boundary of the inner cube,
            default: ``0.75``
        inner_front: float, optional
            the front boundary of the inner cube,
            default: ``0.25``
        inner_back: float, optional
            the back boundary of the inner cube,
            default: ``0.75``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_hollow_cube`,
            default: :obj:`None`

        Returns
        -------
        tensormesh.Mesh
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
    def gen_sphere(chara_length:float=0.1,
                order:int=1,
                cx:float=0.0, cy:float=0.0, cz:float=0.0, r:float=1.0,
                visualize:bool=False,
                cache_path:Optional[str]=None)->'Mesh':
        """Generate a 3-D mesh of a solid ball (filled sphere).

        Delegates to :func:`~tensormesh.dataset.gen_sphere`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        cx: float, optional
            the x coordinate of the center of the sphere,
            default: ``0.0``
        cy: float, optional
            the y coordinate of the center of the sphere,
            default: ``0.0``
        cz: float, optional
            the z coordinate of the center of the sphere,
            default: ``0.0``
        r: float, optional
            the radius of the sphere,
            default: ``1.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_sphere`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        from ..dataset import gen_sphere
        return gen_sphere(chara_length, order, cx, cy, cz, r, visualize, cache_path)

    @staticmethod
    def gen_hollow_sphere(chara_length:float=0.1,
             order:int=1,
             cx:float=0.0, cy:float=0.0, cz:float=0.0,
             r_inner:float=1.0, r_outer:float=2.0,
             visualize:bool=False,
             cache_path:Optional[str]=None)->'Mesh':
        """Generate a 3-D mesh of a spherical shell (ball with a concentric spherical cavity).

        Delegates to :func:`~tensormesh.dataset.gen_hollow_sphere`.

        Parameters
        ----------
        chara_length: float, optional
            the characteristic length of the mesh,
            default: ``0.1``
        order: int, optional
            the order of the basis function,
            default: ``1``
        cx: float, optional
            the x coordinate of the center of the sphere,
            default: ``0.0``
        cy: float, optional
            the y coordinate of the center of the sphere,
            default: ``0.0``
        cz: float, optional
            the z coordinate of the center of the sphere,
            default: ``0.0``
        r_inner: float, optional
            the inner radius of the sphere,
            default: ``1.0``
        r_outer: float, optional
            the outer radius of the sphere,
            default: ``2.0``
        visualize: bool, optional
            whether to visualize the mesh,
            default: :obj:`False`
        cache_path: str, optional
            the path to save the mesh, if :obj:`None`, it will be decided by :func:`~tensormesh.dataset.gen_hollow_sphere`,
            default: :obj:`None`
        Returns
        -------
        tensormesh.Mesh
            the mesh object
        """
        from ..dataset import gen_hollow_sphere
        return gen_hollow_sphere(chara_length, order, cx, cy, cz, r_inner, r_outer, visualize, cache_path)

def _partition(self, num_parts=None, method='coordinate', devices=None):
    """Partition this mesh into submeshes for distributed assembly.

    Thin syntactic-sugar wrapper around
    :class:`~tensormesh.distributed.DistributedMesh` --
    ``mesh.partition(num_parts=4)`` is equivalent to
    ``DistributedMesh(mesh, num_partitions=4)``.

    **Distributed-by-default convention**

    The returned object is meant to be paired with
    :func:`~tensormesh.distributed.distributed_element_assemble_per_rank`
    inside a ``torch.distributed`` context. Each rank's assembly only
    runs on its own submesh; cross-rank halo contributions are routed via
    ``all_to_all_single``. Total cluster compute matches single-process,
    instead of every rank duplicating the full work.

    For a single-process driver (no ``init_process_group``) the same
    object works with :func:`~tensormesh.distributed.distributed_element_assemble`
    (multi-threaded one-process pseudo-distributed assembly) -- mostly
    useful for tests and small examples.

    Parameters
    ----------
    num_parts : int, optional
        Number of partitions. Default: ``torch.cuda.device_count()``,
        or 2 if no CUDA devices are available.
    method : str, optional
        Partitioning method. ``'coordinate'`` (RCB, default),
        ``'metis'`` (requires pymetis), ``'spectral'``.
    devices : list of torch.device, optional
        Device per partition. Defaults to ``cuda:0..cuda:N`` or all CPU.

    Returns
    -------
    DistributedMesh

    Examples
    --------
    Single-process driver::

        mesh = tm.Mesh.gen_rectangle(chara_length=0.1)
        dmesh = mesh.partition(num_parts=4)
        K_dist = distributed_element_assemble(LaplaceElementAssembler, dmesh)

    Multi-rank distributed (each rank does only its own submesh)::

        dist.init_process_group(backend='nccl', rank=rank, world_size=N)
        mesh = tm.Mesh.gen_rectangle(chara_length=0.1)
        dmesh = mesh.partition(num_parts=N)
        K_dist = distributed_element_assemble_per_rank(
            LaplaceElementAssembler, dmesh, rank=rank,
        )
    """
    from ..distributed import DistributedMesh
    return DistributedMesh(self, num_partitions=num_parts,
                            method=method, devices=devices)


Mesh.partition = _partition

Mesh.__autodoc__ = [i for i in dir(Mesh) if not i.startswith("_")]

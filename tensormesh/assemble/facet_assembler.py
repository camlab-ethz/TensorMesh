
from abc import abstractmethod
import torch 
import torch.nn as nn
import numpy as np
import inspect
from typing import Callable, Optional, Dict, Tuple, Iterable, List

from tensormesh.element.element_type import element_type2element

from .projector import ReduceProjector, SparseProjector
from ..element import   element_type2order, \
                        element_type2dimension,\
                        Transformation
from ..nn import BufferDict, BufferList
from ..mesh import Mesh
from ..vmap import vmap
class FacetAssembler(nn.Module):
    r"""The FacetAssembler is used to assemble the operator on the nodes of the mesh

    The output when calling the NodeAssembler is a vector, which is of shape :math:`[|\mathcal V|]` or :math:`[|\mathcal V|\times H]`, 
    where :math:`|\mathcal V|` is the number of nodes and :math:`H` is the number of degrees of freedom per node.

    
    Examples
    --------
    >>> import torch
    >>> from tensormesh import FacetAssembler
    >>> from tensormesh.functional.elasticity import voigt_shape_val
    >>> 
    >>> # Create a facet assembler for applying traction force
    >>> class TractionAssembler(FacetAssembler):
    ...     def forward(self, u):
    ...         # Define downward traction force vector
    ...         f = torch.tensor([0.0, -1.0]).type(u.dtype).to(u.device)
    ...         # Convert shape functions to Voigt notation matrix
    ...         N = voigt_shape_val(u, dim=2)  
    ...         # Return force vector for each basis function
    ...         return f @ N
    ...
    >>> # Usage:
    >>> assembler = TractionAssembler()
    >>> u = torch.tensor(0.5)  # Shape function value at a point
    >>> force = assembler(u)  # Returns force vector [0.0, -0.5]
    

    Attributes
    ----------
    quadrature_weights : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`tensormesh.shape.element_types`.
        Each :obj:`element_type` corresponds to a 1D tensor of shape :math:`[Q]`, where :math:`Q` is the number of quadrature points`
        quadrature_weights of each element type
    quadrature_points : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`tensormesh.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[Q, D]`, where :math:`Q` is the number of quadrature points and :math:`D` is the dimension of the mesh
        quadrature_points of each element type
    shape_val : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`tensormesh.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[Q, B]`, where :math:`Q` is the number of quadrature points and :math:`B` is the number of basis functions
        shape_val of each element type
    projector : BufferDict[str, Projector]
        The element type is the key, which should be one of :meth:`tensormesh.shape.element_types`.
        Each :obj:`element_type` corresponds to a projector from element to nodes,
        each  projector is a :meth:`tensormesh.assemble.Projector` object, could be considered as a sparse matrix
        
        .. math::

            \mathcal P_e: \mathbb{R}_{\text{sparse}}^{|\mathcal C_e| \times B_e} \rightarrow \mathbb{R}^{|\mathcal V|}

        where :math:`\mathcal C` is the set of elements, :math:`B` is the number of basis, :math:`\mathcal V` is the set of nodes/vertices/points.

        projector from element to edge
    elements : BufferDict[str, torch.Tensor]
        The element type is the key, which should be one of :meth:`tensormeshe.shape.element_types`.
        Each :obj:`element_type` corresponds to a 2D tensor of shape :math:`[N, B]`, where :math:`N` is the number of elements and :math:`B` is the number of basis functions
        element connectivity of each element type
    n_points : int
        number of points
    dimension : int
        dimension of the mesh, either :math:`1` or :math:`2` or :math:`3`
    element_types : list[str]
        element types, e.g. :obj:`["triangle6", "quad9"]`

    """
    projector:nn.ModuleDict # Dict[str, Projector]
    facet_mask:nn.ModuleDict # Dict[str, List[torch.Tensor]]
    transformation:nn.ModuleDict # Dict[str, Transformation]
    dimension:int 
    element_types:List[str]
    n_points:int


    __autodoc__ = [
        '__call__',
        'forward',
        '__post_init__',
        'from_assembler',
        'from_mesh',
    ]
    def __init__(self, 
                   facet_mask:nn.ModuleDict,
                   projector:nn.ModuleDict, 
                   transformation:nn.ModuleDict,
                   *args,
                   **kwargs):
        super().__init__()
   
        element_types = list(projector.keys())
        dimension     = element_type2dimension[element_types[0]]

        self.projector          = projector
        self.facet_mask         = facet_mask
        self.transformation    = transformation
        
        self.dimension          = dimension
        self.element_types      = element_types
        self.n_points           = next(iter(transformation.values())).n_points # type:ignore

        self.__post_init__(*args,**kwargs)
            
    @property
    def device(self):
        return self.quadrature_weights.device

    @property
    def dtype(self):
        return self.quadrature_weights.dtype

    def type(self,  dtype:torch.dtype):
        super().__doc__
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self
    
    def __call__(self, points:Optional[torch.Tensor] = None, 
                       func:Optional[Callable] = None,
                       point_data:Optional[Dict[str,torch.Tensor]] = None, 
                       )->torch.Tensor:
        r"""
        Parameters
        ----------
        points: torch.Tensor 
            2D tensor of shape :math:`[|\mathcal V|, D]`, where :math:`\mathcal V` is the set of nodes/vertices/points, :math:`D` is the dimension of the domain
            the coordinates of the points
        func: function or None, optional
            the bilinear function, when it's None the forward function will be used,
            if you want to reuse the same element assembler for different bilinear function, you can pass the bilinear function here
        point_data: Dict[str, torch.Tensor], optional
            tensor of shape :math:`[|\mathcal V|, ...]`, where :math:`\mathcal V` is the set of nodes/vertices/points
        batch_size: int or None, optional
            the batch size of quadrature points
            if :obj:`int` is given, the quadrature points will be divided into batches
            if :obj:`None` is given, the quadrature points will not be divided into batches
            default is :obj:`None`

        Returns
        -------
        torch.Tensor
            a torch.sparse_matrix of shape :math:`[|\mathcal V|]` or :math:`[|\mathcal V|\times H]`, 
            where :math:`|\mathcal V|` is the number of nodes and :math:`H` is the number of degrees of freedom per node.

        """
        if point_data is None:
            point_data = {}
       

        if points is not None: 
            self = self.type(points.dtype).to(points.device)
            for element_type in self.element_types:
                trans:Transformation = self.transformation[element_type] # type:ignore
                trans.update_points(points)
        else:
            points = next(iter(self.transformation.values())).points # type:ignore
            
        point_data["x"] = points # type:ignore

        for key, value in point_data.items():
            assert value.shape[0] == self.n_points, f"the shape of {key} should be [n_point, ...], but got {value.shape}"
 
        fn        = self.forward if func is None else func

        signature = inspect.signature(fn)

        parallel_fn = vmap(vmap(fn))

        integral = None
      
        for element_type in self.element_types:
            
            trans:Transformation = self.transformation[element_type] # type: ignore
            proj                 = self.projector[element_type] # type: ignore
            m:torch.Tensor       = self.facet_mask[element_type] # type:ignore [n_element, n_facet]
           
            ele_point_data       = {k:v[trans.elements] for k,v in point_data.items()}
   
            if trans.element.is_mix_facet: # for pyramid and prism
                tri_m, quad_m       = self.facet_mask[element_type] # type:ignore [n_element, n_facet]
                # prepare arguments
                tri_args = []
                quad_args= []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        
                        tri_shape_val, quad_shape_val = trans.facet_shape_val 
                        tri_shape_val = tri_shape_val.repeat(trans.n_elements, 1, 1, 1)[m] # [n_selected_tri_facet, n_quadrature_per_tri_facet, n_basis]
                        quad_shape_val= quad_shape_val.repeat(trans.n_elements, 1, 1, 1)[m] # [n_selected_quad_facet, n_quadrature_per_quad_facet, n_basis] 
                        tri_args.append(tri_shape_val)
                        quad_args.append(quad_shape_val)

                    elif key in ["gradu", "gradv"]:

                        tri_shape_grad, quad_shape_grad = trans.shape_grad
                        tri_args.append(tri_shape_grad[tri_m])
                        quad_args.append(quad_shape_grad[tri_m])
                        
                    elif key in ele_point_data:

                        tri_shape_val, quad_shape_val = trans.facet_shape_val 
                        tri_point_data = torch.einsum("eb...,fqb->efq...",ele_point_data[key], tri_shape_val)
                        quad_point_data= torch.einsum("eb...,fqb->efq...",ele_point_data[key], quad_shape_val)
               
                        tri_point_data = tri_point_data[tri_m] # [n_selected_tri_facet, n_quadrature_per_tri_facet, ...]
                        quad_point_data= quad_point_data[quad_m]# [n_selected_quad_facet, n_quadrature_per_quad_facet, ...]

                        tri_args.append(tri_point_data)
                        quad_args.append(quad_point_data)

                    elif key.startswith("grad") and key[4:] in ele_point_data: # "key"->"gradkey"
                        
                        tri_shape_grad, quad_shape_grad = trans.shape_grad
                        tri_grad_data = torch.einsum("eb...,efqbd->efq...d",ele_point_data[key[4:]], tri_shape_grad)
                        quad_grad_data= torch.einsum("eb...,efqbd->efq...d",ele_point_data[key[4:]], quad_shape_grad)
                        
                        tri_grad_data = tri_grad_data[tri_m]   # [n_selected_tri_facet, n_quadrature_per_tri_facet, ...., n_dim]
                        quad_grad_data= quad_grad_data[quad_m] # [n_selected_quad_facet, n_quadrature_per_quad_facet, ...., n_dim]

                        tri_args.append(tri_grad_data)
                        quad_args.append(quad_grad_data)

                    else:
                        raise NotImplementedError(f"key {key} is not implemented")

                # parallel dispatch 
                tri_integral = parallel_fn(*tri_args) 
                quad_integral= parallel_fn(*quad_args)
                # tri_integral [n_selected_tri_facet, n_quadrature_per_tri_facet, n_basis, ...]
                # quad_integral [n_selected_quad_facet, n_quadrature_per_quad_facet, n_basis, ...]

                tri_jxw, quad_jxw = trans.JxW 
                tri_jxw = tri_jxw[m]
                quad_jxw= quad_jxw[m]
                tri_integral = torch.einsum('fqb..., fq->fb...', tri_integral, tri_jxw) # [n_selected_tri_facet, n_basis, ...]
                quad_integral= torch.einsum('fqb..., fq->fb...', quad_integral, quad_jxw) # [n_selected_quad_facet, n_basis, ...]
             
                _integral = torch.cat([tri_integral, quad_integral], dim=0) # [n_selected_tri_facet+n_selected_quad_facet, n_basis, ...]
                
                _integral = proj(_integral) # [n_points, ...]

                integral  = _integral if integral is None else integral + _integral

            else: # same facet type
                m:torch.Tensor = self.facet_mask[element_type].item() # type:ignore [n_element, n_facet]

                # prepare arguments
                args = []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        args.append(trans.facet_shape_val.repeat(trans.n_elements, 1, 1, 1)[m]) # type:ignore [n_selected_facet, n_quadrature_per_facet, n_basis]

                    elif key in ["gradu", "gradv"]:
                        args.append(trans.facet_shape_grad[m]) # [n_selected_facet, n_qudrature_per_facet, n_basis, n_dim]

                    elif key in ele_point_data:
                        _ele_point_data = torch.einsum("eb...,qb->eq...",ele_point_data[key], trans.shape_val) # [n_element, n_facet, n_quadrature, ...]
                        _ele_point_data = _ele_point_data[m] # [n_selected_facet, n_quadrature, ...]
                        args.append(_ele_point_data)
    
                    elif key.startswith("grad") and key[4:] in ele_point_data: # "key"->"gradkey"
                        _ele_grad_data  = torch.einsum("eb...,eqbd->eq...d",ele_point_data[key[4:]], trans.facet_shape_grad) # [n_element, n_facet, n_quadrature, ..., n_dim]
                        _ele_grad_data  = _ele_grad_data[m] # [n_selected_facet, n_quadrature, ..., n_dim]
                        args.append(_ele_grad_data)

                    else:
                        raise NotImplementedError(f"key {key} is not implemented")
                
                # parallel dispatch 
                _integral = parallel_fn(*args) # [n_selected_facet, n_quadrature_per_facet, n_basis, ...] 
              
                _integral = torch.einsum('fqb..., fq->fb...', _integral, trans.FxW[m]) # [n_selected_facet, n_basis, ...]

                _integral = proj(_integral) #  [n_points, ...]

                integral  = _integral if integral is None else integral + _integral 

        return integral.flatten() # type: ignore

    def __post_init__(self, *args, **kwargs):
        r"""Override this function to precompute some data after the initialization
        """
        pass

    def __str__(self):
        n_element = {k:trans.n_elements for k, trans in self.transformation.items()}
        n_basis   = {k:trans.n_basis  for k, trans  in self.transformation.items()}
        return (
            f"{self.__class__.__name__}(\n"
            f"    element_types: {self.element_types}\n"
            f"    n_element: {n_element}\n"
            f"    n_point: {self.n_points}\n"
            f"    n_basis: {n_basis}\n"
            f"    n_dim: {self.dimension}\n"
            f"    n_quadrature: {' '.join(f'{k}:{v.n_quadrature}' for k, v in self.transformation.items())}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    @abstractmethod
    def forward(self, *args):
        r"""The weak form of the operator, you should override this function.
        Similar to the :meth:`torch:torch.nn.Module.forward` function, you can use :method: `tensormesh.assemble.ElementAssembler.__call__` to call this function

        Parameters
        ----------
        u : torch.Tensor, optional
            1D tensor shape :math:`[B]`, where :math:`B` is the number of basis
        v : torch.Tensor, optional
            1D tensor shape :math:`[B]`, where :math:`B` is the number of basis
        gradu : torch.Tensor, optional
            2D tensor shape :math:`[B,D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        gradv : torch.Tensor, optional
            2D tensor shape :math:`[B,D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        x : torch.Tensor, optional
            2D tensor shape :math:`[D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        gradx : torch.Tensor, optional
            3D tensor shape :math:`[D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension
        **point_data : Dict[str, torch.Tensor], optional
            The point_data are passed by __call__
            if the point data :obj:`"example_key"` passed in is of shape :math:`[|\mathcal V|, ...]`, 
            then the point data :obj:`"example_key"` passed in will be of shape :math:`[ ...]`,
            and the point data :obj:`"gradexample_key"` passed in will be of shape :math:`[ ..., D]`,
            where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension

        Returns
        -------
        torch.Tensor
            1D tensor of shape :math:`[B]` or 2D tensor of shape :math:`[B, H]`, where :math:`B` is the number of basis, :math:`H` is the number of degree of freedom per point

        """
        raise NotImplementedError(f"forward is not implemented")
    
    @classmethod
    def from_assembler(cls, obj, *args, **kwargs):
        r"""Build an FacetAssembler from another :meth:`tensormesh.assemble.NodeAssembler` or :meth:`tensormesh.assemble.ElementAssembler`.
        It's much faster than :meth:`tensormesh.assemble.NodeAssembler.from_mesh`.
        When you already have an NodeAssembler or ElementAssembler, you can use this function to build another NodeAssembler sharig the same mesh

        Parameters
        ----------
        obj: tensormesh.assemble.NodeAssembler or tensormesh.assemble.ElementAssembler
            an :meth:`tensormesh.assemble.NodeAssembler` or :meth:`torch_fem.assemble.ElementAssembler` object
        
        Returns
        -------
        torch_fem.assemble.NodeAssembler
            the new node_assembler sharing the same mesh
        """
        err_msg = f"the object {obj} should inheritate from NodeAssembler"
        assert isinstance(obj, FacetAssembler), err_msg
        return cls(
                    obj.facet_mask,
                    obj.projector, 
                    obj.transformation,
                    *args,**kwargs
                )

    @classmethod 
    def from_elements(cls,  points:torch.Tensor,
                            elements:Dict[str,torch.Tensor], 
                            boundary_mask:torch.Tensor, 
                            quadrature_order:int = 2, 
                            device:str|torch.device="cpu", 
                            dtype:torch.dtype=torch.float32,
                            project:str = "reduce",
                            *args,**kwargs):
        r"""Build an :meth:`torch_fem.assemble.NodeAssembler` from element connectivity.
        It's slower than :meth:`torch_fem.assemble.NodeAssembler.from_assembler`.

        Parameters
        ----------
        points: torch.Tensor
        elements: Dict[str, torch.Tensor] 
            the element connectivity, the key is the element type, the value is the element connectivity
            e.g. {"tri3": torch.tensor([[0, 1, 2], [1, 2, 3]])}
        n_points: int
            the number of points
        boundary_mask: torch.Tensor
            1 D tensor boolean of shape :math:`[n_points]`, the boundary mask of the mesh
            the boundary mask of the mesh
        quadrature_order: int 
            the order should be poisitive integer,
        
        Returns
        -------
        torch_fem.assemble.NodeAssembler
            the new node assembler use the topology of the mesh
        """
        n_points           = points.shape[0] # TODO: move transformation to the __call__
        projector          = {}
        facet_mask         = {}
        trasnformations    = {}
        
        # compute the facet_mask -> facet_quadrature
        for element_type, value in elements.items(): # type: ignore
            element = element_type2element(element_type)
            if element.is_mix_facet:
                is_boundary_element = boundary_mask[value].any(-1)
                boundary_elements   = value[is_boundary_element]                          # [n_element, n_basis]

                trans               = Transformation(
                                        points,
                                        boundary_elements,
                                        element_type, 
                                        quadrature_order)
                
                tri_boundary_facet_candidate, quad_boundary_facet_candidate = trans.facets
                # tri_boundary_facet_candidate [n_element, n_tri_facet, n_vertex_per_tri_facet]
                # quad_boundary_facet_candidate [n_element, n_quad_facet, n_vertex_per_quad_facet]
                is_tri_boundary_facet = boundary_mask[tri_boundary_facet_candidate].all(-1)     # [n_element, n_tri_facet]
                tri_boundary_facet    = tri_boundary_facet_candidate[is_tri_boundary_facet]     # [n_selected_tri_facet, n_vertex_per_tri_facet]
                is_quad_boundary_facet= boundary_mask[quad_boundary_facet_candidate].all(-1)    # [n_element, n_quad_facet]
                quad_boundary_facet   = quad_boundary_facet_candidate[is_quad_boundary_facet]   # [n_selected_quad_facet, n_vertex_per_quad_facet]
                n_selected_tri_facet, n_vertex_per_facet = tri_boundary_facet.shape
                n_selected_quad_facet, n_vertex_per_facet= quad_boundary_facet.shape
                n_basis               = element_type2order[element_type]

                elements[element_type]          = boundary_elements
                trasnformations[element_type]   = trans

                if project == "reduce":
                    projector[element_type]         = ReduceProjector(
                                                        indices   = torch.cat([tri_boundary_facet.flatten(),quad_boundary_facet.flatten()]), # [n_selected_facet, n_vertex_per_facet]
                                                        from_shape = (n_selected_tri_facet + n_selected_quad_facet, n_basis),
                                                        to_shape   = (n_points,)
                                                    )
                elif project == "sparse":
                    projector[element_type]         = SparseProjector(
                                                        from_ = torch.arange(n_selected_tri_facet * n_basis * n_vertex_per_facet + n_selected_quad_facet * n_basis * n_vertex_per_facet),
                                                        to_ = torch.cat([tri_boundary_facet.flatten(), quad_boundary_facet.flatten()]),
                                                        from_shape = (n_selected_facet, n_basis),
                                                        to_shape = (n_points,)
                                                    )

                facet_mask[element_type]        = BufferList([is_tri_boundary_facet, is_quad_boundary_facet])
            
            else: # same facet type
                is_boundary_element = boundary_mask[value].any(-1)
                boundary_elements   = value[is_boundary_element]                          # [n_element, n_basis]
                
                trans               = Transformation(
                                                    points, 
                                                    boundary_elements,
                                                    element_type, 
                                                    quadrature_order)
                
                boundary_facet_candidate = trans.facets                                   # [n_element, n_facet, n_vertex_per_facet]
                is_boundary_facet   = boundary_mask[boundary_facet_candidate].all(-1)     # [n_element, n_facet]
                boundary_facet      = boundary_facet_candidate[is_boundary_facet]         # [n_selected_facet, n_vertex_per_facet]       
                n_selected_facet, n_vertex_per_facet = boundary_facet.shape
                n_basis             = trans.n_basis

                elements[element_type]          = boundary_elements
                facet_mask[element_type]        = BufferList([is_boundary_facet])
                trasnformations[element_type]   = trans
   
                if project == "reduce":
                    projector[element_type]         = ReduceProjector(
                                                        indices   = boundary_facet.flatten(),
                                                        from_shape = (n_selected_facet, n_basis),
                                                        to_shape   = (n_points,)
                                                    )
                elif project == "sparse":
                    projector[element_type]         = SparseProjector(
                                                        from_ = torch.arange(n_selected_facet * n_basis * n_vertex_per_facet),
                                                        to_ = boundary_facet.flatten(),
                                                        from_shape = (n_selected_facet, n_basis),
                                                        to_shape = (n_points,)
                                                    )
                else:
                    raise ValueError(f"project should be either 'reduce' or 'sparse', but got {project}")
        
        facet_mask         = nn.ModuleDict(facet_mask)
        projector          = nn.ModuleDict(projector)
        transformation    = nn.ModuleDict(trasnformations)

        assembler = cls(
                   facet_mask,
                   projector,
                   transformation,
                   *args,**kwargs)
        
        assembler = assembler.type(dtype).to(device)
        return assembler

    @classmethod
    def from_mesh(cls, mesh:Mesh,  
                       boundary_mask:Optional[str|torch.Tensor]=None, 
                       quadrature_order:int=2,
                       project:str = "reduce",
                       *args,**kwargs):
        r"""Build an :meth:`torch_fem.assemble.NodeAssembler` from a mesh :meth:`torch_fem.mesh.Mesh`.
        It's slower than :meth:`torch_fem.assemble.NodeAssembler.from_assembler`.
        Because it will precompute the projection matrix $\mathcal P_{\mathcal V}$

        Parameters
        ----------
        mesh: torch_fem.mesh.mesh.Mesh
            a meth:`torch_fem.mesh.Mesh` object
        quadrature_order: int
            the order should be poisitive integer,
            default is :obj:`2`
        boundary_mask: str or torch.Tensor or None, optional
            the boundary mask of the mesh, if None, use mesh.boundary_mask,
            if str, use mesh.point_data[boundary_mask],
            if torch.Tensor, use it directly,
            default is :obj:`None`
        project: str, optional
            the projection method, either "reduce" or "sparse",
            default is :obj:`"reduce"`

        Returns
        -------
        torch_fem.assemble.NodeAssembler
            the new node assembler use the topology of the mesh
        """
        points:torch.Tensor   = mesh.points # type:ignore
        elements              = mesh.elements()
        n_points              = points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements} # type:ignore

        if boundary_mask is None:
            boundary_mask = mesh.boundary_mask
        elif isinstance(boundary_mask, str):
            boundary_mask = mesh.point_data[boundary_mask]
        assert boundary_mask.dim() == 1 and boundary_mask.shape[0] == n_points

        return cls.from_elements(points, 
                                 elements, # type:ignore 
                                 boundary_mask,
                                 quadrature_order, 
                                 mesh.device, 
                                 mesh.dtype,
                                 project,
                                 *args,**kwargs)
      

FacetAssembler.type.__doc__ = nn.Module.type.__doc__
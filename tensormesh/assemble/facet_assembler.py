from abc import abstractmethod
import inspect
from typing import Callable, Optional, Dict, List

import torch
import torch.nn as nn

from tensormesh.element.element_type import element_type2element

from .projector import ReduceProjector, SparseProjector
from .topology import boundary_facet_mask
from ..element import element_type2dimension, Transformation
from ..nn import BufferList
from ..mesh import Mesh
from ..vmap import vmap


class FacetAssembler(nn.Module):
    r"""Assemble an integrand over boundary facets of a mesh.

    :class:`FacetAssembler` mirrors :class:`NodeAssembler` but integrates over
    :math:`\partial \Omega` instead of :math:`\Omega`. Override
    :meth:`forward` to define a per-quadrature-point integrand; calling the
    assembler returns a flattened tensor of shape :math:`[|\mathcal V|]` or
    :math:`[|\mathcal V| \times H]` (vector-valued problems with :math:`H`
    DOFs per node).

    Typical uses include Neumann tractions, penalty contact, surface
    tension, and Robin boundary conditions.

    Examples
    --------
    Constant downward traction on the boundary:

    .. code-block:: python

        import torch
        from tensormesh import Mesh, FacetAssembler

        class TractionAssembler(FacetAssembler):
            def forward(self, v):
                t = torch.tensor([0.0, -1.0], dtype=v.dtype, device=v.device)
                return t * v        # contribution at one quadrature point

        mesh = Mesh.gen_rectangle()
        f = TractionAssembler.from_mesh(mesh)(mesh.points)

    Attributes
    ----------
    projector : torch.nn.ModuleDict
        Maps each ``element_type`` to a
        :class:`~tensormesh.assemble.projector.Projector` that scatters
        per-facet basis contributions onto the node vector.
    facet_mask : torch.nn.ModuleDict
        Maps each ``element_type`` to a :class:`~tensormesh.nn.BufferList`
        of boolean masks marking which facets of which elements lie on the
        selected boundary (one mask per facet type for mixed-facet shapes,
        otherwise a list of one).
    transformation : torch.nn.ModuleDict
        Maps each ``element_type`` to its cached :class:`~tensormesh.Transformation`,
        providing ``facet_shape_val``, ``facet_shape_grad``, and ``FxW``.
    n_points : int
        Number of mesh points (length of the output vector for scalar problems).
    dimension : int
        Spatial dimension of the mesh, one of ``1``, ``2``, ``3``.
    element_types : list[str]
        Element-type strings present in the mesh.
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
    def device(self) -> torch.device:
        """Device on which the assembler's buffers live."""
        return next(iter(self.transformation.values())).device  # type: ignore

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the assembler's buffers (``float32`` or ``float64``)."""
        return next(iter(self.transformation.values())).dtype  # type: ignore

    def type(self, dtype: torch.dtype):
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
        r"""Integrate the facet form and scatter into a global node vector.

        Parameters
        ----------
        points : torch.Tensor, optional
            Nodal coordinates of shape :math:`[|\mathcal V|, D]`. If ``None``,
            the points stored in the cached :class:`Transformation` are used.
        func : Callable, optional
            Facet integrand to use *in place of* :meth:`forward`.
        point_data : dict[str, torch.Tensor], optional
            Nodal fields, each of shape :math:`[|\mathcal V|, ...]`. Keys
            can appear as ``forward`` parameters and as gradients
            (``"grad"+key``).

        Returns
        -------
        torch.Tensor
            1D tensor of shape :math:`[|\mathcal V|]` (scalar problems) or
            flattened ``[|\mathcal V| \times H]`` (vector problems).
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
                        tri_shape_val = tri_shape_val.repeat(trans.n_elements, 1, 1, 1)[tri_m] # [n_selected_tri_facet, n_quadrature_per_tri_facet, n_basis]
                        quad_shape_val= quad_shape_val.repeat(trans.n_elements, 1, 1, 1)[quad_m] # [n_selected_quad_facet, n_quadrature_per_quad_facet, n_basis]
                        tri_args.append(tri_shape_val)
                        quad_args.append(quad_shape_val)

                    elif key in ["gradu", "gradv"]:

                        tri_shape_grad, quad_shape_grad = trans.facet_shape_grad
                        tri_args.append(tri_shape_grad[tri_m])
                        quad_args.append(quad_shape_grad[quad_m])
                        
                    elif key in ele_point_data:

                        tri_shape_val, quad_shape_val = trans.facet_shape_val 
                        tri_point_data = torch.einsum("eb...,fqb->efq...",ele_point_data[key], tri_shape_val)
                        quad_point_data= torch.einsum("eb...,fqb->efq...",ele_point_data[key], quad_shape_val)
               
                        tri_point_data = tri_point_data[tri_m] # [n_selected_tri_facet, n_quadrature_per_tri_facet, ...]
                        quad_point_data= quad_point_data[quad_m]# [n_selected_quad_facet, n_quadrature_per_quad_facet, ...]

                        tri_args.append(tri_point_data)
                        quad_args.append(quad_point_data)

                    elif key.startswith("grad") and key[4:] in ele_point_data: # "key"->"gradkey"
                        
                        tri_shape_grad, quad_shape_grad = trans.facet_shape_grad
                        tri_grad_data = torch.einsum("eb...,efqbd->efq...d",ele_point_data[key[4:]], tri_shape_grad)
                        quad_grad_data= torch.einsum("eb...,efqbd->efq...d",ele_point_data[key[4:]], quad_shape_grad)
                        
                        tri_grad_data = tri_grad_data[tri_m]   # [n_selected_tri_facet, n_quadrature_per_tri_facet, ...., n_dim]
                        quad_grad_data= quad_grad_data[quad_m] # [n_selected_quad_facet, n_quadrature_per_quad_facet, ...., n_dim]

                        tri_args.append(tri_grad_data)
                        quad_args.append(quad_grad_data)

                    elif key == "n":

                        tri_normal, quad_normal = trans.facet_normal
                        tri_args.append(tri_normal[tri_m])    # [n_selected_tri_facet, n_quadrature_per_tri_facet, n_dim]
                        quad_args.append(quad_normal[quad_m]) # [n_selected_quad_facet, n_quadrature_per_quad_facet, n_dim]

                    else:
                        raise NotImplementedError(f"key {key} is not implemented")

                # parallel dispatch 
                tri_integral = parallel_fn(*tri_args) 
                quad_integral= parallel_fn(*quad_args)
                # tri_integral [n_selected_tri_facet, n_quadrature_per_tri_facet, n_basis, ...]
                # quad_integral [n_selected_quad_facet, n_quadrature_per_quad_facet, n_basis, ...]

                tri_jxw, quad_jxw = trans.FxW
                tri_jxw = tri_jxw[tri_m]
                quad_jxw= quad_jxw[quad_m]
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
                        _ele_point_data = torch.einsum("eb...,fqb->efq...", ele_point_data[key], trans.facet_shape_val)
                        args.append(_ele_point_data[m])  # [n_selected_facet, n_quadrature_per_facet, ...]

                    elif key.startswith("grad") and key[4:] in ele_point_data:  # "key" -> "gradkey"
                        _ele_grad_data = torch.einsum("eb...,efqbd->efq...d", ele_point_data[key[4:]], trans.facet_shape_grad)
                        args.append(_ele_grad_data[m])  # [n_selected_facet, n_quadrature_per_facet, ..., n_dim]

                    elif key == "n":
                        args.append(trans.facet_normal[m])  # [n_selected_facet, n_quadrature_per_facet, n_dim]

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
        r"""Define the facet integrand at a single quadrature point.

        Subclasses must override this method. Vmap dispatches the
        per-quadrature-point function over all selected facets, so write
        it as if evaluating at *one* facet quadrature point. Unlike
        :class:`ElementAssembler.forward`, the basis arguments here keep
        the basis dimension (the inner ``vmap(vmap(...))`` covers facet +
        quadrature only, not basis).

        Parameters
        ----------
        u, v : torch.Tensor, optional
            Shape value on the facet — 1D tensor of shape ``[B]``.
        gradu, gradv : torch.Tensor, optional
            Shape gradient in physical coordinates — 2D tensor of shape ``[B, D]``.
        x : torch.Tensor, optional
            Physical coordinate at the quadrature point — 1D tensor of shape ``[D]``.
        gradx : torch.Tensor, optional
            Gradient of ``x`` w.r.t. reference coordinates — 2D tensor of shape ``[D, D]``.
        n : torch.Tensor, optional
            Outward unit normal of the facet at the quadrature point — 1D
            tensor of shape ``[D]`` (see
            :attr:`~tensormesh.Transformation.facet_normal`). A ``point_data``
            key named ``n`` takes precedence.
        **point_data : torch.Tensor
            Any key passed to ``__call__`` via ``point_data``: if the nodal
            tensor has shape :math:`[|\mathcal V|, ...]`, the value handed
            to ``forward`` has the trailing ``[...]`` shape, and its
            counterpart ``"grad"+key`` has shape ``[..., D]``.

        Returns
        -------
        torch.Tensor
            1D tensor of shape ``[B]`` (scalar problems) or 2D tensor of
            shape ``[B, H]`` (vector problems with ``H`` degrees of freedom
            per node).
        """
        raise NotImplementedError("forward is not implemented")
    
    @classmethod
    def from_assembler(cls, obj, *args, **kwargs):
        r"""Build a :class:`FacetAssembler` sharing topology with ``obj``.

        Much faster than :meth:`from_mesh` since the facet mask, projector,
        and cached :class:`Transformation` are reused as-is.

        Parameters
        ----------
        obj : FacetAssembler
            An existing facet assembler whose boundary topology should be reused.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.

        Returns
        -------
        FacetAssembler
            A new assembler sharing the same boundary topology.
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
        r"""Build a :class:`FacetAssembler` from raw connectivity tensors.

        Slower than :meth:`from_assembler` because the boundary topology is
        rebuilt from scratch.

        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape :math:`[|\mathcal V|, D]` listing node coordinates.
        elements : dict[str, torch.Tensor]
            Connectivity keyed by element-type string, e.g.
            ``{"triangle": tensor([[0, 1, 2], [1, 2, 3]])}``.
        boundary_mask : torch.Tensor
            1D boolean tensor of shape :math:`[|\mathcal V|]` marking which
            nodes lie on the boundary. A facet is selected iff it is a
            boundary facet of ``elements`` (referenced by a single cell) and
            *all* of its nodes are flagged.
        quadrature_order : int, optional
            Positive integer; defaults to ``2``.
        device : torch.device or str, optional
            Device of the assembler; defaults to ``"cpu"``.
        dtype : torch.dtype, optional
            Floating dtype; defaults to :obj:`torch.float32`.
        project : {'reduce', 'sparse'}, optional
            Projection backend; defaults to ``"reduce"``.

        Returns
        -------
        FacetAssembler
            A new assembler that owns the given boundary topology.
        """
        n_points           = points.shape[0] # TODO: move transformation to the __call__
        projector          = {}
        facet_mask         = {}
        trasnformations    = {}

        # Only cells touching a flagged node can own a selected facet, and every cell
        # sharing such a facet holds all of its nodes too, so facet incidence among
        # these cells tells boundary facets from interior ones. A node mask alone
        # cannot: an interior facet whose nodes all lie on the boundary would be
        # selected once from each side.
        candidate_elements = {element_type: value[boundary_mask[value].any(-1)]
                              for element_type, value in elements.items()}
        on_boundary        = boundary_facet_mask(candidate_elements)

        # compute the facet_mask -> facet_quadrature
        for element_type, boundary_elements in candidate_elements.items(): # type: ignore
            element = element_type2element(element_type)
            if element.is_mix_facet:
                trans               = Transformation(
                                        points,
                                        boundary_elements,
                                        element_type, 
                                        quadrature_order)
                
                tri_boundary_facet_candidate, quad_boundary_facet_candidate = trans.facets
                # tri_boundary_facet_candidate  [n_boundary_element, n_tri_facet, n_basis_per_tri_facet]
                # quad_boundary_facet_candidate [n_boundary_element, n_quad_facet, n_basis_per_quad_facet]
                tri_on_boundary, quad_on_boundary = on_boundary[element_type]
                is_tri_boundary_facet = boundary_mask[tri_boundary_facet_candidate].all(-1) & tri_on_boundary    # [n_boundary_element, n_tri_facet]
                is_quad_boundary_facet= boundary_mask[quad_boundary_facet_candidate].all(-1) & quad_on_boundary  # [n_boundary_element, n_quad_facet]
                n_selected_tri_facet  = int(is_tri_boundary_facet.sum().item())
                n_selected_quad_facet = int(is_quad_boundary_facet.sum().item())
                n_basis               = trans.n_basis                                            # n_basis_per_cell

                # For each selected facet, fetch the *whole cell* connectivity. This is needed
                # because the integrand is computed on cell-wise shape functions; entries that
                # do not belong to the facet are exactly zero for Lagrange bases, so scattering
                # them via these cell dofs only adds zeros to "off-facet" global dofs.
                tri_elem_idx          = is_tri_boundary_facet.nonzero(as_tuple=True)[0]          # [n_selected_tri_facet]
                quad_elem_idx         = is_quad_boundary_facet.nonzero(as_tuple=True)[0]         # [n_selected_quad_facet]
                tri_cell_dofs         = boundary_elements[tri_elem_idx]                          # [n_selected_tri_facet,  n_basis_per_cell]
                quad_cell_dofs        = boundary_elements[quad_elem_idx]                         # [n_selected_quad_facet, n_basis_per_cell]

                trasnformations[element_type]   = trans

                if project == "reduce":
                    projector[element_type]         = ReduceProjector(
                                                        indices    = torch.cat([tri_cell_dofs.flatten(), quad_cell_dofs.flatten()]), # [(n_sel_tri + n_sel_quad) * n_basis_per_cell]
                                                        from_shape = (n_selected_tri_facet + n_selected_quad_facet, n_basis),
                                                        to_shape   = (n_points,)
                                                    )
                elif project == "sparse":
                    n_entries = (n_selected_tri_facet + n_selected_quad_facet) * n_basis
                    projector[element_type]         = SparseProjector(
                                                        from_ = torch.arange(n_entries),
                                                        to_   = torch.cat([tri_cell_dofs.flatten(), quad_cell_dofs.flatten()]),
                                                        from_shape = (n_selected_tri_facet + n_selected_quad_facet, n_basis),
                                                        to_shape = (n_points,)
                                                    )

                facet_mask[element_type]        = BufferList([is_tri_boundary_facet, is_quad_boundary_facet])
            
            else: # same facet type
                trans               = Transformation(
                                                    points, 
                                                    boundary_elements,
                                                    element_type, 
                                                    quadrature_order)
                
                boundary_facet_candidate = trans.facets                                   # [n_boundary_element, n_facet, n_basis_per_facet]
                is_boundary_facet   = boundary_mask[boundary_facet_candidate].all(-1) & on_boundary[element_type][0]  # [n_boundary_element, n_facet]
                n_selected_facet    = int(is_boundary_facet.sum().item())
                n_basis             = trans.n_basis                                       # n_basis_per_cell

                # For each selected facet, fetch the *whole cell* connectivity so that the
                # projector indices align with the cell-basis dimension of the integrand
                # produced in __call__. Lagrange basis functions vanish on facets they do
                # not belong to, so scattering the corresponding zero entries is harmless.
                selected_elem_idx   = is_boundary_facet.nonzero(as_tuple=True)[0]         # [n_selected_facet]
                cell_dofs_per_facet = boundary_elements[selected_elem_idx]                # [n_selected_facet, n_basis_per_cell]

                facet_mask[element_type]        = BufferList([is_boundary_facet])
                trasnformations[element_type]   = trans
   
                if project == "reduce":
                    projector[element_type]         = ReduceProjector(
                                                        indices    = cell_dofs_per_facet.flatten(),
                                                        from_shape = (n_selected_facet, n_basis),
                                                        to_shape   = (n_points,)
                                                    )
                elif project == "sparse":
                    projector[element_type]         = SparseProjector(
                                                        from_ = torch.arange(n_selected_facet * n_basis),
                                                        to_   = cell_dofs_per_facet.flatten(),
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
        r"""Build a :class:`FacetAssembler` from a :class:`~tensormesh.Mesh`.

        Slower than :meth:`from_assembler` because the boundary topology is
        rebuilt from connectivity.

        Parameters
        ----------
        mesh : tensormesh.Mesh
            Source mesh; connectivity, points, and (default) boundary mask
            are read from it.
        boundary_mask : str, torch.Tensor, or None, optional
            Boundary selector. ``None`` (default) uses ``mesh.boundary_mask``;
            ``str`` keys into ``mesh.point_data``; a tensor is used verbatim
            and must be 1D boolean of length ``n_points``.
        quadrature_order : int, optional
            Positive integer; defaults to ``2``.
        project : {'reduce', 'sparse'}, optional
            Projection backend; defaults to ``"reduce"``.

        Returns
        -------
        FacetAssembler
            A new assembler that owns the boundary topology of the mesh.
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
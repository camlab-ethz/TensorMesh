from abc import abstractmethod
import inspect
import math
from typing import Optional, Callable, List, Mapping, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

from .projector import ReduceProjector, SparseProjector, Projector
from ..nn import BufferDict
from ..element import element_type2dimension, Transformation
from ..sparse import SparseMatrix
from ..mesh import Mesh
from ..vmap import vmap

class InputBroadcast:
    element:Optional[int]
    quadrature:Optional[int]
    u:Optional[int]
    v:Optional[int]

    def __init__(self, element:bool, 
                        quadrature:bool,
                        u:bool, 
                        v:bool):
        self.element    = 0 if element else None 
        self.quadrature = 0 if quadrature else None 
        self.u          = 0 if u else None 
        self.v          = 0 if v else None



class ElementAssembler(nn.Module):
    r"""Assemble an element-wise bilinear form into a global sparse matrix.

    :class:`ElementAssembler` is an :class:`torch:torch.nn.Module` whose
    :meth:`forward` defines the integrand of a bilinear form
    :math:`a(u, v) = \int_\Omega f(u, v)\, \mathrm{d}\Omega`. Calling the
    assembler integrates :math:`f` over every quadrature point of every
    element and scatters the result into a :class:`~tensormesh.sparse.SparseMatrix`
    of shape :math:`[|\mathcal V|, |\mathcal V|]` (scalar problems) or
    :math:`[|\mathcal V| \times H, |\mathcal V| \times H]` (vector problems
    with :math:`H` degrees of freedom per node).

    Subclasses are usually built from a mesh:

    * :meth:`from_mesh` — build from a :class:`~tensormesh.Mesh` (slower; precomputes
      the projection tensor :math:`\mathcal P_{\mathcal E}`).
    * :meth:`from_assembler` — share the topology of another assembler (fast).

    The schematic of one ``__call__`` is

    .. math::

        K \overset{\text{bsr matrix}}{\leftarrow}\hat K_\text{global},
        \qquad
        \hat K_{\text{global}}^{nkl} = \mathcal P_{\mathcal E}^{nhij}\, \hat K_{\text{local}}^{hklij},
        \qquad
        \hat K_{ij} = \int_\Omega f(u, v)\, \mathrm{d}\Omega,

    where

    * :math:`\hat K_{\text{global}}` are the non-zero entries of the global
      Galerkin matrix, :math:`K_{\text{global}} \in \mathbb R^{|\mathcal E|\times d \times d}`;
    * :math:`\hat K_{\text{local}}` is the per-element Galerkin block,
      :math:`K_{\text{local}} \in \mathbb R^{|\mathcal C|\times h \times h \times d \times d}`;
    * :math:`\mathcal P_{\mathcal E} \in \mathbb R_{\text{sparse}}^{|\mathcal E|\times|\mathcal C|\times h \times h}`
      is the projection (assemble) tensor from local to global;
    * :math:`\mathcal C` indexes elements/cells, :math:`\mathcal E` indexes
      edges (the unique ``(i, j)`` pairs of basis indices), :math:`\mathcal V`
      indexes nodes/vertices, and :math:`h` is the number of basis functions
      per element.

    Examples
    --------
    Mass matrix :math:`M_{ij} = \int_\Omega u_i v_j\, \mathrm{d}\Omega`:

    .. code-block:: python

        import tensormesh
        class MassAssembler(tensormesh.ElementAssembler):
            def forward(self, u, v):
                return u * v
        mesh = tensormesh.Mesh.gen_rectangle()
        assembler = MassAssembler.from_mesh(mesh)
        M = assembler(mesh.points)

    Laplace stiffness :math:`K_{ij} = \int_\Omega \nabla u_i \cdot \nabla v_j\, \mathrm{d}\Omega`:

    .. code-block:: python

        import tensormesh
        import tensormesh.functional as F
        class LaplaceAssembler(tensormesh.ElementAssembler):
            def forward(self, gradu, gradv):
                return F.dot(gradu, gradv)
        mesh = tensormesh.Mesh.gen_circle()
        assembler = LaplaceAssembler.from_mesh(mesh)
        K = assembler(mesh.points)

    Attributes
    ----------
    projector : torch.nn.ModuleDict
        Maps each ``element_type`` to a
        :class:`~tensormesh.assemble.projector.Projector` that scatters
        per-element contributions of shape :math:`[|\mathcal C_e|, B_e, B_e]`
        into the global edge vector of shape :math:`[|\mathcal E|]`.
    transformation : torch.nn.ModuleDict
        Maps each ``element_type`` to a :class:`~tensormesh.Transformation`
        caching per-element Jacobians, shape values, shape gradients, and
        ``JxW`` at quadrature points.
    elements : tensormesh.nn.BufferDict
        Maps each ``element_type`` to its connectivity tensor of shape
        :math:`[|\mathcal C|, B]`, e.g. ``{"triangle6": tensor([[0, 1, 2], ...])}``.
    edges : torch.Tensor
        Long tensor of shape :math:`[2, |\mathcal E|]` listing the unique
        ``(row, col)`` index pairs of the assembled sparse matrix; produced
        by deduplicating all per-element basis-pair indices.
    n_points : int
        Number of mesh points (size of one matrix dimension).
    dimension : int
        Spatial dimension of the mesh, one of ``1``, ``2``, ``3``.
    element_types : list[str]
        Element-type strings present in the mesh, e.g. ``["triangle6", "quad9"]``.
    """

    projector:nn.ModuleDict      # Dict[str, Projector]
    """:no-index:"""
    transformation:nn.ModuleDict # Dict[str, Transformation]
    """:no-index:"""
    elements:BufferDict          # Dict[str, torch.Tensor]
    """:no-index:"""
    edges:torch.Tensor  
    """:no-index:"""         
    dimension:int
    """:no-index:"""
    element_types:List[str]
    """:no-index:"""
    n_points:int
    """:no-index:"""

    __autodoc__ = [
        '__call__',
        'forward',
        '__post_init__',
        'from_assembler',
        'from_mesh',
    ]
    def __init__(self,
                        projector:nn.ModuleDict, 
                        transformation:nn.ModuleDict,
                        elements:BufferDict,
                        edges:torch.Tensor,
                        *args,
                        **kwargs):
        super().__init__()
        element_types = list(projector.keys())
        dimension     = element_type2dimension[element_types[0]]

        self.projector          = projector
        self.transformation     = transformation
        self.elements           = elements
        self.register_buffer("edges", edges)
        self.dimension          = dimension
        self.element_types      = list(elements.keys())
        self.n_points           = next(iter(self.transformation.values())).n_points # type: ignore
        self.__post_init__(*args,**kwargs)

    @property
    def device(self) -> torch.device:
        r"""Device on which the assembler's buffers live."""
        return next(iter(self.transformation.values())).device  # type: ignore

    @property
    def dtype(self) -> torch.dtype:
        r"""Floating dtype of the assembler's buffers (``float32`` or ``float64``)."""
        return next(iter(self.transformation.values())).dtype  # type: ignore

    def type(self, dtype:torch.dtype):
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self

    def _integrate(self, batch_integral, jxw, n_element, n_basis, use_element_parallel):
        if use_element_parallel:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [{n_element}, batch_size, {n_basis},{n_basis}] or [{n_element}, batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 4 or batch_integral.dim() == 6, error_msg
            assert batch_integral.shape[0] == n_element, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg 
            assert batch_integral.shape[3] == n_basis, error_msg
            if batch_integral.dim() == 6:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("eqij...,eq->eij...", batch_integral, jxw)
        else:
            error_msg = f"the shape returned by forward function is {[*batch_integral.shape]} which is not supported, should either be [batch_size, {n_basis},{n_basis}] or [batch_size,{n_basis},{n_basis}, dof_per_point, dof_per_point]"
            assert batch_integral.dim() == 3 or batch_integral.dim() == 5, error_msg
            assert batch_integral.shape[1] == n_basis, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg
            if batch_integral.dim() == 5:
                assert batch_integral.shape[-1] == batch_integral.shape[-2], error_msg
            batch_integral = torch.einsum("qij...,eq->eij...", batch_integral, jxw)

        return batch_integral
    
    def _build_output(self, integral):
        r"""Wrap the per-edge integral tensor in a :class:`SparseMatrix`.

        Parameters
        ----------
        integral : torch.Tensor
            Per-edge integral of shape :math:`[|\mathcal E|, ...]`.

        Returns
        -------
        SparseMatrix
            COO sparse matrix; block-COO if ``integral`` is 3D (vector problems).
        """
        if integral.dim() == 1:
            return SparseMatrix(integral, self.edges[0], self.edges[1], shape=(self.n_points, self.n_points))
        elif integral.dim() ==  3:
            return SparseMatrix.from_block_coo(integral, self.edges[0], self.edges[1], shape=(self.n_points, self.n_points))
        else:
            raise Exception(f"the shape of integral is supposed to be  1D or 3D, but got {integral.shape}")

    def __call__(self, points:Optional[torch.Tensor] = None, 
                       func:Optional[Callable] = None,
                       point_data:Optional[Mapping[str, torch.Tensor]] = None, 
                       element_data:Optional[Union[Mapping[str, Mapping[str,torch.Tensor]], Mapping[str,torch.Tensor]]] = None, 
                       scalar_data:Optional[Mapping[str, torch.Tensor]] = None,
                       batch_size:int = -1):
        r"""Assemble the bilinear form into a global sparse matrix.

        Parameters
        ----------
        points : torch.Tensor, optional
            Nodal coordinates of shape :math:`[|\mathcal V|, D]`. If ``None``,
            the points stored in the cached :class:`Transformation` are used.
        func : Callable, optional
            Bilinear integrand to use *in place of* :meth:`forward`. Useful
            when reusing the same topology with different forms.
        point_data : Mapping[str, torch.Tensor], optional
            Nodal fields, each of shape :math:`[|\mathcal V|, ...]`. Their
            keys can appear as parameters of ``forward`` (e.g. ``"kappa"``)
            and as gradients (``"gradkappa"``).
        element_data : Mapping[str, ...], optional
            Per-element data. Either ``{key: {element_type: tensor}}``
            (mixed-element meshes) or ``{key: tensor}`` when only one
            element type is present.
        scalar_data : Mapping[str, scalar or torch.Tensor], optional
            Global scalars passed verbatim to ``forward`` (no broadcasting).
        batch_size : int, optional
            Batch size for quadrature points. ``-1`` (default) processes all
            quadrature points at once; positive values split them into
            chunks for memory-bound problems.

        Returns
        -------
        SparseMatrix
            Sparse matrix of shape :math:`[|\mathcal V|, |\mathcal V|]`, or
            block sparse of shape :math:`[|\mathcal V| \times H, |\mathcal V| \times H]`
            with :math:`H` degrees of freedom per node (inferred from the
            rank of the ``forward`` return).
        """
        assert isinstance(point_data, dict) or point_data is None, f"point_data should be a dict, but got {type(point_data)}. Please pass  in extra parameter using key-value pairs"
        # make sure point data is Dict[str, torch.Tensor]
        if point_data is None:
            point_data = {}

        # make sure element data is Dict[str,Dict[str,  torch.Tensor]]
        if element_data is None:
            element_data = {element_type:{} for element_type in self.element_types} # 
        else:
            if not isinstance(next(iter(element_data.values())), dict):
                assert len(self.element_types) == 1
                element_type = self.element_types[0]
                element_data = {key:{element_type:value} for key, value in element_data.items()} # type:ignore
                for  key in element_data:
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0], f"the shape of {key} should be [{self.elements[element_type].shape[0]}, ...], but got {element_data[key][element_type].shape[0]}"
            else:
                for key, value in element_data.items():
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0]
        
        # make sure scalar data is Dict[str, torch.Tensor]
        if scalar_data is None:
            scalar_data = {}
        else:
            scalar_data = {k:torch.tensor(v) for k,v in scalar_data.items()}


        # make sure points is torch.Tensor
        if points is None:
            points = next(iter(self.transformation.values())).points # type: ignore
        else:
            for element_type in self.element_types:
                assert points.shape[1] == self.transformation[element_type].dim, f"the dimension of points should be {self.transformation[element_type].dimension}, but got {points.shape[1]}"
                self.transformation[element_type].update_points(points) # type: ignore
        

        point_data["x"] = points # type:ignore [n_point, n_dim]

        self = self.type(points.dtype).to(points.device) # type:ignore

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], f"the shape of {key} should be [n_point, ...], but got {value.shape}"
        
        # parse signature

        fn = self.forward if func is None else func

        signature = inspect.signature(fn)

        broadcast_fns = [
            (lambda x: x=="u"     , InputBroadcast(False, True, True,  False)), # [:        , n_quadrature, n_u_basis,         :]
            (lambda x: x=="v"     , InputBroadcast(False, True, False,  True)), # [:        , n_quadrature,         :, n_v_basis]
            (lambda x: x=="gradu" , InputBroadcast(True,  True, True,  False)), # [n_element, n_quadrature, n_u_basis,         :, n_dim]
            (lambda x: x=="gradv" , InputBroadcast(True,  True, False,  True)), # [n_element, n_quadrature,         :, n_v_basis, n_dim]
            (lambda x: x in element_data.keys(),
                                    InputBroadcast(True, False, False, False)), # [n_element,            :,         :,         :]
            (lambda x: x in scalar_data.keys(),
                                    InputBroadcast(True, True,  True,  True )),
            (lambda x: x in point_data.keys(),
                                    InputBroadcast(True, True,  False, False)),  # [n_element, n_quadrature,        :,         :]
            (lambda x: x in {"grad" + key for key in point_data.keys()},
                                    InputBroadcast(True, True, False,  False)),  # [n_element, n_quadrature,        :,          :, n_dim]
                
        ]

        element_dims    = []
        quadrature_dims = []
        u_dims          = []
        v_dims          = []
    
        for key in signature.parameters:
            is_match = False
            for condition, broadcast in broadcast_fns:
                if condition(key):
                    element_dims.append(broadcast.element)
                    quadrature_dims.append(broadcast.quadrature)
                    u_dims.append(broadcast.u)
                    v_dims.append(broadcast.v)
                    is_match = True
                    break
            if not is_match:
                raise ValueError(f"{key} is not supported, please use `u`, `v`, `gradu`, `gradv` or more keys provided by point_data, element_data or scalar_data")
            

        element_dims    = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)
        u_dims          = tuple(u_dims)
        v_dims          = tuple(v_dims)
        if all([x is None for x in element_dims]):
            # if all is shape_val
            parallel_fn = vmap(
                vmap(
                    vmap(
                        fn, 
                        in_dims = v_dims
                    ),
                    in_dims = u_dims
                ),
                in_dims = quadrature_dims
            )
            use_element_parallel = False
        else:
            parallel_fn = vmap(
                vmap(
                    vmap(
                        vmap(
                            fn, 
                            in_dims = v_dims
                        ),
                        in_dims = u_dims
                    ),
                    in_dims = quadrature_dims
                ),
                in_dims=element_dims
            )
            use_element_parallel = True

        integral:Optional[torch.Tensor] = None
      
        for element_type in self.element_types:
            element_integral = None
            
            trans:Transformation  = self.transformation[element_type] # type:ignore
            proj:Projector        = self.projector[element_type] # type:ignore
            n_quadrature          = trans.n_quadrature

            if batch_size in (-1, None):
                n_batch      = 1
                n_batch_size = n_quadrature
            else:
                n_batch_size = batch_size
                n_batch      = math.ceil(n_quadrature / batch_size)
            elements:torch.Tensor = self.elements[element_type]
            ele_point_data = {k:v[elements] for k,v in point_data.items()}

            for i in range(n_batch):
                # prepare arguments
                shape_val            = trans.batch_shape_val(i*n_batch_size, n_batch_size) 
                shape_grad, jxw      = trans.batch_shape_grad_jxw(
                                                quadrature_start = i*n_batch_size, 
                                                quadrature_batch = n_batch_size)
                # shape_val : [quadrature_batch, n_basis]
                # shape_grad: [element_batch, quadrature_batch, n_basis, dim]
                # jxw       : [element_batch, quadrature_batch]

                args = []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        args.append(shape_val) 
                    elif key in ["gradu", "gradv"]:
                        args.append(shape_grad)
                    elif key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eq...",ele_point_data[key], shape_val))
                        # point data : [element_batch, quadrature_batch, ...]
                    elif key.startswith("grad") and key[4:] in ele_point_data: # grad point data
                        args.append(torch.einsum("eb...,eqbd->eq...d",ele_point_data[key[4:]], shape_grad)) 
                        # grad point data : [element_batch, quadrature_batch, ..., dim]
                    elif key in element_data: # type:ignore
                        args.append(element_data[key][element_type]) # type:ignore
                    elif key in scalar_data: # type:ignore
                        args.append(scalar_data[key])
                    else:
                        raise NotImplementedError(f"key {key} is not implemented")

                # parallel dispatch 
                batch_integral = parallel_fn(*args) # [n_element, batch_size, n_basis, n_basis, ...] or [n_batch, batch_size, n_basis, ...]


                batch_integral = self._integrate(batch_integral, jxw, trans.n_elements, trans.n_basis, use_element_parallel)

                element_integral = batch_integral if element_integral is None else element_integral + batch_integral
            
            integral = proj(element_integral) if integral is None else integral + proj(element_integral) # type:ignore [n_edge, ...]

        return self._build_output(integral)

    def energy(self, points:Optional[torch.Tensor] = None, 
                       func:Optional[Callable] = None,
                       point_data:Optional[Mapping[str, torch.Tensor]] = None, 
                       element_data:Optional[Union[Mapping[str, Mapping[str,torch.Tensor]], Mapping[str,torch.Tensor]]] = None, 
                       scalar_data:Optional[Mapping[str, torch.Tensor]] = None,
                       batch_size:int = -1):
        r"""
        Calculates the total potential energy of the system by integrating the element energy density over the domain.

        This method is designed for energy-based variational problems (e.g., hyperelasticity, phase field, optimization).
        It computes the global integral:
        
        .. math::
            E = \int_\Omega \psi(\mathbf{u}, \nabla \mathbf{u}, \dots) d\Omega

        where :math:`\psi` is the energy density function (defined in :meth:`element_energy` or passed as `func`).
        The method handles efficient parallel integration using quadrature rules and PyTorch's `vmap`.

        **How it works (Simplified):**
        
        1. You provide nodal data (like displacement `u`) in `point_data`.
        2. You define an `element_energy` function that takes arguments like `u` (value) or `graddisplacement` (gradient) and returns the scalar energy density for a **single quadrature point**.
        3. This method automatically matches your arguments, broadcasts them over all elements and quadrature points, computes the gradients if requested, and performs the numerical integration.

        Parameters
        ----------
        points : torch.Tensor, optional
            The current positions of the nodes (shape: `[N_nodes, Dim]`). 
            If not provided, uses the mesh's initial points.
        func : Callable, optional
            A function that computes energy density at a single quadrature point.
            Signature: `func(arg1, arg2, ...) -> torch.Tensor (scalar)`.
            If `None`, uses `self.element_energy`.
            
            **Supported Arguments for `func`:**
            
            *   **`u`**, **`v`**, ... : Values interpolated from `point_data` (e.g., passing `point_data={'u': ...}` allows requesting `u`).
            *   **`gradu`**, **`gradv`**, ...: Gradients interpolated from `point_data` (e.g., passing `point_data={'u': ...}` allows requesting `gradu`).
            *   **`key`** in `element_data`: Element-wise constant or quadrature-varying data.
            *   **`key`** in `scalar_data`: Global scalar constants.

        point_data : Dict[str, torch.Tensor], optional
            Nodal fields (e.g., displacement, phase field).
            Shape: `[N_nodes, ...]` or `[N_nodes, Dim]`.
        
        element_data : Dict[str, Tensor] or Dict[str, Dict[str, Tensor]], optional
            Data defined on elements.
            Can be:
            
            *   **Constant per element**: Shape `[N_elements, ...]`.
            *   **Varying per quadrature point** (e.g., history variables): Shape `[N_elements, N_quad, ...]`.
            
        scalar_data : Dict[str, Tensor], optional
            Global constants (e.g., time step `dt`).
        
        batch_size : int, optional
            Batch size for processing quadrature points to save memory. Default is -1 (process all at once).

        Returns
        -------
        torch.Tensor
            The total scalar energy (shape: `[]`). 
            Since this operation is differentiable, you can call `.backward()` on it to compute forces (gradients w.r.t `points` or `point_data`).

        Examples
        --------
        
        **1. Hyperelasticity (Neo-Hookean Energy)**
        
        .. code-block:: python
        
            class NeoHookean(ElementAssembler):
                def element_energy(self, graddisplacement):
                    # graddisplacement is [Dim, Dim] tensor at one quad point
                    F = torch.eye(3) + graddisplacement
                    J = torch.det(F)
                    # ... compute psi ...
                    return psi

            model = NeoHookean.from_mesh(mesh)
            
            # Compute energy
            # Auto-differentiable w.r.t u
            u = torch.zeros_like(mesh.points, requires_grad=True)
            E = model.energy(point_data={"displacement": u})
            
            # Compute Forces (Internal Force Vector)
            E.backward()
            F_int = u.grad
            
        """
        assert isinstance(point_data, dict) or point_data is None, f"point_data should be a dict"
        if point_data is None: point_data = {}
        if element_data is None: element_data = {element_type:{} for element_type in self.element_types}
        else:
            if not isinstance(next(iter(element_data.values())), dict):
                assert len(self.element_types) == 1
                element_type = self.element_types[0]
                element_data = {key:{element_type:value} for key, value in element_data.items()}
                for key in element_data:
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0]
            else:
                for key, value in element_data.items():
                    for element_type in self.element_types:
                        assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0]
        
        if scalar_data is None: scalar_data = {}
        else: scalar_data = {k:torch.tensor(v) for k,v in scalar_data.items()}

        if points is None:
            points = next(iter(self.transformation.values())).points
        else:
            for element_type in self.element_types:
                self.transformation[element_type].update_points(points)
        
        point_data["x"] = points
        self = self.type(points.dtype).to(points.device)

        fn = self.element_energy if func is None else func
        signature = inspect.signature(fn)

        # Broadcasting rules are inferred from argument names.
        # For element_data we support both:
        # - per-element constant: [n_elem, ...]
        # - per-(element,quadrature): [n_elem, n_quad, ...] (e.g. history variables)
        broadcast_fns = [
            (lambda x: x in element_data.keys(), InputBroadcast(True, False, False, False)),
            (lambda x: x in scalar_data.keys(), InputBroadcast(True, True, True, True)),
            (lambda x: x in point_data.keys(), InputBroadcast(True, True, False, False)),
            (lambda x: x in {"grad" + key for key in point_data.keys()}, InputBroadcast(True, True, False, False)),
        ]

        element_dims = []
        quadrature_dims = []
        
        for key in signature.parameters:
            # Special-case element_data: auto-detect whether it varies per quadrature
            if key in element_data:
                is_quad = False
                for etype in self.element_types:
                    n_quad = self.transformation[etype].n_quadrature
                    data = element_data[key][etype]
                    if data.dim() >= 2 and data.shape[1] == n_quad:
                        is_quad = True
                    else:
                        is_quad = False
                        break
                element_dims.append(0)
                quadrature_dims.append(0 if is_quad else None)
                continue

            is_match = False
            for condition, broadcast in broadcast_fns[1:]:
                if condition(key):
                    element_dims.append(broadcast.element)
                    quadrature_dims.append(broadcast.quadrature)
                    is_match = True
                    break
            if not is_match:
                raise ValueError(f"{key} is not supported for energy calculation.")

        element_dims = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)
        
        parallel_fn = vmap(
            vmap(fn, in_dims=quadrature_dims),
            in_dims=element_dims
        )

        total_energy = 0.0
        
        for element_type in self.element_types:
            trans = self.transformation[element_type]
            n_quadrature = trans.n_quadrature
            if batch_size in (-1, None):
                n_batch      = 1
                n_batch_size = n_quadrature
            else:
                n_batch_size = batch_size
                n_batch      = math.ceil(n_quadrature / batch_size)
            elements = self.elements[element_type]
            ele_point_data = {k:v[elements] for k,v in point_data.items()}

            for i in range(n_batch):
                q_start = i * n_batch_size
                q_end = q_start + n_batch_size

                # IMPORTANT: energy() needs element-wise shape gradients (include element dim).
                # batch_shape_grad_jxw returns gradients without the element dimension for some elements,
                # which breaks einsum broadcasting. Use full tensors and slice quadrature instead.
                shape_val = trans.shape_val[q_start:q_end, :]                     # [q, b]
                shape_grad = trans.shape_grad[:, q_start:q_end, :, :]            # [e, q, b, d]
                jxw = trans.JxW[:, q_start:q_end]                                # [e, q]
                
                args = []
                for key in signature.parameters:
                    if key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eq...", ele_point_data[key], shape_val))
                    elif key.startswith("grad") and key[4:] in ele_point_data:
                        args.append(torch.einsum("eb...,eqbd->eq...d", ele_point_data[key[4:]], shape_grad))
                    elif key in element_data:
                        args.append(element_data[key][element_type])
                    elif key in scalar_data:
                        args.append(scalar_data[key])
                    else:
                        raise NotImplementedError(f"key {key} not implemented")
                
                batch_energy_density = parallel_fn(*args) # [n_elem, n_quad]
                batch_energy = (batch_energy_density * jxw).sum()
                total_energy += batch_energy
        
        return total_energy

    def element_energy(self, **kwargs):
        r"""
        Override this method to define the energy density at a single quadrature point.
        
        This method is called automatically by :meth:`energy` using `vmap` to parallelize over all quadrature points.
        
        Parameters
        ----------
        **kwargs : torch.Tensor
            Arguments matching the variable names requested.
            Common arguments:
            
            *   **u**, **v**: Value of field at quadrature point.
            *   **gradu**, **gradv**: Gradient of field at quadrature point.
            *   **element_data_key**: Value of element data (constant or interpolated).
            
        Returns
        -------
        torch.Tensor
            Scalar energy density (:math:`\psi`) for this quadrature point.
        """
        raise NotImplementedError("element_energy is not implemented")

    @abstractmethod
    def forward(self, **kwargs):
        r"""Define the integrand of the bilinear form at a single quadrature point.

        Subclasses must override this method. The library uses
        :func:`torch.vmap` to lift the per-quadrature-point function over
        all quadrature points and all elements, so write it as if you were
        evaluating at *one* point. Parameters are dispatched by name;
        return values are integrated against ``JxW`` and scattered by the
        caller (see ``ElementAssembler.__call__``).

        Parameters
        ----------
        u, v : torch.Tensor, optional
            Shape value at the quadrature point — 0D tensor of shape ``[]``.
        gradu, gradv : torch.Tensor, optional
            Shape gradient in physical coordinates — 1D tensor of shape ``[D]``.
        x : torch.Tensor, optional
            Physical coordinate at the quadrature point — 1D tensor of shape ``[D]``.
        gradx : torch.Tensor, optional
            Gradient of ``x`` w.r.t. reference coordinates — 2D tensor of shape ``[D, D]``.
        **point_data : torch.Tensor
            Any key passed to ``__call__`` via ``point_data``: if the nodal
            tensor has shape :math:`[|\mathcal V|, ...]`, the value handed
            to ``forward`` has the trailing ``[...]`` shape, and its
            counterpart ``"grad"+key`` has shape ``[..., D]``.

        Returns
        -------
        torch.Tensor
            Either a 2D tensor of shape ``[B, B]`` (scalar problems) or a 4D
            tensor of shape ``[B, B, H, H]`` (vector problems with ``H``
            degrees of freedom per node).
        """
        raise NotImplementedError("forward is not implemented")
        
    def __post_init__(self):
        r"""Override this function to precompute some data after the initialization
        """
        pass

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    element_types: {self.element_types}\n"
            f"    n_element: {' '.join(f'{k}:{v.shape[0]}' for k, v in self.elements.items())}\n"
            f"    n_point: {self.n_points}\n"
            f"    n_basis: {' '.join(f'{k}:{v.shape[1]}' for k, v in self.elements.items())}\n"
            f"    n_dim: {self.dimension}\n"
            f"    n_quadrature: {' '.join(f'{k}:{v.n_quadrature}' for k, v in self.transformation.items())}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    @classmethod
    def from_assembler(cls, obj, *args, **kwargs):
        r"""Build an :class:`ElementAssembler` that shares topology with ``obj``.

        Much faster than :meth:`from_mesh` since the projector and the
        cached :class:`Transformation` are reused as-is.

        Parameters
        ----------
        obj : ElementAssembler
            An existing assembler whose mesh topology should be reused.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.

        Returns
        -------
        ElementAssembler
            A new assembler sharing the same mesh.
        """
        assert isinstance(obj, ElementAssembler), f"obj must be an instance of ElementAssembler, but got {type(obj)}"
        return cls(
                   obj.projector, 
                   obj.transformation,
                   obj.elements,
                   obj.edges,
                   *args,**kwargs)

    @classmethod
    def from_mesh(cls, mesh:Mesh,
                        quadrature_order:int = 2,
                        project:str = 'reduce',
                        *args,
                        **kwargs):
        r"""Build an :class:`ElementAssembler` from a :class:`~tensormesh.Mesh`.

        Slower than :meth:`from_assembler` because the projection tensor
        :math:`\mathcal P_{\mathcal E}` is precomputed from the connectivity.

        Parameters
        ----------
        mesh : tensormesh.Mesh
            Source mesh; both connectivity and points are taken from it.
        quadrature_order : int, optional
            Positive integer; defaults to ``2``.
        project : {'reduce', 'sparse'}, optional
            Backend used for the element-to-edge scatter; defaults to ``'reduce'``.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.

        Returns
        -------
        ElementAssembler
            A new assembler that owns the mesh topology.
        """

        assert project in ['reduce', 'sparse']

        points:torch.Tensor   = mesh.points # type: ignore
        elements              = mesh.elements() # type:ignore
        n_points:int          = points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}

        projector          = {}
        transformations    = {}
        

        ######################
        # compute the edges
        ######################
        # Per-type stack-then-flatten preserves the original
        # ``[e0:b0b0, e0:b0b1, ..., e0:bNbN, e1:b0b0, ...]`` ordering
        # that the downstream ``reshape(n_element, n_basis, n_basis)``
        # consumer expects. Across types we ``cat`` so mixed-type meshes
        # (quad+tri, hex+tet, ...) with different per-type lengths
        # concatenate cleanly instead of failing the stack same-size check.
        elem_u, elem_v = [], []
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            per_u, per_v = [], []
            for i in range(n_basis):
                for j in range(n_basis):
                    per_u.append(value[:, i])
                    per_v.append(value[:, j])
            elem_u.append(torch.stack(per_u, -1).flatten())
            elem_v.append(torch.stack(per_v, -1).flatten())

        elem_u = torch.cat(elem_u)
        elem_v = torch.cat(elem_v)
        elem_u, elem_v = elem_u.cpu().numpy().copy(), elem_v.cpu().numpy().copy()
        tmp = scipy.sparse.coo_matrix(( # used to remove duplicated edges
            np.ones_like(elem_u), # data
            (elem_u, elem_v), # (row, col)
        ), shape = (n_points,  n_points)).tocsr().tocoo()
        edge_u, edge_v = tmp.row, tmp.col
        num_edges  = len(edge_u)
        eids_csr = scipy.sparse.coo_matrix((
            np.arange(num_edges), (edge_u, edge_v)
        ), shape=(n_points, n_points)).tocsr()    
        elem_eids     = np.array(eids_csr[elem_u, elem_v].copy()).ravel()

        ptr = 0
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            elem_eids  = np.array(eids_csr[elem_u[ptr:ptr+n_element*n_basis*n_basis], elem_v[ptr:ptr+n_element*n_basis*n_basis]].copy()).ravel()
            ptr += n_element * n_basis * n_basis


            if project == "reduce":
                projector[element_type] = ReduceProjector( # [n_element, n_basis, n_basis] -> [num_edges]
                                            indices    = torch.from_numpy(elem_eids),
                                            from_shape = (n_element, n_basis, n_basis), 
                                            to_shape = (num_edges,),
                                        )
            elif project == "sparse":
                projector[element_type] = SparseProjector( # [n_element, n_basis, n_basis] -> [num_edges]
                                            from_  = torch.arange(len(elem_eids), device=points.device),
                                            to_    = torch.from_numpy(elem_eids),
                                            from_shape = (n_element, n_basis, n_basis), 
                                            to_shape = (num_edges,),
                                        )
            else:
                raise ValueError(f"project must be 'reduce' or 'sparse', but got {project}")

            transformations[element_type] = Transformation(
                                        points  = points,
                                        elements = value,
                                        element_type   = element_type, 
                                        quadrature_order= quadrature_order
                                    )
            
         
        edges = torch.from_numpy(np.stack([edge_u, edge_v], 0))


        projector          = nn.ModuleDict(projector)
        transformations    = nn.ModuleDict(transformations)
        elements           = BufferDict({k:v.long() for k,v in elements.items()})
    

        assembler = cls(
                   projector, 
                   transformations,
                   elements,
                   edges,
                   *args,**kwargs)
        assembler = assembler.type(mesh.dtype).to(mesh.device)
        return assembler
      
ElementAssembler.type.__doc__ = nn.Module.type.__doc__
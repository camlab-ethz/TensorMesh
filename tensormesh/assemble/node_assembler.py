
from abc import abstractmethod
import torch 
import torch.nn as nn
import numpy as np
from functools import reduce, partial
import inspect
from typing import Dict, Optional, Callable, Literal, Union
from .element_assembler import InputBroadcast
from .projector import ReduceProjector, SparseProjector, Projector
from ..element import Transformation, element_type2dimension
from ..nn import BufferDict
from ..mesh import Mesh
from ..vmap import vmap


class InputBroadcast:
    element:Optional[int]
    quadrature:Optional[int]
    v:Optional[int]

    def __init__(self, element:bool, 
                        quadrature:bool,
                        v:bool):
        self.element    = 0 if element else None 
        self.quadrature = 0 if quadrature else None 
        self.v          = 0 if v else None



class NodeAssembler(nn.Module):
    r"""The NodeAssembler is used to assemble the operator on the nodes of the mesh

    The output when calling the NodeAssembler is a vector, which is of shape :math:`[|\mathcal V|]` or :math:`[|\mathcal V|\times H]`, 
    where :math:`|\mathcal V|` is th

    Examples
    --------
    1. assemble the mass vector

    .. math::

        f_i = \int_\Omega v_i \text dv

    .. code-block:: python

        import tensormesh
        class MassAssembler(tensormesh.NodeAssembler):
            def forward(self, v):
                return v
        mesh = tensormesh.Mesh.gen_rectangle()
        assembler = MassAssembler.from_mesh(mesh)
        f = assembler(mesh.points)

    2. assemble the traction vector

    .. math::

        f_i = \int_\Omega t \cdot v_i \text dv

    where :math:`t` is the traction vector.

    .. code-block:: python

        import tensormesh
        import tensormesh.functional as F
        class TractionAssembler(tensormesh.NodeAssembler):
            def forward(self, v, t):
                return F.dot(t, v)
        mesh = tensormesh.Mesh.gen_circle()
        t = torch.ones(mesh.n_points, 2) # unit traction in x,y directions
        assembler = TractionAssembler.from_mesh(mesh)
        f = assembler(mesh.points, t)
    
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

    projector: nn.ModuleDict # Dict[str, Projector]
    transformation: nn.ModuleDict # Dict[str, Transformation]
    elements: BufferDict # Dict[str, torch.Tensor]

    dimension: int
    element_types: list[str]
    n_points: int

    __autodoc__ = [
        '__call__',
        'forward',
        '__post_init__',
        'from_assembler',
        'from_mesh',
        'compile',
        'reset_compile',
    ]
    def __init__(self, 
                   projector:nn.ModuleDict, 
                   transformation:nn.ModuleDict,
                   elements:BufferDict,
                   *args, **kwargs):
        super().__init__()

        element_types = list(projector.keys())
        dimension     = element_type2dimension[element_types[0]]

        self.projector          = projector
        self.transformation     = transformation
        self.elements           = elements
        
        self.dimension          = dimension
        self.element_types      = element_types
        self.n_points           = next(iter(elements.values())).shape[0]
        
        # Compile options
        self._compile: bool = False
        self._compile_options: Dict = {}
        self._compiled_call_fn: Optional[Callable] = None

        self.__post_init__(*args, **kwargs)
        
    def _integrate(self, batch_integral, jxw, n_element, n_basis, use_element_parallel):
        if not use_element_parallel:
            error_msg = f"the shape returned by forward function is {batch_integral.shape} which is not supported, should either be [batch_size,{n_basis}] or [batch_size,{n_basis}, dof_per_point]"
            assert batch_integral.dim() == 2 or batch_integral.dim() == 3, error_msg
            assert batch_integral.shape[1] == n_basis, error_msg
            batch_integral = torch.einsum("qi...,eq->ei...", batch_integral, jxw) # [n_element, n_basis, ...]
        else:
            error_msg = f"the shape returned by forward function is {batch_integral.shape} which is not supported, should either be [{n_element},batch_size,{n_basis}] or [{n_element},batch_size,{n_basis}, dof_per_point]"
            assert batch_integral.dim() == 3 or batch_integral.dim() == 4, error_msg
            assert batch_integral.shape[0] == n_element, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg
            batch_integral = torch.einsum("eqb...,eq->eb...", batch_integral, jxw) # [n_element, n_basis, ...]
        return batch_integral
    
    @property
    def device(self) -> torch.device:
        """Returns the device of the assembler."""
        return next(iter(self.transformation.values())).device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the assembler."""
        return next(iter(self.transformation.values())).dtype

    def type(self,  dtype:torch.dtype):
        super().__doc__
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self
    
    def _build_compiled_call(self, 
                             point_data_keys: list,
                             scalar_data_keys: list,
                             element_type: str) -> Callable:
        """Build a compiled function for the entire call path.
        
        This function directly uses broadcast operations instead of vmap,
        which is more efficient when compiled with torch.compile.
        
        Performance optimizations:
        - Uses broadcast + sum instead of einsum for better GPU performance
        - Avoids einsum overhead for simple tensor contractions
        - Uses matmul for 2D contractions where applicable
        """
        trans = self.transformation[element_type]
        proj = self.projector[element_type]
        elements = self.elements[element_type]
        fn = self.forward
        signature = inspect.signature(fn)
        param_keys = list(signature.parameters.keys())
        
        # Pre-compute static data
        shape_val = trans.batch_shape_val(0, trans.n_quadrature)
        shape_grad, jxw = trans.batch_shape_grad_jxw(
            quadrature_start=0, quadrature_batch=trans.n_quadrature
        )
        # Pre-transpose shape_val for matmul: [n_quad, n_basis] -> [n_basis, n_quad]
        shape_val_T = shape_val.T
        
        def compiled_call(point_data_tensors: list, scalar_data_tensors: list) -> torch.Tensor:
            """Optimized call path using direct broadcast (no vmap)."""
            # Build ele_point_data
            ele_point_data = {k: v[elements] for k, v in zip(point_data_keys, point_data_tensors)}
            scalar_data_dict = {k: v for k, v in zip(scalar_data_keys, scalar_data_tensors)}
            
            # Build args with proper shapes for broadcast
            args = []
            for key in param_keys:
                if key == "v":
                    args.append(shape_val)  # [n_quad, n_basis]
                elif key == "gradv":
                    args.append(shape_grad)  # [n_element, n_quad, n_basis, n_dim]
                elif key in ele_point_data:
                    # Interpolate to quadrature points: [n_element, n_basis] @ [n_basis, n_quad] -> [n_element, n_quad]
                    # Use matmul instead of einsum for better performance
                    args.append(torch.matmul(ele_point_data[key], shape_val_T))
                elif key.startswith("grad") and key[4:] in ele_point_data:
                    # Gradient at quadrature points: [n_element, n_basis] -> [n_element, n_quad, n_dim]
                    # Use broadcast + sum instead of einsum: 3.7x faster
                    # einsum("eb,eqbd->eqd") == (x[:, None, :, None] * shape_grad).sum(dim=2)
                    args.append((ele_point_data[key[4:]][:, None, :, None] * shape_grad).sum(dim=2))
                elif key in scalar_data_dict:
                    args.append(scalar_data_dict[key])
            
            # Call forward directly - it should handle broadcast automatically
            # The forward function is written for scalar inputs but works with broadcast
            # because PyTorch broadcasting rules apply
            batch_integral = fn(*args)  # [n_element, n_quad, n_basis]
            
            # Integrate over quadrature points using broadcast + sum instead of einsum
            # einsum("eqb,eq->eb") == (result * jxw[:, :, None]).sum(dim=1)
            # This is ~3x faster than einsum
            batch_integral = (batch_integral * jxw[:, :, None]).sum(dim=1)  # [n_element, n_basis]
            
            # Project to nodes
            return proj(batch_integral).flatten()
        
        return compiled_call

    def __call__(self, 
                 points:Optional[torch.Tensor] = None, 
                 func:Optional[Callable] = None,
                 point_data:Optional[Dict[str, torch.Tensor]]=None, 
                 scalar_data:Optional[Dict[str, torch.Tensor]]=None,
                 batch_size:int=1)->torch.Tensor:
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
        scalar_data: Dict[str, torch.Tensor|float|int], optional
            scalar data that should not be broadcasted, will be directly passed to forward
        batch_size: int or None, optional
            the batch size of quadrature points
            if :obj:`int` is given, the quadrature points will be divided into batches
            if :obj:`None` is given, the quadrature points will not be divided into batches
            default is :obj:`1`

        Returns
        -------
        torch.Tensor
            a torch.sparse_matrix of shape :math:`[|\mathcal V|]` or :math:`[|\mathcal V|\times H]`, 
            where :math:`|\mathcal V|` is the number of nodes and :math:`H` is the number of degrees of freedom per node.

        """
        if point_data is None:
            point_data = {}
        if scalar_data is None:
            scalar_data = {}

        if points is None:
            points = next(iter(self.transformation.values())).points # type:ignore [n_point, n_dim]
        else:
            for element_type in self.element_types:
                assert points.shape[1] == self.transformation[element_type].dim, f"the dimension of the points should be {self.transformation[element_type].dimension}, but got {points.shape[1]}"
                trans:Transformation   = self.transformation[element_type] # type:ignore
                trans.update_points(points) # type:ignore

        point_data["x"] = points # type:ignore

        self = self.type(points.dtype).to(points.device) # type:ignore

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], f"the shape of {key} should be [n_point, ...], but got {value.shape}"
 
        # Use fast path if enabled (bypasses vmap, uses direct broadcast)
        if self._compile and len(self.element_types) == 1 and func is None:
            element_type = self.element_types[0]
            point_data_keys = sorted([k for k in point_data.keys() if k != "x"])
            scalar_data_keys = sorted(scalar_data.keys())
            cache_key = f"call_{element_type}_{tuple(point_data_keys)}_{tuple(scalar_data_keys)}"
            
            if self._compiled_call_fn is None or getattr(self, '_compiled_cache_key', None) != cache_key:
                # Build fast call function (uses broadcast, not vmap)
                raw_fn = self._build_compiled_call(point_data_keys, scalar_data_keys, element_type)
                # Optionally compile with torch.compile for additional optimization
                if self._compile_options.get("mode") != "disable":
                    self._compiled_call_fn = torch.compile(raw_fn, **self._compile_options)
                else:
                    self._compiled_call_fn = raw_fn
                self._compiled_cache_key = cache_key
            
            # Call fast function
            point_data_tensors = [point_data[k] for k in point_data_keys]
            scalar_data_tensors = [scalar_data[k] for k in scalar_data_keys]
            return self._compiled_call_fn(point_data_tensors, scalar_data_tensors)

        # Original vmap path
        fn = self.forward if func is None else func
        signature = inspect.signature(fn)

        broadcast_fns = [
            (lambda x: x=="v"     , InputBroadcast(False, True,  True)), # [:        , n_quadrature, n_v_basis]
            (lambda x: x=="gradv" , InputBroadcast(True,  True,  True)), # [n_element, n_quadrature, n_v_basis, n_dim]
            (lambda x: x in scalar_data.keys(),
                                    InputBroadcast(True, True,  True)),
            (lambda x: x in point_data.keys(),
                                    InputBroadcast(True, True,  False)),  # [n_element, n_quadrature,        :]
            (lambda x: x in {"grad" + key for key in point_data.keys()},
                                    InputBroadcast(True, True,  False)),  # [n_element, n_quadrature,        :, n_dim]
                
        ]

        element_dims    = []
        quadrature_dims = []
        v_dims          = []
    
        for key in signature.parameters:
            is_match = False
            for condition, broadcast in broadcast_fns:
                if condition(key):
                    element_dims.append(broadcast.element)
                    quadrature_dims.append(broadcast.quadrature)
                    v_dims.append(broadcast.v)
                    is_match = True
                    break
            if not is_match:
                raise ValueError(f"{key} is not supported, please use  `v`, `gradv` or more keys provided by point_data, element_data or scalar_data")
            

        element_dims    = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)
        v_dims          = tuple(v_dims)
        
        # Determine use_element_parallel based on element_dims
        use_element_parallel = not all([x is None for x in element_dims])
        
        if all([x is None for x in element_dims]):
            parallel_fn = vmap(vmap(fn, in_dims=v_dims), in_dims=quadrature_dims)
        else:
            parallel_fn = vmap(vmap(vmap(fn, in_dims=v_dims), in_dims=quadrature_dims), in_dims=element_dims)

        integral:Optional[torch.Tensor] = None # [n_points, ...]
      
        for element_type in self.element_types:
            trans:Transformation = self.transformation[element_type] # type:ignore
            proj:ReduceProjector       = self.projector[element_type]      # type:ignore
            element_integral:Optional[torch.Tensor] = None           # [n_element, n_basis, ...]
            n_batch      = trans.n_quadrature // batch_size if batch_size is not None else 1
            n_batch_size = batch_size if batch_size is not None else trans.n_quadrature
            ele_point_data = {k:v[self.elements[element_type]] for k,v in point_data.items()}

            for i in range(n_batch):
                shape_val       = trans.batch_shape_val(i*n_batch_size, n_batch_size)
                shape_grad, jxw = trans.batch_shape_grad_jxw(
                                    quadrature_start = i*n_batch_size, 
                                    quadrature_batch = n_batch_size) 

                args = []
                for key in signature.parameters:
                    if key in ["v"]:
                        args.append(shape_val) 
                    elif key in ["gradv"]:
                        args.append(shape_grad)
                    elif key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eq...",ele_point_data[key], shape_val))
                        # point data : [element_batch, quadrature_batch, ...]
                    elif key.startswith("grad") and key[4:] in ele_point_data: # grad point data
                        args.append(torch.einsum("eb...,eqbd->eq...d",ele_point_data[key[4:]], shape_grad)) 
                        # grad point data : [element_batch, quadrature_batch, ..., dim]
                    elif key in scalar_data: # type:ignore
                        args.append(scalar_data[key])
                    else:
                        raise NotImplementedError(f"key {key} is not implemented")

                batch_integral = parallel_fn(*args) # [n_element, batch_size, n_basis, ...] or [batch_size,  n_basis, ...]

                batch_integral = self._integrate(batch_integral, jxw, trans.n_elements, trans.n_basis, use_element_parallel)

                element_integral = batch_integral if element_integral is None else element_integral + batch_integral

            assert element_integral is not None
            integral = proj(element_integral) if integral is None else integral + proj(element_integral)
    
        assert integral is not None
        return integral.flatten() # [n_points * n_dim]

    def __post_init__(self, *args, **kwargs):
        r"""Override this function to precompute some data after the initialization
        """
        pass

    def compile(self, 
                mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "disable"] = "disable",
                fullgraph: bool = False,
                dynamic: Optional[bool] = None,
                backend: str = "inductor",
                **kwargs) -> "NodeAssembler":
        r"""Enable fast mode for the assembler to speed up computation.
        
        When compile mode is enabled, the ``__call__`` method bypasses vmap and uses
        direct broadcast operations, achieving up to 5-30x speedup.
        
        By default (``mode="disable"``), only the vmap bypass is enabled without 
        ``torch.compile``. This provides the best performance for most cases.
        Set ``mode="default"`` or other modes to additionally enable ``torch.compile``.

        Examples
        --------
        .. code-block:: python

            # Enable fast mode (recommended, no torch.compile overhead)
            assembler = MassAssembler.from_mesh(mesh).compile()
            
            # Enable with torch.compile for potential additional optimization
            assembler = MassAssembler.from_mesh(mesh).compile(mode="default")
            
            # Use normally - automatically uses fast path
            result = assembler(point_data={'phi': phi, 'f': f})
            
            # Disable for debugging (can set breakpoints in forward)
            assembler.reset_compile()

        Parameters
        ----------
        mode : str, optional
            Compilation mode, one of:
            
            - ``"disable"``: Only bypass vmap, no torch.compile (fastest startup, recommended)
            - ``"default"``: Also use torch.compile with default settings
            - ``"reduce-overhead"``: torch.compile with reduced Python overhead
            - ``"max-autotune"``: torch.compile with maximum optimization
            - ``"max-autotune-no-cudagraphs"``: Like max-autotune but without CUDA graphs
            
            Default is ``"disable"``
        fullgraph : bool, optional
            Whether to compile the entire graph. Default is ``False``
        dynamic : bool or None, optional
            Whether to use dynamic shapes. Default is ``None`` (auto-detect)
        backend : str, optional
            Compilation backend. Default is ``"inductor"``
        **kwargs : dict
            Additional keyword arguments passed to ``torch.compile``
        
        Returns
        -------
        NodeAssembler
            Returns self for method chaining
            
        See Also
        --------
        reset_compile : Disable fast mode and use vmap path
        is_compiled : Check if fast mode is enabled
        """
        self._compile = True
        self._compile_options = {
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend,
            **kwargs
        }
        self._compiled_call_fn = None
        return self

    def flat_mode(self) -> "NodeAssembler":
        r"""Enable the fast broadcast-based implementation without torch.compile.
        
        This allows to bypass vmap and use optimized broadcast operations.
        It is equivalent to calling compile(mode="disable").
        
        Returns
        -------
        NodeAssembler
            Returns self for method chaining
        """
        return self.compile(mode="disable")
    
    def reset_compile(self) -> "NodeAssembler":
        r"""Disable torch.compile and clear the compiled function cache.
        
        This is useful for debugging or when you want to switch back to 
        the non-compiled version.

        Examples
        --------
        .. code-block:: python

            # Disable compile for debugging
            assembler.reset_compile()
            
            # Now you can set breakpoints in forward()

        Returns
        -------
        NodeAssembler
            Returns self for method chaining
        """
        self._compile = False
        self._compile_options = {}
        self._compiled_call_fn = None
        return self
    
    @property
    def is_compiled(self) -> bool:
        r"""Check if the assembler is in compile mode.
        
        Returns
        -------
        bool
            True if compile mode is enabled, False otherwise
        """
        return self._compile

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    element_types: {self.element_types}\n"
            f"    n_element: {' '.join(f'{k}:{v.shape[0]}' for k, v in self.elements.items())}\n"
            f"    n_point: {self.n_points}\n"
            f"    n_basis: {' '.join(f'{k}:{v.shape[1]}' for k, v in self.elements.items())}\n"
            f"    n_dim: {self.dimension}\n"
            f"    n_quadrature: {' '.join(f'{k}:{v.shape[0]}' for k, v in self.quadrature_weights.items())}\n"
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
        v : torch.Tensor, optional
            1D tensor shape :math:`[]`
        gradv : torch.Tensor, optional
            2D tensor shape :math:`[D]`, :math:`D` is the dimension of the dimension
        x : torch.Tensor, optional
            2D tensor shape :math:`[D]`, :math:`D` is the dimension of the dimension
        gradx : torch.Tensor, optional
            3D tensor shape :math:`[D, D]`, :math:`D` is the dimension of the dimension
        **point_data : Dict[str, torch.Tensor], optional
            The point_data are passed by __call__
            if the point data :obj:`"example_key"` passed in is of shape :math:`[|\mathcal V|, ...]`, 
            then the point data :obj:`"example_key"` passed in will be of shape :math:`[ ...]`,
            and the point data :obj:`"gradexample_key"` passed in will be of shape :math:`[ ..., D]`,
            where :math:`B` is the number of basis, :math:`D` is the dimension of the dimension

        Returns
        -------
        torch.Tensor
            0D tensor of shape :math:`[]` or 1D tensor of shape :math:`[H]`, :math:`H` is the number of degree of freedom per point

        """
        raise NotImplementedError(f"forward is not implemented")
    
    @classmethod
    def from_assembler(cls, obj, *args,**kwargs):
        r"""Build an NodeAssembler from another :meth:`tensormesh.assemble.NodeAssembler` or :meth:`tensormesh.assemble.ElementAssembler`.
        It's much faster than :meth:`tensormesh.assemble.NodeAssembler.from_mesh`.
        When you already have an NodeAssembler or ElementAssembler, you can use this function to build another NodeAssembler sharig the same mesh

        Parameters
        ----------
        obj: tensormesh.assemble.NodeAssembler or tensormesh.assemble.ElementAssembler
            an :meth:`tensormesh.assemble.NodeAssembler` or :meth:`tensormesh.assemble.ElementAssembler` object
        
        Returns
        -------
        tensormesh.assemble.NodeAssembler
            the new node_assembler sharing the same mesh
        """
        err_msg = f"the object {obj} should inheritate from NodeAssembler"
        assert isinstance(obj, NodeAssembler), err_msg
        return cls(
                obj.projector, 
                obj.transformation,
                obj.elements,
                *args, **kwargs
                )

    @classmethod 
    def from_elements(cls, 
                      points:torch.Tensor,
                      elements:Dict[str, torch.Tensor], 
                      quadrature_order:int = 2, 
                      device:torch.device|str="cpu", 
                      dtype:torch.dtype=torch.float32,
                      project:str = "reduce",
                      *args,**kwargs):
        r"""Build an :meth:`tensormesh.assemble.NodeAssembler` from element connectivity.
        It's slower than :meth:`tensormesh.assemble.NodeAssembler.from_assembler`.

        Parameters
        ----------
        points:torch.Tensor
            2D tensor of shape :math:`[|\mathcal V|, D]`, where :math:`\mathcal V` is the set of nodes/vertices/points, :math:`D` is the dimension of the domain
        elements: Dict[str, torch.Tensor] or torch.Tensor
            the element connectivity, the key is the element type, the value is the element connectivity
            e.g. {"tri3": torch.tensor([[0, 1, 2], [1, 2, 3]])}
        quadrature_order: int 
            the order should be poisitive integer, default is :obj:`2`
        device: torch.device or str
            the device of the assembler, default is :obj:`"cpu"`
        dtype: torch.dtype
            the data type of the assembler, default is :obj:`torch.float32`
        project: str
            the projection method, either "reduce" or "sparse", default is "reduce"
            "reduce" uses :meth:`torch:torch.Tensor.index_add_` to assemble the matrix, which is more memory efficient
            "sparse" uses sparse matrix multiplication to assemble the matrix, which is faster but requires more memory

                    
        Returns
        -------
        tensormesh.assemble.NodeAssembler
            the new node assembler use the topology of the mesh
        """
        projector          = {}
        tranformation      = {}
        
        n_points = points.shape[0]

        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            # quadrature_weights[element_type], quadrature_points[element_type] =\
            # get_quadrature(element_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
            # shape_val[element_type] = get_shape_val(element_type, quadrature_points[element_type]) # [n_quadrature, n_basis]

            if project == "reduce":
                projector[element_type] = ReduceProjector(
                    indices   = value.flatten(),
                    from_shape = (n_element, n_basis),
                    to_shape   = (n_points,)
                )
            elif project == "sparse":
                projector[element_type] = SparseProjector(
                    from_ = torch.arange(n_element * n_basis, device=value.device).reshape(n_element, n_basis).flatten(),
                    to_   = value.flatten(),
                    from_shape = (n_element, n_basis),
                    to_shape   = (n_points,)
                )
            else:
                raise ValueError(f"project should be either 'reduce' or 'sparse', but got {project}")
            tranformation[element_type] = Transformation(
                points, 
                value,
                element_type,
                quadrature_order
            )

        projector          = nn.ModuleDict(projector)
        transformation     = nn.ModuleDict(tranformation)
        elements           = BufferDict(elements) # type:ignore

        assembler = cls(
                   projector, 
                   transformation,
                   elements,*args, **kwargs) # type:ignore

        
        assembler = assembler.type(dtype).to(device)
        return assembler

    @classmethod
    def from_mesh(cls, mesh:Mesh,  
                  quadrature_order:int = 2, 
                  project:str = "reduce",
                  *args, **kwargs):
        r"""Build an :meth:`tensormesh.assemble.NodeAssembler` from a mesh :meth:`tensormesh.mesh.Mesh`.
        It's slower than :meth:`tensormesh.assemble.NodeAssembler.from_assembler`.
        Because it will precompute the projection matrix $\mathcal P_{\mathcal V}$

        Parameters
        ----------
        mesh: tensormesh.mesh.mesh.Mesh
            a meth:`tensormesh.mesh.Mesh` object
        quadrature_order: int or None
            the order should be poisitive integer, default is :obj:`2`
        project: str
            the projection method, either "reduce" or "sparse",
            default is :obj:`"reduce"`
        *args: tuple
            Additional positional arguments passed to the assembler constructor
        **kwargs: dict
            Additional keyword arguments passed to the assembler constructor
        
        Returns
        -------
        tensormesh.assemble.NodeAssembler
            the new node assembler use the topology of the mesh
        """
        elements            = mesh.elements()
        assert isinstance(mesh.points, torch.Tensor)
        points              = mesh.points
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}

        return cls.from_elements(points, 
                                 elements, 
                                 quadrature_order, 
                                 mesh.device, 
                                 mesh.dtype,
                                 project,
                                 *args, **kwargs)
      

NodeAssembler.type.__doc__ = nn.Module.type.__doc__
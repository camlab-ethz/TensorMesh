from abc import abstractmethod
import inspect
import math
from typing import Dict, Optional, Callable, Literal

import torch
import torch.nn as nn

from .projector import ReduceProjector, SparseProjector
from ..element import Transformation, element_type2dimension
from ..nn import BufferDict
from ..mesh import Mesh
from ..vmap import vmap


class InputBroadcast:
    """Per-argument vmap mapping for :class:`NodeAssembler.forward`.

    Each attribute is the ``in_dims`` index for the corresponding vmap layer
    (element / quadrature / v), or ``None`` to broadcast over that layer.
    """
    element: Optional[int]
    quadrature: Optional[int]
    v: Optional[int]

    def __init__(self, element: bool, quadrature: bool, v: bool):
        self.element    = 0 if element else None
        self.quadrature = 0 if quadrature else None
        self.v          = 0 if v else None



class NodeAssembler(nn.Module):
    r"""Assemble an element-wise linear form into a global node vector.

    :class:`NodeAssembler` is the linear-form counterpart of
    :class:`ElementAssembler`. Override :meth:`forward` to define the
    integrand :math:`l(v) = \int_\Omega f(v)\, \mathrm{d}\Omega`; calling
    the assembler returns a 1-D :class:`torch.Tensor` of shape
    :math:`[|\mathcal V|]`, or :math:`[|\mathcal V| \times H]` for
    vector-valued problems with :math:`H` degrees of freedom per node.

    Subclasses are usually built from a mesh:

    * :meth:`from_mesh` — build from a :class:`~tensormesh.Mesh`.
    * :meth:`from_elements` — build from raw connectivity tensors.
    * :meth:`from_assembler` — share topology with another assembler.

    Examples
    --------
    Load vector :math:`f_i = \int_\Omega v_i\, \mathrm{d}\Omega`:

    .. code-block:: python

        import tensormesh
        class OneAssembler(tensormesh.NodeAssembler):
            def forward(self, v):
                return v
        mesh = tensormesh.Mesh.gen_rectangle()
        f = OneAssembler.from_mesh(mesh)(mesh.points)

    Traction-style load :math:`f_i = \int_\Omega \mathbf t \cdot v_i\, \mathrm{d}\Omega`:

    .. code-block:: python

        import tensormesh
        import tensormesh.functional as F
        class TractionAssembler(tensormesh.NodeAssembler):
            def forward(self, v, t):
                return F.dot(t, v)
        mesh = tensormesh.Mesh.gen_circle()
        t = torch.ones(mesh.n_points, 2)  # unit traction in x, y
        assembler = TractionAssembler.from_mesh(mesh)
        f = assembler(mesh.points, point_data={"t": t})

    Attributes
    ----------
    projector : torch.nn.ModuleDict
        Maps each ``element_type`` to a
        :class:`~tensormesh.assemble.projector.Projector` that scatters
        per-element basis contributions of shape :math:`[|\mathcal C_e|, B_e]`
        onto the node vector of shape :math:`[|\mathcal V|]`.
    transformation : torch.nn.ModuleDict
        Maps each ``element_type`` to a :class:`~tensormesh.Transformation`
        caching shape values, shape gradients, and ``JxW`` at quadrature points.
    elements : tensormesh.nn.BufferDict
        Maps each ``element_type`` to its connectivity tensor of shape
        :math:`[|\mathcal C|, B]`.
    n_points : int
        Number of mesh points (length of the output vector for scalar problems).
    dimension : int
        Spatial dimension of the mesh, one of ``1``, ``2``, ``3``.
    element_types : list[str]
        Element-type strings present in the mesh.
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
        """Device on which the assembler's buffers live."""
        return next(iter(self.transformation.values())).device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the assembler's buffers (``float32`` or ``float64``)."""
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
        r"""Assemble the linear form into a global node vector.

        Parameters
        ----------
        points : torch.Tensor, optional
            Nodal coordinates of shape :math:`[|\mathcal V|, D]`. If ``None``,
            the points stored in the cached :class:`Transformation` are used.
        func : Callable, optional
            Linear integrand to use *in place of* :meth:`forward`.
        point_data : dict[str, torch.Tensor], optional
            Nodal fields, each of shape :math:`[|\mathcal V|, ...]`. Keys
            can appear as ``forward`` parameters (e.g. ``"f"``) and as
            gradients (``"gradf"``).
        scalar_data : dict[str, scalar or torch.Tensor], optional
            Global scalars passed verbatim to ``forward``.
        batch_size : int, optional
            Batch size for quadrature points. Defaults to ``1`` (process
            one quadrature point at a time); pass ``-1`` to process all
            quadrature points at once, which also caches the shape gradients
            and ``JxW`` on the :class:`~tensormesh.Transformation` for later calls.

        Returns
        -------
        torch.Tensor
            1D tensor of shape :math:`[|\mathcal V|]` (scalar problems) or
            flattened ``[|\mathcal V| \times H]`` (vector problems with
            ``H`` degrees of freedom per node).
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
            # the compiled function holds this geometry's shape gradients: rebuild when it changes
            geometry = self.transformation[element_type]._geometry_version
            cache_key = f"call_{element_type}_{tuple(point_data_keys)}_{tuple(scalar_data_keys)}_{geometry}"
            
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
            if batch_size in (-1, None):
                n_batch      = 1
                n_batch_size = trans.n_quadrature
            else:
                n_batch_size = batch_size
                n_batch      = math.ceil(trans.n_quadrature / batch_size)
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
            f"    n_quadrature: {' '.join(f'{k}:{v.n_quadrature}' for k, v in self.transformation.items())}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    @abstractmethod
    def forward(self, *args):
        r"""Define the integrand of the linear form at a single quadrature point.

        Subclasses must override this method. The library uses
        :func:`torch.vmap` to lift the per-quadrature-point function over
        all quadrature points and all elements, so write it as if you were
        evaluating at *one* point.

        Parameters
        ----------
        v : torch.Tensor, optional
            Shape value at the quadrature point — 0D tensor of shape ``[]``.
        gradv : torch.Tensor, optional
            Shape gradient in physical coordinates — 1D tensor of shape ``[D]``.
        x : torch.Tensor, optional
            Physical coordinate — 1D tensor of shape ``[D]``.
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
            Either a 1D tensor of shape ``[B]`` (scalar problems) or a 2D
            tensor of shape ``[B, H]`` (vector problems with ``H`` degrees
            of freedom per node).
        """
        raise NotImplementedError("forward is not implemented")
    
    @classmethod
    def from_assembler(cls, obj, *args,**kwargs):
        r"""Build a :class:`NodeAssembler` that shares topology with ``obj``.

        Much faster than :meth:`from_mesh` since the projector and cached
        :class:`Transformation` are reused as-is.

        Parameters
        ----------
        obj : NodeAssembler
            An existing node assembler whose mesh topology should be reused.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.

        Returns
        -------
        NodeAssembler
            A new assembler sharing the same mesh.
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
        r"""Build a :class:`NodeAssembler` from raw connectivity tensors.

        Slower than :meth:`from_assembler` because the projection backend
        is built from scratch.

        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape :math:`[|\mathcal V|, D]` listing node coordinates.
        elements : dict[str, torch.Tensor]
            Connectivity keyed by element-type string, e.g.
            ``{"triangle": tensor([[0, 1, 2], [1, 2, 3]])}``.
        quadrature_order : int, optional
            Positive integer; defaults to ``2``.
        device : torch.device or str, optional
            Device of the assembler; defaults to ``"cpu"``.
        dtype : torch.dtype, optional
            Floating dtype; defaults to :obj:`torch.float32`.
        project : {'reduce', 'sparse'}, optional
            Projection backend; ``"reduce"`` (default) uses
            :meth:`torch.Tensor.index_add_` and is memory-efficient,
            ``"sparse"`` uses a sparse mat-vec product and is faster but
            uses more memory.

        Returns
        -------
        NodeAssembler
            A new assembler that owns the given topology.
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
        r"""Build a :class:`NodeAssembler` from a :class:`~tensormesh.Mesh`.

        Slower than :meth:`from_assembler` because the projection backend
        :math:`\mathcal P_{\mathcal V}` is precomputed from connectivity.

        Parameters
        ----------
        mesh : tensormesh.Mesh
            Source mesh; both connectivity and points are taken from it.
        quadrature_order : int, optional
            Positive integer; defaults to ``2``.
        project : {'reduce', 'sparse'}, optional
            Projection backend; defaults to ``"reduce"``.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.

        Returns
        -------
        NodeAssembler
            A new assembler that owns the mesh topology.
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

from abc import abstractmethod
import torch 
import torch.nn as nn
import numpy as np
import scipy.sparse
from functools import reduce, partial
import inspect

from .projector import Projector
from ..quadrature import get_quadrature
from ..shape import get_shape_val, get_shape_grad, element_type2order, element_type2dimension
from ..nn import BufferDict

class NodeAssembler(nn.Module):
    def __init__(self, 
                    quadrature_weights,
                   quadrature_points,
                   shape_val,
                   projector, 
                   elements):
        super().__init__()

        element_types = list(quadrature_weights.keys())
        dimension     = element_type2dimension[element_types[0]]

        self.quadrature_weights = quadrature_weights
        self.quadrature_points  = quadrature_points
        self.shape_val          = shape_val
        self.projector          = projector
        self.elements           = elements
        
        self.dimension          = dimension
        self.element_types      = element_types

        self.precompute()
        
    def integrate(self, batch_integral, jxw, n_element, n_basis, use_element_parallel):
        if not use_element_parallel:
            error_msg = f"the shape returned by forward function is {batch_integral.shape} which is not supported, should either be [batch_size,{n_basis}] or [batch_size,{n_basis}, dof_per_point]"
            assert batch_integral.shape[1] == n_basis, error_msg
            assert batch_integral.dim() == 2 or batch_integral.dim() == 3, error_msg
            batch_integral = torch.einsum("qi...,eq->ei...", batch_integral, jxw) # [n_element, n_basis, ...]
        else:
            error_msg = f"the shape returned by forward function is {batch_integral.shape} which is not supported, should either be [{n_element},batch_size,{n_basis}] or [{n_element},batch_size,{n_basis}, dof_per_point]"
            assert batch_integral.shape[0] == n_element, error_msg
            assert batch_integral.shape[2] == n_basis, error_msg
            assert batch_integral.dim() == 3 or batch_integral.dim() == 4, error_msg
            batch_integral = torch.einsum("eqb...,eq->eb...", batch_integral, jxw) # [n_element, n_basis, ...]
        return batch_integral
    
    def build_output(self, integral):
        return integral.flatten()
    
    @property
    def device(self):
        return self.quadrature_weights.device

    @property
    def dtype(self):
        return self.quadrature_weights.dtype

    def type(self,  dtype):
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self
    
    def __call__(self, points, func=None,point_data=None, batch_size=None):
        """
            Parameters:
            -----------
                points: torch.Tensor of shape [n_point, n_dim]
                    the coordinates of the points
                func: function|None when it's None the forward function will be used
                point_data: Dict[str, torch.Tensor] of shape [n_point, ...]
                batch_size: int|None
                    the batch size of quadrature points
                    if int is given, the quadrature points will be divided into batches
                    if None is given, the quadrature points will not be divided into batches
            Returns:
            --------
                A: (edata, col, row)
        """
        if point_data is None:
            point_data = {}
        point_data["x"] = points

        self = self.type(points.dtype).to(points.device)

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], f"the shape of {key} should be [n_point, ...], but got {value.shape}"
 

        signature = inspect.signature(self.forward)

        fn = None
        
        use_element_parallel = None

        integral = None
      
        for element_type in self.element_types:
            element_integral = None
            n_quadrature = self.quadrature_weights[element_type].shape[0]
            n_batch      = n_quadrature // batch_size if batch_size is not None else 1
            n_batch_size = batch_size if batch_size is not None else n_quadrature
            n_basis      = self.shape_val[element_type].shape[1]
            n_element    = self.elements[element_type].shape[0]
            ele_point_data = {k:v[self.elements[element_type]] for k,v in point_data.items()}
            ele_coords   = points[self.elements[element_type]] # [n_element, n_basis, n_dim]
            for i in range(n_batch):
                shape_val = self.shape_val[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size, n_basis]
                w         = self.quadrature_weights[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size]
                quadrature_points = self.quadrature_points[element_type][i * n_batch_size: (i+1) * n_batch_size] # [batch_size, n_dim]
                shape_grad, jxw = get_shape_grad(element_type, w, quadrature_points, ele_coords) # [n_element, batch_size, n_basis, n_dim], [n_element, n_batch]
                
                # prepare arguments
                args = []
                for key in signature.parameters:
                    if key in ["u", "v"]:
                        args.append(shape_val)
                    elif key in ["gradu", "gradv"]:
                        args.append(shape_grad)
                    elif key in ele_point_data:
                        args.append(torch.einsum("eb...,qb->eqb...",ele_point_data[key], shape_val))
                    elif key.startswith("grad") and key[4:] in ele_point_data:
                        args.append(torch.einsum("eb...,eqbd->eqb...d",ele_point_data[key[4:]], shape_grad))
                    else:
                        raise NotImplementedError(f"key {key} is not implemented")



                # parallel dispatch 

                if fn is None:
                    element_dims = []
                    quadrature_dims = []
                    for key in signature.parameters:
                        if key in ["u", "v"]:
                            element_dims.append(None)
                            quadrature_dims.append(0)
                        else:
                            element_dims.append(0)
                            quadrature_dims.append(0)
                    
                    element_dims = tuple(element_dims)
                    quadrature_dims = tuple(quadrature_dims)

                    fn = self.forward if func is None else func
                   
                    if all([x is None for x in element_dims]):
                        # if all is shape_val
                        fn = torch.vmap(fn, in_dims=quadrature_dims)
                        use_element_parallel = False
                    else:
                        fn = torch.vmap(
                            torch.vmap(
                                fn,
                                in_dims = quadrature_dims
                            ),
                            in_dims=element_dims
                        )
                        use_element_parallel = True

                batch_integral = fn(*args) # [n_element, batch_size, n_basis, n_basis, ...] or [n_batch, batch_size, n_basis, ...]

                batch_integral = self.integrate(batch_integral, jxw, n_element, n_basis, use_element_parallel)

                if element_integral is None:
                    element_integral = batch_integral
                else:
                    element_integral += batch_integral
    
            if integral is None:
                integral = self.projector[element_type](element_integral) # [n_edge, ...]
            else:
                integral += self.projector[element_type](element_integral) # [n_edge, ...]

        return self.build_output(integral)

    def precompute(self):
        pass

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
        raise NotImplementedError(f"forward is not implemented")
    
    @classmethod
    def from_assembler(cls, obj):
        assert isinstance(obj, ElementAssembler)
        return cls(
                obj.quadrature_weights,
                obj.quadrature_points,
                obj.shape_val,
                obj.projector, 
                obj.elements
        )

    @classmethod
    def from_mesh(cls, mesh,  quadrature_order=None):
        elements = mesh.elements()
        n_points = mesh.points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}

        quadrature_weights = {}
        quadrature_points  = {}
        shape_val          = {}
        projector          = {}
        
        for element_type, value in elements.items():
            n_element, n_basis = value.shape
            quadrature_weights[element_type], quadrature_points[element_type] =\
            get_quadrature(element_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
            shape_val[element_type] = get_shape_val(element_type, quadrature_points[element_type]) # [n_quadrature, n_basis]
            projector[element_type] = Projector(
                from_ = torch.arange(n_element * n_basis),
                to_   = value.flatten(),
                from_shape = (n_element, n_basis),
                to_shape   = (n_points,)
            )

        quadrature_weights = BufferDict(quadrature_weights)
        quadrature_points  = BufferDict(quadrature_points)
        shape_val          = BufferDict(shape_val)
        projector          = BufferDict(projector)
        elements           = BufferDict(elements)

        return cls(quadrature_weights,
                   quadrature_points,
                   shape_val,
                   projector, 
                   elements)

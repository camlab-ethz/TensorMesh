from abc import ABC, abstractmethod
import torch 
import torch.nn as nn
import numpy as np
import scipy.sparse
from functools import reduce, partial
import inspect

from ..quadrature import get_quadrature
from ..shape import get_shape_val, get_shape_grad
from ..sparse import SparseMatrix
from ..utils import is_float

class Projector(nn.Module):
    def __init__(self, from_, to_, from_shape, to_shape, dtype = None):
        """
            Parameters:
            -----------
                from_: torch.tensor[n_edges]
                to_: torch.tensor[n_edges]
                from_shape: tuple
                to_shape: tuple 
        """
        super().__init__()
        if dtype is None:
            dtype = from_.dtype
        projection = torch.sparse_coo_tensor(
            torch.stack([to_,from_],0),
            torch.ones_like(from_,dtype=dtype),
            size = (np.prod(to_shape), np.prod(from_shape))
        ).to_sparse_csr()
        self.register_buffer("projection", projection)
        self.from_shape = from_shape
        self.to_shape   = to_shape

    def type(self, dtype):
        if dtype != self.dtype:
            self.projection = self.projection.type(dtype)
        return self

    @property
    def device(self):
        return self.projection.device

    @property
    def dtype(self):
        return self.projection.dtype

    def __call__(self, x):
        """
            Parameters:
            -----------
                x: torch.tensor[*from_shape, ...]
            Returns:
            --------
                y: torch.tensor[*to_shape, ....]
        """
        assert self.dtype == x.dtype, f"the dtype of x must be {self.dtype}, but got {x.dtype}"
        assert self.device == x.device, f"the device of x must be {self.device}, but got {x.device}"
        assert x.shape[:len(self.from_shape)] == self.from_shape, f"the shape of x must be [{self.from_shape}, ...], but got {x.shape}"

        dim_shape = x.shape[len(self.from_shape):]
        x = x.reshape(np.prod(self.from_shape), -1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            x = (self.projection @ x).squeeze(-1)
        else:
            x = self.projection @ x
        x = x.reshape(*self.to_shape, *dim_shape)
        return x

class ElementAssembler(nn.Module):
    def __init__(self, elements, n_point, cell_type, quadrature_order=1):
        """
            Parameters:
            -----------
                elements: torch.Tensor of shape [n_elements, n_basis]
                    the cells of the mesh
                n_point: int
                    the number of points in the mesh
                cell_type: str
                    the type of the element, e.g., 'triangle', 'tetra', 'hexa'
                quadrature_order: int
                    the order of quadrature points
        """
        super().__init__()
        quadrature_weights, quadrature_points =\
            get_quadrature(cell_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
        n_quadrature = quadrature_points.shape[0]

        shape_val = get_shape_val(cell_type, quadrature_points) # [n_quadrature, n_basis]
        # gen ele2msh_edge
        n_element, n_basis = elements.shape
        elem_u, elem_v = [], []
        for i in range(n_basis):
            for j in range(n_basis):
                elem_u.append(elements[:, i])
                elem_v.append(elements[:, j])
        elem_u, elem_v = torch.stack(elem_u, -1).flatten(), torch.stack(elem_v, -1).flatten() # [num_elements * num_basis * num_basis]
        elem_u, elem_v = elem_u.cpu().numpy().copy(), elem_v.cpu().numpy().copy()
        tmp = scipy.sparse.coo_matrix(( # used to remove duplicated edges
            np.ones_like(elem_u), # data
            (elem_u, elem_v), # (row, col)
        ), shape = (n_point,  n_point)).tocsr().tocoo()
        edge_u, edge_v = tmp.row, tmp.col
        num_edges  = len(edge_u)
        eids_csr = scipy.sparse.coo_matrix((
            np.arange(num_edges), (edge_u, edge_v)
        ), shape=(n_point, n_point)).tocsr()
     
        elem_eids     = np.array(eids_csr[elem_u, elem_v].copy()).ravel()

        self.register_buffer("quadrature_weights", quadrature_weights)
        self.register_buffer("quadrature_points", quadrature_points)
        self.register_buffer("shape_val", shape_val)
        self.register_buffer("elements", elements)
        self.cell_type    = cell_type
        self.n_dim        = self.quadrature_points.shape[1]
        self.n_quadrature = n_quadrature 
        self.n_element    = n_element
        self.n_point      = n_point
        self.n_basis      = n_basis
        self.ele2msh_edge = Projector(
            from_ = torch.arange(n_element * n_basis * n_basis), 
            to_    = torch.from_numpy(elem_eids),
            from_shape = (n_element, n_basis, n_basis), 
            to_shape = (num_edges,),
        ).type(quadrature_weights.dtype)
        self.register_buffer("row", torch.from_numpy(edge_u))
        self.register_buffer("col", torch.from_numpy(edge_v))

        self.precompute()

    @property
    def device(self):
        return self.quadrature_weights.device

    @property
    def dtype(self):
        return self.quadrature_weights.dtype

    def type(self, dtype):
        if dtype != self.dtype:
            for name, buffer in self._buffers.items():
                if is_float(buffer):
                    self.register_buffer(name, buffer.type(dtype))
            self.ele2msh_edge = self.ele2msh_edge.type(dtype)
        return  self

    def __call__(self, points):
        """
            Parameters:
            -----------
                points: torch.Tensor of shape [n_point, n_dim]
                    the coordinates of the points
            Returns:
            --------
                A: (edata, col, row)
        """
        self = self.type(points.dtype).to(points.device)

        quadrature_weights = self.quadrature_weights 
        quadrature_points  = self.quadrature_points
        projector          = self.ele2msh_edge
        shape_val          = self.shape_val

        valid_keys = {"u","v","gradu","gradv"}
        signature = inspect.signature(self.forward)
        for key in signature.parameters.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid argument {key} in forward function. Expect one of {valid_keys}")
        
        element_coords = points[self.elements]
        shape_grad, jxw = get_shape_grad(self.cell_type, quadrature_weights, quadrature_points, element_coords) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature]
        
        element_dims    = []
        quadrature_dims = []
        arguments       = []
        for key in signature.parameters.keys():
            if key in ["u", "v"]:
                element_dims.append(None)
                quadrature_dims.append(0)
                arguments.append(shape_val)
            elif key in ["gradu", "gradv"]:
                element_dims.append(0)
                quadrature_dims.append(0)
                arguments.append(shape_grad)
            else:
                raise NotImplementedError(f"key {key} is not implemented")
        element_dims    = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)


        # fn = partial(self.forward, self=self)
        fn = self.forward

        if all([x is None for x in element_dims]):
            # if all is shape_val
            integral = torch.vmap(fn, in_dims=quadrature_dims)(*arguments) # [n_quadrature, n_basis, n_basis, ...]
            assert integral.shape[:3] == (self.n_quadrature, self.n_basis, self.n_basis), f"the shape returned by forward function is {[*integral.shape]} which is not supported, should either be [{self.n_quadrature},{self.n_basis},{self.n_basis}] or [{self.n_quadrature},{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"
            assert integral.dim() == 3 or integral.dim() == 5, f"the shape returned by forward function is {[*integral.shape[1:]]} which is not supported, should either be [{self.n_basis},{self.n_basis}] or [{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"

            integral = torch.einsum("qij...,eq->eij...", integral, jxw) # [n_element, n_basis, n_basis, ...]

        else:
            parallel_fn = torch.vmap(
                torch.vmap(
                    fn,
                    in_dims = quadrature_dims
                ),
                in_dims=element_dims
            )

            integral = parallel_fn(*arguments) # [n_element, n_quadrature, n_basis, n_basis, ...]

            assert integral.shape[:4] == (self.n_element, self.n_quadrature, self.n_basis, self.n_basis), f"the shape returned by forward function is {[*integral.shape[4:]]} which is not supported, should either be [{self.n_element}, {self.n_quadrature}, {self.n_basis},{self.n_basis}] or [{self.n_element}, {self.n_quadrature},{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"
            assert integral.dim() == 4 or integral.dim() == 6, f"the shape returned by forward function is {[self.n_basis, self.n_basis, *integral.shape[4:]]} which is not supported, should either be [{self.n_basis},{self.n_basis}] or [{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"
            if integral.dim() == 6:
                assert integral.shape[-1] == integral.shape[-2], f"the shape returned by forward function is {[self.n_basis, self.n_basis, *integral.shape[4:]]} which is not supported, should either be [{self.n_basis},{self.n_basis}] or [{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"


            integral = torch.einsum("eqij...,eq->eij...", integral, jxw) # [n_element, n_basis, n_basis, ...]

        integral = projector(integral) # [n_edge, ...]

        if integral.dim() == 1:
            return SparseMatrix(integral, self.row, self.col, shape=(points.shape[0], points.shape[0]))
        elif integral.dim() ==  3:
            return SparseMatrix.from_block_coo(integral, self.row, self.col, shape=(points.shape[0], points.shape[0]))
        else:
            raise Exception(f"the shape returned by forward function is {[self.n_basis, self.n_basis,*integral.shape[1:]]} which is not supported, should either be [{self.n_basis},{self.n_basis}] or [{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]")

    @abstractmethod
    def forward(self, gradu, gradv):
        """The weak form of the operator
            Parameters:
            -----------
                gradu: torch.Tensor of shape [n_basis, n_dim]
                    the gradient of the test function
                gradv: torch.Tensor of shape [n_basis, n_dim]
                    the gradient of the trial function
            Returns:
            --------
                torch.Tensor of shape [n_basis, n_basis]
        """
        raise NotImplementedError(f"forward is not implemented")
        return gradu @ gradv
    
    def precompute(self):
        pass

    def __str__(self):
        return (
            f"ElementAssembler(\n"
            f"    cell_type: {self.cell_type}\n"
            f"    n_element: {self.n_element}\n"
            f"    n_point: {self.n_point}\n"
            f"    n_basis: {self.n_basis}\n"
            f"    n_dim: {self.n_dim}\n"
            f"    n_quadrature: {self.n_quadrature}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    @classmethod
    def from_mesh(cls, mesh, cell_type = None, quadrature_order=1):
        elements = mesh.elements(cell_type)
        n_point  = mesh.n_point
        if cell_type is None:
            cell_type = mesh.default_cell_type
        return cls(elements, n_point, cell_type, quadrature_order)



class NodeAssembler(nn.Module):
    def __init__(self, elements, n_point, cell_type, quadrature_order=1):
        super().__init__()
        quadrature_weights, quadrature_points =\
            get_quadrature(cell_type, quadrature_order) # [n_quadrature], [n_quadrature, n_dim]
        n_quadrature = quadrature_points.shape[0]

        shape_val = get_shape_val(cell_type, quadrature_points) # [n_quadrature, n_basis]

        n_element, n_basis = elements.shape


        self.register_buffer("quadrature_weights", quadrature_weights)
        self.register_buffer("quadrature_points", quadrature_points)
        self.register_buffer("shape_val", shape_val)
        self.register_buffer("elements", elements)
        self.cell_type   = cell_type
        self.n_dim       = self.quadrature_points.shape[1]
        self.n_element   = n_element
        self.n_basis     = n_basis
        self.n_quadrature = self.quadrature_points.shape[0]
        self.n_point     = n_point

        self.ele2msh_node = Projector(
            from_ = torch.arange(n_element * n_basis),
            to_   = elements.flatten(),
            from_shape = (n_element, n_basis),
            to_shape   = (n_point,)
        )

        self.precompute()

    @property
    def dtype(self):
        return self.quadrature_weights.dtype
    
    @property
    def device(self):
        return self.quadrature_weights.device
    
    def type(self, dtype):
        if dtype != self.dtype:
            for name, buffer in self.named_buffers():
                if is_float(buffer):
                    self.register_buffer(name, buffer.type(dtype))
            self.ele2msh_node = self.ele2msh_node.type(dtype)
        
        return self

    def __call__(self, points, point_data=None):
        """
            Parameters:
            -----------
                points: torch.Tensor of shape [n_point, n_dim]
                    the coordinates of the points
            Returns:
            --------
                x: (n_point, ...)
        """
        shape_val = self.shape_val.type(points.dtype).to(points.device)         # [n_quadrature, n_basis]
        projector = self.ele2msh_node.type(points.dtype).to(points.device)
        self.quadrature_points = self.quadrature_points.type(points.dtype).to(points.device)
        self.quadrature_weights = self.quadrature_weights.type(points.dtype).to(points.device)


        valid_keys = {"x","u","v","gradu","gradv"} 
        if isinstance(point_data, dict):
            point_data = {k: v[self.elements].type(points.dtype).to(points.device) for k, v in point_data.items()}
            valid_keys.update(point_data.keys())

        signature = inspect.signature(self.forward)
        for key in signature.parameters.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid argument {key} in forward function. Expect one of {valid_keys}")
        
       
        element_coords = points[self.elements]
        shape_grad, jxw = get_shape_grad(self.cell_type, self.quadrature_weights, self.quadrature_points, element_coords) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature]
        
        element_dims    = []
        quadrature_dims = []
        arguments       = []

        for key in signature.parameters.keys():
            if key in ["u", "v"]:
                element_dims.append(None)
                quadrature_dims.append(0)
                arguments.append(shape_val)
            elif key in ["gradu", "gradv"]:
                element_dims.append(0)
                quadrature_dims.append(0)
                arguments.append(shape_grad)
            else:
                raise NotImplementedError(f"key {key} is not implemented")
            
        fn = self.forward

        element_dims    = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)

        if all([x is None for x in element_dims]):
            # if all is shape_val
            integral = torch.vmap(fn, in_dims=quadrature_dims)(*arguments) # [n_quadrature, n_basis ...]
            assert integral.shape[:2] == (self.n_quadrature, self.n_basis), f"the shape returned by forward function is {[*integral.shape[1:]]} which is not supported, should either be [{self.n_basis}] or [{self.n_basis}, dof_per_point]"
            assert integral.dim() == 2 or integral.dim() == 3, f"the shape returned by forward function is {[*integral.shape[1:]]} which is not supported, should either be [{self.n_basis}] or [{self.n_basis}, dof_per_point]"

            integral = torch.einsum("qi...,eq->ei...", integral, jxw) # [n_element, n_basis, ...]

        else:
            parallel_fn = torch.vmap(
                torch.vmap(
                    fn,
                    in_dims = quadrature_dims
                ),
                in_dims=element_dims
            )

            integral = parallel_fn(*arguments) # [n_element, n_quadrature, n_basis, ...]

            assert integral.shape[:3] == (self.n_element, self.n_quadrature, self.n_basis), f"the shape returned by forward function is {integral.shape} which is not supported, should either be [{self.n_element},{self.n_quadrature},{self.n_basis},{self.n_basis}] or [{self.n_element},{self.n_quadrature},{self.n_basis},{self.n_basis}, dof_per_point, dof_per_point]"
            assert integral.dim() == 3 or integral.dim() == 4, f"the shape returned by forward function is {[self.n_basis,*integral.shape[3:]]} which is not supported, should either be [{self.n_basis}] or [{self.n_basis}, dof_per_point]"

            integral = torch.einsum("eqb...,eq->eb...", integral, jxw) # [n_element, n_basis, ...]

        integral = projector(integral) # [n_node, ...]

        return integral.flatten()
        
    def __str__(self):
        return (
            f"NodeAssembler(\n"
            f"    cell_type: {self.cell_type}\n"
            f"    n_element: {self.n_element}\n"
            f"    n_point: {self.n_point}\n"
            f"    n_basis: {self.n_basis}\n"
            f"    n_dim: {self.n_dim}\n"
            f"    n_quadrature: {self.n_quadrature}\n"
            f"    forward: \n{inspect.getsource(self.forward)}"
            f")"
        )
    
    def __repr__(self):
        return str(self)

    def forward(self, *args):
        raise NotImplementedError(f"forward is not implemented")
    
    def precompute(self):
        pass

    @classmethod
    def from_mesh(cls, mesh, cell_type = None, quadrature_order=1):
        elements = mesh.elements(cell_type)
        n_point  = mesh.n_point
        if cell_type is None:
            cell_type = mesh.default_cell_type
        return cls(elements, n_point, cell_type, quadrature_order)

import torch
import torch.nn as nn
from typing import Optional, Tuple, Type, Union
from .types import Tensorx1, Tensorx2, Tensorx3, Tensorx4
from .element import Element
from .element_type import element_type2element, element_type2order


class Transformation(nn.Module):
    """Reference-to-physical mapping for one element-type block of a mesh.

    Holds the per-element Jacobian, shape-function values, shape-function
    gradients, quadrature points/weights, and their facet analogues — all the
    geometric ingredients an assembler needs to integrate a weak form.
    Subclass of :class:`torch.nn.Module`, so every cached tensor is a buffer
    and follows the module's ``.to()``/``.cuda()``/``.double()``.

    Examples
    --------
    .. code-block:: python

        import tensormesh as tm

        mesh = tm.Mesh.gen_rectangle(chara_length=0.1)        # triangle mesh
        transform = tm.Transformation(
            mesh.points,
            mesh.cells["triangle"],
            element_type="triangle",
            quadrature_order=2,
        )

        # Geometric quantities at quadrature points
        J   = transform.jacobian      # [n_elem, n_q, dim, dim]
        phi = transform.shape_val     # [n_q, n_basis]
        gphi= transform.shape_grad    # [n_elem, n_q, n_basis, dim]
        jxw = transform.JxW           # [n_elem, n_q]  (|det J| · w)

    Attributes
    ----------
    elements : torch.Tensor
        Element connectivity tensor of shape :math:`[N_e, N_b]` where:
        
        * :math:`N_e` = number of elements
        * :math:`N_b` = number of basis functions per element
        
    points : torch.Tensor 
        Point coordinates tensor of shape :math:`[N_p, D]` where:
        
        * :math:`N_p` = number of points
        * :math:`D` = spatial dimension
        
    element : Type[Element]
        Element type class (e.g. Triangle, Tetrahedron)
        
    basis_order : int
        Order of basis functions
        
    quadrature_order : int
        Order of quadrature rule
    """

    elements:torch.Tensor 
    """Element connectivity tensor of shape :math:`[N_e, N_b]` where:
    
    * :math:`N_e` = number of elements
    * :math:`N_b` = number of basis functions per element
    
    :no-index:
    """

    points:torch.Tensor
    """Point coordinates tensor of shape :math:`[N_p, D]` where:
    
    * :math:`N_p` = number of points
    * :math:`D` = spatial dimension
    
    :no-index:
    """

    element:Type[Element]
    """Element type class (e.g. Triangle, Tetrahedron)
    
    :no-index:
    """

    basis_order:int
    """Order of basis functions
    
    :no-index:
    """

    quadrature_order:int
    """Order of quadrature rule
    
    :no-index:
    """

    def __init__(self, 
                 points:torch.Tensor,
                 elements:torch.Tensor,
                 element_type:str, 
                 quadrature_order:int=2):
        """
        Initialize transformation.


        Examples
        --------
        .. code-block:: python

            # Single linear triangle
            points   = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
            elements = torch.tensor([[0, 1, 2]])
            transform = Transformation(points, elements, "triangle", quadrature_order=2)

        Parameters
        ----------
        points : torch.Tensor
            Point coordinates tensor of shape :math:`[N_p, D]`.
        elements : torch.Tensor
            Element connectivity tensor of shape :math:`[N_e, N_b]`.
        element_type : str
            Element type string (e.g. ``"triangle"``, ``"triangle6"``,
            ``"tetra"``, ``"hexahedron27"``). See
            :data:`tensormesh.element_types` for the full list.
        quadrature_order : int, optional
            Order of the quadrature rule, by default 2.
        """
        super().__init__()
        self.register_buffer("elements", elements)
        self.element:Type[Element] = element_type2element(element_type)
        self.basis_order:int = element_type2order[element_type]
        self.quadrature_order= quadrature_order
        
        self.update_points(points)

    def update_points(self, points:torch.Tensor):
        """Replace the cached node coordinates and reset derived buffers.

        Use this after deforming the mesh — e.g. in a moving-mesh or
        topology-optimization loop — when the connectivity does not change
        but the geometry does.

        Examples
        --------
        .. code-block:: python

            new_points = transform.points + displacement
            transform.update_points(new_points)

        Parameters
        ----------
        points : torch.Tensor
            Point coordinates tensor of shape :math:`[N_p, D]` where
            :math:`N_p` is the number of points and :math:`D` the spatial
            dimension.
        """
        dtype, device            = points.dtype, points.device
        self.register_buffer("points", points)
        self.n_points            = points.shape[0] 
        # self.element_coords      = self.points[self.elements]


    @property
    def dim(self)->int:
        """Get the spatial dimension of the mesh.

        Returns
        -------
        int
            The spatial dimension :math:`D` (1, 2, or 3)
        """
        return self.points.shape[-1] # type:ignore

    @property 
    def device(self)->torch.device:
        """Get the device of the mesh tensors.

        Returns
        -------
        torch.device
            The device (CPU/GPU) where the mesh tensors are stored
        """
        return self.points.device
    
    @property
    def dtype(self)->torch.dtype:
        """Get the data type of the mesh tensors.

        Returns
        -------
        torch.dtype
            The data type (e.g. torch.float32) of the mesh tensors
        """
        return self.points.dtype

    def _apply(self, fn, recurse: bool = True):
        # Keep geometry buffers (points / jacobians / quadrature) real even
        # when the enclosing assembler is cast to a complex dtype. The
        # bilinear form's complex content enters through ``point_data`` /
        # ``element_data`` coefficients (multiplied in the assembler's
        # ``forward``), not through the mesh geometry itself — see
        # ROADMAP item 2.
        def _real_only(t):
            out = fn(t)
            if out.is_complex():
                real_dtype = (
                    torch.float32 if out.dtype == torch.complex64 else torch.float64
                )
                out = out.real.to(real_dtype)
            return out
        return super()._apply(_real_only, recurse=recurse)
    
    @property
    def element_coords(self)->torch.Tensor:
        """Get the physical coordinates of element nodes.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            coords = transform.element_coords
            # coords[e]    -> [n_basis, dim] coordinates of all nodes in element e
            # coords[e, b] -> [dim] coordinate of basis node b in element e

        Returns
        -------
        torch.Tensor
            Element node coordinates tensor of shape :math:`[N_e, N_b, D]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_b` = number of basis functions per element  
            * :math:`D` = spatial dimension
        """
        return self.points[self.elements] # type:ignore [n_element, n_basis, dim]
    
    @property
    def n_elements(self)->int:
        """Get the number of elements in the mesh.

        Returns
        -------
        int
            Number of elements :math:`N_e`
        """
        return self.elements.shape[0]

    @property
    def basis(self)->Tensorx1:
        """Get the basis function nodes in reference coordinates.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle", basis_order=2)
            basis = transform.basis  # Get quadratic basis nodes
            # For triangles: 6 nodes (3 vertices + 3 edge midpoints)

        Returns
        -------
        torch.Tensor
            Basis node coordinates tensor of shape :math:`[N_b, D]` where:
            
            * :math:`N_b` = number of basis functions
            * :math:`D` = spatial dimension
        """
        return self.element.get_basis(self.basis_order).type(self.dtype).to(self.device) # [n_basis, dim]
    
    @property
    def n_basis(self)->int:
        """Get the number of basis functions per element.

        Returns
        -------
        int
            Number of basis functions :math:`N_b`
        """
        return self.basis.shape[0]

    @property 
    def quadrature(self)->Tensorx2:
        """Get quadrature weights and points.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle", quadrature_order=2)
            weights, points = transform.quadrature
            # weights: quadrature weights for numerical integration
            # points: quadrature point coordinates in reference element

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Quadrature weights tensor of shape :math:`[N_q]`
            - Quadrature points tensor of shape :math:`[N_q, D]`
            
            where:
            
            * :math:`N_q` = number of quadrature points
            * :math:`D` = spatial dimension
        """
        if "_quadrature_w" not in self._buffers: # type: ignore
            w, q =  self.element.get_quadrature(self.quadrature_order, self.dtype, self.device) # [n_quadrature], [n_quadrature, dim]
            self.register_buffer("_quadrature_w", w)
            self.register_buffer("_quadrature_q", q)
        return self._quadrature_w, self._quadrature_q # type: ignore [n_quadrature], [n_quadrature, dim]
    
    @property
    def n_quadrature(self)->int:
        """Get the number of quadrature points per element.

        Returns
        -------
        int
            Number of quadrature points :math:`N_q`
        """
        return self.quadrature[0].shape[0]

    @property 
    def shape_val(self)->Tensorx1:
        r"""Get shape function values at quadrature points.

        The shape functions :math:`N_i(\boldsymbol{\xi})` are evaluated at the quadrature points
        in the reference element.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            N = transform.shape_val
            # N[i,j] is value of jth shape function at ith quadrature point

        Returns
        -------
        torch.Tensor
            Shape function values tensor of shape :math:`[N_q, N_b]` where:
            
            * :math:`N_q` = number of quadrature points
            * :math:`N_b` = number of basis functions
        """
        if "_shape_val" not in self._buffers: # type: ignore
            _, p      = self.quadrature
            shape_val = self.element.eval_shape_val(p, self.basis_order, self.quadrature_order)
            self.register_buffer("_shape_val", shape_val)
        return self._shape_val  # type:ignore [n_quadrature, n_basis]

    @property 
    def shape_grad(self)->Tensorx1:
        r"""Get shape function gradients at quadrature points.

        The shape function gradients :math:`\nabla N_i(\mathbf{x})` are evaluated at the quadrature points
        in physical coordinates.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            dN = transform.shape_grad
            # dN[e,q,i,d] is derivative of ith shape function w.r.t.
            # dth coordinate at qth quadrature point in eth element

        Returns
        -------
        torch.Tensor
            Shape function gradients tensor of shape :math:`[N_e, N_q, N_b, D]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_q` = number of quadrature points
            * :math:`N_b` = number of basis functions
            * :math:`D` = spatial dimension
        """
        if "_shape_grad" not in self._buffers: # type: ignore
            _, q = self.quadrature
            shape_grad, jacobian = self.element.eval_shape_grad(
                self.element_coords, q, self.basis_order, self.quadrature_order)
            self.register_buffer("_shape_grad", shape_grad)
            self.register_buffer("_jacobian", jacobian)
        return self._shape_grad 
    
    @property
    def jacobian(self)->Tensorx1:
        r"""Get Jacobian matrices at basis points,

        The Jacobian matrix :math:`\mathbf{J} = \frac{\partial\mathbf{x}}{\partial\boldsymbol{\xi}}`
        maps derivatives from reference to physical coordinates.

        The Jacobian matrix maps derivatives between reference and physical coordinates:

        .. math::
            \frac{\partial}{\partial x_i} = \sum_{j=1}^D J^{-1}_{ij} \frac{\partial}{\partial \xi_j}

        where :math:`J_{ij} = \frac{\partial x_i}{\partial \xi_j}` and :math:`D` is the spatial dimension.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            J = transform.jacobian
            # J[e,q,:,:] is Jacobian matrix at qth quadrature point
            # in eth element

        Returns
        -------
        torch.Tensor
            Jacobian matrices tensor of shape :math:`[N_e, N_q, D, D]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_q` = number of quadrature points 
            * :math:`D` = spatial dimension
        """
        if "_jacobian" not in self._buffers: # type: ignore
            self.shape_grad
        return self._jacobian # type:ignore [n_element, n_quadrature, dim, dim]
    
    @property 
    def facets(self)->Union[Tensorx1,Tensorx2]:
        """Get element facet connectivity.

        For elements with uniform facets (e.g. all triangles or all quads), returns a single tensor.
        For elements with mixed facets (e.g. prism or pyramid), returns separate tensors for triangular 
        and quadrilateral facets.

        Examples
        --------
        .. code-block:: python

            # For a tetrahedral mesh (uniform triangular facets)
            transform = Transformation(points, elements, "tetra")
            facets = transform.facets  # Single tensor of triangular facets
            
            # For a prism mesh (mixed tri/quad facets)
            transform = Transformation(points, elements, "wedge") 
            tri_facets, quad_facets = transform.facets  # Separate tensors

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Facet connectivity tensor of shape :math:`[N_f, N_b]` where:

                * :math:`N_f` = number of facets
                * :math:`N_b` = number of basis functions per facet

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet tensor of shape :math:`[N_{tf}, N_{tb}]`
                * Quadrilateral facet tensor of shape :math:`[N_{qf}, N_{qb}]`
                
                where:

                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tb}` = number of basis functions per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets  
                * :math:`N_{qb}` = number of basis functions per quadrilateral facet
        """
        if self.element.is_mix_facet:
            tri_facets, quad_facets = self.element.element_to_facet(self.elements, self.basis_order) # type: ignore
            return (tri_facets, quad_facets)
        else:
            facets = self.element.element_to_facet(self.elements, self.basis_order) # type: ignore
            assert isinstance(facets, torch.Tensor)
            return facets

    @property
    def facet_quadrature(self)->Union[Tensorx2,Tensorx4]:
        """Get quadrature weights and points for element facets.

        Provides quadrature rules for numerical integration over element facets.
        For mixed facet elements, returns separate quadrature rules for triangular
        and quadrilateral facets.

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            weights, points = transform.facet_quadrature
            
            # For prism mesh (mixed facets)
            transform = Transformation(points, elements, "wedge")
            tri_w, tri_p, quad_w, quad_p = transform.facet_quadrature

        Returns
        -------
        If element has uniform facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Weights tensor of shape :math:`[N_f, N_q]`
                * Points tensor of shape :math:`[N_f, N_q, D]`
                
                where:

                * :math:`N_f` = number of facets
                * :math:`N_q` = number of quadrature points per facet
                * :math:`D` = spatial dimension

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                * Triangular facet weights of shape :math:`[N_{tf}, N_{tq}]`
                * Triangular facet points of shape :math:`[N_{tf}, N_{tq}, D]`
                * Quadrilateral facet weights of shape :math:`[N_{qf}, N_{qq}]`
                * Quadrilateral facet points of shape :math:`[N_{qf}, N_{qq}, D]`
                
                where:

                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tq}` = quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets
                * :math:`N_{qq}` = quadrature points per quadrilateral facet
                * :math:`D` = spatial dimension
        """
        if self.element.is_mix_facet:
            tri_m, tri_q, quad_m, quad_q, tri_mask = self.element.get_facet_quadrature(self.quadrature_order, transform=True) # type:ignore
            tri_m = tri_m.type(self.dtype).to(self.device)
            tri_q = tri_q.type(self.dtype).to(self.device)
            quad_m= quad_m.type(self.dtype).to(self.device)
            quad_q= quad_q.type(self.dtype).to(self.device)
            return (tri_m, tri_q, quad_m, quad_q)
        else:
            m, q = self.element.get_facet_quadrature(self.quadrature_order, transform=True) # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
            m = m.type(self.dtype).to(self.device)
            q = q.type(self.dtype).to(self.device)
            return m, q
        
    @property 
    def facet_shape_val(self)->Union[Tensorx1,Tensorx2]:
        r"""Get shape function values at facet quadrature points.

        Evaluates the element shape functions at the quadrature points on each facet.
        For mixed facet elements, returns separate evaluations for triangular and
        quadrilateral facets.

        The shape functions :math:`\phi_i(\mathbf{x})` are evaluated at quadrature points on each facet:

        .. math::

            \phi_i(\mathbf{x}_q^f) = \phi_i(\mathbf{F}^f(\mathbf{r}_q))

        where:

        * :math:`\mathbf{x}_q^f` = physical coordinates of quadrature point :math:`q` on facet :math:`f`
        * :math:`\mathbf{F}^f` = mapping from reference to physical facet
        * :math:`\mathbf{r}_q` = reference coordinates of quadrature point :math:`q`

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            shape_vals = transform.facet_shape_val
            
            # For prism mesh (mixed facets)
            transform = Transformation(points, elements, "wedge")
            tri_vals, quad_vals = transform.facet_shape_val

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Shape function values tensor of shape :math:`[N_f, N_q, N_b]` where:

                * :math:`N_f` = number of facets
                * :math:`N_q` = number of quadrature points per facet
                * :math:`N_b` = number of basis functions

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet values of shape :math:`[N_{tf}, N_{tq}, N_b]`
                * Quadrilateral facet values of shape :math:`[N_{qf}, N_{qq}, N_b]`
                
                where:

                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tq}` = quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets
                * :math:`N_{qq}` = quadrature points per quadrilateral facet
                * :math:`N_b` = number of basis functions
        """

        if self.element.is_mix_facet:
            if "_facet_shape_val_tri" not in self._buffers: # type: ignore
                tri_m, tri_p, quad_m, quad_p, tri_mask = self.facet_quadrature # type:ignore
                n_tri_facet, n_quadrature_per_tri_facet, _   = tri_p.shape
                n_quad_facet, n_quadrature_per_quad_facet, _ = quad_p.shape
                tri_shape_val = self.element.eval_shape_val(tri_p, self.basis_order, self.quadrature_order)
                quad_shape_val= self.element.eval_shape_val(quad_p, self.basis_order, self.quadrature_order)
                tri_shape_val = tri_shape_val.reshape(n_tri_facet, n_quadrature_per_tri_facet, self.n_basis)
                quad_shape_val= quad_shape_val.reshape(n_quad_facet, n_quadrature_per_quad_facet, self.n_basis)

                self.register_buffer("_facet_shape_val_tri", tri_shape_val)
                self.register_buffer("_facet_shape_val_quad", quad_shape_val)
            
            return self._facet_shape_val_tri, self._facet_shape_val_quad # type:ignore [n_tri_facet, n_quadrature_per_tri_facet, n_basis], [n_quad_facet, n_quadrature_per_quad_facet, n_basis]

        else:
            if "_facet_shape_val" not in self._buffers: # type: ignore
                _, p = self.facet_quadrature # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
                n_facet, n_quadrature_per_facet, dim = p.shape 
                p = p.reshape(-1, dim)
                shape_val = self.element.eval_shape_val(p, self.basis_order, self.quadrature_order)
                shape_val = shape_val.reshape(n_facet, n_quadrature_per_facet, self.n_basis)
                self.register_buffer("_facet_shape_val", shape_val)

            return self._facet_shape_val # type:ignore [n_facet, n_quadrature_per_facet, n_basis]
              
    @property 
    def facet_shape_grad(self)->Union[Tensorx1, Tensorx2]:
        r"""Get shape function gradients at facet quadrature points.

        Evaluates the gradients of element shape functions at quadrature points on each facet.
        For mixed facet elements, returns separate gradient evaluations for triangular and
        quadrilateral facets.

        The shape function gradients :math:`\nabla\phi_i` are evaluated at quadrature points on each facet.
        For a facet :math:`f`, the gradients are computed as:

        .. math::
            \nabla\phi_i(\mathbf{x}_q^f) = \mathbf{J}^{-T}(\mathbf{x}_q^f) \nabla\hat{\phi}_i(\boldsymbol{\xi}_q^f)

        where:

        * :math:`\mathbf{x}_q^f` is the physical coordinate of quadrature point :math:`q` on facet :math:`f`
        * :math:`\boldsymbol{\xi}_q^f` is the reference coordinate of the quadrature point
        * :math:`\mathbf{J}` is the Jacobian matrix
        * :math:`\hat{\phi}_i` is the reference element shape function

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            grads = transform.facet_shape_grad
            
            # For prism mesh (mixed facets)
            transform = Transformation(points, elements, "wedge")
            tri_grads, quad_grads = transform.facet_shape_grad

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Shape function gradients tensor of shape :math:`[N_f, N_q, N_b, D]` where:

                * :math:`N_f` = number of facets
                * :math:`N_q` = number of quadrature points per facet
                * :math:`N_b` = number of basis functions
                * :math:`D` = spatial dimension

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet gradients of shape :math:`[N_{tf}, N_{tq}, N_b, D]`
                * Quadrilateral facet gradients of shape :math:`[N_{qf}, N_{qq}, N_b, D]`
                
                where:

                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tq}` = quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets
                * :math:`N_{qq}` = quadrature points per quadrilateral facet
                * :math:`N_b` = number of basis functions
                * :math:`D` = spatial dimension
        """
        if self.element.is_mix_facet:

            if "_facet_shape_grad_tri" not in self._buffers: # type: ignore

                tri_m, tri_q, quad_m, quad_q, tri_mask = self.facet_quadrature # type:ignore
                n_tri_facet, n_quadrature_per_tri_facet, _   = tri_q.shape
                n_quad_facet, n_quadrature_per_quad_facet, _ = quad_q.shape

                tri_q = tri_q.reshape(-1, self.element.dim)
                quad_q= quad_q.reshape(-1, self.element.dim)
                tri_shape_grad, tri_cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, tri_q, self.basis_order
                ) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, dim, dim]
                quad_shape_grad, quad_cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, quad_q, self.basis_order
                )
                tri_shape_grad    = tri_shape_grad.reshape(
                    self.n_elements, n_tri_facet, n_quadrature_per_tri_facet, self.n_basis, self.element.dim)
                quad_shape_grad   = quad_shape_grad.reshape(
                    self.n_elements, n_quad_facet, n_quadrature_per_quad_facet, self.n_basis, self.element.dim)


                self.register_buffer("_facet_shape_grad_tri", tri_shape_grad)
                self.register_buffer("_facet_shape_grad_quad", quad_shape_grad)
                self.register_buffer("_facet_cell_jacobian_tri", tri_cell_jacobian)
                self.register_buffer("_facet_cell_jacobian_quad", quad_cell_jacobian)

            return self._facet_shape_grad_tri, self._facet_shape_grad_quad # type:ignore [n_tri_facet, n_quadrature_per_tri_facet, n_basis, dim], [n_quad_facet, n_quadrature_per_quad_facet, n_basis, dim]
        
        else:

            if "_facet_shape_grad" not in self._buffers: # type:ignore
                w, q = self.facet_quadrature # type:ignore [n_facet, n_quadrature_per_facet], [n_facet, n_quadrature_per_facet, dim]
                n_facet, n_quadrature_per_facet, _ = q.shape
                
                q    = q.reshape(-1, self.element.dim)
                shape_grad, cell_jacobian = self.element.eval_shape_grad(
                    self.element_coords, q, self.basis_order
                ) # [n_element, n_quadrature, n_basis, n_dim], [n_element, n_quadrature, dim, dim]
                
                shape_grad    = shape_grad.reshape(
                    self.n_elements, n_facet, n_quadrature_per_facet, self.n_basis, self.element.dim)
                
                self.register_buffer("_facet_shape_grad", shape_grad)
                self.register_buffer("_facet_cell_jacobian", cell_jacobian)

            return self._facet_shape_grad # type:ignore

    @property
    def facet_jacobian(self)->Union[Tensorx1,Tensorx2]:
        r"""Get Jacobian matrices at facet quadrature points.

        The facet Jacobian matrix maps derivatives from reference facet coordinates to physical coordinates.
        For mixed facet elements, returns separate Jacobians for triangular and quadrilateral facets.

        The Jacobian matrix :math:`\mathbf{J}_f = \frac{\partial\mathbf{x}}{\partial\boldsymbol{\xi}_f}`
        where :math:`\mathbf{x}` are physical coordinates and :math:`\boldsymbol{\xi}_f` are reference facet coordinates.

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            J = transform.facet_jacobian
            # J[e,f,q] is Jacobian at qth quadrature point
            # on fth facet of eth element
            
            # For prism mesh (mixed facets)
            transform = Transformation(points, elements, "wedge")
            J_tri, J_quad = transform.facet_jacobian

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Facet Jacobian tensor of shape :math:`[N_e, N_f, N_q, D-1, D]` where:

                * :math:`N_e` = number of elements
                * :math:`N_f` = number of facets per element
                * :math:`N_q` = number of quadrature points per facet
                * :math:`D` = spatial dimension

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet Jacobians of shape :math:`[N_e, N_{tf}, N_{tq}, D-1, D]`
                * Quadrilateral facet Jacobians of shape :math:`[N_e, N_{qf}, N_{qq}, D-1, D]`
                
                where:

                * :math:`N_e` = number of elements
                * :math:`N_{tf}` = number of triangular facets per element
                * :math:`N_{tq}` = quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets per element
                * :math:`N_{qq}` = quadrature points per quadrilateral facet
                * :math:`D` = spatial dimension
        """

        if self.element.is_mix_facet:

            if "_facet_jacobian_tri" not in self._buffers: # type: ignore

                tri_facet_jacobian, quad_facet_jacobian = self.element.eval_facet_jacobian(
                    self.element_coords, self.basis_order, self.quadrature_order)
                
                self.register_buffer("_facet_jacobian_tri", tri_facet_jacobian)
                self.register_buffer("_facet_jacobian_quad", quad_facet_jacobian)

            return self._facet_jacobian_tri, self._facet_jacobian_quad # type:ignore [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim-1, dim], [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim-1, dim]


        else:

            if "_facet_jacobian" not in self._buffers: # type:ignore
                facet_jacobian = self.element.eval_facet_jacobian(
                    self.element_coords, self.basis_order, self.quadrature_order)
                
                assert isinstance(facet_jacobian, torch.Tensor)
                self.register_buffer("_facet_jacobian", facet_jacobian) 
     
            return self._facet_jacobian # type:ignore [n_element, n_facet, n_quadrature_per_facet, dim-1, dim]
    @property 
    def nanson_scale(self)->Union[Tensorx1,Tensorx2]:
        r"""Get Nanson scale factors for element facets.

        The Nanson scale factor :math:`\mathbf{n}` is used to transform area elements from reference to physical space:

        .. math::

            \mathbf{n} = \det(\mathbf{J}) \mathbf{J}^{-T} \mathbf{N} w

        where:

        * :math:`\mathbf{J}` is the facet-to-cell Jacobian matrix
        * :math:`\mathbf{N}` is the outward unit normal in reference space
        * :math:`w` is the quadrature weight

        Examples
        --------
        .. code-block:: python

            # For elements with uniform facets (e.g. tetrahedra)
            transform = Transformation(points, elements, "tetra")
            n = transform.nanson_scale  # Single tensor for all facets
            
            # For elements with mixed facets (e.g. prisms)
            transform = Transformation(points, elements, "wedge")
            n_tri, n_quad = transform.nanson_scale  # Separate tensors for tri/quad facets

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Nanson scale tensor of shape :math:`[N_e, N_f, N_q]` where:

                * :math:`N_e` = number of elements
                * :math:`N_f` = number of facets per element
                * :math:`N_q` = number of quadrature points per facet

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet scales of shape :math:`[N_e, N_{tf}, N_{tq}]`
                * Quadrilateral facet scales of shape :math:`[N_e, N_{qf}, N_{qq}]`

                where:

                * :math:`N_e` = number of elements
                * :math:`N_{tf}` = number of triangular facets per element
                * :math:`N_{tq}` = quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets per element
                * :math:`N_{qq}` = quadrature points per quadrilateral facet

        References
        ----------
        Wikiversity, *Continuum mechanics — Volume change and area change*:
        https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change
        """
      
       
        if self.element.is_mix_facet:
            if "_nanson_scale_tri" not in self._buffers:
                tri_w, tri_q, quad_w, quad_q = self.facet_quadrature # type:ignore

                # compute the facet cell jacobian
                tri_j, quad_j = self.element.eval_facet_cell_jacobian(
                    self.element_coords, 
                    self.basis_order, 
                    self.quadrature_order)
            
                tri_inv_j = torch.inverse(tri_j) # [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim, dim]
                quad_inv_j= torch.inverse(quad_j)# [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim, dim]
                # compute the nanson scale
                normals = self.element.get_outwards_facet_normal() # [n_facet_per_element, dim]
                tri_normals = normals[self.element.get_tri_mask()] # [n_tri_facet, dim]
                quad_normals= normals[self.element.get_quad_mask()]# [n_quad_facet, dim]

                det_tri_j = torch.linalg.det(tri_j) # [n_element, n_tri_facet, n_quadrature_per_tri_facet]
                det_quad_j= torch.linalg.det(quad_j)# [n_element, n_quad_facet, n_quadrature_per_quad_facet]

                tri_nanson  = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', tri_normals, tri_inv_j)  # [n_element, n_tri_facet, n_quadrature_per_tri_facet, dim]
                ) # [n_element, n_tri_facet, n_quadrature_per_tri_facet]
                quad_nanson = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', quad_normals, quad_inv_j) # [n_element, n_quad_facet, n_quadrature_per_quad_facet, dim]
                ) # [n_element, n_quad_facet, n_quadrature_per_quad_facet]

                # TODO: check if the absoulte value should be added to the determinant
                tri_nanson = torch.einsum("efq,efq,fq->efq",tri_nanson , det_tri_j , tri_w)
                quad_nanson= torch.einsum("efq,efq,fq->efq",quad_nanson, det_quad_j, quad_w)

                self.register_buffer("_nanson_scale_tri", tri_nanson)
                self.register_buffer("_nanson_scale_quad", quad_nanson)

            return self._nanson_scale_tri, self._nanson_scale_quad # type:ignore [n_element, n_tri_facet, n_quadrature_per_tri_facet], [n_element, n_quad_facet, n_quadrature_per_quad_facet]
        
        else: # single facet shape 

            if "_nanson_scale" not in self._buffers:

                j = self.element.eval_facet_cell_jacobian(
                        self.element_coords, 
                        self.basis_order, 
                        self.quadrature_order)
                
                inv_j = torch.inverse(j) # [n_element, n_facet, n_quadrature_per_facet, dim, dim]
                det_j = torch.linalg.det(j) # [n_element, n_facet, n_quadrature_per_facet]

                normals = self.element.get_outwards_facet_normal() # [n_facet, dim]

                nanson_scale = torch.linalg.norm(
                    torch.einsum('fi,efqji->efqj', normals, inv_j) # [n_element, n_facet, n_quadrature_per_facet, dim]
                ) # [n_element, n_facet, n_quadrature_per_facet]

                nanson_scale = torch.einsum("efq,efq,fq->efq", nanson_scale, det_j, self.facet_quadrature[0]) # [n_element, n_facet, n_quadrature_per_facet]

                self.register_buffer("_nanson_scale", nanson_scale)

            return self._nanson_scale # type:ignore [n_element, n_facet, n_quadrature_per_facet]


        
      



    ###############
    # Abbreviation
    ############### 


    # basis 
    phi     = shape_val
    gradphi = shape_grad
    J       = jacobian 
    
    @property 
    def detJ(self)->Tensorx1:
        r"""Get the determinant of the Jacobian matrix at quadrature points.

        The determinant :math:`\lvert J \rvert = \det\!\left(\frac{\partial\mathbf{x}}{\partial\boldsymbol{\xi}}\right)`
        is the local scaling factor between reference and physical coordinates.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            detJ = transform.detJ
            # detJ[e,q] is Jacobian determinant at qth quadrature point
            # in eth element

        Returns
        -------
        torch.Tensor
            Jacobian determinant tensor of shape :math:`[N_e, N_q]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_q` = number of quadrature points
        """
        return torch.linalg.det(self.jacobian)
    
    @property 
    def JxW(self)->Tensorx1:
        r"""Get the Jacobian determinant times quadrature weights.

        The JxW term :math:`\vert J\vert w` combines the Jacobian determinant with quadrature weights
        for numerical integration over physical elements:

        .. math::
            \int_\Omega f(\mathbf{x}) d\mathbf{x} = \sum_{e=1}^{N_e} \sum_{q=1}^{N_q} f(\mathbf{x}_q^e) \vert J_q^e \vert w_q

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            jxw = transform.JxW
            # Integrate function over mesh:
            integral = torch.sum(f * jxw)  # f evaluated at quadrature points

        Returns
        -------
        torch.Tensor
            JxW tensor of shape :math:`[N_e, N_q]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_q` = number of quadrature points
        """
        if "_jxw" not in self._buffers: # type: ignore
            w, _       = self.quadrature
            jxw        = torch.einsum('q,eq->eq', w, self.detJ.abs())
            self.register_buffer("_jxw", jxw)
        return self._jxw # type:ignore [n_element, n_quadrature]

    G     = J 
    detG  = detJ 
    GxW   = JxW

    F     = facet_jacobian

    @property
    def detF(self)->Union[Tensorx1, Tensorx2]:
        r"""Get determinant of facet Jacobian matrices.

        For a facet with normal :math:`\mathbf{n}`, the facet Jacobian determinant is:

        .. math::
            |F| = \sqrt{\det(F^T F)}

        where :math:`F` is the facet Jacobian matrix mapping from reference to physical facets.

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            detF = transform.detF  # Single tensor for tri facets
            
            # For prism mesh (mixed tri/quad facets)
            transform = Transformation(points, elements, "wedge")
            tri_detF, quad_detF = transform.detF  # Separate tensors

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                Facet Jacobian determinant tensor of shape :math:`[N_e, N_f, N_q]` where:
                
                * :math:`N_e` = number of elements
                * :math:`N_f` = number of facets per element
                * :math:`N_q` = number of quadrature points per facet

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet determinants of shape :math:`[N_e, N_{tf}, N_{tq}]`
                * Quadrilateral facet determinants of shape :math:`[N_e, N_{qf}, N_{qq}]`
                
                where:

                * :math:`N_e` = number of elements
                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tq}` = number of quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets
                * :math:`N_{qq}` = number of quadrature points per quadrilateral facet
        """
        if self.element.is_mix_facet:
            tri_fj, quad_fj = self.facet_jacobian # type:ignore
            tri_jac_det = torch.sqrt(torch.linalg.det(
                torch.einsum('efqij,efqjk->efqik', tri_fj, tri_fj)))
            quad_jac_det= torch.sqrt(torch.linalg.det(
                torch.einsum('efqij,efqjk->efqik', quad_fj, quad_fj)))
            return tri_jac_det, quad_jac_det
        else:
            fj = self.facet_jacobian # type:ignore
           
            return torch.sqrt(torch.linalg.det(fj @ fj.mT))
   
    @property 
    def FxW(self)->Union[Tensorx1,Tensorx2]:
        r"""Get facet Jacobian determinant times quadrature weights.

        The FxW term :math:`|F|w` combines the facet Jacobian determinant with quadrature weights
        for numerical integration over physical facets:

        .. math::
            \int_{\Gamma} f(\mathbf{x}) d\Gamma = \sum_{e=1}^{N_e} \sum_{f=1}^{N_f} \sum_{q=1}^{N_q} f(\mathbf{x}_q^{ef}) |F_q^{ef}| w_q

        Examples
        --------
        .. code-block:: python

            # For tetrahedral mesh (triangular facets)
            transform = Transformation(points, elements, "tetra")
            fxw = transform.FxW  # Single tensor for tri facets
            
            # For prism mesh (mixed tri/quad facets) 
            transform = Transformation(points, elements, "wedge")
            tri_fxw, quad_fxw = transform.FxW  # Separate tensors
            
            # Integrate function over facets:
            integral = torch.sum(f * fxw)  # f evaluated at facet quadrature points

        Returns
        -------
        If element has uniform facets:
            torch.Tensor
                FxW tensor of shape :math:`[N_e, N_f, N_q]` where:
                
                * :math:`N_e` = number of elements
                * :math:`N_f` = number of facets per element
                * :math:`N_q` = number of quadrature points per facet

        If element has mixed facets:
            Tuple[torch.Tensor, torch.Tensor]
                * Triangular facet FxW of shape :math:`[N_e, N_{tf}, N_{tq}]`
                * Quadrilateral facet FxW of shape :math:`[N_e, N_{qf}, N_{qq}]`
                
                where:

                * :math:`N_e` = number of elements
                * :math:`N_{tf}` = number of triangular facets
                * :math:`N_{tq}` = number of quadrature points per triangular facet
                * :math:`N_{qf}` = number of quadrilateral facets
                * :math:`N_{qq}` = number of quadrature points per quadrilateral facet
        """
        if self.element.is_mix_facet:
            if "_jxf_tri" not in self._buffers: # type: ignore
                tri_w, tri_q, quad_w, quad_q = self.facet_quadrature # type:ignore
                tri_detF, quad_detF = self.detF
                tri_jxf = torch.einsum('fq,efq->efq', tri_w, tri_detF.abs())
                quad_jxf= torch.einsum('fq,efq->efq', quad_w, quad_detF.abs())
                self.register_buffer("_jxf_tri", tri_jxf)
                self.register_buffer("_jxf_quad", quad_jxf)
            return self._jxf_tri, self._jxf_quad # type:ignore [n_element, n_facet, n_quadrature_per_facet]
        else:
            if "_jxf" not in self._buffers: # type: ignore
                w, q = self.facet_quadrature # type:ignore
                detF = self.detF
                jxf  = torch.einsum('fq,efq->efq', w, detF.abs()) 
                self.register_buffer("_jxf", jxf)
            return self._jxf # type:ignore [n_element, n_facet, n_quadrature_per_facet]
    # facet normal
    n = nanson_scale
    

    ######################
    # Efficient Methods
    ######################
    def batch_quadrature(self, start:int, batch:int)->Tensorx2:
        r"""Get a batch of quadrature points and weights.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            
            # Get first 10 quadrature points
            w, q = transform.batch_quadrature(start=0, batch=10)
            
            # Get all quadrature points after index 5
            w, q = transform.batch_quadrature(start=5, batch=-1)

        Parameters
        ----------
        start : int
            The starting index of the batch
        batch : int
            The number of quadrature points in the batch.
            If -1, returns all points starting from `start`

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * Quadrature weights tensor of shape :math:`[N_b]`
            * Quadrature points tensor of shape :math:`[N_b, D]`
            
            where:

            * :math:`N_b` = batch size
            * :math:`D` = spatial dimension
        """
        if start < 0:
            start += self.n_quadrature  
        assert start >= 0 and start < self.n_quadrature
        if start == 0 and batch == -1:
            return self.quadrature
        w, q = self.quadrature # [n_quadrature], [n_quadrature, dim]
        return w[start:start+batch], q[start:start+batch] # [batch], [batch, dim]
    
    def batch_elements_coords(self, start:int, batch:int)->Tensorx1:
        r"""Get physical coordinates for a batch of elements.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            
            # Get coordinates for first 5 elements
            coords = transform.batch_elements_coords(start=0, batch=5)
            
            # Get coordinates for all elements after index 10
            coords = transform.batch_elements_coords(start=10, batch=-1)

        Parameters
        ----------
        start : int
            The starting element index
        batch : int
            The number of elements in the batch.
            If -1, returns all elements starting from `start`

        Returns
        -------
        torch.Tensor
            Element coordinates tensor of shape :math:`[N_b, N_n, D]` where:

            * :math:`N_b` = batch size
            * :math:`N_n` = nodes per element
            * :math:`D` = spatial dimension
        """
        if start < 0:
            start += self.n_elements
        if start == 0 and batch == -1:
            return self.element_coords
        else:
            elements:torch.Tensor        = self.elements[start:start+batch] # type:ignore [batch, n_basis]
            elements_coords:torch.Tensor = self.points[elements] # type:ignore [batch, n_basis, dim]
            return elements_coords

    def batch_shape_val(self, start:int, batch:int)->Tensorx1:
        r"""Get shape function values for a batch of quadrature points.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            
            # Get shape values at first 10 quadrature points
            vals = transform.batch_shape_val(start=0, batch=10)
            
            # Get shape values at all points after index 5
            vals = transform.batch_shape_val(start=5, batch=-1)

        Parameters
        ----------
        start : int
            The starting quadrature point index
        batch : int
            The number of quadrature points in the batch

        Returns
        -------
        torch.Tensor
            Shape function values tensor of shape :math:`[N_b, N_s]` where:

            * :math:`N_b` = batch size
            * :math:`N_s` = shape functions per element
        """
        return self.shape_val[start:start+batch] #  [batch, n_basis]

    def batch_shape_grad_jxw(self, 
                            element_start:int = 0,
                            element_batch:int = -1,
                            quadrature_start:int = 0, 
                            quadrature_batch:int = -1)->Tensorx2:
        r"""Get shape function gradients and Jacobian weights for batches of elements and quadrature points.

        Examples
        --------
        .. code-block:: python

            transform = Transformation(points, elements, "triangle")
            
            # Get values for first 5 elements and first 10 quadrature points
            grads, jxw = transform.batch_shape_grad_jxw(
                element_start=0, element_batch=5,
                quadrature_start=0, quadrature_batch=10
            )
            
            # Get values for all elements and quadrature points
            grads, jxw = transform.batch_shape_grad_jxw()

        Parameters
        ----------
        element_start : int, optional
            The starting element index, by default 0
        element_batch : int, optional
            The number of elements in the batch.
            If -1, uses all elements after start, by default -1
        quadrature_start : int, optional
            The starting quadrature point index, by default 0
        quadrature_batch : int, optional
            The number of quadrature points in the batch.
            If -1, uses all points after start, by default -1

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * Shape gradients tensor of shape :math:`[N_e, N_q, N_s, D]`
            * Jacobian weights tensor of shape :math:`[N_e, N_q]`
            
            where:

            * :math:`N_e` = number of elements in batch
            * :math:`N_q` = number of quadrature points in batch
            * :math:`N_s` = shape functions per element
            * :math:`D` = spatial dimension
        """
        if "_shape_grad" not in self._buffers or "_jxw" not in self._buffers: # type: ignore
            w, q = self.batch_quadrature(quadrature_start, quadrature_batch)
            e    = self.batch_elements_coords(element_start, element_batch)
            shape_grad, jacobian = self.element.eval_shape_grad(
                e, q, self.basis_order, self.quadrature_order)
            jxw  = torch.einsum('q,eq->eq', w, torch.linalg.det(jacobian).abs())
            return shape_grad, jxw # [n_element, n_batch, n_basis, dim], [n_element, n_batch, dim, dim]
        else:
            return (self.shape_grad[:, quadrature_start:quadrature_start+quadrature_batch], 
                    self.JxW[:, quadrature_start:quadrature_start+quadrature_batch]) #type:ignore [n_element, n_batch, n_basis, dim], [n_element, n_batch, dim, dim]
        

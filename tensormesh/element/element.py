
from functools import lru_cache, reduce
from logging import warning
import torch
import math
from functools import wraps
from typing import List, Tuple, Optional, Type, Sequence, Union
from .polynomial import Polynomial, \
                        Polynomials
from .basis import  lin_basis,\
                    tri_basis,\
                    quad_basis,\
                    tet_basis,\
                    hex_basis,\
                    pyr_basis,\
                    pri_basis,\
                    facet_basis_index_2d, \
                    tet_facet_basis_index, \
                    hex_facet_basis_index,\
                    mix_facet_basis_index, \
                    edge_index
from .quadrature import lin_quadrature,\
                        tri_quadrature,\
                        tet_quadrature,\
                        quad_quadrature,\
                        hex_quadrature,\
                        pyr_quadrature,\
                        pri_quadrature,\
                        facet_quadrature_2d,\
                        tet_facet_quadrature,\
                        hex_facet_quadrature,\
                        mix_facet_quadrature_3d
from .normal import outwards_normal_2d,\
                    outwards_normal_3d
# from .plot import plot_1d,\
#                   plot_2d,\
#                   plot_3d 
from .types import Tensorx1, Tensorx2, Tensorx3, Tensorx4, Tensorx5






class Element:
    """Base class for the seven reference shapes (line, triangle, quad, tet,
    hex, pyramid, prism).

    Each subclass is a thin namespace of class methods — ``get_basis``,
    ``get_quadrature``, ``get_facet_type``, ``reorder``, … — that the
    assembler and mesh I/O paths call internally to obtain
    interpolation-node layouts, shape functions, quadrature rules, and the
    Gmsh/VTK ↔ TensorMesh permutation.

    Subclasses are not instantiated; ``Triangle``, ``Hexahedron``, etc. are
    used directly as types. See :doc:`/user_guide/elements_and_quadrature`
    for the full API tour and :ref:`node-ordering-gallery` for a visual
    comparison of the node-numbering conventions.
    """

    #: Coordinates of element vertices
    #: Shape: :math:`[V, D]` where :math:`V` = number of vertices and :math:`D` = spatial dimension
    points: torch.Tensor

    #: Vertex indices for the element
    #: Shape: :math:`[V, 1]` where :math:`V` = number of vertices
    vertex: torch.Tensor 

    #: Edge connectivity defining pairs of vertex indices
    #: Shape: :math:`[E, 2]` where :math:`E` = number of edges
    edge: torch.Tensor

    #: Face connectivity defining vertex indices for each face.
    #: Shape: Tuple of tuples, each inner tuple contains vertex indices 
    #: for one face.
    #:
    #: Example:
    #:     ((0,1,2), (1,2,3)) for two triangular faces
    face: Optional[Tuple[Tuple[int,...],...]]

    #: Cell connectivity defining vertex indices for volumetric elements
    #: Shape: :math:`[N_c, V_c]` where :math:`N_c` = number of cells and :math:`V_c` = vertices per cell
    cell: Optional[torch.Tensor]

    #: Spatial dimension of the element (1D, 2D, or 3D)
    dim: int

    #: Number of vertices in the element
    n_vertex: int 

    #: Number of edges in the element
    n_edge: int

    #: Number of faces in the element
    n_face: int

    #: Number of cells in the element 
    n_cell: int
   
    is_mix_facet:bool = False
    """Boolean indicating whether the element has mixed facet types.

    False indicates uniform facets (e.g. all triangles or all quads).
    True indicates mixed facets (e.g. prism or pyramid).
    """

    @classmethod
    def reorder(cls, elements: torch.Tensor, to_gmsh: bool = True) -> torch.Tensor:
        r"""
        Reorder element connectivity between **TensorMesh internal ordering** and **Gmsh/VTK ordering**.

        TensorMesh stores connectivity for some tensor-product elements (e.g. Quad, Hex) in a
        **lexicographic** node order that matches the basis / tensor-product polynomial layout.
        However, file formats like **VTK/VTU** (and Gmsh conventions) expect a different ordering.

        This helper provides a single, centralized conversion point so that:
        - internal FEM kernels always see TensorMesh ordering
        - IO / visualization (VTK/VTU) can export in Gmsh/VTK ordering

        Parameters
        ----------
        elements:
            Connectivity tensor of shape ``[..., n_nodes]`` (typically ``[n_elem, n_nodes]``).
        to_gmsh:
            - ``False``: convert **Gmsh/VTK -> TensorMesh** (used when reading meshes)
            - ``True``:  convert **TensorMesh -> Gmsh/VTK** (used when writing VTK/VTU)

        Returns
        -------
        torch.Tensor
            Reordered connectivity tensor with the same shape/dtype/device as input.
        """
        perm = cls.get_gmsh_permutation(elements.shape[-1], device=elements.device)
        if to_gmsh:
            perm = torch.argsort(perm)
        return elements.index_select(-1, perm)

    @classmethod
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        r"""
        Return a permutation ``perm`` (shape ``[n_nodes]``) mapping **Gmsh/VTK -> TensorMesh**.

        Given an input connectivity ``conn_gmsh[..., :]`` in Gmsh/VTK ordering, the internal ordering is:

        ``conn_internal = conn_gmsh.index_select(-1, perm)``

        Notes
        -----
        Subclasses should override this for element types where TensorMesh ordering differs from
        the external Gmsh/VTK ordering. If not overridden, the default is identity.
        """
        # Default: identity (meaning Gmsh/VTK ordering == TensorMesh ordering)
        return torch.arange(n_nodes, device=device, dtype=torch.long)

    @classmethod
    def classify_nodes(cls, order: int) -> Tuple[List[Tuple[str, int, int]], int]:
        r"""Classify the reference nodes of the order-``order`` Lagrange space.

        Each reference node (row of :meth:`get_basis`) is matched **by
        coordinates** to a vertex, to an interior point of an edge, or to
        a face/cell interior point — no assumption on the internal node
        ordering is made. The topological DOF maps
        (``tensormesh.assemble.topology.lagrange_dofmap``) and the
        node-ordering check (:meth:`tensormesh.Mesh.check_node_ordering`)
        are built on this classification.

        Parameters
        ----------
        order : int
            Polynomial order of the Lagrange space (``>= 1``).

        Returns
        -------
        slots : List[Tuple[str, int, int]]
            One entry per reference node, slot-aligned with :meth:`get_basis`:
            ``("vertex", local_vertex, 0)``; ``("edge", local_edge, j)`` with
            ``j in 1..order-1`` counted from ``edge[local_edge, 0]`` towards
            ``edge[local_edge, 1]``; or ``("interior", running_index, 0)``.
        n_interior : int
            Number of interior (neither vertex nor edge) nodes.
        """
        ref = cls.get_basis(order, torch.float64)          # [nb, D]
        verts = cls.points.to(torch.float64)               # [n_vertex, D]
        edges = cls.edge.tolist()                          # [n_edge, 2]
        slots, n_interior = [], 0
        for x in ref:
            dist = (verts - x).norm(dim=1)
            if dist.min() < 1e-8:
                slots.append(("vertex", int(dist.argmin()), 0))
                continue
            hit = None
            for local_edge, (a, b) in enumerate(edges):
                va, vb = verts[a], verts[b]
                tv = vb - va
                t = float(torch.dot(x - va, tv) / torch.dot(tv, tv))
                if 1e-8 < t < 1 - 1e-8 and (va + t * tv - x).norm() < 1e-8:
                    j = round(t * order)
                    assert abs(t * order - j) < 1e-6 and 1 <= j <= order - 1, (
                        f"{cls.__name__} order-{order} edge node at t={t} does "
                        f"not sit on the uniform lattice"
                    )
                    hit = ("edge", local_edge, j)
                    break
            if hit is not None:
                slots.append(hit)
            else:
                slots.append(("interior", n_interior, 0))
                n_interior += 1
        return slots, n_interior

    # @abstractmethod
    @classmethod
    def get_facet_type(cls)->Union[Type['Element'],Tuple[Type['Element'],Type['Element']]]:
        """
        Get the element type(s) for the facets of this element.

        This method returns the element type(s) that make up the facets (faces) of the current element.
        For example, a triangle element has line segment facets, while a prism has both triangular and
        quadrilateral facets.

        Examples
        --------
        >>> from tensormesh.element import Triangle, Line
        >>> Triangle.get_facet_type()
        Line

        >>> from tensormesh.element import Prism, Triangle, Quadrilateral 
        >>> facet_types = Prism.get_facet_type()
        >>> facet_types == (Triangle, Quadrilateral)
        True

        >>> from tensormesh.element import Tetrahedron
        >>> Tetrahedron.get_facet_type()
        Triangle

        Returns
        -------
        Union[Type['Element'], Tuple[Type['Element'], Type['Element']]]
            The element type(s) for the facets. Returns either:

            - A single Element type for elements with uniform facets (e.g. Line for Triangle)
            - A tuple of Element types for elements with mixed facets (e.g. (Triangle, Quadrilateral) for Prism)

        """
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->torch.Tensor:
        """
        Get the basis node coordinates for the element.

        This method returns the coordinates of the basis nodes that define the element's shape functions.
        The number and location of basis nodes depends on the polynomial order.

        For example:
        - For order=1 (linear), returns vertices of the element
        - For order=2 (quadratic), returns vertices plus midpoints of edges
        - For higher orders, returns additional nodes in the interior

        The basis nodes are used to construct the element's shape functions and define the geometric
        mapping between reference and physical elements.

        Examples
        --------
        >>> from tensormesh.element import Triangle
        >>> basis = Triangle.get_basis(order=1)
        >>> basis.shape
        torch.Size([3, 2])  # 3 vertices in 2D
        >>> basis
        tensor([[0., 0.],
                [1., 0.], 
                [0., 1.]])

        >>> # Quadratic basis has additional nodes at edge midpoints
        >>> basis = Triangle.get_basis(order=2) 
        >>> basis.shape
        torch.Size([6, 2])  # 3 vertices + 3 edge midpoints

        >>> from tensormesh.element import Tetrahedron
        >>> basis = Tetrahedron.get_basis(order=1)
        >>> basis.shape
        torch.Size([4, 3])  # 4 vertices in 3D
        >>> basis
        tensor([[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])

        Parameters
        ----------
        order: int 
            the order of the basis
        dtype: torch.dtype
            the float data type of the polynomial
        device: torch.device
            the device of the polynomial

        Returns
        -------
        basis: torch.Tensor 
            2D Tensor of shape [B, D], 
            where B is the number of basis nodes and D is the dimension
        """
        raise NotImplementedError()
    
    # @abstractmethod
    @classmethod 
    def get_facet(cls, order:int=1)->Union[Tensorx1,Tensorx2]:
        """
        Get the facet connectivity for the element.

        For elements with uniform facets (e.g. Triangle, Quadrilateral), returns a single tensor 
        containing facet connectivity. For elements with mixed facets (e.g. Prism), returns a tuple 
        of tensors containing facet connectivity for each facet type.

        The facet connectivity defines how the basis nodes are connected to form each facet of the element.
        For example, for a triangle element with linear basis (order=1), the facet connectivity would be::

            [[0,1],  # First edge connects nodes 0 and 1
             [1,2],  # Second edge connects nodes 1 and 2  
             [2,0]]  # Third edge connects nodes 2 and 0

        For a prism element which has both triangular and quadrilateral facets, would return::

            (tri_facets, quad_facets)

        Where tri_facets contains connectivity for the triangular faces and quad_facets contains
        connectivity for the quadrilateral faces.

        Examples
        --------
        >>> from tensormesh.element import Triangle
        >>> facets = Triangle.get_facet(order=1)
        >>> facets
        tensor([[0, 1],  # First edge connects nodes 0 and 1
                [1, 2],  # Second edge connects nodes 1 and 2
                [2, 0]]) # Third edge connects nodes 2 and 0

        >>> # Higher order elements have more nodes per facet
        >>> facets = Triangle.get_facet(order=2)
        >>> facets.shape
        torch.Size([3, 3])  # 3 edges with 3 nodes each

        >>> from tensormesh.element import Prism
        >>> tri_facets, quad_facets = Prism.get_facet(order=1)
        >>> tri_facets.shape  # Two triangular faces
        torch.Size([2, 3])
        >>> quad_facets.shape  # Three quadrilateral faces 
        torch.Size([3, 4])

        Parameters
        ----------
        order: int 
            the order of the basis

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            For elements with uniform facets, a 2D tensor of shape
            :math:`[F, B_f]` where :math:`F` is the number of facets and
            :math:`B_f` the number of basis nodes per facet.

            For elements with mixed facets, a pair
            ``(tri_facet, quad_facet)`` of shapes :math:`[F_t, B_{ft}]` and
            :math:`[F_q, B_{fq}]`, respectively.
        """
        raise NotImplementedError()

    @classmethod
    @lru_cache()
    def get_edge(cls, order:int)->Tensorx1:
        """
        Get the edge basis indices for the element.

        Returns the connectivity array for the edges of the element. Each row represents an edge
        and contains the indices of the basis nodes along that edge.

        Examples
        --------
        >>> from tensormesh.element import Triangle
        >>> edges = Triangle.get_edge(order=1)
        >>> edges
        tensor([[0, 1],  # First edge connects nodes 0 and 1
                [1, 2],  # Second edge connects nodes 1 and 2
                [2, 0]]) # Third edge connects nodes 2 and 0

        >>> # Higher order elements have more nodes per edge
        >>> edges = Triangle.get_edge(order=2)
        >>> edges.shape
        torch.Size([3, 3])  # 3 edges with 3 nodes each

        Parameters
        ----------
        order: int 
            the order of the basis

        Returns
        -------
        edge: torch.Tensor
            2D Tensor of shape [n_edge, 2] containing edge connectivity
        """
        return edge_index(cls.vertex, cls.edge, order)

    # @abstractmethod
    @classmethod
    def get_polynomial(cls, 
                       order:int=1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        """
        Get the polynomial form for the element.

        For a given order n, returns a polynomial with terms up to maximum total degree n.
        The polynomial will have n_vars equal to the dimension of the element.

        For example, 
        
        * For a 2D element with order=1:
          :math:`P(x,y) = a_0 + a_1x + a_2y`

        * For order=2:
          :math:`P(x,y) = a_0 + a_1x + a_2y + a_3x^2 + a_4xy + a_5y^2`

        The exact form and number of terms depends on the specific element type.

        Examples
        --------
        >>> from tensormesh.element import Triangle
        >>> poly = Triangle.get_polynomial(order=1)
        >>> poly.n_vars  # 2D element
        2
        >>> poly.n_terms  # Linear terms: 1, x, y
        3

        >>> poly = Triangle.get_polynomial(order=2)
        >>> poly.n_terms  # Quadratic terms: 1, x, y, x^2, xy, y^2
        6

        >>> from tensormesh.element import Tetrahedron
        >>> poly = Tetrahedron.get_polynomial(order=1)
        >>> poly.n_vars  # 3D element
        3
        >>> poly.n_terms  # Linear terms: 1, x, y, z
        4

        Parameters
        ----------
        order: int 
            the order of the basis
        dtype: torch.dtype
            the float data type of the polynomial
        device: torch.device
            the device of the polynomial

        Returns
        -------
        polynomial: Polynomial
            A polynomial object with number of variables equal to the element dimension, 
            
        """
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        """
        Get the quadrature points and weights for the element.

        For elements with single type of facets (e.g. Tetrahedron):
            * Returns a tuple of two tensors containing quadrature weights and points

        For elements with mixed facets (e.g. Prism):
            * Returns a tuple of four tensors containing quadrature weights and points for each facet type 

        Examples
        --------
        >>> from tensormesh.element import Triangle
        >>> weights, points = Triangle.get_quadrature(order=1)
        >>> weights.shape  # Number of quadrature points
        torch.Size([1])
        >>> points.shape  # [n_points, dimension]
        torch.Size([1, 2])

        >>> weights, points = Triangle.get_quadrature(order=2)
        >>> weights.shape  # More quadrature points for higher order
        torch.Size([3])
        >>> points.shape
        torch.Size([3, 2])

        >>> from tensormesh.element import Tetrahedron
        >>> weights, points = Tetrahedron.get_quadrature(order=1)
        >>> weights.shape
        torch.Size([1])
        >>> points.shape  # 3D points
        torch.Size([1, 3])
            
        Parameters
        ----------
        order: int 
            quadrature order
        dtype: torch.dtype
            the float data type of the polynomial
        device: torch.device
            the device of the polynomial

        Returns
        -------
        quadrature_weights : torch.Tensor
            1D tensor of shape :math:`[N_q]` where :math:`N_q` is the
            number of quadrature points.
        quadrature_points : torch.Tensor
            2D tensor of shape :math:`[N_q, D]` where :math:`D` is the
            element dimension (1, 2, or 3).
        """
        raise NotImplementedError()

    # @abstractmethod
    @classmethod
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )-> Union[Tensorx2,Tensorx4]:
        """Get the quadrature points and weights for the facets of the element.

        This method returns quadrature rules for integrating over element facets. The return format depends on whether the element has uniform or mixed facet types.

        **For elements with uniform facets** (e.g. Tetrahedron):
            * Returns a tuple of (weights, points) tensors
            * weights: quadrature weights for each facet point
            * points: quadrature point coordinates 

        **For elements with mixed facets** (e.g. Prism):
            * Returns a tuple of (tri_weights, tri_points, quad_weights, quad_points)
            * tri_weights: quadrature weights for triangular facets
            * tri_points: quadrature points for triangular facets  
            * quad_weights: quadrature weights for quadrilateral facets
            * quad_points: quadrature points for quadrilateral facets

        The quadrature points can be returned either in the reference element coordinates (transform=True) or in facet-local coordinates (transform=False).
        
        Examples
        --------
        >>> # Example for tetrahedron element with uniform triangular facets
        >>> weights, points = Tetrahedron.get_facet_quadrature(order=1)
        >>> weights.shape  # [n_facet=4, n_quad_points=1] 
        torch.Size([4, 1])
        >>> points.shape  # [n_facet=4, n_quad_points=1, dim=3]
        torch.Size([4, 1, 3])

        >>> # Example for prism element with mixed facets (triangular and quadrilateral)
        >>> tri_w, tri_p, quad_w, quad_p = Prism.get_facet_quadrature(order=1)
        >>> tri_w.shape  # [n_tri_facet=2, n_quad_points=1]
        torch.Size([2, 1]) 
        >>> tri_p.shape  # [n_tri_facet=2, n_quad_points=1, dim=3]
        torch.Size([2, 1, 3])
        >>> quad_w.shape # [n_quad_facet=3, n_quad_points=1]
        torch.Size([3, 1])
        >>> quad_p.shape # [n_quad_facet=3, n_quad_points=1, dim=3] 
        torch.Size([3, 1, 3])

        >>> # Example with transform=False returns points in facet-local coordinates
        >>> weights, points = Triangle.get_facet_quadrature(order=1, transform=False)
        >>> weights.shape  # [n_quad_points=1]
        torch.Size([1])
        >>> points.shape  # [n_quad_points=1, dim-1=1] 
        torch.Size([1, 1])

        Parameters
        ----------
        order : int, optional
            Quadrature order. Default is 1.
        transform : bool, optional
            Whether to return the transformed facet quadrature.
            If True, returns points in reference element coordinates with shape :math:`[N_f, N_q, D]`, where :math:`N_f` = number of facets, :math:`N_q` = quadrature points per facet, and :math:`D` = spatial dimension.
            If False, returns points in facet-local coordinates with shape :math:`[N_q, D-1]`, where :math:`N_q` = quadrature points per facet and :math:`D` = spatial dimension.
        dtype : torch.dtype, optional
            The float data type of the polynomial.
        device : torch.device, optional
            The device of the polynomial.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            For elements with uniform facets, a pair
            ``(weights, points)`` whose shapes depend on ``transform``:

            * ``transform=True`` — weights of shape :math:`[N_f, N_q]`,
              points of shape :math:`[N_f, N_q, D]` (cell coordinates).
            * ``transform=False`` — weights of shape :math:`[N_q]`,
              points of shape :math:`[N_q, D-1]` (facet-local coordinates).

            For elements with mixed facets, a 4-tuple
            ``(tri_weights, tri_points, quad_weights, quad_points)`` with
            the obvious analogues of the shapes above for the triangular
            and quadrilateral facet kinds.
        """
        raise NotImplementedError()

    @classmethod
    @lru_cache()
    def get_basis_fns(cls, 
                      order:int = 1, 
                      dtype:torch.dtype = torch.float32,
                      device:torch.device = torch.device('cpu')
                      )->Polynomials:
        """Get the basis functions for this element type.

        Returns polynomial basis functions that interpolate values at the element nodes.
        The basis functions have the Kronecker delta property, meaning each basis function
        is 1 at its corresponding node and 0 at all other nodes.

        For example, for a linear triangle element (order=1), returns 3 basis functions
        corresponding to the 3 vertices. Each basis function is 1 at one vertex and 0
        at the other vertices.

        For higher orders, returns basis functions for both vertex and edge nodes.
        The number and location of nodes depends on the specific element type.

        The basis functions are returned as a Polynomials object containing the
        polynomial coefficients and exponents. This allows efficient evaluation
        and manipulation of the basis functions.

        Examples
        --------
        .. code-block:: python

            >>> from tensormesh.element import Triangle
            >>> basis_fns = Triangle.get_basis_fns(order=1)
            >>> basis_fns
            Polynomials([
                1 - x - y,  # First basis function, 1 at vertex 0 (0,0)
                x,          # Second basis function, 1 at vertex 1 (1,0)
                y           # Third basis function, 1 at vertex 2 (0,1)
            ])

            >>> # Evaluate basis functions at a point
            >>> point = torch.tensor([[0.5, 0.25]])
            >>> values = basis_fns(point)
            >>> values
            tensor([[0.25, 0.50, 0.25]])  # Values sum to 1

            >>> # Higher order basis functions
            >>> basis_fns = Triangle.get_basis_fns(order=2)
            >>> basis_fns
            Polynomials([
                # Vertex basis functions
                2*x^2 + 2*y^2 - 3*x - 3*y + 1,  # Node 0 
                2*x^2 - x,                       # Node 1
                2*y^2 - y,                       # Node 2
                # Edge basis functions  
                4*x*y,                           # Node 3
                -4*x*y + 4*x,                    # Node 4 
                -4*x*y + 4*y                     # Node 5
            ])

            >>> from tensormesh.element import Tetrahedron
            >>> basis_fns = Tetrahedron.get_basis_fns(order=1)
            >>> basis_fns
            Polynomials([
                1 - x - y - z,  # First basis function
                x,              # Second basis function  
                y,              # Third basis function
                z               # Fourth basis function
            ])

        Parameters
        ----------
        order : int 
            The order of the basis functions.
        dtype : torch.dtype
            The float data type of the polynomial.
        device : torch.device
            The device to place the polynomial on.

        Returns
        -------
        Polynomials
            The basis functions of the element. Shape is :math:`[N_b]` where :math:`N_b` is the number of basis functions, with :math:`N_{vars}=D` variables and :math:`N_{exp}=N_b` expansion terms.
        """
        def adaptive_inv(x:torch.Tensor)->torch.Tensor:
            eps = 1e-7
            assert x.dim() == 2, f"Input tensor must be 2D, got {x.dim()}D"
            assert x.shape[0] == x.shape[1], f"Input tensor must be square, got shape {x.shape}"
            if order >= 3:
                result = torch.linalg.pinv(x)
            else:
                result = torch.linalg.inv(x)
            if torch.dist(x @ result, torch.eye(x.shape[0], dtype=x.dtype, device=x.device)) > eps:
                warning(f"basis functions are not accurate for {cls} with order {order}")
            return result
        
        basis = cls.get_basis(order, dtype, device) # [n_basis, n_dim] n_dim ~ n_vars
        
        n_basis = basis.shape[0]
        # basis [order + 1, 1]
        
        polys = cls.get_polynomial(order, dtype, device).repeat(n_basis) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        
        V = polys.get_exp_terms(basis) # [n_basis, n_basis]
        coef = adaptive_inv(V).T # [n_basis, n_basis] 
        polys.reset_coef(coef)
        polys = polys.to(device)
        return polys

    @classmethod
    @lru_cache()
    def get_basis_grad_fns(cls, 
                                 order:int = 1, 
                                 dtype:torch.dtype = torch.float32,
                                 device:torch.device = torch.device('cpu')
                                 )->Polynomials:
        r"""Get the gradient of basis functions with respect to reference coordinates.

        For a basis function :math:`\phi(x_1,\ldots,x_n)`, computes the gradient:

        .. math::

            \nabla \phi = \left[\frac{\partial \phi}{\partial x_1}, \ldots, \frac{\partial \phi}{\partial x_n}\right]

        The gradient is computed by taking the partial derivative of each basis function
        with respect to each coordinate direction :math:`x_i` in the reference element.

        Examples
        --------
        >>> # Get gradients of linear basis functions for a triangle
        >>> from tensormesh.element import Triangle
        >>> grad_fns = Triangle.get_basis_grad_fns(order=1)
        >>> points = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        >>> grads = grad_fns(points)  # [n_points=2, dim=2, n_basis=3]
        >>> grads.shape
        torch.Size([2, 2, 3])

        >>> # Get gradients of quadratic basis functions for a tetrahedron
        >>> from tensormesh.element import Tetrahedron
        >>> grad_fns = Tetrahedron.get_basis_grad_fns(order=2)
        >>> points = torch.tensor([[0.25, 0.25, 0.25]])
        >>> grads = grad_fns(points)  # [n_points=1, dim=3, n_basis=10]
        >>> grads.shape
        torch.Size([1, 3, 10])

        Parameters
        ----------
        order : int 
            The order of the basis functions
        dtype : torch.dtype
            The float data type of the polynomial
        device : torch.device
            The device to place the polynomial on

        Returns
        -------
        Polynomials
            The gradient basis functions. Shape is :math:`[D, N_b]` where
            - :math:`D` = spatial dimension
            - :math:`N_b` = number of basis functions
            - n_vars = :math:`D`
            - n_terms = :math:`N_b`
        """
        basis_fns = cls.get_basis_fns(order, dtype, device) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        return basis_fns.grad() # Polynomials [dim, n_basis] n_vars=dim n_terms = n_basis

    @classmethod 
    def eval_cell_jacobian(cls, 
                        quadrature:torch.Tensor, 
                        element_coords:torch.Tensor, 
                        basis_order:int=1)->torch.Tensor:
        r"""Evaluate the Jacobian matrix at quadrature points for each element.

        The Jacobian matrix represents the linear transformation from reference coordinates to physical coordinates.
        It is computed by multiplying the element coordinates by the gradients of the basis functions.

        For each quadrature point q and element e, computes:

        .. math::

            J_{e,q,i,j} = \sum_b \text{coords}_{e,b,j} \frac{\partial \phi_b}{\partial x_i}(q)

        where:

        * :math:`\text{coords}_{e,b,j}` is the j-th coordinate of basis node b in element e
        * :math:`\phi_b` is basis function b  
        * :math:`\frac{\partial \phi_b}{\partial x_i}` is the derivative of basis function b with respect to reference coordinate i

        Examples
        --------
        .. code-block:: python

            element = Triangle
            quadrature = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
            element_coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
            jacobian = element.eval_cell_jacobian(quadrature, element_coords)
            print(jacobian.shape)  # torch.Size([1, 2, 2, 2])

        Parameters
        ----------
        quadrature : torch.Tensor
            Shape :math:`[N_q, D]` or :math:`[N_e, N_q, D]` where:
            
            * :math:`N_q` = number of quadrature points
            * :math:`N_e` = number of elements
            * :math:`D` = spatial dimension
            
            Quadrature points for each element
        element_coords : torch.Tensor 
            Shape :math:`[N_e, N_b, D]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_b` = number of basis nodes
            * :math:`D` = spatial dimension
            
            Physical coordinates of element nodes
        basis_order : int, optional
            Order of basis functions, by default 1

        Returns
        -------
        torch.Tensor
            Shape :math:`[N_e, N_q, D, D]` where:
            
            * :math:`N_e` = number of elements
            * :math:`N_q` = number of quadrature points
            * :math:`D` = spatial dimension
            
            Jacobian matrix evaluated at each quadrature point for each element
        """
        assert quadrature.dtype == element_coords.dtype, f"quadrature dtype {quadrature.dtype} != element_coords dtype {element_coords.dtype}"
        assert quadrature.device == element_coords.device, f"quadrature device {quadrature.device} != element_coords device {element_coords.device}"
        dtype  = quadrature.dtype
        device = quadrature.device

        shape_grad_fns = cls.get_basis_grad_fns(basis_order, dtype, device) # Polynomials  [dim, n_basis] n_vars=dim n_terms = n_basis
        
        if quadrature.dim() == 2:
            n_basis        = shape_grad_fns.shape[1]
            shape_grad     = shape_grad_fns.map(quadrature) # [n_quadrature, dim, n_basis]
            cell_jacobian  = torch.einsum("ebj,qib->eqij", element_coords, shape_grad)
        else:
            n_element, n_quadrature_per_element, dim = quadrature.shape
            assert n_element == element_coords.shape[0]
            n_basis        = element_coords.shape[1]
            shape_grad     = shape_grad_fns.map(quadrature.reshape(-1, dim)) # [n_element,n_quadrature, dim, n_basis]
            shape_grad     = shape_grad.reshape(n_element, n_quadrature_per_element, dim, n_basis)
            cell_jacobian  = torch.einsum("ebj,eqib->eqij", element_coords, shape_grad)

        return cell_jacobian

    @classmethod 
    def eval_shape_val(cls, 
                       quadrature_points:Optional[torch.Tensor] = None, 
                       order:int = 1, 
                       quadrature_order:int=1,
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->torch.Tensor:
        r"""
        Evaluates the shape functions at quadrature points.

        For a given set of quadrature points, evaluates each basis function phi_i at those points.
        Returns a matrix where entry (q,i) is the value of basis function i at quadrature point q.

        The shape functions have two key properties:

        1. Partition of unity: The functions sum to 1 at any point
        2. Interpolation property: phi_i(x_j) = delta_ij at the nodes

        For example, for linear elements (order=1):

        .. math::
            \phi_1(x) &= 1-x \quad \text{(equals 1 at x=0, 0 at x=1)} \\
            \phi_2(x) &= x \quad \text{(equals 0 at x=0, 1 at x=1)}

        For quadratic elements (order=2), includes additional mid-node basis functions.

        If quadrature_points are not provided, uses default quadrature rule of specified order.

        Examples
        --------
        .. code-block:: python

            import torch
            from tensormesh.element import Element

            # Create quadrature points
            quad_points = torch.tensor([[0.0], [0.5], [1.0]])
            
            # Evaluate shape functions at quadrature points
            shape_vals = Element.eval_shape_val(quad_points, order=1)
            print(shape_vals.shape)  # [3, 2] for linear element

        Parameters
        ----------
        quadrature_points : Optional[torch.Tensor]
            Tensor of shape :math:`[N_q, D]` containing quadrature points, where :math:`N_q` = number of quadrature points and :math:`D` = spatial dimension
        order : int
            Order of the basis functions
        quadrature_order : int 
            The order for quadrature, if quadrature is not None, quadrature_order will be ignored
        dtype : torch.dtype
            The float data type of the shape value, if quadrature_points is not None, the dtype will be ignored.
            Default is torch.float32
        device : torch.device
            The device of the shape value, if quadrature_points is not None, the device will be ignored.
            Default is torch.device('cpu')

        Returns
        -------
        torch.Tensor
            Tensor of shape :math:`[N_q, N_b]` containing shape function values, where :math:`N_q` = number of quadrature points and :math:`N_b` = number of basis functions
        """
        dtype = quadrature_points.dtype if quadrature_points is not None else dtype
        device= quadrature_points.device if quadrature_points is not None else device
        if quadrature_points is None:
            _, quadrature_points = cls.get_quadrature(quadrature_order, dtype, device)
        basis_fns = cls.get_basis_fns(order, dtype, device) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        shape_val = basis_fns.map(quadrature_points) # [n_quadrature, n_basis]
        return shape_val 
    
    @classmethod 
    def eval_shape_grad(cls, 
                        element_coords:torch.Tensor, 
                        quadrature:Optional[torch.Tensor] = None, 
                        basis_order:int=1, 
                        quadrature_order:int=1
                        )->Tensorx2:
        r"""
        Evaluates the shape function gradients and cell Jacobians at quadrature points.

        For each element, computes:

        1. Shape function gradients in reference coordinates
        2. Cell Jacobian mapping from reference to physical coordinates 
        3. Transforms gradients to physical coordinates using inverse Jacobian

        The shape function gradients are used in weak form assembly:

        .. math::
            \nabla \phi_i = \sum_j J^{-1}_{jk} \frac{\partial \phi_i}{\partial \xi_k}

        where:

        - :math:`\phi_i` are the basis functions
        - :math:`J` is the Jacobian matrix
        - :math:`\xi_k` are reference coordinates

        The cell Jacobian :math:`J_{ij} = \frac{\partial x_i}{\partial \xi_j}` maps derivatives between reference and physical coordinates.

        For linear elements (order=1), the Jacobian is constant within each element.
        For higher order elements, the Jacobian varies spatially.

        Examples
        --------
        .. code-block:: python

            import torch
            from tensormesh.element import Element

            # Create element coordinates
            element_coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
            
            # Evaluate shape function gradients and Jacobian
            shape_grad, cell_jacobian = Element.eval_shape_grad(element_coords, basis_order=1)
            print(shape_grad.shape)      # [1, n_quad, 3, 2] 
            print(cell_jacobian.shape)   # [1, n_quad, 2, 2]

        Parameters
        ----------
        element_coords: torch.Tensor 
            Tensor of shape :math:`[N_e, N_b, D]` containing element vertex coordinates, where:
            - :math:`N_e` = number of elements
            - :math:`N_b` = number of basis functions
            - :math:`D` = spatial dimension
        quadrature: torch.Tensor
            Tensor of shape :math:`[N_q, D]` containing quadrature points for each element, where:
            - :math:`N_q` = number of quadrature points
            - :math:`D` = spatial dimension
        basis_order: int 
            Order for basis functions
        quadrature_order: int 
            Order for quadrature, if element_coords is not None, the quadrature_order will be ignored,
            default is 1

        Returns
        -------
        shape_grad: torch.Tensor
            Tensor of shape :math:`[N_e, N_q, N_b, D]` containing shape function gradients, where:
            - :math:`N_e` = number of elements
            - :math:`N_q` = number of quadrature points
            - :math:`N_b` = number of basis functions
            - :math:`D` = spatial dimension
        cell_jacobian: torch.Tensor 
            Tensor of shape :math:`[N_e, N_q, D, D]` containing cell Jacobians, where:
            - :math:`N_e` = number of elements
            - :math:`N_q` = number of quadrature points
            - :math:`D` = spatial dimension
        """
  
        dtype = element_coords.dtype
        device= element_coords.device
        if quadrature is None:
            _, quadrature = cls.get_quadrature(quadrature_order, dtype, device) # [n_quadrature, dim]

        shape_grad_fns = cls.get_basis_grad_fns(basis_order, dtype, device) # Polynomials  [dim, n_basis]
    
        shape_grad     = shape_grad_fns.map(quadrature) # [n_qudrature, dim, n_basis]

        cell_jacobian  = torch.einsum("ebj,qib->eqij", element_coords, shape_grad) # TODO: check the order
        
        inv_cell_jacobian = torch.inverse(cell_jacobian)

        shape_grad     = torch.einsum("qib, eqji->eqbj", shape_grad, inv_cell_jacobian) 

        return shape_grad, cell_jacobian

    @classmethod
    @lru_cache()
    def get_local_facet_mapping_fns(cls, 
                                    dtype:torch.dtype=torch.float32,
                                    device:torch.device=torch.device('cpu')
                                    )->Polynomials:
        r"""Maps points from the reference facet to the facet of the reference element.

        For a facet :math:`f`, defines a mapping :math:`\gamma_f(r)` from reference facet coordinates :math:`r` 
        to reference element coordinates :math:`\xi`:

        .. math::
            \gamma_f: \hat{F} \rightarrow F

        where:
        - :math:`\hat{F}` is the reference facet (e.g. line for 2D, triangle for 3D)
        - :math:`F` is the facet of the reference element
        - :math:`\gamma_f(r)` is a linear mapping

        Examples
        --------
        .. code-block:: python

            import torch
            from tensormesh.element import Triangle
            
            # Get mapping functions for triangle element facets
            facet_fns = Triangle.get_local_facet_mapping_fns()
            
            # Map points from reference facet to element facet
            ref_points = torch.tensor([[0.0], [0.5], [1.0]])  # Points on reference line
            mapped_points = facet_fns.map(ref_points)  # Points on triangle facet

        Parameters
        ----------
        dtype: torch.dtype
            the float data type of the polynomials
            default is torch.float32
        device: torch.device
            the device of the polynomials
            default is torch.device('cpu')

        Returns
        -------
        local_facet_fns : Polynomials
            Polynomial functions mapping reference facet to element facet coordinates.
            Shape is :math:`[N_f, D]` where:
            
            - :math:`N_f` = number of facets
            - :math:`D` = spatial dimension
            
            Each polynomial has:
            
            - :math:`D-1` variables (reference facet coordinates)
            - :math:`D` terms (affine transformation)
            
            The mapping takes the form:
            
            .. math::
                \gamma_f(r) = c + \sum_{i=1}^{D-1} \alpha_i r_i
                
            where :math:`c, \alpha_i \in \mathbb{R}^D` are the coefficients
        """
        facet = {
            2 : cls.edge, 
            3 : cls.face 
        }[cls.dim]
    
        facet_dim = cls.dim - 1
    
        if cls.dim == 2:
            origin_facet_coords = Line.points[Line.vertex] # [2(points), 1, 1(dim-1)]
            origin_facet_coords = origin_facet_coords.permute(1, 0, 2) # [1, 2(points), 1(dim-1)]
            transf_facet_coords = cls.points[facet]    # [n_facet, n_edge, 1(dim-1)]
        elif cls.dim == 3:
            origin_facet_coords = Triangle.points[Triangle.vertex] # [3(points), n_triangle, 2(dim-1)]
            origin_facet_coords = origin_facet_coords.permute(1, 0, 2) # [n_triangle, 3(points), 2(dim-1)]
            transf_facet_coords = cls.points[torch.tensor([x[:3] for x in facet])] # [n_triangle, 3(points), 3(dim)]
        else:
            raise Exception(f"Invalid dim {cls.dim}")
        
        transf_facet_coords = transf_facet_coords.type(dtype).to(device) # [n_facet, n_vertex_per_facet, dim]   
       
        polys = Polynomial.lin_exp(facet_dim, dtype, device) # n_vars=dim-1(n_vars), n_terms=dim
        polys = polys.repeat(len(facet), cls.dim)         # [n_face, dim] n_vars=dim-1 n_terms=dim
       
        origin_facet_coords = origin_facet_coords.repeat(len(facet), 1, 1) 
        origin_facet_coords = origin_facet_coords.type(dtype).to(device)    # [n_face, n_vertex_per_facet, dim-1]

        # lstsq returns the trailing axis indexed by output coordinate; polys._coef
        # expects it indexed by basis term, so swap the last two axes.
        result = torch.linalg.lstsq(polys.get_exp_terms(origin_facet_coords), transf_facet_coords)
        x      = result.solution.transpose(-1, -2).contiguous()  # [n_facet, dim, n_terms]

        assert x.shape == (len(facet), cls.dim, polys.n_terms), \
            f"facet mapping coef shape {tuple(x.shape)} != {(len(facet), cls.dim, polys.n_terms)}"
        polys.reset_coef(x)
        return polys
    
    @classmethod 
    @lru_cache()
    def get_local_facet_mapping_grad_fns(cls, 
                                         dtype:torch.dtype=torch.float32,
                                         device:torch.device=torch.device('cpu')
                                         )->Polynomials:
        r"""Computes the gradient of the facet mapping function with respect to the reference coordinates.

        For a facet mapping function :math:`\gamma_f(r)` that maps from reference coordinates :math:`r` to global coordinates,
        computes the Jacobian matrix:

        .. math::
            \frac{\partial \gamma_f (r)}{\partial r}

        This gives the derivatives of each global coordinate with respect to each reference coordinate.

        Examples
        --------
        .. code-block:: python

            element = Triangle()
            facet_mapping_grad = element.get_local_facet_mapping_grad_fns()
            # Returns polynomial tensor with derivatives of mapping
            # Shape: [1, 3, 2] for 2D triangle element (1 ref coord, 3 edges, 2 global coords)
        Parameters
        ----------
        dtype: torch.dtype
            the float data type of the polynomials
            default is torch.float32
        device: torch.device
            the device of the polynomials
            default is torch.device('cpu')

        Returns
        -------
        PolynomialTensor
            Tensor containing the Jacobian with shape :math:`[D-1, N_f, D]` where:
            
            * :math:`D-1` = number of reference coordinates (gradient dimension)
            * :math:`N_f` = number of facets
            * :math:`D` = spatial dimension (global coordinates)
            
            The polynomial has :math:`n_{vars}=D-1` variables and :math:`n_{terms}=D` terms.
        """
        local_facet_fns = cls.get_local_facet_mapping_fns(dtype, device) # Polynomials [n_facet, dim] n_vars=dim-1 n_terms = dim
        return local_facet_fns.grad() # [dim-1(gradient), n_facet, dim(global_coords)]

    @classmethod 
    @lru_cache()
    def get_outwards_facet_normal(cls, 
                                  dtype:torch.dtype=torch.float32,
                                  device:torch.device=torch.device('cpu')
                                  )->Tensorx1:
        """Computes the outward unit normal vectors for each facet of the element.

        This returns the geometric normal vectors without the Nanson scale factor.
        For 2D elements, returns normals for each edge.
        For 3D elements, returns normals for each face.

        Examples
        --------
        .. code-block:: python

            element = Triangle()
            normals = element.get_outwards_facet_normal()
            # Returns tensor of shape [3, 2] containing outward normals
            # for each edge of the triangle

        Parameters
        ----------
        dtype: torch.dtype
            The float data type for the output tensor
            Default is torch.float32
        device: torch.device
            The device to place the output tensor on
            Default is torch.device('cpu')

        Returns
        -------
        outward_facet_normal : torch.Tensor
            Tensor of shape :math:`[N_f, D]` containing the outward unit normal vector for each facet, where:
            
            * :math:`N_f` = number of facets (edges in 2D, faces in 3D)
            * :math:`D` = spatial dimension (2 or 3)
        """
        outwards_normal_fns = {
            2: outwards_normal_2d,
            3: outwards_normal_3d
        }
        facet = {
            2: cls.edge,
            3: cls.face
        }
        return outwards_normal_fns[cls.dim](cls.points.type(dtype).to(device), facet[cls.dim])

    @classmethod
    @lru_cache()
    def get_n_facet(cls) -> int:
        """Number of facets of the reference element.

        Edges for 2D shapes, faces for 3D shapes.

        Returns
        -------
        int
            Number of facets.
        """
        dim2nfacet = {
            2: len(cls.edge),
            3: len(cls.face),
        }
        return dim2nfacet[cls.dim]

    @classmethod
    @lru_cache()
    def get_n_basis(cls, order: int = 1) -> int:
        """Number of basis (interpolation) nodes at the given order.

        Parameters
        ----------
        order : int, optional
            Polynomial order. Defaults to 1.

        Returns
        -------
        int
            Number of basis nodes.
        """
        return cls.get_basis(order).shape[0]

    @classmethod
    def eval_facet_cell_jacobian(cls, 
                                 element_coords:torch.Tensor,
                                 basis_order:int=1,
                                 quadrature_order:int=1
                                 )->Union[Tensorx1,Tensorx2]:
        r"""
        Evaluates the Jacobian matrix of the mapping from reference to physical coordinates at facet quadrature points.

        For each element and facet, computes the Jacobian matrix :math:`\mathbf{J} = \frac{\partial\mathbf{x}}{\partial\boldsymbol{\xi}}` 
        at the quadrature points on that facet. The Jacobian describes how the reference element coordinates 
        :math:`\boldsymbol{\xi}` map to physical coordinates :math:`\mathbf{x}`.

        For mixed elements with both triangular and quadrilateral facets (e.g. pyramids), returns separate Jacobians
        for each facet type.

        The Jacobian matrix components are:

        .. math::

            J_{ij} = \frac{\partial x_i}{\partial \xi_j}

        Examples
        --------
        .. code-block:: python

            # Create a triangle element
            element = Triangle()
            # Element coordinates for a single triangle 
            coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
            # Evaluate facet Jacobians using linear basis and 2nd order quadrature
            jacobians = element.eval_facet_cell_jacobian(coords, basis_order=1, quadrature_order=2)
            # jacobians shape: [1, 3, 2, 2, 2] for 1 element, 3 edges, 2 quad points per edge

        Parameters
        ----------
        element_coords : torch.Tensor
            3D Tensor of shape :math:`[N_e, N_b, D]` containing the physical coordinates of element nodes,
            where :math:`N_e` = number of elements, :math:`N_b` = number of basis nodes, :math:`D` = spatial dimension
        basis_order : int
            Order of the basis functions used for the mapping
        quadrature_order : int 
            Order of quadrature rule used for facet integration

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            For elements with uniform facets, a 5D tensor of shape
            :math:`[N_e, N_f, N_q, D, D]`.

            For elements with mixed facets (:class:`Prism`,
            :class:`Pyramid`), a pair
            ``(tri_facet_cell_jacobian, quad_facet_cell_jacobian)`` of shapes
            :math:`[N_e, N_{tf}, N_{tq}, D, D]` and
            :math:`[N_e, N_{qf}, N_{qq}, D, D]`.

        Notes
        -----
        The Jacobian is used to:

        * Transform integrals from reference to physical space:
          
          .. math::
             
             \int_K f(\mathbf{x}) \,d\mathbf{x} = \int_{\hat{K}} f(\mathbf{x}(\boldsymbol{\xi})) \vert\det(\mathbf{J})\vert \,d\boldsymbol{\xi}

        * Compute geometric quantities like normal vectors and surface measures
        * Map derivatives between reference and physical coordinates:

          .. math::
             
             \nabla_{\mathbf{x}} = \mathbf{J}^{-T} \nabla_{\boldsymbol{\xi}}
        """
        device = element_coords.device
        dtype  = element_coords.dtype
        if cls.is_mix_facet:
            tri_w, tri_q, quad_w, quad_q = cls.get_facet_quadrature(
                                            quadrature_order, 
                                            dtype=dtype, device=device)
            n_tri_facet, n_quadrature_per_tri_facet, _ = tri_q.shape
            n_quad_facet, n_quadrature_per_quad_facet, _ = quad_q.shape
            q = torch.cat([tri_q.reshape(-1,  3), quad_q.reshape(-1, 3)], 0)
            j = cls.eval_cell_jacobian(q, element_coords, basis_order)
            tri_j = j[:, :n_tri_facet * n_quadrature_per_tri_facet]
            quad_j= j[:, n_tri_facet * n_quadrature_per_tri_facet:]
            tri_j = tri_j.reshape(-1, n_tri_facet, n_quadrature_per_tri_facet, cls.dim, cls.dim)
            quad_j= quad_j.reshape(-1, n_quad_facet, n_quadrature_per_quad_facet, cls.dim, cls.dim)
            return tri_j, quad_j
        else:
            w, q = cls.get_facet_quadrature(quadrature_order, dtype=dtype, device=device) 
            n_facet, n_quadrature_per_facet, _ = q.shape
            q    = q.reshape(-1, cls.dim)
            j = cls.eval_cell_jacobian(q, element_coords, basis_order) # [n_element, n_facet*n_quadrature_per_facet, dim, dim]
            return j.reshape(-1, n_facet, n_quadrature_per_facet, cls.dim, cls.dim)

    @classmethod 
    def eval_facet_jacobian(cls,
                            element_coords:torch.Tensor, 
                            basis_order:int=1,
                            quadrature_order:int=1,
                            )->Union[Tensorx1,Tensorx2]:
        r"""Evaluate the Jacobian mapping from reference facet coordinates to physical coordinates.

        For each facet of the element, computes the Jacobian matrix that maps from local facet 
        coordinates to global physical coordinates.

        The facet Jacobian is used to:

        * Transform integrals over facets from reference to physical space:
          
          .. math::
             
             \int_F f(\mathbf{x}) \,ds = \int_{\hat{F}} f(\mathbf{x}(\boldsymbol{\xi})) \|\mathbf{J}\mathbf{t}\| \,d\boldsymbol{\xi}

        * Compute geometric quantities like normal vectors and surface measures
        * Map derivatives between reference facet and physical coordinates:

          .. math::
             
             \nabla_{\mathbf{x}} = \mathbf{J}^{-T} \nabla_{\boldsymbol{\xi}}

        where :math:`\mathbf{t}` is a tangent vector to the facet.

        For mixed elements like pyramids that have both triangular and quadrilateral facets,
        returns separate Jacobians for each facet type.

        Examples
        --------
        .. code-block:: python

            # Create a triangle element
            element_coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
            # Evaluate facet Jacobian with linear basis and 2nd order quadrature 
            facet_jacobian = Triangle.eval_facet_jacobian(element_coords,
                                                        basis_order=1,
                                                        quadrature_order=2)
            facet_jacobian.shape  # torch.Size([1, 3, 2, 1, 2])
        Parameters
        ----------
        element_coords : torch.Tensor :math:`[N_e, N_b, D]`
            Boundary element coordinates, where :math:`N_e` = number of elements, :math:`N_b` = number of basis functions, :math:`D` = spatial dimension
        basis_order : int
            The order of the basis
        quadrature_order : int
            The order of quadrature rule to use

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            For elements with uniform facets, a 5D tensor of shape
            :math:`[N_e, N_f, N_q, D-1, D]` where :math:`N_e` is the number
            of elements, :math:`N_f` the number of facets per element,
            :math:`N_q` the number of quadrature points per facet, and
            :math:`D` the spatial dimension.

            For elements with mixed facets (:class:`Prism`,
            :class:`Pyramid`), a pair
            ``(tri_facet_jacobian, quad_facet_jacobian)`` of shapes
            :math:`[N_e, N_{tf}, N_{tq}, D-1, D]` and
            :math:`[N_e, N_{qf}, N_{qq}, D-1, D]`, respectively.
        """

        dtype = element_coords.dtype
        device= element_coords.device
  
        facet_quadrature_orig     = cls.get_facet_quadrature(order=quadrature_order, 
                                                             transform=False, 
                                                             dtype=dtype, 
                                                             device=device) 

        if cls.is_mix_facet:
            assert len(facet_quadrature_orig) == 4

            tri_j, quad_j = cls.eval_facet_cell_jacobian(element_coords, basis_order, quadrature_order)

            facet_mapping_grad_fns = cls.get_local_facet_mapping_grad_fns(dtype, device) # Polynomials [dim-1(gradient), n_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim

            tri_mask  = cls.get_tri_mask() # [n_facet]

            tri_facet_mapping_grad_fns = facet_mapping_grad_fns[:, tri_mask]   # Polynomials [dim-1(gradient), n_tri_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim
            quad_facet_mapping_grad_fns= facet_mapping_grad_fns[:, ~tri_mask]  # Polynomials [dim-1(gradient), n_quad_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim

            n_tri_facet = tri_facet_mapping_grad_fns.shape[1]
            n_quad_facet= quad_facet_mapping_grad_fns.shape[1]

            assert isinstance(tri_facet_mapping_grad_fns, Polynomials) and tri_facet_mapping_grad_fns.shape == (cls.dim-1, n_tri_facet, cls.dim) 
            assert isinstance(quad_facet_mapping_grad_fns, Polynomials) and quad_facet_mapping_grad_fns.shape == (cls.dim-1, n_quad_facet, cls.dim)

            tri_w, tri_q,quad_w, quad_q = facet_quadrature_orig 
            # triw, triq [n_quadrature_per_tri_facet] [n_quadrature_per_tri_facet, dim-1]

            tri_facet_mapping_grad = tri_facet_mapping_grad_fns.map(tri_q)  # [n_quadrature_per_tri_face,  dim-1(gradient), n_tri_facet, dim(global_coords)]
            quad_facet_mapping_grad= quad_facet_mapping_grad_fns.map(quad_q)# [n_quadrature_per_quad_face, dim-1(gradient), n_quad_facet, dim(global_coords)]

            tri_facet_mapping_grad = torch.einsum("efqij, qgfi-> efqgj",tri_j, tri_facet_mapping_grad)   #[n_element, n_tri_facet, n_quadrature_per_tri_face, dim-1(gradient), dim(global_coords)]
            quad_facet_mapping_grad= torch.einsum("efqij, qgfi-> efqgj",quad_j, quad_facet_mapping_grad) #[n_element, n_quad_facet, n_quadrature_per_quad_face, dim-1(gradient), dim(global_coords)]

            return tri_facet_mapping_grad, quad_facet_mapping_grad

        else:
            assert len(facet_quadrature_orig) == 2
            w, q                   = facet_quadrature_orig
    
            j                      = cls.eval_facet_cell_jacobian(element_coords, basis_order, quadrature_order)

            facet_mapping_grad_fns = cls.get_local_facet_mapping_grad_fns(dtype, device)        # PolynomialTensor [dim-1(gradient), n_facet, dim(global_coords)] n_vars=dim-1 n_terms = dim
            facet_mapping_grad     = facet_mapping_grad_fns.map(q) # [n_quadrature_per_face, dim-1(gradient), n_facet, dim(global_coords)]
            facet_mapping_grad     = torch.einsum("efqij, qkfi-> efqkj",j, facet_mapping_grad) #[n_element, n_facet, n_quadrature_per_face, dim-1(gradient), dim(global_coords)]

            return facet_mapping_grad

    @classmethod
    def element_to_facet(cls, 
                         elements:torch.Tensor,
                         order:int,
                         )->Union[Tensorx1,Tensorx2]:
        """
        Maps element basis functions to facet basis functions.

        For each facet of an element, extracts the subset of element basis functions that are active on that facet.

        For mixed elements with both triangular and quadrilateral facets, returns separate tensors for each facet type.

        For example:

        .. code-block:: python

            # Hexahedral element (order=1)
            hex_element = torch.randn(8)  # 8 basis functions
            hex_facets = element_to_facet(hex_element, order=1)  # [6, 4] tensor
            
            # Tetrahedral element (order=2)
            tet_element = torch.randn(10)  # 10 basis functions
            tet_facets = element_to_facet(tet_element, order=2)  # [4, 6] tensor
            
            # Wedge element (order=2)
            wedge_element = torch.randn(18)  # 18 basis functions
            tri_facets, quad_facets = element_to_facet(wedge_element, order=2)
            # tri_facets: [2, 6] tensor for triangular faces
            # quad_facets: [3, 9] tensor for quadrilateral faces

        Parameters
        ----------
        elements : torch.Tensor
            2D tensor of shape :math:`[N_e, N_b]` containing element basis function values
            or 1D tensor of shape :math:`[N_b]` for a single element,
            where:

            - :math:`N_e` = number of elements
            - :math:`N_b` = number of basis functions

        order : int
            Order of the basis functions

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            For elements with uniform facets (``is_mix_facet`` is ``False``),
            a tensor of shape :math:`[N_e, N_f, N_b]` (3D input) or
            :math:`[N_f, N_b]` (1D input), where :math:`N_e` is the number
            of elements, :math:`N_f` the number of facets, and :math:`N_b`
            the number of basis functions per facet.

            For elements with mixed facets, a pair
            ``(tri_facet, quad_facet)`` of shapes
            :math:`[\ldots, N_{tf}, N_{tb}]` and
            :math:`[\ldots, N_{qf}, N_{qb}]`, respectively.
        """
        assert elements.shape[-1] == cls.get_n_basis(order)
        

        if cls.is_mix_facet:
            tri_facet, quad_facet = cls.get_facet(order)
            tri_facet  = elements[..., tri_facet] # [n_element, n_tri_facet, n_basis_per_tri_facet]
            quad_facet = elements[..., quad_facet]# [n_element, n_quad_facet, n_basis_per_quad_facet]
            return tri_facet, quad_facet
        
        else: # facet of the same shape 
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            return elements[..., facet]

    @classmethod
    def element_to_edge(cls, 
                         elements:torch.Tensor,
                         order:int,
                         unique:bool = True)->Tensorx1:
        """
        Extract edge values from element values.

        Maps element basis function values to edge basis function values using the edge connectivity.

        Parameters
        ----------
        elements : torch.Tensor
            ND tensor of shape [..., N_b] containing basis function values,
            where N_b is the number of basis functions. The leading dimensions
            can be arbitrary (e.g., batch dimensions).

        order : int
            Order of the basis functions
        
        unique : bool, optional
            If True, returns only unique edges after sorting edge values.
            If False, returns all edges including duplicates.
            Default is True.

        Returns
        -------
        edge : torch.Tensor
            (N+1)D tensor of shape [..., N_ed, N_eb] containing edge basis function values,
            where:

            - [...] = input tensor's leading dimensions
            - N_ed = number of edges
            - N_eb = number of basis functions per edge
        """
        assert elements.shape[-1] == cls.get_n_basis(order)

        edges = cls.get_edge(order) # [n_edge, 2]
        edges = elements[..., edges].reshape(-1, 2) # [n_all_edges, 2]
        
        if unique:
            
            edges = edges.sort(-1)[0]
        
            edges = torch.unique(edges, dim=0)
          
        return edges
        

    @classmethod 
    @lru_cache()
    def get_tri_mask(cls)->torch.Tensor:
        """
        Returns a boolean mask indicating which faces are triangular.

        For elements with mixed face types (triangular and quadrilateral faces), returns a boolean tensor 
        marking which faces are triangular (True) vs quadrilateral (False).

        The mask has length equal to the total number of faces and True values for each triangular face.

        Examples
        --------
        For a prism element with 2 triangular faces and 3 quadrilateral faces:

        .. code-block:: python

            >>> Element.get_tri_mask()
            tensor([True, True, False, False, False])

        Returns
        -------
        tri_mask: torch.Tensor 
            1D boolean Tensor of shape :math:`[N_f]` where :math:`N_f` = number of facets
        """
        assert cls.is_mix_facet 
        return torch.tensor([len(face) == 3 for face in cls.face])

    @classmethod
    @lru_cache()
    def get_quad_mask(cls)->torch.Tensor:
        """
        Returns a boolean mask indicating which faces are quadrilateral.

        For elements with mixed face types (triangular and quadrilateral faces), returns a boolean tensor
        marking which faces are quadrilateral (True) vs triangular (False).

        The mask has length equal to the total number of faces and True values for each quadrilateral face.

        Examples
        --------
        For a prism element with 2 triangular faces and 3 quadrilateral faces:

        .. code-block:: python

            >>> Element.get_quad_mask()
            tensor([False, False, True, True, True])

        Returns
        -------
        quad_mask: torch.Tensor 
            1D boolean Tensor of shape :math:`[N_f]` where :math:`N_f` = number of facets
        """
        assert cls.is_mix_facet 
        return torch.tensor([len(face) == 4 for face in cls.face])

    @classmethod 
    @lru_cache()
    def get_contour(cls, order:int)->torch.Tensor:
        """
        Returns the contour points for a 2D element at the specified order.

        For a 2D element, returns points along the element boundary at the given polynomial order.
        The points are ordered counterclockwise around the boundary.

        Examples
        --------
        Get contour points for a quadrilateral at order 2:

        >>> quad = Quadrilateral()
        >>> contour = quad.get_contour(2)
        >>> contour.shape
        torch.Size([8, 2])

        Parameters
        ----------
        order : int
            The polynomial order. Must be >= 1.

        Returns
        -------
        contour : torch.Tensor
            Tensor of shape :math:`[N_p, 2]` containing the :math:`(x,y)` coordinates of points 
            along the element boundary, ordered counterclockwise, where :math:`N_p` = number of points.

        Notes
        -----
        - Only implemented for 2D elements (dim=2)
        - Points are distributed according to the polynomial order
        - Points are ordered counterclockwise starting from the first vertex
        """
        assert order >= 1
        assert cls.dim == 2, f"Only 2d elements has contour, but got {cls.dim}d element"
        raise NotImplementedError()
        

class Line(Element):
    """1D line reference element on :math:`[0, 1]`.

    The facet of every 2D element (edge of a triangle or quad) is a
    :class:`Line`. Type strings: ``"line"`` (linear), ``"line3"``,
    ``"line4"``, …, ``"line11"``.
    """

    points = torch.tensor([[0.0],[1.0]]) # 2x1
    vertex = torch.tensor([[0], [1]]) # 2x1
    edge   = torch.tensor([[0, 1]]) # 1x2
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = 0
    n_cell   = 0

    @classmethod
    @lru_cache()
    @wraps(Element.get_basis)
    def get_basis(cls, order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                    )->Tensorx1:
        return lin_basis(
                cls.points.type(dtype).to(device), 
                cls.vertex, cls.edge, order)

    @classmethod 
    @lru_cache()
    @wraps(Element.get_polynomial)
    def get_polynomial(cls, 
                       order:int=1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(1, order, dtype, device)

    @classmethod 
    @lru_cache()
    @wraps(Element.get_quadrature)
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return lin_quadrature(order, dtype, device)

class Triangle(Element):
    """2D triangle reference element with vertices at
    :math:`(0,0), (1,0), (0,1)`.

    Simplex ordering: vertex nodes first, then edge nodes, then interior
    nodes. The facets are line segments. Type strings: ``"triangle"``
    (linear), ``"triangle6"``, ``"triangle10"``, …, ``"triangle66"``.
    """

    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0]]) # 3x2
    vertex = torch.tensor([[0], [1], [2]]) # 3x1
    edge   = torch.tensor([[1, 2], [0, 2], [0, 1]]) # 3x2
    face   = ((0, 1, 2),) # 1x3
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

    @classmethod
    @lru_cache()
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return tri_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         torch.tensor(cls.face), 
                         order
                         )

    @classmethod
    @lru_cache()
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.poly_exp(2, order, dtype, device)
    
    @classmethod
    @lru_cache()
    def get_facet(cls, order:int=1)->Tensorx1:
        facet = facet_basis_index_2d(cls.vertex, cls.edge, order)
        return facet

    @classmethod
    @lru_cache() 
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return tri_quadrature(order, dtype, device)

    @classmethod
    @lru_cache()
    def get_facet_quadrature(cls, 
                             order:int=1, 
                             transform=True, 
                             dtype:torch.dtype=torch.float32,
                             device:torch.device=torch.device('cpu')
                             )->Tensorx2:
        return facet_quadrature_2d(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    order, transform)
        
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Line
    
    @classmethod 
    @lru_cache()
    def get_contour(cls, order:int)->torch.Tensor:
        if order == 1:
            return torch.tensor([0, 1, 2])
        elif order > 1:
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            facet = facet[:, 2:] # facet without boundaries
            return torch.cat([
                torch.tensor([0]),
                facet[2],
                torch.tensor([1]),
                facet[0],
                torch.tensor([2]),
                facet[1].flip([0])
            ])
        else:
            raise Exception(f"Invalid order {order}")

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Triangle elements.
        """
        # n_nodes = (n+1)(n+2)/2  =>  n is polynomial order
        disc = 9 + 8 * n_nodes
        n = int((-3 + math.isqrt(disc)) // 2)
        if (n + 1) * (n + 2) // 2 != n_nodes:
            raise ValueError(f"Triangle.get_gmsh_permutation: invalid n_nodes={n_nodes}")

        def _reorder_list(p: List[int], order: int) -> List[int]:
            # Inner recursion terminator: order-0 triangle has exactly one node
            if order == 0:
                return p
            idx0 = [p[0], p[1], p[2]]
            if order <= 1:
                return idx0
            n_edge_nodes = order - 1
            edges = p[3: 3 + 3 * n_edge_nodes]
            e12 = edges[0:n_edge_nodes]
            e23 = edges[n_edge_nodes:2 * n_edge_nodes]
            e31 = edges[2 * n_edge_nodes:3 * n_edge_nodes]
            e13 = list(reversed(e31))
            idx1 = e23 + e13 + e12
            if order <= 2:
                return idx0 + idx1
            inner = p[3 + 3 * n_edge_nodes:]
            # For triangles, removing the boundary reduces the order by 3 (matches the legacy implementation)
            idx2 = _reorder_list(inner, order - 3)
            return idx0 + idx1 + idx2

        perm = torch.tensor(_reorder_list(list(range(n_nodes)), n), dtype=torch.long, device=device)
        return perm

class Quadrilateral(Element):
    """2D quadrilateral reference element on :math:`[0, 1]^2`.

    Tensor-product Lagrange nodes in **lexicographic** order — note this
    differs from the Gmsh/VTK boundary-spiral order, so loading external
    high-order quads must go through :meth:`Element.reorder`. The facets
    are line segments. Type strings: ``"quad"`` (linear), ``"quad8"``,
    ``"quad9"``, ``"quad16"``, …, ``"quad121"``.
    """

    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]) # 4x2
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # 4x2
    face   = ((0, 1, 2, 3),) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

   
    @classmethod
    @lru_cache()
    def get_basis(cls, 
                  order:int = 1, 
                  dtype:torch.dtype = torch.float32,
                  device:torch.device = torch.device('cpu')
                  )->Tensorx1:
        return quad_basis(
                    cls.points.type(dtype).to(device), 
                    cls.vertex, 
                    cls.edge, 
                    torch.tensor(cls.face), 
                    order)
    
    @classmethod
    @lru_cache()
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32,
                       device:torch.device=torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(2, order, dtype, device)

    @classmethod
    @lru_cache()
    def get_facet(cls, order:int=1):
        facet = facet_basis_index_2d(cls.vertex, cls.edge, order)
        return facet

    @classmethod 
    @lru_cache()
    def get_quadrature(cls, 
                       order:int = 1, 
                       dtype:torch.dtype=torch.float32, 
                       device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return quad_quadrature(order, dtype, device)

    @classmethod
    @lru_cache()
    def get_contour(cls, order:int)->torch.Tensor:
        # Returns indices for CCW contour: 0->1->3->2
        # (0,0) -> (1,0) -> (1,1) -> (0,1)
        return torch.tensor([0, 1, 3, 2])
       
    @classmethod
    @lru_cache()
    def get_facet_quadrature(cls,  
                             order:int = 1, 
                             transform:bool = True, 
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return facet_quadrature_2d(
                cls.get_local_facet_mapping_fns(dtype, device), 
                order, transform)

    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Line
    
    @classmethod
    @lru_cache()
    def get_coutour(cls, order:int)->torch.Tensor:
        if order == 1:
            return torch.tensor([0, 1, 3, 2])
        elif order > 1:
            facet = cls.get_facet(order) # [n_facet, n_basis_per_facet]
            facet = facet[:, 2:] # facet without boundaries
            return torch.cat([
                torch.tensor([0]),
                facet[0],
                torch.tensor([1]),
                facet[2],
                torch.tensor([3]),
                facet[3].flip([0]),
                torch.tensor([2]),
                facet[1].flip([0])
            ])
        else:
            raise Exception(f"Invalid order {order}")

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Quadrilateral elements.
        """
        side = int(math.isqrt(n_nodes))
        if side * side != n_nodes:
            raise ValueError(f"Quadrilateral.get_gmsh_permutation: invalid n_nodes={n_nodes}")

        def _reorder_list(p: List[int], side_n: int) -> List[int]:
            idx0 = [p[0], p[1], p[3], p[2]]
            if side_n <= 2:
                return idx0
            n_edge_nodes = side_n - 2
            edges = p[4: 4 + 4 * n_edge_nodes]
            e12 = edges[0:n_edge_nodes]
            e23 = edges[n_edge_nodes:2 * n_edge_nodes]
            e34 = edges[2 * n_edge_nodes:3 * n_edge_nodes]
            e41 = edges[3 * n_edge_nodes:4 * n_edge_nodes]
            e14 = list(reversed(e41))
            e43 = list(reversed(e34))
            idx1 = e12 + e14 + e23 + e43
            if side_n <= 3:
                return idx0 + idx1 + [p[-1]]
            inner = p[4 + 4 * n_edge_nodes:]
            idx2 = _reorder_list(inner, side_n - 2)
            return idx0 + idx1 + idx2

        perm = torch.tensor(_reorder_list(list(range(n_nodes)), side), dtype=torch.long, device=device)
        return perm

class Tetrahedron(Element):
    """3D tetrahedron reference element with vertices at
    :math:`(0,0,0), (1,0,0), (0,1,0), (0,0,1)`.

    Simplex ordering: vertex / edge / face / interior groupings, with
    indices within each group following the TensorMesh convention (see
    :ref:`node-ordering-gallery`). The facets are triangles. Type
    strings: ``"tetra"`` (linear), ``"tetra10"``, ``"tetra20"``, …,
    ``"tetra286"``.
    """

    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 4x3
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]) # 6x2
    face   = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)) # 4x3
    cell   = torch.tensor([[0, 1, 2, 3]]) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    @classmethod
    @lru_cache()
    @wraps(Element.get_basis)
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return tet_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         torch.tensor(cls.face), 
                         cls.cell,  # type: ignore
                         order)

    @classmethod
    @lru_cache() 
    def get_polynomial(cls, 
                        order:int = 1, 
                        dtype:torch.dtype = torch.float32,
                        device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.poly_exp(3, order, dtype, device)
    
    @classmethod
    @lru_cache() 
    def get_quadrature(cls, 
                        order:int = 1, 
                        dtype:torch.dtype=torch.float32,
                        device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return tet_quadrature(order, dtype, device)
       
    @classmethod 
    @lru_cache()
    def get_facet(cls, order: int = 1) -> Tensorx1:
        index = tet_facet_basis_index(
            cls.vertex, 
            cls.edge, 
            torch.tensor(cls.face), 
            order)
        return index
    
    @classmethod
    @lru_cache()
    def get_facet_quadrature(cls, 
                                order:int = 1, 
                                transform:bool = True, 
                                dtype:torch.dtype = torch.float32,  
                                device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return tet_facet_quadrature(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    order, 
                    transform)
    
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Triangle

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Tetrahedron elements.
        """
        # Supported node counts:
        # - Tet4 (p1), Tet10 (p2), Tet20 (p3) via legacy implementation below
        # - Tet35 (p4) via explicit permutation (legacy did not handle p4 reliably)
        if n_nodes == 35:
            return torch.tensor(
                [
                    0, 1, 2, 3, 19, 20, 21, 10, 11, 12,
                    18, 17, 16, 13, 14, 15, 7, 8, 9, 4,
                    5, 6, 28, 29, 30, 23, 24, 22, 25, 27,
                    26, 31, 33, 32, 34,
                ],
                device=device,
                dtype=torch.long,
            )

        if n_nodes not in (4, 10, 20):
            # fall back to identity for unsupported higher orders
            return torch.arange(n_nodes, device=device, dtype=torch.long)

        # Infer order from node count
        order_map = {4: 1, 10: 2, 20: 3}
        n = order_map[n_nodes]

        def _reorder_list(p: List[int], order: int) -> List[int]:
            idx0 = [p[0], p[1], p[2], p[3]]
            if order <= 1:
                return idx0

            n_edge_nodes = order - 1
            edges = p[4: 4 + 6 * n_edge_nodes]
            # legacy split order: 12, 23, 31, 41, 43, 42 (1-based labels)
            e12 = edges[0:n_edge_nodes]
            e23 = edges[n_edge_nodes:2 * n_edge_nodes]
            e31 = edges[2 * n_edge_nodes:3 * n_edge_nodes]
            e41 = edges[3 * n_edge_nodes:4 * n_edge_nodes]
            e43 = edges[4 * n_edge_nodes:5 * n_edge_nodes]
            e42 = edges[5 * n_edge_nodes:6 * n_edge_nodes]

            e13 = list(reversed(e31))
            e14 = list(reversed(e41))
            e24 = list(reversed(e42))
            e34 = list(reversed(e43))

            if order == 2:
                idx1 = e24 + e34 + e23 + e14 + e13 + e12
            else:
                # order == 3 in legacy implementation
                idx1 = e34 + e24 + e23 + e14 + e13 + e12

            if order <= 2:
                return idx0 + idx1

            # Faces are handled as a flat tail in legacy implementation (order 3 supported)
            faces = p[4 + 6 * n_edge_nodes:]
            # split into 4 faces equally
            chunk = len(faces) // 4
            f123 = faces[0:chunk]
            f124 = faces[chunk:2 * chunk]
            f134 = faces[2 * chunk:3 * chunk]
            f234 = faces[3 * chunk:4 * chunk]
            idx2 = f234 + f134 + f124 + f123
            return idx0 + idx1 + idx2

        perm = torch.tensor(_reorder_list(list(range(n_nodes)), n), dtype=torch.long, device=device)
        return perm

class Hexahedron(Element):
    """3D hexahedron reference element on :math:`[0, 1]^3`.

    Tensor-product Lagrange nodes in **lexicographic** order — note this
    differs substantially from Gmsh/VTK, which walks along edges. The
    facets are quadrilaterals. Type strings: ``"hexahedron"`` (linear),
    ``"hexahedron20"``, ``"hexahedron27"``, ``"hexahedron64"``, …,
    ``"hexahedron1000"``.
    """

    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]) # 8x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]) # 12x2
    face   = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6), (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7)) # 6x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]) # 1x8
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    @classmethod
    @lru_cache()
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32,
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return hex_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge,
                         torch.tensor(cls.face), 
                         cls.cell,  # type: ignore
                         order)
    
    @classmethod
    @lru_cache() 
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.tens_exp(3, order, dtype, device)
        
    @classmethod 
    @lru_cache()
    def get_quadrature(cls, 
                        order:int = 1, 
                        dtype:torch.dtype = torch.float32,
                        device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return hex_quadrature(order, dtype, device) 
       
    @classmethod 
    @lru_cache()
    def get_facet(cls, order:int=1)->Tensorx1:
        index = hex_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    torch.tensor(cls.face), 
                    order)
        return index

    @classmethod
    @lru_cache()
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx2:
        return hex_facet_quadrature(
            cls.get_local_facet_mapping_fns(dtype, device), 
            order, transform)
        
    @classmethod 
    def get_facet_type(cls)->Type['Element']:
        return Quadrilateral

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Hexahedron elements.

        Supports Hex8 (linear) and common Lagrange orders used in examples (p2/p3/p4).
        For other high orders, falls back to identity.
        """
        if n_nodes == 27:  # p2
            return torch.tensor(
                [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26],
                device=device,
                dtype=torch.long,
            )
        if n_nodes == 64:  # p3
            return torch.tensor(
                [
                    0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 14, 15, 18, 19, 10, 11,
                    24, 25, 28, 29, 30, 31, 26, 27, 12, 13, 16, 17, 22, 23, 20, 21,
                    40, 41, 42, 43, 44, 45, 46, 47, 36, 37, 38, 39, 48, 49, 50, 51,
                    32, 33, 34, 35, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                ],
                device=device,
                dtype=torch.long,
            )
        if n_nodes == 125:  # p4
            return torch.tensor(
                [
                    0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 17, 18, 19, 23, 24, 25,
                    11, 12, 13, 32, 33, 34, 38, 39, 40, 41, 42, 43, 35, 36, 37,
                    14, 15, 16, 20, 21, 22, 29, 30, 31, 26, 27, 28,
                    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                    53, 54, 55, 56, 57, 58, 59, 60, 61,
                    80, 81, 82, 83, 84, 85, 86, 87, 88,
                    44, 45, 46, 47, 48, 49, 50, 51, 52,
                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
                ],
                device=device,
                dtype=torch.long,
            )
        if n_nodes != 8:
            return torch.arange(n_nodes, device=device, dtype=torch.long)

        # VTK Hex8 order (standard): bottom face CCW then top face CCW
        # 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
        #
        # TensorMesh internal points for Hexahedron are lexicographic:
        # 0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) 4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
        #
        # So to convert VTK -> internal, we need: [0,1,3,2,4,5,7,6]
        return torch.tensor([0, 1, 3, 2, 4, 5, 7, 6], device=device, dtype=torch.long)

class Pyramid(Element):
    """3D pyramid reference element with a unit square base and apex at
    :math:`(0, 0, 1)`.

    Mixed-facet element: one quadrilateral base + four triangular sides
    (see :attr:`Element.is_mix_facet`). Type strings: ``"pyramid"``
    (linear), ``"pyramid14"``.
    """

    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4]]) # 5x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 8x2
    face   = ((0, 1, 2, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)) # 5x4
    cell   = torch.tensor([[0, 1, 2, 3, 4]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    is_mix_facet = True
   
    @classmethod
    @lru_cache()
    def get_basis(cls, 
                  order:int = 1, 
                  dtype:torch.dtype = torch.float32,
                  device:torch.device = torch.device('cpu')
                  )->Tensorx1:
        return pyr_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         cls.face, # type: ignore
                         cls.cell, # type: ignore
                         order) 

    @classmethod 
    @lru_cache()
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,  
                       device:torch.device= torch.device('cpu')
                       )->Polynomial:
        return Polynomial.pyr_exp(order, dtype, device)

    @classmethod
    @lru_cache()
    def get_quadrature(cls, 
                        order:int=1, 
                        dtype:torch.dtype=torch.float32, 
                        device:torch.device=torch.device('cpu')
                       )->Tensorx2:
        return pyr_quadrature(order, dtype, device)
       
    @classmethod
    @lru_cache() 
    def get_facet(cls, order:int=1)->Tensorx2:
        assert isinstance(cls.face, tuple)
        tri_facet, quad_facet = mix_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    cls.face, 
                    order) # type: ignore
        return tri_facet, quad_facet

    @classmethod
    @lru_cache()
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx4:
        tri_w, tri_q,\
        quad_w,quad_q = mix_facet_quadrature_3d(
                    cls.get_local_facet_mapping_fns(dtype, device), 
                    cls.face, order, transform)
        return (tri_w.type(dtype), 
                tri_q.type(dtype), 
                quad_w.type(dtype), 
                quad_q.type(dtype))

    @classmethod 
    def get_facet_type(cls)->Tuple[Type['Element'],Type['Element']]:
        return Triangle, Quadrilateral

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Pyramid elements.

        Currently supports Pyramid5 (linear). Higher order pyramids fall back to identity.
        """
        if n_nodes != 5:
            return torch.arange(n_nodes, device=device, dtype=torch.long)

        # VTK Pyramid5 order: base quad around then apex
        # base: 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) apex:4
        # TensorMesh Pyramid points are lexicographic on the base:
        # 0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) apex:4
        # VTK -> internal: [0,1,3,2,4]
        return torch.tensor([0, 1, 3, 2, 4], device=device, dtype=torch.long)

class Prism(Element):
    """3D triangular prism (wedge) reference element — triangle base
    :math:`\\times` line height.

    Mixed-facet element: two triangular caps + three quadrilateral sides
    (see :attr:`Element.is_mix_facet`). Note the *string* convention
    ``"wedge"`` differs from the *class* name ``Prism`` (carried over from
    meshio). Type strings: ``"wedge"`` (linear), ``"wedge18"``,
    ``"wedge40"``, ``"wedge75"``, …, ``"wedge550"``.
    """

    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5]]) # 6x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 9x2
    face   = ((0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5), (1, 2, 4, 5),(3,4,5)) # 4x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

    is_mix_facet = True
 
    @classmethod
    @lru_cache()
    def get_basis(cls, 
                  order:int=1, 
                  dtype:torch.dtype=torch.float32, 
                  device:torch.device=torch.device('cpu')
                  )->Tensorx1:
        return pri_basis(cls.points.type(dtype).to(device), 
                         cls.vertex, 
                         cls.edge, 
                         cls.face,  # type: ignore
                         cls.cell,  # type: ignore
                         order)

    @classmethod
    @lru_cache() 
    def get_polynomial(cls, 
                       order:int = 1, 
                       dtype:torch.dtype = torch.float32,  
                       device:torch.device = torch.device('cpu')
                       )->Polynomial:
        return Polynomial.pri_exp(order, dtype, device)
   
    @classmethod 
    @lru_cache()
    def get_quadrature(cls, 
                       order:int = 1,
                       dtype:torch.dtype = torch.float32,
                       device:torch.device = torch.device('cpu')
                       )->Tensorx2:
        return pri_quadrature(order, dtype, device)
       
    @classmethod
    @lru_cache()
    def get_facet(cls, order:int=1)->Tensorx2:
        assert isinstance(cls.face, tuple)
        tri_facet, quad_facet = mix_facet_basis_index(
                    cls.vertex, 
                    cls.edge, 
                    cls.face, 
                    order)
        return tri_facet, quad_facet

    @classmethod 
    @lru_cache()
    def get_facet_quadrature(cls, 
                             order:int = 1, 
                             transform:bool = True,
                             dtype:torch.dtype = torch.float32,
                             device:torch.device = torch.device('cpu')
                             )->Tensorx4:
        tri_w, tri_q,\
        quad_w,quad_q = mix_facet_quadrature_3d(
                            cls.get_local_facet_mapping_fns(dtype, device), 
                            cls.face, order, transform)
        return (tri_w.type(dtype), 
                tri_q.type(dtype), 
                quad_w.type(dtype), 
                quad_q.type(dtype))
    
    @classmethod 
    def get_facet_type(cls)->Tuple[Type['Element'],Type['Element']]:
        return Triangle, Quadrilateral

    @classmethod
    @lru_cache()
    def get_gmsh_permutation(cls, n_nodes: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Permutation mapping Gmsh/VTK ordering -> TensorMesh internal ordering for Prism/Wedge elements.

        Supports Prism6 (linear) and common Lagrange orders used in examples (p2/p3/p4).
        For other high orders, falls back to identity.
        """
        if n_nodes == 18:  # p2
            return torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16],
                device=device,
                dtype=torch.long,
            )
        if n_nodes == 40:  # p3
            return torch.tensor(
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 9, 8, 18, 19, 22, 23, 21, 20,
                    10, 11, 14, 15, 16, 17, 24, 37, 25, 26, 27, 28, 33, 34, 35, 36,
                    30, 29, 32, 31, 38, 39,
                ],
                device=device,
                dtype=torch.long,
            )
        if n_nodes == 75:  # p4
            return torch.tensor(
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 11, 10, 9,
                    24, 25, 26, 30, 31, 32, 29, 28, 27,
                    12, 13, 14, 18, 19, 20, 21, 22, 23,
                    33, 34, 35, 63, 64, 65, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    54, 55, 56, 57, 58, 59, 60, 61, 62,
                    47, 46, 45, 50, 49, 48, 53, 52, 51,
                    66, 67, 68, 69, 70, 71, 72, 73, 74,
                ],
                device=device,
                dtype=torch.long,
            )
        if n_nodes != 6:
            return torch.arange(n_nodes, device=device, dtype=torch.long)
        # For the linear wedge, TensorMesh reference vertex order matches VTK/Wedge6:
        # bottom triangle (0,1,2) then top triangle (3,4,5).
        return torch.arange(6, device=device, dtype=torch.long)





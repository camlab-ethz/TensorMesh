"""
Batch Poisson Solver using TensorMesh FEM framework.

This module provides a GPU-accelerated batch solver for the Poisson equation:
    -Δu = f  in Ω
    u = 0    on ∂Ω (Dirichlet boundary condition)

The solver uses finite element method (FEM) with TensorMesh's efficient
sparse matrix operations to solve multiple Poisson problems in parallel.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Literal

# Import TensorMesh components
from ...mesh import Mesh
from ...assemble import LaplaceElementAssembler, NodeAssembler
from ...operator import Condenser
from ...sparse import SparseMatrix


class BatchPoissonSolver(nn.Module):
    """
    Batch Poisson Solver using Finite Element Method.
    
    Solves the Poisson equation -Δu = f with zero Dirichlet boundary conditions
    for multiple source terms f simultaneously.
    
    Parameters
    ----------
    mesh : Mesh
        The finite element mesh. Can be 2D (rectangle, circle, etc.) or 3D (cube, etc.)
    device : str or torch.device, optional
        Device to run computations on. Default is 'cuda' if available, else 'cpu'.
    dtype : torch.dtype, optional
        Data type for computations. Default is torch.float32.
    solver_backend : str, optional
        Backend for sparse linear solve. Options: 'scipy', 'petsc', 'torch'.
        Default is None (auto-select).
    
    Examples
    --------
    >>> import torch
    >>> from tensormesh import Mesh
    >>> from tensormesh.dataset.equation import BatchPoissonSolver
    >>> 
    >>> # Create a 2D mesh
    >>> mesh = Mesh.gen_rectangle(chara_length=0.05)
    >>> solver = BatchPoissonSolver(mesh, device='cuda')
    >>> 
    >>> # Create batch of source terms: [batch_size, n_nodes]
    >>> batch_size = 32
    >>> f = torch.randn(batch_size, mesh.n_points, device='cuda')
    >>> 
    >>> # Solve for u
    >>> u = solver.solve(f)  # [batch_size, n_nodes]
    
    Attributes
    ----------
    mesh : Mesh
        The finite element mesh.
    K : SparseMatrix
        The condensed stiffness matrix (Laplacian).
    condenser : Condenser
        The boundary condition condenser.
    """
    
    def __init__(
        self,
        mesh: Mesh,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        solver_backend: Optional[str] = None,
    ):
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.dtype = dtype
        self.solver_backend = solver_backend
        
        # Move mesh to device
        self.mesh = mesh.to(device)
        
        # Build stiffness matrix
        assembler = LaplaceElementAssembler.from_mesh(self.mesh)
        K_full = assembler(self.mesh.points)
        
        # Build mass matrix for RHS assembly: F = M @ f
        from tensormesh.assemble import MassElementAssembler
        mass_assembler = MassElementAssembler.from_mesh(self.mesh)
        self.M_full = mass_assembler(self.mesh.points)
        
        # Setup boundary conditions (zero Dirichlet on boundary)
        dirichlet_value = torch.zeros(
            self.mesh.boundary_mask.sum().item(), 
            device=device, 
            dtype=dtype
        )
        self.condenser = Condenser(self.mesh.boundary_mask, dirichlet_value)
        
        # Condense stiffness matrix (remove boundary DOFs)
        self.K, _ = self.condenser(K_full, torch.zeros(self.mesh.n_points, device=device, dtype=dtype))
        
        # Get inner DOF mask for extracting F_inner from F_full
        self.is_inner = ~self.mesh.boundary_mask
        
        # Cache dimensions
        self.n_points = self.mesh.n_points
        self.n_inner = self.K.shape[0]
        self.dim = self.mesh.points.shape[-1]
        
        # Pre-compute LU factorization for fast batch solves
        self._lu_cache = None
        self._precompute_lu()
    
    def _precompute_lu(self):
        """Pre-compute and cache LU factorization of stiffness matrix."""
        import scipy.sparse
        import scipy.sparse.linalg
        
        # Convert K to scipy sparse matrix
        K = self.K
        edata = K.edata.detach().cpu().numpy()
        row = K.row.detach().cpu().numpy()
        col = K.col.detach().cpu().numpy()
        shape = K.shape
        
        A_csc = scipy.sparse.coo_matrix((edata, (row, col)), shape=shape).tocsc()
        
        # Compute LU factorization for CPU
        self._lu_cache = scipy.sparse.linalg.splu(A_csc)
        
        # For CUDA, cache the cupy LU factorization
        self._cupy_lu = None
        if self.device.type == "cuda":
            try:
                import cupy
                import cupyx.scipy.sparse
                import cupyx.scipy.sparse.linalg
                from tensormesh.sparse.utils import tensor2cupy
                
                cupy.cuda.Device(self.device.index or 0).use()
                
                # Convert to cupy sparse matrix and compute LU
                edata_cp = tensor2cupy(K.edata.detach())
                row_cp = tensor2cupy(K.row.detach())
                col_cp = tensor2cupy(K.col.detach())
                A_cupy = cupyx.scipy.sparse.coo_matrix(
                    (edata_cp, (row_cp, col_cp)), shape=shape
                ).tocsc()
                
                # Pre-compute LU factorization on GPU
                self._cupy_lu = cupyx.scipy.sparse.linalg.splu(A_cupy)
                print(f"[BatchPoissonSolver] CuPy LU factorization cached on GPU")
            except Exception as e:
                print(f"[BatchPoissonSolver] CuPy LU failed: {e}, falling back to CPU")
    
    def _solve_with_cached_lu(self, F_condensed: torch.Tensor) -> torch.Tensor:
        """Solve using cached LU factorization."""
        
        if self.device.type == "cuda" and self._cupy_lu is not None:
            # Use cached CuPy LU on GPU (supports batch RHS)
            import cupy
            from tensormesh.sparse.utils import tensor2cupy, cupy2tensor
            
            cupy.cuda.Device(self.device.index or 0).use()
            
            # Convert torch tensor to cupy array
            F_cp = tensor2cupy(F_condensed)
            
            # Solve using cached LU (splu.solve supports 2D RHS)
            u_cp = self._cupy_lu.solve(F_cp)
            
            # Convert back to torch tensor
            u_inner = cupy2tensor(u_cp)
            return u_inner
        else:
            # Use scipy LU on CPU
            F_cpu = F_condensed.detach().cpu().numpy()
            u_inner_np = self._lu_cache.solve(F_cpu)
            u_inner = torch.tensor(u_inner_np, dtype=F_condensed.dtype, device=self.device)
            return u_inner
    
    def assemble_rhs(self, f: torch.Tensor) -> torch.Tensor:
        """
        Assemble the right-hand side vector from source term values at nodes.
        
        This performs the FEM integration: F_i = ∫ f(x) * φ_i(x) dx
        where φ_i are the basis functions.
        
        Parameters
        ----------
        f : torch.Tensor
            Source term values at mesh nodes.
            Shape: [n_nodes] or [batch_size, n_nodes]
        
        Returns
        -------
        torch.Tensor
            Assembled RHS vector(s) at inner (non-boundary) nodes.
            Shape: [n_inner] or [batch_size, n_inner]
        """
        # Handle batch dimension
        if f.dim() == 1:
            f = f.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = f.shape[0]
        
        # Assemble RHS: need to transpose for NodeAssembler
        # f: [batch_size, n_nodes] -> [n_nodes, batch_size]
        f_T = f.T  # [n_nodes, batch_size]
        
        # Use the source term assembler
        F = self._f_assembler(self.mesh.points, point_data={"f": f_T})
        # F: [n_nodes, batch_size]
        
        # Condense (remove boundary DOFs)
        _, F_condensed = self.condenser(self.K, F)
        # F_condensed: [n_inner, batch_size]
        
        # Transpose back: [batch_size, n_inner]
        F_out = F_condensed.T
        
        if squeeze_output:
            F_out = F_out.squeeze(0)
        
        return F_out
    
    def solve(
        self, 
        f: torch.Tensor,
        tol: float = 1e-6,
        max_iter: int = 10000,
    ) -> torch.Tensor:
        """
        Solve the Poisson equation -Δu = f with zero Dirichlet BC.
        
        Parameters
        ----------
        f : torch.Tensor
            Source term values at mesh nodes.
            Shape: [n_nodes] or [batch_size, n_nodes]
        tol : float, optional
            Tolerance for iterative solver. Default is 1e-6.
        max_iter : int, optional
            Maximum iterations for iterative solver. Default is 10000.
        
        Returns
        -------
        torch.Tensor
            Solution values at all mesh nodes (including boundary).
            Shape: [n_nodes] or [batch_size, n_nodes]
        """
        # Handle batch dimension
        if f.dim() == 1:
            f = f.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = f.shape[0]
        
        # Assemble RHS using mass matrix: F = M @ f
        # f: [batch_size, n_nodes] -> [n_nodes, batch_size]
        f_T = f.T
        
        # F_full = M @ f_T, result: [n_nodes, batch_size]
        F_full = self.M_full @ f_T
        
        # Extract inner DOFs (condense)
        # F_condensed: [n_inner, batch_size]
        F_condensed = F_full[self.is_inner]
        
        # Solve the linear system using cached LU: K @ u_inner = F_condensed
        u_inner = self._solve_with_cached_lu(F_condensed)
        # u_inner: [n_inner, batch_size]
        
        # Recover full solution (add back boundary DOFs with zero values)
        u_full = self.condenser.recover(u_inner)
        # u_full: [n_nodes, batch_size]
        
        # Transpose to batch-first format: [batch_size, n_nodes]
        u_out = u_full.T
        
        if squeeze_output:
            u_out = u_out.squeeze(0)
        
        return u_out
    
    def forward(self, f: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - alias for solve()."""
        return self.solve(f, **kwargs)
    
    @classmethod
    def create_2d(
        cls,
        domain: Literal["rectangle", "circle", "L"] = "rectangle",
        chara_length: float = 0.05,
        order: int = 1,
        element_type: str = "tri",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        **mesh_kwargs,
    ) -> "BatchPoissonSolver":
        """
        Create a 2D Poisson solver with specified domain.
        
        Parameters
        ----------
        domain : str
            Domain type: 'rectangle', 'circle', or 'L'.
        chara_length : float
            Characteristic mesh element length.
        order : int
            Polynomial order of elements.
        element_type : str
            Element type ('tri' for triangles, 'quad' for quadrilaterals).
        device : str or torch.device, optional
            Device for computations.
        dtype : torch.dtype, optional
            Data type for computations.
        **mesh_kwargs : dict
            Additional arguments passed to Mesh generator.
        
        Returns
        -------
        BatchPoissonSolver
            Configured solver instance.
        """
        if domain == "rectangle":
            mesh = Mesh.gen_rectangle(
                chara_length=chara_length, 
                order=order, 
                element_type=element_type,
                **mesh_kwargs
            )
        elif domain == "circle":
            mesh = Mesh.gen_circle(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        elif domain == "L":
            mesh = Mesh.gen_L(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        else:
            raise ValueError(f"Unknown 2D domain: {domain}")
        
        return cls(mesh, device=device, dtype=dtype)
    
    @classmethod
    def create_3d(
        cls,
        domain: Literal["cube", "sphere", "cylinder"] = "cube",
        chara_length: float = 0.1,
        order: int = 1,
        element_type: str = "tet",
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        **mesh_kwargs,
    ) -> "BatchPoissonSolver":
        """
        Create a 3D Poisson solver with specified domain.
        
        Parameters
        ----------
        domain : str
            Domain type: 'cube', 'sphere', or 'cylinder'.
        chara_length : float
            Characteristic mesh element length.
        order : int
            Polynomial order of elements.
        element_type : str
            Element type ('tet' for tetrahedra, 'hex' for hexahedra).
        device : str or torch.device, optional
            Device for computations.
        dtype : torch.dtype, optional
            Data type for computations.
        **mesh_kwargs : dict
            Additional arguments passed to Mesh generator.
        
        Returns
        -------
        BatchPoissonSolver
            Configured solver instance.
        """
        if domain == "cube":
            mesh = Mesh.gen_cube(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        elif domain == "sphere":
            mesh = Mesh.gen_sphere(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        elif domain == "cylinder":
            mesh = Mesh.gen_cylinder(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        else:
            raise ValueError(f"Unknown 3D domain: {domain}")
        
        return cls(mesh, device=device, dtype=dtype)


class _SourceTermAssembler(NodeAssembler):
    """
    Internal assembler for integrating source term f over elements.
    
    Computes F_i = ∫ f(x) * φ_i(x) dx for each node i.
    """
    
    def forward(self, v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        v : torch.Tensor
            Basis function values at quadrature points. Shape: [n_qpts_per_elem]
        f : torch.Tensor
            Source term values at nodes. Shape: [n_nodes] or [n_nodes, batch_size]
        
        Returns
        -------
        torch.Tensor
            Contribution to RHS. Shape: same as f at this node.
        """
        # v: basis function value (scalar or [n_qpts])
        # f: source term value at this node
        return v * f


# Convenience function
def solve_poisson_batch(
    f: torch.Tensor,
    mesh: Optional[Mesh] = None,
    domain: str = "rectangle",
    chara_length: float = 0.05,
    dim: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    **solver_kwargs,
) -> Tuple[torch.Tensor, Mesh]:
    """
    Convenience function to solve batch Poisson equations.
    
    Parameters
    ----------
    f : torch.Tensor
        Source term values at mesh nodes.
        Shape: [n_nodes] or [batch_size, n_nodes]
    mesh : Mesh, optional
        Pre-built mesh. If None, creates one from domain specification.
    domain : str
        Domain type if mesh is None.
    chara_length : float
        Mesh characteristic length if mesh is None.
    dim : int
        Spatial dimension (2 or 3) if mesh is None.
    device : str or torch.device, optional
        Device for computations.
    dtype : torch.dtype, optional
        Data type.
    **solver_kwargs : dict
        Additional arguments for solver.solve().
    
    Returns
    -------
    u : torch.Tensor
        Solution values. Shape: [n_nodes] or [batch_size, n_nodes]
    mesh : Mesh
        The mesh used for solving.
    """
    if mesh is None:
        if dim == 2:
            solver = BatchPoissonSolver.create_2d(
                domain=domain,
                chara_length=chara_length,
                device=device,
                dtype=dtype,
            )
        else:
            solver = BatchPoissonSolver.create_3d(
                domain=domain,
                chara_length=chara_length,
                device=device,
                dtype=dtype,
            )
    else:
        solver = BatchPoissonSolver(mesh, device=device, dtype=dtype)
    
    u = solver.solve(f, **solver_kwargs)
    return u, solver.mesh


"""
Batch Linear Elasticity Solver using TensorMesh FEM framework.

This module provides a GPU-accelerated batch solver for the linear elasticity equations:
    -∇·σ = f  in Ω
    u = 0     on ∂Ω (Dirichlet boundary condition)

where:
    σ = λ tr(ε) I + 2μ ε  (Hooke's law)
    ε = (∇u + (∇u)^T) / 2  (infinitesimal strain)

The solver uses finite element method (FEM) with TensorMesh's efficient
sparse matrix operations to solve multiple elasticity problems in parallel.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Literal

# Import TensorMesh components
from ...mesh import Mesh
from ...assemble import LinearElasticityElementAssembler, MassElementAssembler, NodeAssembler
from ...operator import Condenser
from ...sparse import SparseMatrix


class BatchLinearElasticitySolver(nn.Module):
    """
    Batch Linear Elasticity Solver using Finite Element Method.
    
    Solves the linear elasticity equation -∇·σ = f with zero Dirichlet boundary 
    conditions for multiple body force fields f simultaneously.
    
    Parameters
    ----------
    mesh : Mesh
        The finite element mesh. Can be 2D or 3D.
    E : float, optional
        Young's modulus. Default is 1.0.
    nu : float, optional
        Poisson's ratio. Default is 0.3.
    device : str or torch.device, optional
        Device to run computations on. Default is 'cuda' if available.
    dtype : torch.dtype, optional
        Data type for computations. Default is torch.float64.
    solver_backend : str, optional
        Backend for sparse linear solve. Options: 'scipy', 'petsc', 'torch'.
    
    Examples
    --------
    >>> import torch
    >>> from tensormesh import Mesh
    >>> from tensormesh.dataset.equation import BatchLinearElasticitySolver
    >>> 
    >>> # Create a 2D mesh
    >>> mesh = Mesh.gen_rectangle(chara_length=0.05)
    >>> solver = BatchLinearElasticitySolver(mesh, E=1.0, nu=0.3, device='cuda')
    >>> 
    >>> # Create batch of body forces: [batch_size, n_nodes, dim]
    >>> batch_size = 32
    >>> f = torch.randn(batch_size, mesh.n_points, 2, device='cuda')
    >>> 
    >>> # Solve for displacement u: [batch_size, n_nodes, dim]
    >>> u = solver.solve(f)
    
    Attributes
    ----------
    mesh : Mesh
        The finite element mesh.
    K : SparseMatrix
        The condensed stiffness matrix.
    condenser : Condenser
        The boundary condition condenser.
    dim : int
        Spatial dimension (2 or 3).
    """
    
    def __init__(
        self,
        mesh: Mesh,
        E: float = 1.0,
        nu: float = 0.3,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        solver_backend: Optional[str] = None,
        cuda_solver: Literal['bicgstab', 'lu'] = 'bicgstab',
    ):
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.dtype = dtype
        self.solver_backend = solver_backend
        self.cuda_solver = cuda_solver
        self.E = E
        self.nu = nu
        
        # Move mesh to device
        self.mesh = mesh.to(device)
        self.dim = self.mesh.points.shape[-1]
        
        # Build stiffness matrix using linear elasticity assembler
        assembler = LinearElasticityElementAssembler.from_mesh(
            self.mesh, E=E, nu=nu
        )
        K_full = assembler(self.mesh.points)
        
        # Build mass matrix for RHS assembly: F = M @ f
        mass_assembler = MassElementAssembler.from_mesh(self.mesh)
        self.M_full = mass_assembler(self.mesh.points)
        
        # Setup boundary conditions (zero Dirichlet on all boundary DOFs)
        # For vector problems, we need to expand the boundary mask to all DOFs
        boundary_mask_scalar = self.mesh.boundary_mask  # [n_points]
        
        # Expand to vector DOFs: [n_points * dim]
        # DOF ordering: [u0_x, u0_y, u1_x, u1_y, ...] for 2D
        boundary_mask_vector = boundary_mask_scalar.unsqueeze(-1).expand(-1, self.dim).reshape(-1)
        
        dirichlet_value = torch.zeros(
            boundary_mask_vector.sum().item(),
            device=device,
            dtype=dtype
        )
        self.condenser = Condenser(boundary_mask_vector, dirichlet_value)
        
        # Condense stiffness matrix (remove boundary DOFs)
        n_dofs = self.mesh.n_points * self.dim
        self.K, _ = self.condenser(K_full, torch.zeros(n_dofs, device=device, dtype=dtype))
        
        # Get inner DOF mask
        self.is_inner = ~boundary_mask_vector
        
        # Cache dimensions
        self.n_points = self.mesh.n_points
        self.n_dofs = n_dofs
        self.n_inner = self.K.shape[0]
        
        # Pre-compute LU factorization or prepare for iterative solver
        self._lu_cache = None
        self._cupy_lu = None
        self._cupy_K = None
        self._precompute_lu()
    
    def _precompute_lu(self):
        """Pre-compute and cache LU factorization of stiffness matrix (CPU only or when cuda_solver='lu')."""
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
        
        # For CUDA with LU solver, cache the cupy LU factorization
        if self.device.type == "cuda" and self.cuda_solver == 'lu':
            try:
                import cupy
                import cupyx.scipy.sparse
                import cupyx.scipy.sparse.linalg
                from tensormesh.sparse.utils import tensor2cupy
                
                cupy.cuda.Device(self.device.index or 0).use()
                
                edata_cp = tensor2cupy(K.edata.detach())
                row_cp = tensor2cupy(K.row.detach())
                col_cp = tensor2cupy(K.col.detach())
                A_cupy = cupyx.scipy.sparse.coo_matrix(
                    (edata_cp, (row_cp, col_cp)), shape=shape
                ).tocsc()
                
                self._cupy_lu = cupyx.scipy.sparse.linalg.splu(A_cupy)
                print(f"[BatchLinearElasticitySolver] CuPy LU factorization cached on GPU")
            except Exception as e:
                print(f"[BatchLinearElasticitySolver] CuPy LU failed: {e}, falling back to CPU")
        
        # For CUDA with BiCGSTAB, cache the cupy sparse matrix
        if self.device.type == "cuda" and self.cuda_solver == 'bicgstab':
            try:
                import cupy
                import cupyx.scipy.sparse
                from tensormesh.sparse.utils import tensor2cupy
                
                cupy.cuda.Device(self.device.index or 0).use()
                
                edata_cp = tensor2cupy(K.edata.detach())
                row_cp = tensor2cupy(K.row.detach())
                col_cp = tensor2cupy(K.col.detach())
                self._cupy_K = cupyx.scipy.sparse.coo_matrix(
                    (edata_cp, (row_cp, col_cp)), shape=shape
                ).tocsr()
            except Exception as e:
                print(f"[BatchLinearElasticitySolver] CuPy matrix setup failed: {e}")
    
    def _solve_with_cached_lu(self, F_condensed: torch.Tensor, tol: float = 1e-6, max_iter: int = 10000) -> torch.Tensor:
        """Solve linear system. Uses BiCGSTAB on CUDA (default) or LU factorization."""
        
        if self.device.type == "cuda":
            import cupy
            import cupyx.scipy.sparse.linalg
            from tensormesh.sparse.utils import tensor2cupy, cupy2tensor
            
            cupy.cuda.Device(self.device.index or 0).use()
            
            if self.cuda_solver == 'bicgstab':
                # Use BiCGSTAB iterative solver
                F_cp = tensor2cupy(F_condensed)
                
                # Handle batch dimension
                if F_cp.ndim == 1:
                    u_cp, info = cupyx.scipy.sparse.linalg.bicgstab(
                        self._cupy_K, F_cp, tol=tol, maxiter=max_iter
                    )
                    if info != 0:
                        print(f"[BiCGSTAB] Warning: convergence info = {info}")
                else:
                    # Batch solve: iterate over columns
                    u_cp = cupy.zeros_like(F_cp)
                    for i in range(F_cp.shape[1]):
                        u_cp[:, i], info = cupyx.scipy.sparse.linalg.bicgstab(
                            self._cupy_K, F_cp[:, i], tol=tol, maxiter=max_iter
                        )
                
                u_inner = cupy2tensor(u_cp)
                return u_inner
            elif self._cupy_lu is not None:
                # Use cached LU factorization
                F_cp = tensor2cupy(F_condensed)
                u_cp = self._cupy_lu.solve(F_cp)
                u_inner = cupy2tensor(u_cp)
                return u_inner
            else:
                # Fallback to CPU
                F_cpu = F_condensed.detach().cpu().numpy()
                u_inner_np = self._lu_cache.solve(F_cpu)
                u_inner = torch.tensor(u_inner_np, dtype=F_condensed.dtype, device=self.device)
                return u_inner
        else:
            # CPU: use LU factorization
            F_cpu = F_condensed.detach().cpu().numpy()
            u_inner_np = self._lu_cache.solve(F_cpu)
            u_inner = torch.tensor(u_inner_np, dtype=F_condensed.dtype, device=self.device)
            return u_inner
    
    def solve(
        self,
        f: torch.Tensor,
        tol: float = 1e-6,
        max_iter: int = 10000,
    ) -> torch.Tensor:
        """
        Solve the linear elasticity equation -∇·σ = f with zero Dirichlet BC.
        
        Parameters
        ----------
        f : torch.Tensor
            Body force values at mesh nodes.
            Shape: [n_nodes, dim] or [batch_size, n_nodes, dim]
        tol : float, optional
            Tolerance for iterative solver. Default is 1e-6.
        max_iter : int, optional
            Maximum iterations for iterative solver. Default is 10000.
        
        Returns
        -------
        torch.Tensor
            Displacement values at all mesh nodes (including boundary).
            Shape: [n_nodes, dim] or [batch_size, n_nodes, dim]
        """
        # Handle batch dimension
        if f.dim() == 2:
            f = f.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = f.shape[0]
        
        # Flatten spatial dimension: [batch, n_nodes, dim] -> [batch, n_dofs]
        f_flat = f.reshape(batch_size, -1)  # [batch, n_nodes * dim]
        
        # Transpose for processing: [n_dofs, batch]
        f_T = f_flat.T
        
        # Assemble RHS using mass matrix
        # For vector problems with scalar mass matrix, we need to expand
        # M is [n_nodes, n_nodes], f is [n_nodes * dim]
        # We need to apply M to each component separately
        
        # Reshape to [n_nodes, dim, batch]
        f_reshaped = f_T.reshape(self.n_points, self.dim, batch_size)
        
        # Apply mass matrix to each component
        F_assembled = torch.zeros_like(f_reshaped)
        for d in range(self.dim):
            F_assembled[:, d, :] = self.M_full @ f_reshaped[:, d, :]
        
        # Flatten back: [n_dofs, batch]
        F_full = F_assembled.reshape(self.n_dofs, batch_size)
        
        # Condense (extract inner DOFs)
        F_condensed = F_full[self.is_inner]  # [n_inner, batch]
        
        # Solve linear system
        u_inner = self._solve_with_cached_lu(F_condensed, tol=tol, max_iter=max_iter)  # [n_inner, batch]
        
        # Recover full solution
        u_full = self.condenser.recover(u_inner)  # [n_dofs, batch]
        
        # Transpose and reshape: [batch, n_nodes, dim]
        u_out = u_full.T.reshape(batch_size, self.n_points, self.dim)
        
        if squeeze_output:
            u_out = u_out.squeeze(0)
        
        return u_out
    
    def forward(self, f: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - alias for solve()."""
        return self.solve(f, **kwargs)
    
    def compute_strain(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute strain field from displacement field.
        
        Parameters
        ----------
        u : torch.Tensor
            Displacement at nodes. Shape: [n_nodes, dim] or [batch, n_nodes, dim]
            
        Returns
        -------
        torch.Tensor
            Strain tensor at each node. Shape: [n_nodes, dim, dim] or [batch, n_nodes, dim, dim]
        """
        # This would require computing gradients using FEM shape functions
        # For now, return a placeholder
        raise NotImplementedError(
            "Strain computation requires gradient interpolation. "
            "Use the analytical solution's strain() method for exact strains."
        )
    
    def compute_stress(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute stress field from displacement field using Hooke's law.
        
        Parameters
        ----------
        u : torch.Tensor
            Displacement at nodes. Shape: [n_nodes, dim] or [batch, n_nodes, dim]
            
        Returns
        -------
        torch.Tensor
            Stress tensor at each node. Shape: [n_nodes, dim, dim] or [batch, n_nodes, dim, dim]
        """
        raise NotImplementedError(
            "Stress computation requires strain computation first. "
            "Use the analytical solution's stress() method for exact stresses."
        )
    
    @classmethod
    def create_2d(
        cls,
        domain: Literal["rectangle", "circle", "L", "hollow_rectangle"] = "rectangle",
        chara_length: float = 0.05,
        order: int = 1,
        element_type: str = "tri",
        E: float = 1.0,
        nu: float = 0.3,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        **mesh_kwargs,
    ) -> "BatchLinearElasticitySolver":
        """
        Create a 2D linear elasticity solver with specified domain.
        
        Parameters
        ----------
        domain : str
            Domain type: 'rectangle', 'circle', 'L', or 'hollow_rectangle'.
        chara_length : float
            Characteristic mesh element length.
        order : int
            Polynomial order of elements.
        element_type : str
            Element type ('tri' or 'quad').
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        device : str or torch.device, optional
            Device for computations.
        dtype : torch.dtype, optional
            Data type for computations.
        **mesh_kwargs : dict
            Additional arguments passed to Mesh generator.
        
        Returns
        -------
        BatchLinearElasticitySolver
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
        elif domain == "hollow_rectangle":
            mesh = Mesh.gen_hollow_rectangle(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        else:
            raise ValueError(f"Unknown 2D domain: {domain}")
        
        return cls(mesh, E=E, nu=nu, device=device, dtype=dtype)
    
    @classmethod
    def create_3d(
        cls,
        domain: Literal["cube", "sphere", "cylinder", "hollow_cube"] = "cube",
        chara_length: float = 0.1,
        order: int = 1,
        element_type: str = "tet",
        E: float = 1.0,
        nu: float = 0.3,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        **mesh_kwargs,
    ) -> "BatchLinearElasticitySolver":
        """
        Create a 3D linear elasticity solver with specified domain.
        
        Parameters
        ----------
        domain : str
            Domain type: 'cube', 'sphere', 'cylinder', or 'hollow_cube'.
        chara_length : float
            Characteristic mesh element length.
        order : int
            Polynomial order of elements.
        element_type : str
            Element type ('tet' or 'hex').
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        device : str or torch.device, optional
            Device for computations.
        dtype : torch.dtype, optional
            Data type for computations.
        **mesh_kwargs : dict
            Additional arguments passed to Mesh generator.
        
        Returns
        -------
        BatchLinearElasticitySolver
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
        elif domain == "hollow_cube":
            mesh = Mesh.gen_hollow_cube(
                chara_length=chara_length,
                order=order,
                element_type=element_type,
                **mesh_kwargs
            )
        else:
            raise ValueError(f"Unknown 3D domain: {domain}")
        
        return cls(mesh, E=E, nu=nu, device=device, dtype=dtype)


def solve_linear_elasticity_batch(
    f: torch.Tensor,
    mesh: Optional[Mesh] = None,
    domain: str = "rectangle",
    chara_length: float = 0.05,
    dim: int = 2,
    E: float = 1.0,
    nu: float = 0.3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    **solver_kwargs,
) -> Tuple[torch.Tensor, Mesh]:
    """
    Convenience function to solve batch linear elasticity equations.
    
    Parameters
    ----------
    f : torch.Tensor
        Body force values at mesh nodes.
        Shape: [n_nodes, dim] or [batch_size, n_nodes, dim]
    mesh : Mesh, optional
        Pre-built mesh. If None, creates one from domain specification.
    domain : str
        Domain type if mesh is None.
    chara_length : float
        Mesh characteristic length if mesh is None.
    dim : int
        Spatial dimension (2 or 3) if mesh is None.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    device : str or torch.device, optional
        Device for computations.
    dtype : torch.dtype, optional
        Data type.
    **solver_kwargs : dict
        Additional arguments for solver.solve().
    
    Returns
    -------
    u : torch.Tensor
        Displacement values. Shape: [n_nodes, dim] or [batch_size, n_nodes, dim]
    mesh : Mesh
        The mesh used for solving.
    """
    if mesh is None:
        if dim == 2:
            solver = BatchLinearElasticitySolver.create_2d(
                domain=domain,
                chara_length=chara_length,
                E=E,
                nu=nu,
                device=device,
                dtype=dtype,
            )
        else:
            solver = BatchLinearElasticitySolver.create_3d(
                domain=domain,
                chara_length=chara_length,
                E=E,
                nu=nu,
                device=device,
                dtype=dtype,
            )
    else:
        solver = BatchLinearElasticitySolver(mesh, E=E, nu=nu, device=device, dtype=dtype)
    
    u = solver.solve(f, **solver_kwargs)
    return u, solver.mesh


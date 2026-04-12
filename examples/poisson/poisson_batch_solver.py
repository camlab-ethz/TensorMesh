"""
Batch Poisson Solver Example using TensorMesh.

This example demonstrates how to:
1. Generate batch source terms f using PoissonMultiFrequency (2D/3D)
2. Solve the Poisson equation -Δu = f using FEM
3. Compare with analytical solutions

The Poisson equation:
    -Δu = f  in Ω = [0,1]^d
    u = 0    on ∂Ω (Dirichlet BC)

Usage:
    python poisson_batch_solver.py --mode 2d      # 2D example
    python poisson_batch_solver.py --mode 3d      # 3D example  
    python poisson_batch_solver.py --mode bench   # Benchmark
    python poisson_batch_solver.py --mode all     # All examples
"""

import sys
import os
import importlib.util

sys.path.insert(0, "../..")

import torch
import numpy as np
from time import perf_counter
from typing import Optional


# =============================================================================
# Module loading (avoids matplotlib version conflicts in some environments)
# =============================================================================

def _load_module(mod_path, mod_name):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Load Poisson equation generators
_poisson_mod = _load_module(
    os.path.join(_ROOT, "tensormesh/dataset/equation/poisson.py"),
    "poisson_mod"
)
PoissonMultiFrequency = _poisson_mod.PoissonMultiFrequency
PoissonMultiFrequency3D = _poisson_mod.PoissonMultiFrequency3D


# =============================================================================
# Batch Poisson Solver Class
# =============================================================================

class SimpleBatchPoissonSolver:
    """
    A simplified batch Poisson solver using TensorMesh FEM.
    
    Solves: -Δu = f on [0,1]^d with u=0 on boundary
    
    Parameters
    ----------
    mesh : Mesh
        TensorMesh mesh object.
    device : str or torch.device
        Device to run computations on.
    
    Example
    -------
    >>> solver = SimpleBatchPoissonSolver(mesh, device='cuda')
    >>> u = solver.solve(f)  # f: [batch_size, n_nodes], u: [batch_size, n_nodes]
    """
    
    def __init__(self, mesh, device: Optional[str] = None):
        from tensormesh.assemble import LaplaceElementAssembler, MassElementAssembler
        from tensormesh.operator import Condenser
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.mesh = mesh.to(self.device)
        
        # Build stiffness matrix (Laplacian)
        assembler = LaplaceElementAssembler.from_mesh(self.mesh)
        K_full = assembler(self.mesh.points)
        
        # Setup Dirichlet BC (u=0 on boundary)
        n_boundary = self.mesh.boundary_mask.sum().item()
        dirichlet_value = torch.zeros(n_boundary, device=self.device)
        self.condenser = Condenser(self.mesh.boundary_mask, dirichlet_value)
        
        # Condense stiffness matrix
        self.K, _ = self.condenser(
            K_full, 
            torch.zeros(self.mesh.n_points, device=self.device)
        )
        
        # Mass matrix for RHS assembly: F = M @ f  (i.e. ∫ v*f dΩ)
        mass_assembler = MassElementAssembler.from_mesh(self.mesh)
        self.M = mass_assembler()

        self.n_points = self.mesh.n_points
        self.n_inner = self.K.shape[0]
    
    def solve(self, f: torch.Tensor, tol: float = 1e-6, max_iter: int = 10000) -> torch.Tensor:
        """
        Solve -Δu = f with zero Dirichlet BC.
        
        Parameters
        ----------
        f : torch.Tensor
            Source term at mesh nodes. Shape: [n_nodes] or [batch_size, n_nodes]
        tol : float
            Solver tolerance.
        max_iter : int
            Maximum iterations.
        
        Returns
        -------
        torch.Tensor
            Solution at mesh nodes. Shape: [n_nodes] or [batch_size, n_nodes]
        """
        squeeze = f.dim() == 1
        if squeeze:
            f = f.unsqueeze(0)
        
        # f: [batch, n_nodes] -> [n_nodes, batch]
        f_T = f.T

        # Assemble RHS: F = M @ f  (∫ v*f dΩ for each batch)
        F = self.M @ f_T  # [n_nodes, batch]

        # Condense RHS (K already condensed in __init__)
        F_cond = self.condenser.condense_rhs(F)
        
        # Solve K @ u = F (column by column, as solve only accepts 1D rhs)
        if F_cond.dim() == 1:
            u_inner = self.K.solve(F_cond, tol=tol, maxiter=max_iter)
        else:
            cols = [self.K.solve(F_cond[:, i], tol=tol, maxiter=max_iter)
                    for i in range(F_cond.shape[1])]
            u_inner = torch.stack(cols, dim=1)  # [n_inner, batch]

        # Recover full solution
        u_full = self.condenser.recover(u_inner)  # [n_nodes] or [n_nodes, batch]
        u = u_full.T  # [batch, n_nodes]
        
        if squeeze:
            u = u.squeeze(0)
        
        return u


# =============================================================================
# Example Functions
# =============================================================================

def example_2d():
    """2D batch Poisson solver example."""
    from tensormesh import Mesh
    
    print("=" * 60)
    print("2D Batch Poisson Solver Example")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Parameters
    batch_size = 32
    K = 8
    chara_length = 0.02
    
    # Create mesh
    print(f"\nCreating mesh (chara_length={chara_length})...")
    mesh = Mesh.gen_rectangle(chara_length=chara_length)
    print(f"Mesh: {mesh.n_points} nodes, {mesh.n_elements} elements")
    
    # Create solver
    print("Building solver...")
    t0 = perf_counter()
    solver = SimpleBatchPoissonSolver(mesh, device=device)
    print(f"Solver built in {perf_counter() - t0:.3f}s")
    
    # Generate source terms
    print(f"\nGenerating {batch_size} source terms (K={K})...")
    torch.manual_seed(42)
    a = torch.rand((batch_size, K, K), device=device) * 2 - 1
    equation = PoissonMultiFrequency(a=a)
    
    points = mesh.points.to(device)
    f = equation.source_term(points, domain="rectangle")  # [batch, n_nodes]
    print(f"Source terms shape: {f.shape}")
    
    # Solve
    print("\nSolving...")
    t0 = perf_counter()
    u_fem = solver.solve(f)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_solve = perf_counter() - t0
    print(f"Solved in {t_solve:.3f}s ({t_solve/batch_size*1000:.2f}ms per problem)")
    
    # Compare with analytical solution
    u_analytical = equation.solution(points)
    error = (u_fem - u_analytical).abs()
    rel_error = error / (u_analytical.abs().max(dim=1, keepdim=True).values + 1e-10)
    
    print(f"\nResults:")
    print(f"  Max absolute error: {error.max():.6e}")
    print(f"  Mean absolute error: {error.mean():.6e}")
    print(f"  Max relative error: {rel_error.max():.6e}")
    
    # Visualize
    try:
        idx = 0
        mesh.to('cpu').plot(
            {
                "f": f[idx].cpu(),
                "u_fem": u_fem[idx].cpu(),
                "u_analytical": u_analytical[idx].cpu(),
                "error": error[idx].cpu() * 100,
            },
            save_path="poisson_batch_solver_2d.png",
            show_mesh=False,
        )
        print("\nPlot saved to poisson_batch_solver_2d.png")
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
    
    return solver, u_fem, u_analytical


def example_3d():
    """3D batch Poisson solver example."""
    from tensormesh import Mesh
    
    print("\n" + "=" * 60)
    print("3D Batch Poisson Solver Example")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Parameters
    batch_size = 16
    K = 4
    chara_length = 0.02
    
    # Create mesh
    print(f"\nCreating 3D mesh (chara_length={chara_length})...")
    mesh = Mesh.gen_cube(chara_length=chara_length)
    print(f"Mesh: {mesh.n_points} nodes, {mesh.n_elements} elements")
    
    # Create solver
    print("Building solver...")
    t0 = perf_counter()
    solver = SimpleBatchPoissonSolver(mesh, device=device)
    print(f"Solver built in {perf_counter() - t0:.3f}s")
    
    # Generate source terms
    print(f"\nGenerating {batch_size} source terms (K={K})...")
    torch.manual_seed(42)
    a = torch.rand((batch_size, K, K, K), device=device) * 2 - 1
    equation = PoissonMultiFrequency3D(a=a)
    
    points = mesh.points.to(device)
    f = equation.source_term(points, domain="cube")
    print(f"Source terms shape: {f.shape}")
    
    # Solve
    print("\nSolving...")
    t0 = perf_counter()
    u_fem = solver.solve(f)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_solve = perf_counter() - t0
    print(f"Solved in {t_solve:.3f}s ({t_solve/batch_size*1000:.2f}ms per problem)")
    
    # Compare with analytical solution
    u_analytical = equation.solution(points)
    error = (u_fem - u_analytical).abs()
    rel_error = error / (u_analytical.abs().max(dim=1, keepdim=True).values + 1e-10)
    
    print(f"\nResults:")
    print(f"  Max absolute error: {error.max():.6e}")
    print(f"  Mean absolute error: {error.mean():.6e}")
    print(f"  Max relative error: {rel_error.max():.6e}")
    
    # Save to VTK
    try:
        mesh_cpu = mesh.to('cpu')
        mesh_cpu.register_point_data("u_fem", u_fem[0].cpu())
        mesh_cpu.register_point_data("u_analytical", u_analytical[0].cpu())
        mesh_cpu.register_point_data("error", error[0].cpu())
        mesh_cpu.register_point_data("f", f[0].cpu())
        mesh_cpu.save("poisson_batch_solver_3d.vtu")
        print("\nVTK file saved to poisson_batch_solver_3d.vtu")
    except Exception as e:
        print(f"\nVTK save skipped: {e}")
    
    return solver, u_fem, u_analytical


def benchmark():
    """Benchmark batch solving performance."""
    from tensormesh import Mesh
    
    print("\n" + "=" * 60)
    print("Batch Size Scaling Benchmark")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create mesh and solver
    mesh = Mesh.gen_rectangle(chara_length=0.02)
    solver = SimpleBatchPoissonSolver(mesh, device=device)
    points = mesh.points.to(device)
    
    K = 8
    batch_sizes = [1, 4, 16, 64, 256, 1024]
    n_warmup = 2
    n_runs = 5
    
    print(f"\nMesh: {mesh.n_points} nodes, K={K}")
    print("-" * 50)
    print(f"{'Batch Size':>12} {'Total Time (ms)':>18} {'Per Problem (ms)':>18}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        # Generate source terms
        a = torch.rand((batch_size, K, K), device=device) * 2 - 1
        equation = PoissonMultiFrequency(a=a)
        f = equation.source_term(points, domain="rectangle")
        
        # Warmup
        for _ in range(n_warmup):
            _ = solver.solve(f)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = perf_counter()
            _ = solver.solve(f)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(perf_counter() - t0)
        
        mean_time = np.mean(times) * 1000  # ms
        per_problem = mean_time / batch_size
        
        print(f"{batch_size:>12} {mean_time:>18.2f} {per_problem:>18.3f}")


def demo_source_term_generation():
    """Demo: Generate source terms and solutions (no mesh needed)."""
    print("=" * 60)
    print("Source Term Generation Demo (No Mesh)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 2D example
    print("\n--- 2D Poisson ---")
    batch_size = 4
    K = 8
    n_points = 1000
    
    # Random points in [0,1]^2
    points_2d = torch.rand(n_points, 2, device=device)
    
    # Random coefficients
    a = torch.rand(batch_size, K, K, device=device) * 2 - 1
    eq = PoissonMultiFrequency(a=a)
    
    # Generate source terms and solutions
    f = eq.source_term(points_2d, domain="rectangle")
    u = eq.solution(points_2d)
    
    print(f"Points shape: {points_2d.shape}")
    print(f"Source term f shape: {f.shape}")
    print(f"Solution u shape: {u.shape}")
    
    # 3D example
    print("\n--- 3D Poisson ---")
    K3 = 4
    
    # Random points in [0,1]^3
    points_3d = torch.rand(n_points, 3, device=device)
    
    # Random coefficients [batch, K, K, K]
    a3d = torch.rand(batch_size, K3, K3, K3, device=device) * 2 - 1
    eq3d = PoissonMultiFrequency3D(a=a3d)
    
    f3d = eq3d.source_term(points_3d, domain="cube")
    u3d = eq3d.solution(points_3d)
    
    print(f"Points shape: {points_3d.shape}")
    print(f"Source term f shape: {f3d.shape}")
    print(f"Solution u shape: {u3d.shape}")
    
    print("\nDemo complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Poisson Solver Examples")
    parser.add_argument(
        "--mode", 
        choices=["2d", "3d", "bench", "demo", "all"], 
        default="demo",
        help="Which example to run"
    )
    args = parser.parse_args()
    
    # Always run demo (works without full tensormesh)
    if args.mode == "demo":
        demo_source_term_generation()
    else:
        # Try to import full tensormesh
        try:
            from tensormesh import Mesh
            TENSORMESH_OK = True
        except ImportError as e:
            print(f"Error: Could not import tensormesh: {e}")
            print("Running demo mode instead...")
            demo_source_term_generation()
            TENSORMESH_OK = False
        
        if TENSORMESH_OK:
            if args.mode in ["2d", "all"]:
                example_2d()
            if args.mode in ["3d", "all"]:
                example_3d()
            if args.mode in ["bench", "all"]:
                benchmark()
    
    print("\nDone!")

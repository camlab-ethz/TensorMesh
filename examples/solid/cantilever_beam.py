"""
Cantilever Beam Deformation Example
===================================

This example demonstrates the linear elastic deformation of a steel cantilever beam.

Features:
- Using `tensormesh.material` for material properties.
- Using `LinearElasticityElementAssembler` for physics.
- Using `tensormesh.visualization.plot_deformation` for static visualization.
- Pure PyTorch/TensorMesh implementation (no numpy dependency).
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.dataset.mesh import gen_cube
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.material import Steel, Aluminum
from tensormesh.visualization import plot_deformation

def main():
    # 1. Geometry: 2m x 0.2m x 0.2m Beam
    print("Generating Mesh...")
    # Generate a beam using gen_cube (produces tetrahedra by default)
    mesh = gen_cube(chara_length=0.08, left=0.0, right=2.0, bottom=0.0, top=0.2, front=0.0, back=0.2)
    n_nodes = mesh.points.shape[0]
    n_cells = sum([v.shape[0] for v in mesh.cells.values()])
    print(f"Mesh created: {n_nodes} nodes, {n_cells} elements")

    # 2. Material: Steel
    material = Steel
    print(f"Material: {material.name} (E={material.E/1e9} GPa, nu={material.nu})")

    # 3. Assemble Stiffness Matrix
    print("Assembling Stiffness Matrix...")
    assembler = LinearElasticityElementAssembler.from_mesh(mesh, E=material.E, nu=material.nu)
    K = assembler()

    # 4. Boundary Conditions
    # Fix x = 0 face (Dirichlet)
    print("Applying Boundary Conditions...")
    points = mesh.points
    eps = 1e-5
    # Fix left end (x=0)
    fixed_node_mask = torch.abs(points[:, 0]) < eps
    
    # Expand to DOFs (N, 3) -> (N*3)
    dim = 3
    fixed_dof_mask = torch.repeat_interleave(fixed_node_mask, dim)
    
    condenser = Condenser(fixed_dof_mask) # Value defaults to 0.0
    
    # 5. Load
    # Apply downward force at x = 2 face (right end)
    # Total Force F = -100 kN in y direction
    # Distributed over nodes on the right face
    right_mask = torch.abs(points[:, 0] - 2.0) < eps
    right_nodes = torch.where(right_mask)[0]
    
    F_total = -1.0e5 # -100 kN
    f_per_node = F_total / right_nodes.shape[0]
    
    rhs = torch.zeros((n_nodes, mesh.dim))
    rhs[right_nodes, 1] = f_per_node # y-direction
    rhs_flat = rhs.flatten()
    
    # 6. Solve
    print("Solving linear system...")
    K_cond, F_cond = condenser(K, rhs_flat)
    
    # Use internal SparseMatrix solver
    # K_cond is a SparseMatrix, F_cond is a Tensor
    u_cond = K_cond.solve(F_cond)
    
    # Recover full solution
    u_flat = u_cond.float() # Ensure float
    u = condenser.recover(u_flat)
    u_vec = u.reshape(-1, 3)
    
    max_disp = torch.max(torch.norm(u_vec, dim=1)).item()
    print(f"Max displacement: {max_disp*1000:.2f} mm")

    # 7. Visualize (Static Comparison)
    # Save to the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "cantilever_steel.png")
    
    print(f"Generating static comparison plot to {output_file}...")
    
    # Auto-scale to make deformation visible (e.g. 20% of length)
    target_vis_disp = 0.4 
    scale_factor = target_vis_disp / max_disp if max_disp > 0 else 1.0
    print(f"Using scale factor: {scale_factor:.2f}x for visualization")
    
    plot_deformation(mesh, u_vec, output_file, scale_factor=scale_factor, camera_position='xy',
                     fixed_nodes=fixed_node_mask, force_vectors=rhs)


if __name__ == "__main__":
    main()

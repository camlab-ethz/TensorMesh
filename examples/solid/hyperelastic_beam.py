"""
Hyperelastic Beam Example (Neo-Hookean)
=======================================

This example demonstrates large deformation of a rubber beam using a Neo-Hookean material model.
It minimizes the potential energy using LBFGS optimizer.

Physics:
- Compressible Neo-Hookean Hyperelasticity
- Finite Element Method (Total Lagrangian)
- Energy Minimization

"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np # Only for math constants if needed, but we use torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.dataset.mesh import gen_cube
from tensormesh.assemble import ElementAssembler
from tensormesh.material import Rubber
from tensormesh.visualization import plot_deformation

class NeoHookeanModel(ElementAssembler):
    def __post_init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def element_energy(self, gradu):
        """
        Compute energy density for a single quadrature point.
        gradu: [Dim, Dim]
        """
        # Deformation Gradient F = I + grad_u
        dim = gradu.shape[-1]
        I = torch.eye(dim, device=gradu.device, dtype=gradu.dtype)
        F = I + gradu 
        
        # Invariants
        # J = det(F)
        J = torch.det(F)
        
        # I1 = tr(C) = tr(F.T F) = sum(F_ij^2)
        I1 = (F**2).sum()
        
        # Strain Energy Density (Compressible Neo-Hookean)
        # Psi = mu/2 * (I1 - 3) - mu*ln(J) + lam/2 * (ln(J))^2
        
        # Numerical stability for log(J)
        J = torch.clamp(J, min=1e-6)
        logJ = torch.log(J)
        
        Psi = 0.5 * self.mu * (I1 - 3) - self.mu * logJ + 0.5 * self.lam * (logJ**2)
        return Psi

def main():
    # 1. Geometry: 1m x 0.4m x 0.4m Beam (Short & Thick)
    print("Generating Mesh...")
    # Use Quadratic Tetrahedra (order=2)
    mesh = gen_cube(chara_length=0.1, order=2, left=0.0, right=1.0, bottom=0.0, top=0.4, front=0.0, back=0.4)
    n_nodes = mesh.points.shape[0]
    print(f"Mesh created: {n_nodes} nodes")
    print(f"Mesh bounds: {mesh.points.min(0)[0]} to {mesh.points.max(0)[0]}")

    # 2. Material: Rubber
    material = Rubber
    print(f"Material: {material.name} (E={material.E/1e6} MPa, nu={material.nu})")
    
    # 3. Model
    mu, lam = material.lame_params
    print(f"Lame params: mu={mu:.2e}, lam={lam:.2e}")
    model = NeoHookeanModel.from_mesh(mesh, mu=mu, lam=lam)
    
    # 4. Boundary Conditions
    points = mesh.points
    eps = 1e-5
    # Fix left end (x=0)
    fixed_mask = torch.abs(points[:, 0]) < eps
    fixed_indices = torch.where(fixed_mask)[0]
    
    # 5. Load (Torsion)
    # Apply torque at x = 1 face
    right_mask = torch.abs(points[:, 0] - 1.0) < eps
    right_indices = torch.where(right_mask)[0]
    
    # Center of the cross-section
    y_center = 0.2
    z_center = 0.2
    
    # Coordinates relative to center
    dy = points[right_indices, 1] - y_center
    dz = points[right_indices, 2] - z_center
    
    # Torsion Force Field: F = C * (0, -dz, dy)
    # This creates a torque around the x-axis
    # Magnitude
    # Reduced to 3.0e4 to target ~0.2m displacement (reasonable twist)
    C = 3.0e4
    
    f_ext = torch.zeros_like(points)
    f_ext[right_indices, 1] = -dz * C
    f_ext[right_indices, 2] = dy * C
    
    # Ensure pure torque (remove spurious net force due to mesh irregularity)
    # This prevents the beam from bending "up" or "sideways" inadvertently
    f_y_net = f_ext[right_indices, 1].sum()
    f_z_net = f_ext[right_indices, 2].sum()
    f_ext[right_indices, 1] -= f_y_net / len(right_indices)
    f_ext[right_indices, 2] -= f_z_net / len(right_indices)
    
    # 6. Optimization (Energy Minimization)
    u = torch.zeros_like(points, requires_grad=True)
    
    optimizer = optim.LBFGS([u], 
                            lr=1.0, 
                            max_iter=100, 
                            max_eval=120, 
                            tolerance_grad=1e-5, 
                            tolerance_change=1e-5,
                            history_size=100,
                            line_search_fn="strong_wolfe")
    
    print("Starting Optimization...")
    
    def closure():
        optimizer.zero_grad()
        
        # Apply BC by masking: u_active = u * mask
        # This ensures gradient at fixed nodes is zero (dL/du_fixed = 0)
        # and we don't modify leaf variable in-place.
        mask = (~fixed_mask).unsqueeze(1).to(u.device, u.dtype)
        u_active = u * mask
            
        # Internal Energy
        E_int = model.energy(point_data={"u": u_active})
        
        # External Work: W = f_ext . u
        W_ext = (f_ext * u_active).sum()
        
        # Potential Energy
        loss = E_int - W_ext
        
        if loss.requires_grad:
            loss.backward()
            
        return loss

    # Load Stepping (optional but good for robustness)
    n_steps = 10
    f_ext_final = f_ext.clone()
    
    for step in range(1, n_steps + 1):
        # Ramp up load
        f_ext = f_ext_final * (step / n_steps)
        
        loss_val = optimizer.step(closure)
        
        with torch.no_grad():
            max_d = u.norm(dim=1).max().item()
            print(f"Step {step}/{n_steps}: Loss = {loss_val.item():.4e}, Max Disp = {max_d:.4f} m")

    # Final cleanup of BCs
    with torch.no_grad():
        mask = (~fixed_mask).unsqueeze(1).to(u.device, u.dtype)
        u.data = u.data * mask
    
    print("Optimization Complete.")
    
    # 7. Visualize
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "hyperelastic_rubber.png")
    
    u_vec = u.detach()
    max_disp = torch.max(torch.norm(u_vec, dim=1)).item()
    print(f"Final Max displacement: {max_disp*1000:.2f} mm")
    
    # Use 'isometric' view
    plot_deformation(mesh, u_vec, output_file, scale_factor=1.0, camera_position='isometric',
                     fixed_nodes=fixed_mask, force_vectors=f_ext_final)

if __name__ == "__main__":
    main()


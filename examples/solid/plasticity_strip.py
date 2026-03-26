"""
Plasticity Example (J2 Flow Theory)
===================================

This example demonstrates 2D plasticity (J2 Flow Theory with Isotropic Hardening)
using an incremental potential and automatic differentiation.

Physics:
- Small strain assumption (Linear Kinematics)
- Von Mises Yield Criterion
- Linear Isotropic Hardening
- Plane Strain

Implementation:
- Defines `J2PlasticityModel` which stores internal variables (plastic strain, equivalent plastic strain).
- Uses a "Variational Constitutive Update" approach where the stress is derived from an algorithmic potential.
- Solves for displacement using LBFGS energy minimization at each time step.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh, Condenser
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.assemble import ElementAssembler, J2Plasticity
from tensormesh.visualization import mesh_to_pyvista, setup_headless
from tensormesh.material import IsotropicMaterial
import pyvista as pv

def main():
    # 1. Geometry: 2D Rectangle Strip
    L, H = 1.0, 0.2
    print("Generating Mesh...")
    mesh = gen_rectangle(left=0.0, right=L, bottom=0.0, top=H, chara_length=0.02)
    print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.n_elements} elements")
    
    # 2. Material (Steel-like but lower yield for demo)
    # E=200 GPa, Sig0=250 MPa, H=1 GPa
    steel_hardening = IsotropicMaterial("Steel_Hardening", E=200e9, nu=0.3, rho=7850, sigma_y=250e6, H=1e9)
    model = J2Plasticity.from_mesh(mesh, material=steel_hardening)
    
    # 3. Boundary Conditions
    # Relaxed BCs: Left Roller (x=0 fixed), Bottom-Left Pin (x=0, y=0 fixed)
    # This prevents artificial stiffening/singularity at corners
    points = mesh.points
    eps_tol = 1e-5
    
    left_mask = points[:, 0] < eps_tol
    bottom_left_mask = (points[:, 0] < eps_tol) & (points[:, 1] < eps_tol)
    right_mask = points[:, 0] > L - eps_tol
    
    u_total_x = 0.10 # 10% strain (Large deformation to make plasticity obvious)
    n_steps = 20
    
    # Optimizer settings
    u = torch.zeros_like(points, requires_grad=True)
    
    # We will use LBFGS
    optimizer = optim.LBFGS([u], lr=1, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
    
    # Prepare PyVista plotter for animation
    setup_headless()
    plotter = pv.Plotter(off_screen=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "plasticity_strip.mp4")
    plotter.open_movie(output_file)
    
    print("Starting Simulation...")
    
    for step in range(1, n_steps + 1):
        target_disp = u_total_x * (step / n_steps)
        
        # Enforce BCs in closure
        def closure():
            optimizer.zero_grad()
            
            # Apply Dirichlet BCs
            # Left: fixed (0)
            # Right: prescribed x-displacement (target_disp)
            
            # We create a mask of FREE degrees of freedom
            # But here we modify u directly or use penalty?
            # Better: Create u_active where BCs are enforced.
            
            # BCs enforced via u_active = u * mask + val
            
            # Mask approach:
            mask_tensor = torch.ones_like(u)
            mask_tensor[left_mask, 0] = 0 # Fix x on left
            mask_tensor[bottom_left_mask, 1] = 0 # Fix y on bottom-left corner
            mask_tensor[right_mask, 0] = 0 # Fix x on right
            
            val_tensor = torch.zeros_like(u)
            val_tensor[right_mask, 0] = target_disp
            
            u_active = u * mask_tensor + val_tensor
            
            # Prepare element_data from history (transpose dict)
            element_data = {
                'eps_p_n': {etype: h['eps_p'] for etype, h in model.history.items()},
                'alpha_n': {etype: h['alpha'] for etype, h in model.history.items()}
            }
            
            energy = model.energy(point_data={'displacement': u_active}, element_data=element_data)
            
            if energy.requires_grad:
                energy.backward()
                
                # Zero out gradients on BC nodes so optimizer doesn't move them from their initial 0 guess
                # (Assuming u was initialized to 0 and we add val_tensor)
                # Actually u parameter holds the "displacement".
                # If we use u * mask + val, gradients w.r.t u at masked locations are 0.
                # So u values at masked locations won't change.
                pass
                
            return energy

        # Run optimization step
        loss = optimizer.step(closure)
        
        # Update State Variables
        # Need to reconstruct the full u_active to pass to update
        with torch.no_grad():
            mask_tensor = torch.ones_like(u)
            mask_tensor[left_mask, 0] = 0
            mask_tensor[bottom_left_mask, 1] = 0
            mask_tensor[right_mask, 0] = 0
            
            val_tensor = torch.zeros_like(u)
            val_tensor[right_mask, 0] = target_disp
            u_final = u * mask_tensor + val_tensor
            
            # Update u parameter to match u_final (for next step consistency)
            u.data.copy_(u_final)
            
            model.update_state(u_final)
            
            # Get max plastic strain for visualization
            # Just take the first element type's alpha
            # Average over quad points for cell data
            alpha_field = next(iter(model.history.values()))['alpha'] # [N_elem, N_quad]
            alpha_elem = alpha_field.mean(dim=1).cpu().numpy()
            
            max_alpha = alpha_elem.max()
            print(f"Step {step}: Loss={loss.item():.4e}, Max Plastic Strain={max_alpha:.4f}")
            
            # Add frame to video
            # Use mesh_to_pyvista
            # We pass cell_data for plastic strain
            pv_mesh = mesh_to_pyvista(mesh, 
                                      point_data={'displacement': u_final}, 
                                      cell_data={'plastic_strain': alpha_elem})
            
            # Warp
            warped = pv_mesh.warp_by_vector('displacement', factor=1.0)
            
            plotter.clear()
            # Show edges to reveal Quad elements (otherwise shading looks triangular)
            plotter.add_mesh(warped, scalars='plastic_strain', cmap='jet', show_edges=True, clim=[0, u_total_x])
            
            # Visualize BCs (Fixed Nodes) - Blue Cubes
            # Convert masks to numpy for pyvista
            mask_l = left_mask.cpu().numpy()
            mask_bl = bottom_left_mask.cpu().numpy()
            mask_fixed = mask_l | mask_bl
            
            if mask_fixed.any():
                # Get current positions of fixed nodes
                p_fixed = mesh.points[mask_fixed].cpu().numpy() + u_final[mask_fixed].cpu().numpy()
                # Pad to 3D
                if p_fixed.shape[1] == 2: p_fixed = np.pad(p_fixed, ((0,0), (0,1)))
                
                cloud_fixed = pv.PolyData(p_fixed)
                cube = pv.Cube().scale(0.015, 0.015, 0.015)
                glyphs_fixed = cloud_fixed.glyph(scale=False, geom=cube)
                plotter.add_mesh(glyphs_fixed, color='blue', label='Fixed')

            # Visualize Load (Right Edge) - Red Arrows
            mask_r = right_mask.cpu().numpy()
            if mask_r.any():
                p_load = mesh.points[mask_r].cpu().numpy() + u_final[mask_r].cpu().numpy()
                if p_load.shape[1] == 2: p_load = np.pad(p_load, ((0,0), (0,1)))
                
                cloud_load = pv.PolyData(p_load)
                # Direction (1, 0, 0)
                vecs = np.zeros_like(p_load)
                vecs[:, 0] = 1.0
                cloud_load['vectors'] = vecs
                
                arrow = pv.Arrow().scale(0.05, 0.05, 0.05)
                # orient='vectors' aligns arrow with vector data
                glyphs_load = cloud_load.glyph(orient='vectors', scale=False, geom=arrow)
                plotter.add_mesh(glyphs_load, color='red', label='Load')

            plotter.add_text(f"Step {step}, Disp={target_disp:.4f}m", position='upper_left')
            plotter.view_xy()
            plotter.write_frame()
            
    plotter.close()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    main()


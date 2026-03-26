"""
3D Plasticity Example (J2 Flow Theory)
======================================

This example demonstrates 3D plasticity (J2 Flow Theory with Isotropic Hardening)
using the tensormesh J2Plasticity assembler.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh
from tensormesh.dataset.mesh import gen_cube
from tensormesh.assemble import J2Plasticity
from tensormesh.visualization import mesh_to_pyvista, setup_headless
from tensormesh.material import IsotropicMaterial
import pyvista as pv

def main():
    # 1. Geometry: 3D Cube
    L, W, H = 0.5, 0.5, 0.5
    print("Generating 3D Mesh...")
    # Using a denser mesh
    mesh = gen_cube(left=0.0, right=L, bottom=0.0, top=W, front=0.0, back=H, chara_length=0.04)
    print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.n_elements} elements")
    
    # 2. Material
    steel_hardening = IsotropicMaterial("Steel_Hardening", E=200e9, nu=0.3, rho=7850, sigma_y=250e6, H=1e9)
    model = J2Plasticity.from_mesh(mesh, material=steel_hardening)
    
    # 3. Boundary Conditions
    points = mesh.points
    eps_tol = 1e-5
    
    # Masks
    face_left_mask = points[:, 0] < eps_tol
    face_right_mask = points[:, 0] > L - eps_tol
    p0_mask = (points[:, 0] < eps_tol) & (points[:, 1] < eps_tol) & (points[:, 2] < eps_tol)
    p1_mask = (points[:, 0] < eps_tol) & (points[:, 1] > W - eps_tol) & (points[:, 2] < eps_tol)
    
    # Simulation Parameters
    u_total_x = 0.20 # 20% strain (Increased for visible deformation)
    n_loading_steps = 50
    n_unloading_steps = 50
    
    u = torch.zeros_like(points, requires_grad=True)
    
    # Data recording
    displacements = []
    forces = []
    
    # Visualization
    setup_headless()
    plotter = pv.Plotter(off_screen=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "plasticity_3d_combined.mp4")
    
    # Use imageio to write video
    writer = imageio.get_writer(output_file, fps=10)
    
    print("Starting Simulation...")
    
    # Pre-create figure to save time (or create new one each time to be safe)
    # We will create new one to avoid state issues in loop
    
    # Loading then Unloading
    steps = np.concatenate([
        np.linspace(0, u_total_x, n_loading_steps + 1)[1:],
        np.linspace(u_total_x, 0, n_unloading_steps + 1)[1:]
    ])
    
    for i, target_disp in enumerate(steps):
        step_num = i + 1
        is_loading = i < n_loading_steps
        phase_name = "Loading" if is_loading else "Unloading"
        
        # Re-initialize optimizer each step
        optimizer = optim.LBFGS([u], lr=1, max_iter=100, history_size=100, line_search_fn="strong_wolfe")

        # Define BC masks and values (reused in closure and post-processing)
        mask_tensor = torch.ones_like(u)
        val_tensor = torch.zeros_like(u)
        mask_tensor[face_left_mask, 0] = 0
        mask_tensor[p0_mask, 1] = 0
        mask_tensor[p0_mask, 2] = 0
        mask_tensor[p1_mask, 2] = 0
        mask_tensor[face_right_mask, 0] = 0
        val_tensor[face_right_mask, 0] = target_disp

        def closure():
            optimizer.zero_grad()
            u_active = u * mask_tensor + val_tensor
            element_data = {
                'eps_p_n': {etype: h['eps_p'] for etype, h in model.history.items()},
                'alpha_n': {etype: h['alpha'] for etype, h in model.history.items()}
            }
            energy = model.energy(point_data={'displacement': u_active}, element_data=element_data)
            if energy.requires_grad:
                energy.backward()
            return energy

        loss = optimizer.step(closure)
        
        # Calculate Reaction Force
        u_forc = (u * mask_tensor + val_tensor).detach().clone()
        u_forc.requires_grad = True
        
        element_data = {
            'eps_p_n': {etype: h['eps_p'] for etype, h in model.history.items()},
            'alpha_n': {etype: h['alpha'] for etype, h in model.history.items()}
        }
        energy_forc = model.energy(point_data={'displacement': u_forc}, element_data=element_data)
        energy_forc.backward()
        
        # Sum x-forces on right face (Reaction Force)
        reaction_force_x = u_forc.grad[face_right_mask, 0].sum().item()
        
        displacements.append(target_disp)
        forces.append(reaction_force_x)

        with torch.no_grad():
            u_final = u * mask_tensor + val_tensor
            u.data.copy_(u_final)
            model.update_state(u_final)
            
            # Visualization
            alpha_field = next(iter(model.history.values()))['alpha']
            alpha_elem = alpha_field.mean(dim=1).cpu().numpy()
            max_alpha = alpha_elem.max()
            
            print(f"Step {step_num} ({phase_name}): Loss={loss.item():.4e}, Max Plastic Strain={max_alpha:.4f}, Force={reaction_force_x:.4e}")
            
            pv_mesh = mesh_to_pyvista(mesh, 
                                      point_data={'displacement': u_final}, 
                                      cell_data={'plastic_strain': alpha_elem})
            
            warped = pv_mesh.warp_by_vector('displacement', factor=1.0)
            
            plotter.clear()
            # clim updated for larger strain (expecting max plastic strain ~0.4-0.5)
            plotter.add_mesh(warped, scalars='plastic_strain', cmap='jet', show_edges=True, clim=[0, 0.5])
            plotter.add_axes()
            plotter.add_text(f"Step {step_num} ({phase_name})", position='upper_left')
            plotter.view_isometric()
            
            # Render PyVista to image
            pv_img = plotter.screenshot(None, return_img=True)
            
            # Create composite frame using Matplotlib
            fig = plt.figure(figsize=(12, 5))
            
            # Left: PyVista Image
            ax1 = fig.add_subplot(121)
            ax1.imshow(pv_img)
            ax1.axis('off')
            ax1.set_title(f"3D Plasticity (Step {step_num})")
            
            # Right: Force-Displacement Curve
            ax2 = fig.add_subplot(122)
            ax2.plot(displacements, forces, '-b', linewidth=2, label='Force-Disp')
            ax2.scatter([displacements[-1]], [forces[-1]], c='r', s=100, zorder=5, label='Current')
            
            ax2.set_title("Force-Displacement Curve")
            ax2.set_xlabel("Displacement (m)")
            ax2.set_ylabel("Reaction Force (N)")
            ax2.grid(True)
            ax2.legend()
            
            # Set fixed limits to avoid jumping axes
            # Updated estimates for 20% strain
            ax2.set_xlim(-0.005, u_total_x * 1.1)
            ax2.set_ylim(-4e8, 3e8)
            
            # Convert Matplotlib figure to image buffer
            fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape((h, w, 3))
            
            writer.append_data(buf)
            
            plt.close(fig)
            
    plotter.close()
    writer.close()
    print(f"Combined video saved to {output_file}")

if __name__ == "__main__":
    main()

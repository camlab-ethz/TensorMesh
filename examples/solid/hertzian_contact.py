"""
Hertzian Contact Problem (Circle on Flat Block)
===============================================

Geometry:
- Indenter (Left): Full Circle (Tri6), Radius=1.0, centered at (-1.0, 0.0).
- Block (Right): Rectangle (Quad9), fixed at the right boundary.
- Contact interface at x=0.

Method:
- Penalty method with Point-to-Segment contact detection.
- Accurate Von Mises stress calculation from strain.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pyvista as pv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tensormesh import Mesh
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.assemble import LinearElasticityElementAssembler
from tensormesh.visualization import mesh_to_pyvista, setup_headless

def gen_circle_indenter(r=1.0, chara_length=0.1, order=2):
    """Generate a full circle mesh on the left side, touching x=0."""
    import gmsh
    
    # Unique cache name for circle
    cache_path = f".gmsh_cache/circle_indenter_{r}_{chara_length}_{order}.msh"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if not os.path.exists(cache_path):
        gmsh.initialize()
        gmsh.model.add("circle")
        
        # Disk centered at (-r, 0), radius r. 
        # Rightmost point is at (0, 0). Leftmost at (-2r, 0).
        gmsh.model.occ.addDisk(-r, 0, 0, r, r)
        
        gmsh.model.occ.synchronize()
        
        # Add Physical Group to ensure only 2D elements are exported
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 1)
        
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", chara_length)
        gmsh.model.mesh.generate(2)
        gmsh.write(cache_path)
        gmsh.finalize()
    
    mesh = Mesh.from_file(cache_path, reorder=True)
    
    # Register boundary data
    pts = mesh.points
    
    # "Back" surface is the left side of the circle (x approx -2r)
    # Let's pick an arc on the left side, e.g., x < -1.8*r
    is_back = pts[:, 0] < -1.8 * r
    
    # Contact surface is the front part near x=0
    # Let's pick the right hemisphere or just the front arc
    is_contact = pts[:, 0] > -0.5 * r
    
    mesh.register_point_data("is_back", is_back)
    mesh.register_point_data("is_contact_front", is_contact)
    
    return mesh

def main():
    print("=" * 60)
    print("Hertzian Contact - Full Circle Indenter")
    print("=" * 60)
    
    # Parameters
    E = 1000.0
    nu = 0.3
    R = 1.0
    
    # 1. Generate Meshes
    print("Generating meshes...")
    # Indenter: Full Circle - Refine mesh for smoother contact
    indenter = gen_circle_indenter(r=R, chara_length=0.06, order=2)
    
    # Block: Right side - Refine mesh
    # Use generic gen_rectangle but we will implement custom transfinite logic there if needed
    # Or just write custom generation here to ensure symmetry
    
    def gen_symmetric_block(left, right, bottom, top, chara_length, order):
        import gmsh
        cache_path = f".gmsh_cache/rect_sym_{left}_{right}_{bottom}_{top}_{chara_length}_{order}.msh"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        if not os.path.exists(cache_path):
            gmsh.initialize()
            gmsh.model.add("rectangle_sym")
            
            # Create rectangle
            rect = gmsh.model.occ.addRectangle(left, bottom, 0, right-left, top-bottom)
            gmsh.model.occ.synchronize()
            
            # Physical group
            gmsh.model.addPhysicalGroup(2, [rect], 1)
            
            # Transfinite Meshing for perfect symmetry
            # Get lines
            lines = gmsh.model.getEntities(1)
            
            # We want specific number of elements.
            width = right - left
            height = top - bottom
            n_x = int(width / chara_length)
            n_y = int(height / chara_length)
            
            # Ensure even number for symmetry if crossing 0
            if n_y % 2 != 0: n_y += 1
            
            for dim, tag in lines:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                dx = abs(bbox[3] - bbox[0])
                dy = abs(bbox[4] - bbox[1])
                
                # Check if horizontal or vertical
                if dx > dy: # Horizontal
                    gmsh.model.mesh.setTransfiniteCurve(tag, n_x + 1)
                else: # Vertical
                    gmsh.model.mesh.setTransfiniteCurve(tag, n_y + 1)
            
            surfs = gmsh.model.getEntities(2)
            for dim, tag in surfs:
                gmsh.model.mesh.setTransfiniteSurface(tag)
                gmsh.model.mesh.setRecombine(dim, tag)
            
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            gmsh.model.mesh.generate(2)
            gmsh.write(cache_path)
            gmsh.finalize()
            
        mesh = Mesh.from_file(cache_path, reorder=True)
        
        # Register boundary data manually since we bypassed the standard gen_rectangle
        pts = mesh.points
        tol = 1e-4
        is_left = torch.abs(pts[:, 0] - left) < tol
        is_right = torch.abs(pts[:, 0] - right) < tol
        is_bottom = torch.abs(pts[:, 1] - bottom) < tol
        is_top = torch.abs(pts[:, 1] - top) < tol
        
        mesh.register_point_data("is_left_boundary", is_left)
        mesh.register_point_data("is_right_boundary", is_right)
        mesh.register_point_data("is_bottom_boundary", is_bottom)
        mesh.register_point_data("is_top_boundary", is_top)
        
        return mesh

    block = gen_symmetric_block(left=0.0, right=1.0, bottom=-1.0, top=1.0,
                          chara_length=0.06, order=2)
    
    print(f"Indenter (Circle): {indenter.points.shape[0]} nodes")
    print(f"Block (Rectangle): {block.points.shape[0]} nodes")
    
    # 2. Physics Models
    ind_model = LinearElasticityElementAssembler.from_mesh(indenter, E=E, nu=nu)
    blk_model = LinearElasticityElementAssembler.from_mesh(block, E=E, nu=nu)
    
    ind_pts = indenter.points
    blk_pts = block.points
    
    # 3. Boundary Conditions
    # Modified Strategy for better visualization:
    # 1. Fix the BACK (left) of the circle (u=0)
    # 2. Push the RIGHT of the block to the LEFT
    
    ind_back_mask = indenter.point_data["is_back"]
    ind_fix_nodes = torch.nonzero(ind_back_mask, as_tuple=True)[0]
    
    blk_right_mask = block.point_data["is_right_boundary"]
    blk_push_nodes = torch.nonzero(blk_right_mask, as_tuple=True)[0]
    
    # 4. Identification of Contact Surfaces
    # TWO-WAY CONTACT (Symmetric)
    # CRITICAL FIX: Strictly select ONLY boundary nodes to avoid zig-zag master segments
    
    # Surface 1: Indenter Front (Arc)
    # R = 1.0, centered at (-1, 0). 
    # Points on surface satisfy (x+1)^2 + y^2 = R^2
    # Our generated mesh might have slight numerical error, use tolerance.
    # Also restrict to the front face (e.g., x > -0.5)
    dist_from_center = torch.sqrt((ind_pts[:, 0] + 1.0)**2 + ind_pts[:, 1]**2)
    is_surface_circle = torch.abs(dist_from_center - 1.0) < 1e-3
    is_front = ind_pts[:, 0] > -0.5
    surf1_mask = is_surface_circle & is_front
    surf1_indices = torch.nonzero(surf1_mask, as_tuple=True)[0]
    
    # Surface 2: Block Left (Flat)
    # Block starts at x=0.0. Left boundary is x=0.0.
    is_surface_block = torch.abs(block.points[:, 0] - 0.0) < 1e-3
    surf2_mask = is_surface_block
    surf2_indices = torch.nonzero(surf2_mask, as_tuple=True)[0]
    
    print(f"Contact Surface 1 (Circle) Nodes: {len(surf1_indices)}")
    print(f"Contact Surface 2 (Block) Nodes: {len(surf2_indices)}")

    # Pre-sort indices for Master Segment construction
    # Circle: Sort by Y
    surf1_y = ind_pts[surf1_indices, 1]
    surf1_sorted_indices = surf1_indices[torch.argsort(surf1_y)]
    
    # Block: Sort by Y
    surf2_y = blk_pts[surf2_indices, 1]
    surf2_sorted_indices = surf2_indices[torch.argsort(surf2_y)]
    
    # 5. Optimization
    u_ind = torch.zeros_like(ind_pts, requires_grad=True)
    u_blk = torch.zeros_like(blk_pts, requires_grad=True)
    
    displacement = -0.15  # Moderate push
    penalty = 2e6         # Very stiff penalty
    
    optimizer = optim.LBFGS([u_ind, u_blk], lr=1.0, max_iter=200,
                            line_search_fn="strong_wolfe")
    
    def get_onesided_contact_energy(u_slave, slave_ref_pts, slave_indices, u_master, master_ref_pts, master_indices_sorted, master_is_flat_vertical=False):
        # Slave points
        pos_slave = u_slave[slave_indices] + slave_ref_pts[slave_indices]
        
        # Master segments
        pos_master = u_master[master_indices_sorted] + master_ref_pts[master_indices_sorted]
        
        P_j = pos_master[:-1]
        P_jp1 = pos_master[1:]
        
        V = P_jp1 - P_j
        Len_sq = (V**2).sum(dim=1)
        
        if master_is_flat_vertical:
            # Block is Master. V points Up. Normal points Left (-1, 0).
            Normals = torch.stack([-V[:, 1], V[:, 0]], dim=1)
        else:
            # Circle is Master. V points Up. Normal points Right (1, 0).
            Normals = torch.stack([V[:, 1], -V[:, 0]], dim=1)
            
        Normals = Normals / (torch.sqrt(Len_sq).unsqueeze(1) + 1e-8)
        
        S_exp = pos_slave.unsqueeze(1)
        P_j_exp = P_j.unsqueeze(0)
        V_exp = V.unsqueeze(0)
        Len_sq_exp = Len_sq.unsqueeze(0)
        
        W = S_exp - P_j_exp
        dot = (W * V_exp).sum(dim=2)
        t = dot / (Len_sq_exp + 1e-8)
        t_clamped = torch.clamp(t, 0.0, 1.0)
        
        Closest = P_j_exp + t_clamped.unsqueeze(2) * V_exp
        Dist_sq = ((S_exp - Closest)**2).sum(dim=2)
        
        # Find closest master segment for each slave node
        min_dist_sq, min_indices = torch.min(Dist_sq, dim=1)
        
        sel_normals = Normals[min_indices]
        sel_closest = torch.gather(Closest, 1, min_indices.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        
        vec = pos_slave - sel_closest
        # Gap = vec . Normal
        # Positive Gap = Separation
        # Negative Gap = Penetration
        gap = (vec * sel_normals).sum(dim=1)
        
        # Soft contact threshold
        # Activate if Gap < dist_thresh
        # Reduce threshold back to near-zero for visual "touching"
        dist_thresh = 1e-4 
        penetration = dist_thresh - gap
        active = penetration > 0
        
        # if active.sum() > 0:
        #    print(f"Active contacts: {active.sum().item()} / {len(gap)}")
        
        return 0.5 * penalty * (penetration[active]**2).sum()

    def closure():
        optimizer.zero_grad()
        
        # BCs
        u_ind_bc = u_ind.clone()
        u_ind_bc[ind_fix_nodes] = 0  # Fix Circle
        
        u_blk_bc = u_blk.clone()
        u_blk_bc[blk_push_nodes, 0] = displacement  # Push Block Left
        u_blk_bc[blk_push_nodes, 1] = 0             # No vertical slip at push boundary
        
        E1 = ind_model.energy(point_data={'displacement': u_ind_bc})
        E2 = blk_model.energy(point_data={'displacement': u_blk_bc})
        
        # Two-way Contact (Symmetric)
        # Re-enabling to prevent Block nodes from penetrating the Circle
        
        # 1. Circle (Slave) vs Block (Master)
        E_c1 = get_onesided_contact_energy(u_ind_bc, ind_pts, surf1_indices, u_blk_bc, blk_pts, surf2_sorted_indices, master_is_flat_vertical=True)
        
        # 2. Block (Slave) vs Circle (Master)
        # Using Circle as master requires care. The Circle surface is discretized.
        # But with matched mesh density, this helps prevent the Block nodes from diving in between Circle nodes.
        E_c2 = get_onesided_contact_energy(u_blk_bc, blk_pts, surf2_indices, u_ind_bc, ind_pts, surf1_sorted_indices, master_is_flat_vertical=False)
        
        loss = E1 + E2 + E_c1 + E_c2
        
        # Debug printing (only occasionally or if needed)
        # print(f"E1: {E1.item():.2e}, E2: {E2.item():.2e}, Ec1: {E_c1.item():.2e}, Ec2: {E_c2.item():.2e}")
        
        if loss.requires_grad:
            loss.backward()
            
        with torch.no_grad():
            if u_ind.grad is not None:
                u_ind.grad[ind_fix_nodes] = 0
            if u_blk.grad is not None:
                u_blk.grad[blk_push_nodes] = 0
                
        return loss
        
        if loss.requires_grad:
            loss.backward()
            
        with torch.no_grad():
            if u_ind.grad is not None:
                u_ind.grad[ind_fix_nodes] = 0
            if u_blk.grad is not None:
                u_blk.grad[blk_push_nodes] = 0
                
        return loss

    # Pre-sort Circle indices for Master use
    surf1_y = ind_pts[surf1_indices, 1]
    surf1_sorted_indices = surf1_indices[torch.argsort(surf1_y)]

    print("Solving...")
    for i in range(25):
        loss = optimizer.step(closure)
        if i % 5 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4e}")

    # Final visualization
    with torch.no_grad():
        u_ind[ind_fix_nodes] = 0
        u_blk[blk_push_nodes, 0] = displacement
        u_blk[blk_push_nodes, 1] = 0
        
        def compute_element_stress(mesh, u, E, nu):
            # Manual implementation of shape function derivatives for stress
            def get_dNdxi_tri6(xi_val):
                xi, eta = xi_val[0], xi_val[1]
                dNdxi = torch.zeros((6, 2), dtype=torch.float64)
                dNdxi[0, 0] = 4*xi + 4*eta - 3; dNdxi[0, 1] = 4*xi + 4*eta - 3
                dNdxi[1, 0] = 4*xi - 1;         dNdxi[1, 1] = 0
                dNdxi[2, 0] = 0;                dNdxi[2, 1] = 4*eta - 1
                dNdxi[3, 0] = 4 - 8*xi - 4*eta; dNdxi[3, 1] = -4*xi
                dNdxi[4, 0] = 4*eta;            dNdxi[4, 1] = 4*xi
                dNdxi[5, 0] = -4*eta;           dNdxi[5, 1] = 4 - 4*xi - 8*eta
                return dNdxi

            def get_dNdxi_quad9(xi_val):
                xi, eta = xi_val[0], xi_val[1]
                dNdxi = torch.zeros((9, 2), dtype=torch.float64)
                def phi(i, x):
                    return 0.5*x*(x-1) if i==0 else (1-x**2 if i==1 else 0.5*x*(x+1))
                def dphi(i, x):
                    return x-0.5 if i==0 else (-2*x if i==1 else x+0.5)
                idx_map = {0:(0,0), 1:(2,0), 2:(2,2), 3:(0,2), 4:(1,0), 5:(2,1), 6:(1,2), 7:(0,1), 8:(1,1)}
                for k in range(9):
                    i, j = idx_map[k]
                    dNdxi[k, 0] = dphi(i, xi) * phi(j, eta)
                    dNdxi[k, 1] = phi(i, xi) * dphi(j, eta)
                return dNdxi

            if "tri" in mesh.default_element_type:
                dNdxi = get_dNdxi_tri6(torch.tensor([1/3, 1/3], dtype=torch.float64))
            else:
                dNdxi = get_dNdxi_quad9(torch.tensor([0.0, 0.0], dtype=torch.float64))
            
            elems = mesh.elements(mesh.default_element_type)
            elem_nodes = mesh.points[elems]
            elem_u = u[elems]
            
            J = torch.einsum('eki,kj->eij', elem_nodes.double(), dNdxi.double())
            try:
                invJ = torch.inverse(J)
            except:
                invJ = torch.eye(2, dtype=torch.float64).expand(elems.shape[0], -1, -1)
                
            dNdx = torch.einsum('kj,eji->eki', dNdxi.double(), invJ)
            grad_u = torch.einsum('eki,ekj->eij', elem_u.double(), dNdx)
            eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
            
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            trace_eps = eps[:, 0, 0] + eps[:, 1, 1]
            sigma = 2 * mu * eps
            sigma[:, 0, 0] += lam * trace_eps
            sigma[:, 1, 1] += lam * trace_eps
            
            s11, s22, s12 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 0, 1]
            s33 = nu * (s11 + s22)
            von_mises = torch.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 + 6 * s12**2))
            
            node_stress = torch.zeros(mesh.points.shape[0], dtype=torch.float64)
            node_count = torch.zeros(mesh.points.shape[0], dtype=torch.float64)
            elems_flat = elems.flatten()
            vm_rep = von_mises.repeat_interleave(elems.shape[1])
            node_stress.scatter_add_(0, elems_flat, vm_rep)
            node_count.scatter_add_(0, elems_flat, torch.ones_like(vm_rep))
            return (node_stress / (node_count + 1e-8)).float()

        s_ind = compute_element_stress(indenter, u_ind, E, nu)
        s_blk = compute_element_stress(block, u_blk, E, nu)
        
        print("Visualizing...")
        setup_headless()
        
        pv_ind = mesh_to_pyvista(indenter)
        pv_blk = mesh_to_pyvista(block)
        
        # Add displacement for warping
        u_ind_3d = np.pad(u_ind.numpy(), ((0,0),(0,1)))
        u_blk_3d = np.pad(u_blk.numpy(), ((0,0),(0,1)))
        pv_ind.point_data['Displacement'] = u_ind_3d
        pv_blk.point_data['Displacement'] = u_blk_3d
        
        # Warp meshes
        pv_ind = pv_ind.warp_by_vector('Displacement', factor=1.0)
        pv_blk = pv_blk.warp_by_vector('Displacement', factor=1.0)
        
        pv_ind.point_data['VonMises'] = s_ind.numpy()
        pv_blk.point_data['VonMises'] = s_blk.numpy()
        
        # --- Smart Color Scaling ---
        # 1. Identify "Singularity Nodes" (Fixed boundary)
        # These nodes have unphysically high stress. We exclude them from clim calculation.
        mask_ind_valid = np.ones(s_ind.shape[0], dtype=bool)
        mask_ind_valid[ind_fix_nodes.detach().numpy()] = False
        
        # Also exclude neighbors of fixed nodes to be safe (heuristic: high stress outliers)
        # Or just take a percentile of the *rest*
        s_ind_valid = s_ind.numpy()[mask_ind_valid]
        s_blk_valid = s_blk.numpy() # Block has no singularity usually
        
        all_valid_stress = np.concatenate([s_ind_valid, s_blk_valid])
        
        # Use 99th percentile of the VALID data as max
        # This allows the contact stress (which is high but not infinite) to be red/yellow
        clim_max = np.percentile(all_valid_stress, 99.5)
        
        # Setup Plotter
        pv.set_plot_theme("document")
        pl = pv.Plotter(off_screen=True, window_size=[1600, 1000]) # Larger res
        pl.enable_anti_aliasing("ssaa")
        
        # Use 'turbo' for high contrast rainbow-like mapping, easier to see gradients
        cmap = "turbo"
        
        pl.add_mesh(pv_ind, scalars='VonMises', cmap=cmap, clim=[0, clim_max], 
                   show_edges=False, label="Indenter", show_scalar_bar=False)
        pl.add_mesh(pv_blk, scalars='VonMises', cmap=cmap, clim=[0, clim_max], 
                   show_edges=False, label="Block", show_scalar_bar=False)
        
        # Wireframe (more visible)
        pl.add_mesh(pv_ind.extract_all_edges(), color="black", opacity=0.3, line_width=1.0)
        pl.add_mesh(pv_blk.extract_all_edges(), color="black", opacity=0.3, line_width=1.0)
        
        # --- BC Visualization (Uniform Spatial Sampling) ---
        
        def get_spatially_uniform_indices(points, axis=1, n_markers=15):
            """Select indices uniformly spaced along a given axis."""
            vals = points[:, axis]
            v_min, v_max = vals.min(), vals.max()
            # Shrink range slightly to avoid edges
            margin = (v_max - v_min) * 0.05
            targets = np.linspace(v_min + margin, v_max - margin, n_markers)
            
            selected = []
            for t in targets:
                # Find closest node
                dist = np.abs(vals - t)
                idx = np.argmin(dist)
                selected.append(idx)
            return np.unique(selected)

        # 1. Block Push Arrows (Right side)
        blk_push_indices_all = blk_push_nodes.detach().numpy()
        push_pts_all = pv_blk.points[blk_push_indices_all]
        # Filter for uniform Y distribution
        subset_idx = get_spatially_uniform_indices(push_pts_all, axis=1, n_markers=15)
        push_pts = push_pts_all[subset_idx]
        
        pd_push = pv.PolyData(push_pts)
        pd_push["vectors"] = np.tile([-1.0, 0.0, 0.0], (len(push_pts), 1))
        # Thinner, longer arrows
        arrows_push = pd_push.glyph(orient="vectors", scale=False, factor=0.15, geom=pv.Arrow(tip_radius=0.2, shaft_radius=0.05))
        
        # 2. Circle Fixed Cones (Left side) -> Replaced with Translucent Mask
        # Instead of using mesh points which are irregular, we draw a geometric "Clamp" 
        # that represents the fixed region (x < -1.8).
        # We'll draw a Box from x=-2.2 to x=-1.8, spanning Y=[-1, 1] (approx radius)
        
        # Create a geometric box
        clamp_box = pv.Box(bounds=(-2.2, -1.8, -1.1, 1.1, -0.1, 0.1))
        
        # Colors
        bc_push_color = "#F1C40F" # Yellow
        bc_fix_color = "#E74C3C"  # Red
        
        pl.add_mesh(arrows_push, color=bc_push_color, label="Displacement")
        # Add translucent clamp
        pl.add_mesh(clamp_box, color=bc_fix_color, opacity=0.3, show_edges=True, label="Fixed Support")
        
        pl.view_xy()
        pl.camera.zoom(1.3)
        pl.add_title("Hertzian Contact Stress (Von Mises)", font_size=16, color="black")
        
        pl.add_scalar_bar(title="Von Mises Stress", 
                         title_font_size=14, 
                         label_font_size=12, 
                         color="black", 
                         position_x=0.3, position_y=0.05, width=0.4, height=0.06,
                         fmt="%.2f")
        
        pl.add_legend(bcolor=(0.95, 0.95, 0.95), size=(0.15, 0.15), loc='upper right')
        
        out_path = os.path.join(os.path.dirname(__file__), "hertzian_contact.png")
        pl.screenshot(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

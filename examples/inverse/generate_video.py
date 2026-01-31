#!/usr/bin/env python3
"""
Generate MP4 video of topology optimization process.
Uses sparse filter for memory efficiency.
"""
import sys
sys.path.insert(0, '../..')

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import meshio
import pyvista as pv
import os
from tqdm import tqdm
from scipy.spatial import cKDTree

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness


class SIMP3DStiffnessAssembler(ElementAssembler):
    def __post_init__(self, E=16500.0, nu=0.342, penal=3.0, E_min=1e-9):
        self.E0 = E
        self.nu = nu
        self.penal = penal
        self.E_min = E_min
    
    def forward(self, gradu, gradv, rho):
        dim = gradu.shape[0]
        E = self.E_min + (rho ** self.penal) * (self.E0 - self.E_min)
        Ba = voigt_shape_grad(gradu)
        Bb = voigt_shape_grad(gradv)
        C = voigt_stiffness(E, self.nu, dim)
        C = C.to(dtype=gradu.dtype, device=gradu.device)
        return Ba.T @ C @ Bb


def build_sparse_filter(centroids, R):
    """Build sparse density filter using KD-tree."""
    n = centroids.shape[0]
    centroids_np = centroids.cpu().numpy()
    
    # Build KD-tree
    tree = cKDTree(centroids_np)
    
    # Find neighbors within radius
    pairs = tree.query_pairs(R, output_type='ndarray')
    
    # Build sparse matrix data
    rows = []
    cols = []
    vals = []
    
    # Add self-connections
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(R)  # max weight at self
    
    # Add neighbor connections
    for i, j in pairs:
        dist = np.linalg.norm(centroids_np[i] - centroids_np[j])
        weight = R - dist
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([weight, weight])
    
    # Create sparse matrix
    from scipy.sparse import csr_matrix
    H = csr_matrix((vals, (rows, cols)), shape=(n, n))
    H_sum = np.array(H.sum(axis=1)).flatten()
    
    return H, H_sum


def apply_filter(H, H_sum, x, vols):
    """Apply sparse filter."""
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    vols_np = vols.cpu().numpy() if isinstance(vols, torch.Tensor) else vols
    
    filtered = H.dot(x_np * vols_np) / H_sum / vols_np
    return torch.tensor(filtered, dtype=torch.float64)


def main():
    # Parameters
    epochs = 15
    volfrac = 0.15
    penal = 3.0
    move = 0.2
    filter_R = 5.0
    R_fix = 6.0
    
    print("=" * 60)
    print("  TOPOLOGY OPTIMIZATION VIDEO GENERATION")
    print("=" * 60)
    
    # Load mesh
    mesh_data = meshio.read('ge_bracket.vtu')
    nodes = torch.tensor(mesh_data.points, dtype=torch.float64)
    elements = torch.tensor(mesh_data.cells[0].data, dtype=torch.long)
    domain = torch.tensor(mesh_data.cell_data['gmsh:geometrical'][0], dtype=torch.long)
    
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    dim = 3
    
    print(f"  Mesh: {n_nodes} nodes, {n_elements} elements")
    print(f"  Epochs: {epochs}")
    
    # Setup TensorMesh
    meshio_obj = meshio.Mesh(
        points=nodes.cpu().numpy(),
        cells=[('tetra', elements.cpu().numpy())]
    )
    tm_mesh = Mesh(meshio_obj, reorder=True)
    assembler = SIMP3DStiffnessAssembler.from_mesh(tm_mesh, E=16500.0, nu=0.342, penal=penal)
    
    # BC
    constraint_mask = torch.zeros(n_nodes, dim, dtype=torch.bool)
    for d in [1, 3, 4, 5]:
        dom_elements = elements[domain == d]
        dom_nodes = torch.unique(dom_elements)
        center = nodes[dom_nodes].mean(dim=0)
        dist_sq = (nodes[dom_nodes, 0] - center[0])**2 + (nodes[dom_nodes, 1] - center[1])**2
        inner = dist_sq < R_fix**2
        constraint_mask[dom_nodes[inner], :] = True
    
    dbc_mask = constraint_mask.flatten()
    condenser = Condenser(dbc_mask)
    
    # Rigid body modes for AMG
    n_inner = (~dbc_mask).sum().item()
    B_rigid = torch.zeros(n_inner, 3, dtype=torch.float64)
    for i in range(3):
        B_rigid[i::3, i] = 1.0
    
    # Load case (just one for speed)
    load_nodes = torch.unique(elements[(domain == 2) | (domain == 7)])
    n_load = len(load_nodes)
    F = torch.zeros(n_nodes * dim)
    F[load_nodes * dim + 2] = 8000.0 / n_load
    
    # Design domain
    design_mask = (domain == 6)
    n_design = design_mask.sum().item()
    elements_design = elements[design_mask]
    
    print(f"  Design elements: {n_design}")
    
    # Element volumes and centroids (vectorized)
    elem_pts = nodes[elements_design]  # [n_design, 4, 3]
    v0, v1, v2, v3 = elem_pts[:, 0], elem_pts[:, 1], elem_pts[:, 2], elem_pts[:, 3]
    cross_prod = torch.linalg.cross(v2 - v0, v3 - v0)
    vols_design = torch.abs(((v1 - v0) * cross_prod).sum(dim=1)) / 6.0
    centroids = elem_pts.mean(dim=1)
    
    vols = torch.zeros(n_elements)
    vols[design_mask] = vols_design
    
    # Build sparse filter
    print("  Building sparse filter...")
    H, H_sum = build_sparse_filter(centroids, filter_R)
    print(f"  Filter: {H.nnz} non-zeros ({100*H.nnz/n_design**2:.4f}% dense)")
    
    # Initialize
    rho_design = volfrac * torch.ones(n_design)
    rho_full = torch.ones(n_elements)
    rho_full[design_mask] = rho_design
    
    rho_min, rho_max = 0.01, 1.0
    V_0 = volfrac * vols_design.sum()
    
    # Storage for video frames
    frames_dir = 'output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"\n  Running optimization...")
    
    for epoch in tqdm(range(epochs), desc="Optimizing"):
        rho_full[design_mask] = rho_design
        
        # Assemble and solve
        with torch.no_grad():
            K = assembler(tm_mesh.points, element_data={"rho": rho_full})
        
        K_, F_ = condenser(K, F)
        with torch.no_grad():
            u_ = K_.solve(F_, backend='amg', tol=1e-5, B=B_rigid)
        u = condenser.recover(u_)
        
        compliance = torch.inner(F, u).item()
        
        # Analytic sensitivity
        rho_design_penal = rho_design ** (penal - 1.0)
        u_reshaped = u.reshape(-1, dim)
        u_e = u_reshaped[elements_design].reshape(n_design, -1)
        u_e_sq = (u_e ** 2).sum(dim=1)
        sensitivity = -penal * rho_design_penal * u_e_sq
        
        # Filter sensitivity
        sensitivity_filtered = apply_filter(H, H_sum, sensitivity, vols_design)
        
        # OC update
        l1, l2, tol_oc = 0.0, 1e9, 1e-3
        while l2 - l1 > tol_oc:
            l_mid = 0.5 * (l1 + l2)
            B_e = -sensitivity_filtered / l_mid
            rho_new = rho_design * torch.sqrt(B_e.clamp(min=1e-10))
            rho_new = torch.clamp(rho_new, rho_design - move, rho_design + move)
            rho_new = torch.clamp(rho_new, rho_min, rho_max)
            rho_new_filtered = apply_filter(H, H_sum, rho_new, vols_design)
            
            if (rho_new_filtered * vols_design).sum() > V_0:
                l1 = l_mid
            else:
                l2 = l_mid
        
        rho_design = rho_new_filtered.clone()
        
        # Save frame
        rho_full_np = rho_full.cpu().numpy()
        rho_full_np[design_mask.cpu().numpy()] = rho_design.cpu().numpy()
        
        # Create VTK mesh for visualization
        cells_vtk = np.hstack([np.full((n_elements, 1), 4), elements.cpu().numpy()]).flatten()
        cell_types = np.full(n_elements, pv.CellType.TETRA)
        grid = pv.UnstructuredGrid(cells_vtk, cell_types, nodes.cpu().numpy())
        grid.cell_data['density'] = rho_full_np
        
        # Threshold to show only solid material
        thresholded = grid.threshold(0.3, scalars='density')
        
        # Render frame
        plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
        plotter.add_mesh(thresholded, scalars='density', cmap='viridis', 
                        clim=[0, 1], show_scalar_bar=True)
        plotter.add_text(f"Epoch {epoch+1}/{epochs}\nCompliance: {compliance:.0f}", 
                        position='upper_left', font_size=12)
        plotter.camera_position = [(200, 100, 150), (0, 0, 0), (0, 0, 1)]
        plotter.screenshot(f'{frames_dir}/frame_{epoch:04d}.png')
        plotter.close()
        
        tqdm.write(f"  Epoch {epoch}: Compliance = {compliance:.0f}")
    
    print(f"\n  Generating MP4 video...")
    
    # Generate MP4 using ffmpeg
    output_video = 'output/optimization.mp4'
    os.system(f'ffmpeg -y -framerate 3 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_video}')
    
    print(f"  Video saved: {output_video}")
    print("=" * 60)


if __name__ == '__main__':
    main()

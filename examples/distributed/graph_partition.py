import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.visualization.draw_element_value import draw_element_value_2d
from tensormesh.mesh.partition import partition_mesh

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Create a Triangular Mesh
    n = 60
    h = 1.0 / n
    print(f"Creating Triangular Mesh with chara_length={h}...")
    
    mesh = gen_rectangle(chara_length=h, element_type='tri')
    mesh.to(device)

    elem_type = mesh.dim2eletyp[mesh.dim][0]
    n_elems = mesh.cells[elem_type].shape[0]
    print(f"Mesh created with {n_elems} {elem_type} elements.")

    # Run Partitioning
    n_parts = 4
    print(f"Running Spectral Partitioning ({n_parts} parts)...")

    t0 = time.time()
    # partition_mesh returns List[Mesh] with ghost nodes included
    submeshes = partition_mesh(mesh, n_parts=n_parts, method='spectral')

    if device == 'cuda':
        torch.cuda.synchronize()
    t_part = time.time() - t0

    print(f"Partitioning done in {t_part:.4f}s.")
    
    total_elems = sum([m.cells[elem_type].shape[0] for m in submeshes])
    print(f"Total elements in submeshes: {total_elems} (Original: {n_elems})")

    # --- Advanced Visualization: Exploded View ---
    try:
        print("Generating Exploded View for Paper...")
        
        # 1. Identify Shared Nodes (Global IDs)
        # Concatenate all orig_nids from all submeshes
        all_nids = torch.cat([m.point_data['orig_nid'] for m in submeshes])
        # Find IDs that appear more than once
        unique_nids, counts = torch.unique(all_nids, return_counts=True)
        shared_nids = unique_nids[counts > 1]
        
        # 2. Setup Plot
        fig, ax = plt.subplots(figsize=(12, 12))
        cmap = plt.get_cmap('tab10')
        
        # Calculate global center for explosion direction
        global_center = mesh.points.mean(dim=0).cpu().numpy()
        explode_factor = 0.2  # Gap size
        
        for i, sub in enumerate(submeshes):
            # Data preparation
            pts = sub.points.cpu().numpy()
            
            # Calculate explosion shift
            sub_center = pts.mean(axis=0)
            direction = sub_center - global_center
            shift = direction * explode_factor
            shifted_pts = pts + shift
            
            # Identify Ghost Nodes (Shared)
            local_orig_nids = sub.point_data['orig_nid']
            is_shared = torch.isin(local_orig_nids, shared_nids).cpu().numpy()
            
            # 3. Draw Elements (Shifted background)
            sub_cells = {k: v.cpu() for k,v in sub.cells.items() if k==elem_type}
            # Dummy values for API
            n_sub = sub_cells[elem_type].shape[0]
            dummy_vals = {elem_type: torch.zeros(n_sub)}
            
            color = cmap(i)
            
            # Draw filled elements
            draw_element_value_2d(
                shifted_pts, 
                sub_cells, 
                dummy_vals, 
                alpha=0.2, 
                ax=ax,
                color=color,
                edgecolor=color,
                linewidth=0.5
            )
            
            # 4. Draw Nodes
            # Internal nodes: Small dots
            internal_pts = shifted_pts[~is_shared]
            if len(internal_pts) > 0:
                ax.scatter(internal_pts[:, 0], internal_pts[:, 1], 
                           s=5, color=color, alpha=0.4, marker='.')
            
            # Ghost/Interface nodes: Highlighted with black rim
            ghost_pts = shifted_pts[is_shared]
            if len(ghost_pts) > 0:
                ax.scatter(ghost_pts[:, 0], ghost_pts[:, 1], 
                           s=30, facecolors=color, edgecolors='black', linewidth=1.0, 
                           marker='o', zorder=10) # High zorder to be on top
        
        ax.set_title(f"Domain Decomposition with Ghost Nodes (Exploded View)", fontsize=16)
        ax.axis('equal')
        ax.axis('off') # Clean look for paper
        
        output_file = 'graph_partition_exploded.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Result saved to {output_file}")
        
    except ImportError:
        print("Matplotlib not found, skipping visualization.")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

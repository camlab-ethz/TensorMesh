import torch
import time
import matplotlib.pyplot as plt
from tensormesh.dataset.mesh import gen_rectangle
from tensormesh.visualization.draw_element_value import draw_element_value_2d

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    # Create a Triangular Mesh
    # n=50 means roughly 50x50 nodes -> h = 1/50 = 0.02
    n = 50
    h = 1.0 / n
    print(f"Creating Triangular Mesh with chara_length={h}...")
    
    # element_type='tri' for triangular mesh
    mesh = gen_rectangle(chara_length=h, element_type='tri')
    mesh.to(device)

    # Get the main element type (usually 'tri' or 'quad')
    # Use max_dim to identify volume elements
    elem_type = mesh.dim2eletyp[mesh.dim][0]
    n_elems = mesh.cells[elem_type].shape[0]
    print(f"Mesh created with {n_elems} {elem_type} elements.")

    # Run Coloring
    print("Running Parallel Graph Coloring on Element Graph...")
    t0 = time.time()
    colors = mesh.color()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    t_color = time.time() - t0

    n_colors = colors.max().item() + 1
    print(f"Coloring done in {t_color:.4f}s. Used {n_colors} colors.")

    # Visualize
    try:
        print("Visualizing mesh elements...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Prepare values dict for visualization
        values = {elem_type: colors.cpu()}
        
        # Draw elements colored by their ID
        draw_element_value_2d(
            mesh.points.cpu(), 
            {k: v.cpu() for k,v in mesh.cells.items() if k==elem_type}, 
            values, 
            cmap='tab20', 
            ax=ax
        )
        
        ax.set_title(f"Triangular Element Coloring ({n_colors} colors)")
        ax.axis('equal')
        
        output_file = 'graph_coloring_result.png'
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")
    except ImportError:
        print("Matplotlib not found, skipping visualization.")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

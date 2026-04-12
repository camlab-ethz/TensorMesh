import sys
sys.path.append("../..")
import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensormesh.element import Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism

def plot_basis_points_comparison():
    font_size = 14
    scatter_size = 100  # Control size of scatter points
    # Create output directory
    os.makedirs("output/basis_points", exist_ok=True)
    
    # Plot basis points for each element type and order
    elements = [Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism]
    
    for element in elements:
        element_name = element.__name__.lower()
        
        # Create figure with subplots for orders 1-4
        if element.dim == 3:
            fig = plt.figure(figsize=(20, 5))
        else:
            fig = plt.figure(figsize=(16, 4))
            
        for order in range(1, 5):
            # Get basis points for this element and order
            basis = element.get_basis(order)
            n_basis = basis.shape[0]
            
            # Create subplot
            if element.dim == 1:
                ax = fig.add_subplot(1, 4, order)
                # Draw element edges
                edges = element.points[element.edge]
                for edge in edges:
                    ax.plot(edge[:, 0], [0, 0], 'k-', alpha=0.5)
                for i in range(n_basis):
                    ax.scatter(basis[i, 0], 0, s=scatter_size)
                    ax.text(basis[i, 0], 0, f'{i+1}', fontsize=font_size)
                ax.set_xlabel('x')
                ax.grid(True)
                
            elif element.dim == 2:
                ax = fig.add_subplot(1, 4, order)
                for i in range(n_basis):
                    ax.scatter(basis[i, 0], basis[i, 1], s=scatter_size)
                    ax.text(basis[i, 0], basis[i, 1], f'{i+1}', fontsize=font_size)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True)
                
                # Draw element edges
                edges = element.points[element.edge]
                for edge in edges:
                    ax.plot(edge[:, 0], edge[:, 1], 'k-', alpha=0.5)
                    
            else:  # 3D
                ax = fig.add_subplot(1, 4, order, projection='3d')
                for i in range(n_basis):
                    ax.scatter(basis[i, 0], basis[i, 1], basis[i, 2], s=scatter_size)
                    ax.text(basis[i, 0], basis[i, 1], basis[i, 2], f'{i+1}', fontsize=font_size)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                
                # Draw element edges
                edges = element.points[element.edge]
                for edge in edges:
                    ax.plot(edge[:, 0].numpy(), edge[:, 1].numpy(), edge[:, 2].numpy(), 'k-', alpha=0.5)
            
            ax.set_title(f'Order {order} ({n_basis} basis)')
            
        plt.suptitle(f'{element_name.capitalize()} Basis Points Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"output/basis_points/{element_name}_comparison.png")
        plt.close()

if __name__ == "__main__":
    plot_basis_points_comparison()

import torch
import numpy as np
from .. import sparse

def graph_coloring(adjacency: sparse.SparseMatrix, max_iter: int = 100) -> torch.Tensor:
    """
    Parallel graph coloring algorithm (Iterative Conflict Resolution).
    Runs efficiently on GPU.

    Parameters
    ----------
    adjacency : SparseMatrix
        The adjacency matrix of the graph. shape: [n_nodes, n_nodes]
    max_iter : int
        Maximum number of conflict resolution iterations.

    Returns
    -------
    torch.Tensor
        IntTensor of shape [n_nodes] containing the color ID for each node.
    """
    n_nodes = adjacency.shape[0]
    device = adjacency.device
    
    # Random weights for conflict resolution (fixed throughout)
    # If two neighbors pick the same color, the one with lower weight yields.
    node_weights = torch.rand(n_nodes, device=device)
    
    # Initialize colors (0 for all)
    colors = torch.zeros(n_nodes, dtype=torch.long, device=device)
    
    # Get edge list for efficient neighbor lookup
    # edges: [2, n_edges]
    edges = adjacency.edges
    u, v = edges[0], edges[1]
    
    # Remove self-loops if any
    mask = u != v
    u, v = u[mask], v[mask]
    
    # Main loop
    for i in range(max_iter):
        # 1. Detect conflicts
        # Check edges where endpoints have same color
        color_u = colors[u]
        color_v = colors[v]
        
        conflict_mask = (color_u == color_v)
        
        if not conflict_mask.any():
            break
            
        # Get conflicting edges
        conf_u = u[conflict_mask]
        conf_v = v[conflict_mask]
        
        # 2. Resolve conflicts
        # In a conflict pair (u, v), the one with lower weight must change
        # We identify nodes that need to change
        weight_u = node_weights[conf_u]
        weight_v = node_weights[conf_v]
        
        # Nodes to update: u where weight[u] < weight[v]
        update_mask_u = weight_u < weight_v
        nodes_to_update = conf_u[update_mask_u]
        
        # Also handle the other side (v < u) implicitly? 
        # Since edges are symmetric in sparse matrix usually, (u,v) and (v,u) exist.
        # But if adjacency stores symmetric edges explicitly, we process both.
        # If adjacency is strictly upper/lower triangular, we need logic.
        # Assuming adjacency is symmetric (contains both u->v and v->u).
        
        if nodes_to_update.numel() == 0:
            # Should not happen if conflicts exist and weights are random floats (low collision prob)
            break
            
        unique_nodes_to_update = torch.unique(nodes_to_update)
        
        # 3. Re-color conflicting nodes from a small random palette.
        # The palette [0, limit) widens slowly with i to guarantee convergence
        # while keeping the total color count low (~5-8 for planar graphs).
        # We only need *a* valid coloring for FEM assembly; minimizing color
        # count is secondary (though fewer colors → better GPU occupancy).
        limit = 6 + (i // 5)
        new_colors = torch.randint(0, limit, (unique_nodes_to_update.shape[0],), device=device)
        colors[unique_nodes_to_update] = new_colors

    return colors

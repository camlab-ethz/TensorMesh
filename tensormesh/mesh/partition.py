import torch
import numpy as np
import meshio
from typing import Optional, Dict, List, Any, Tuple
from .. import sparse


# ─── Fast coordinate-based partitioning ─────────────────────────────

def _rcb_partition_elements(centroids: torch.Tensor, n_parts: int) -> torch.Tensor:
    """Recursive Coordinate Bisection on element centroids.

    Recursively splits elements along the longest bounding-box axis at
    the median.  O(n log n) — orders of magnitude faster than spectral.

    Parameters
    ----------
    centroids : torch.Tensor
        Element centroids, shape ``[n_elements, dim]``.
    n_parts : int
        Number of partitions (best if power of 2).

    Returns
    -------
    torch.Tensor
        Integer labels ``[n_elements]`` in ``[0, n_parts-1]``.
    """
    n = centroids.shape[0]
    labels = torch.zeros(n, dtype=torch.long)

    # Stack: (indices_into_centroids, partition_offset)
    stack: List[Tuple[torch.Tensor, int, int]] = [
        (torch.arange(n), 0, n_parts)
    ]

    while stack:
        indices, offset, parts_remaining = stack.pop()

        if parts_remaining <= 1 or len(indices) == 0:
            labels[indices] = offset
            continue

        # Pick longest axis of bounding box
        pts = centroids[indices]
        extent = pts.max(dim=0).values - pts.min(dim=0).values
        axis = extent.argmax().item()

        # Median split along that axis
        vals = pts[:, axis]
        median = vals.median()
        left_mask = vals <= median

        # Ensure both sides are non-empty
        if left_mask.all():
            left_mask[vals.argmax()] = False
        elif not left_mask.any():
            left_mask[vals.argmin()] = True

        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        left_parts = parts_remaining // 2
        right_parts = parts_remaining - left_parts

        stack.append((left_idx, offset, left_parts))
        stack.append((right_idx, offset + left_parts, right_parts))

    return labels


def _compute_element_centroids(mesh) -> Tuple[torch.Tensor, List[str], List[int]]:
    """Compute centroids for all top-dimension elements.

    Returns (centroids, target_types, cell_counts).
    """
    dim_key = getattr(mesh, 'max_dim', mesh.dim)
    target_types = mesh.dim2eletyp[dim_key]
    if isinstance(target_types, str):
        target_types = [target_types]

    cell_counts = [mesh.cells[k].shape[0] for k in target_types]
    points = mesh.points  # [n_points, dim]

    centroid_list = []
    for k in target_types:
        cells = mesh.cells[k]  # [n_elem, n_basis]
        elem_pts = points[cells]  # [n_elem, n_basis, dim]
        centroid_list.append(elem_pts.mean(dim=1))  # [n_elem, dim]

    centroids = torch.cat(centroid_list, dim=0)  # [total_elements, dim]
    return centroids, target_types, cell_counts

def _spectral_bisection_gpu(adjacency: sparse.SparseMatrix, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Partition a subgraph using Fiedler vector computed via LOBPCG on GPU.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(part0_global, part1_global)`` — two 1D index tensors that together
        cover the input ``indices``. Either side may be empty when the
        bisection degenerates; the caller in :func:`graph_partition` then
        falls back to a manual median split.
    """
    n = indices.shape[0]
    device = adjacency.device
    empty = torch.empty(0, dtype=indices.dtype, device=device)

    if n == 0:
        return empty, empty
    if n == 1:
        return indices, empty
    
    # 1. Extract subgraph Laplacian
    # Filter edges: both u and v must be in indices
    # To do this efficiently:
    # 1. Create a boolean mask of nodes in subgraph
    mask = torch.zeros(adjacency.shape[0], dtype=torch.bool, device=device)
    mask[indices] = True
    
    row, col = adjacency.row, adjacency.col
    edge_mask = mask[row] & mask[col]
    
    sub_row = row[edge_mask]
    sub_col = col[edge_mask]
    sub_data = adjacency.edata[edge_mask]
    
    # Relabel indices to [0, n-1]
    indices_sorted, _ = torch.sort(indices)
    
    sub_row_mapped = torch.searchsorted(indices_sorted, sub_row)
    sub_col_mapped = torch.searchsorted(indices_sorted, sub_col)
    
    # Build Laplacian L = D - A
    
    # Convert to torch.sparse_coo_tensor for efficient matmul
    sub_indices = torch.stack([sub_row_mapped, sub_col_mapped])
    A_sparse = torch.sparse_coo_tensor(sub_indices, sub_data, (n, n)).coalesce()
    
    # Degree
    degrees = torch.sparse.sum(A_sparse, dim=1).to_dense()
    
    # Solve eigenvalue problem
    try:
        # Use lobpcg
        # Simpler fallback: Use generic method or dense if small.
        if n < 2048:
            L_dense = torch.diag(degrees) - A_sparse.to_dense()
            vals, vecs = torch.linalg.eigh(L_dense)
            fiedler = vecs[:, 1]
        else:
            # For sparse LOBPCG
            idx_d = torch.arange(n, device=device)
            indices_d = torch.stack([idx_d, idx_d])
            L_sparse = torch.sparse_coo_tensor(
                torch.cat([indices_d, sub_indices], dim=1),
                torch.cat([degrees, -sub_data]),
                (n, n)
            ).coalesce()
            
            vals, vecs = torch.lobpcg(L_sparse, k=2, largest=False)
            fiedler = vecs[:, 1]
            
    except Exception as e:
        # Fallback to random bisection if the eigensolver fails
        labels = (torch.rand(n, device=device) > 0.5).long()
        part0_local = torch.nonzero(labels == 0).squeeze(1)
        part1_local = torch.nonzero(labels == 1).squeeze(1)
        return indices_sorted[part0_local], indices_sorted[part1_local]

    # Median cut
    median = torch.median(fiedler)
    labels = (fiedler > median).long()
    
    part0_local = torch.nonzero(labels == 0).squeeze(1)
    part1_local = torch.nonzero(labels == 1).squeeze(1)
    
    part0_global = indices_sorted[part0_local]
    part1_global = indices_sorted[part1_local]
    
    return part0_global, part1_global


def graph_partition(adjacency: sparse.SparseMatrix, n_parts: int, method: str = 'spectral') -> torch.Tensor:
    """
    Partition a graph into balanced subdomains using GPU-accelerated algorithms.
    
    This function divides graph nodes into ``n_parts`` groups such that:
    
    1. Each partition has approximately equal number of nodes (load balance)
    2. The number of edges crossing partitions is minimized (minimal interface)
    
    Parameters
    ----------
    adjacency : SparseMatrix
        The adjacency matrix of the graph. Shape: ``[n_nodes, n_nodes]``.
        Should be symmetric for undirected graphs.
    n_parts : int
        Number of partitions to create. For spectral method, works best 
        when ``n_parts`` is a power of 2.
    method : str, optional
        Partitioning algorithm to use:
        
        - ``'spectral'``: Recursive Spectral Bisection using Fiedler vector.
          Computed via ``torch.lobpcg`` on GPU. Default method.
        - ``'metis'``: Uses pymetis library (requires pymetis installation).
          Falls back to spectral if pymetis is not available.
    
    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``[n_nodes]`` containing partition labels 
        in range ``[0, n_parts-1]``.
    
    Notes
    -----
    The spectral method computes the Fiedler vector (second smallest eigenvector 
    of the graph Laplacian :math:`L = D - A`) and recursively bisects based on 
    median values. This preserves locality in the partitioning.
    
    For small subgraphs (< 2048 nodes), dense eigensolvers are used for stability.
    For larger graphs, LOBPCG is used for efficiency.
    
    Examples
    --------
    >>> from tensormesh.mesh import graph_partition
    >>> # Partition element adjacency into 4 parts
    >>> adj = mesh.element_adjacency()
    >>> labels = graph_partition(adj, n_parts=4, method='spectral')
    >>> print(f"Partition sizes: {[(labels == i).sum().item() for i in range(4)]}")
    """
    n_nodes = adjacency.shape[0]
    device = adjacency.device
    
    if method == 'metis':
        try:
            import pymetis
            adj_scipy = adjacency.to_scipy_coo().tocsr()
            n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adj_scipy)
            return torch.tensor(membership, device=device)
        except ImportError:
            print("pymetis not found, falling back to spectral.")
            method = 'spectral'
            
    if method == 'spectral':
        labels = torch.zeros(n_nodes, dtype=torch.long, device=device)
        
        # Queue of node indices for each part
        parts = [torch.arange(n_nodes, device=device)]
        
        while len(parts) < n_parts:
            # Pick largest part to split
            lengths = [len(p) for p in parts]
            split_idx = np.argmax(lengths)
            indices_to_split = parts.pop(split_idx)
            
            # Bisect
            part0, part1 = _spectral_bisection_gpu(adjacency, indices_to_split)
            
            if len(part0) > 0: parts.append(part0)
            if len(part1) > 0: parts.append(part1)
            
            if len(part0) == 0 or len(part1) == 0:
                n = len(indices_to_split)
                if n > 1:
                    mid = n // 2
                    parts.append(indices_to_split[:mid])
                    parts.append(indices_to_split[mid:])
        
        # Assign labels
        for i, indices in enumerate(parts):
            labels[indices] = i
            
        return labels
    
    raise ValueError(f"Unknown method {method}")

def partition_mesh(mesh, n_parts: int, method: str = 'coordinate', ghost_nodes: bool = True) -> List[Any]:
    """
    Partition a mesh into independent submeshes for parallel computation.

    This function performs element-based domain decomposition, creating
    ``n_parts`` submeshes that can be processed independently (with ghost
    node communication for boundary data exchange).

    Parameters
    ----------
    mesh : Mesh
        The mesh to partition. Must have elements of at least one type.
    n_parts : int
        Number of partitions to create.
    method : str, optional
        Partitioning algorithm:

        - ``'coordinate'``: Recursive Coordinate Bisection on element
          centroids.  Very fast — O(n log n).  Default.
        - ``'spectral'``: Recursive Spectral Bisection using Fiedler vector.
          Better partition quality but much slower.
        - ``'metis'``: Uses pymetis (requires installation).
    ghost_nodes : bool, optional
        Whether to include ghost nodes (shared boundary nodes) in submeshes.
        Currently only ``True`` is supported for element-based partitioning.
        Default: ``True``.

    Returns
    -------
    List[Mesh]
        A list of ``n_parts`` submeshes. Each submesh is a complete ``Mesh``
        object containing:

        - Local nodes and elements with renumbered indices
        - ``point_data['orig_nid']``: Tensor mapping local node indices to
          original global node indices (for data exchange between partitions)

        Returns ``None`` for empty partitions.

    Notes
    -----
    Ghost nodes are nodes shared between partitions (on the interface).
    They are duplicated in each partition that uses them, enabling independent
    local computation. The ``orig_nid`` mapping allows reconstruction of
    global solutions and inter-partition communication.

    Examples
    --------
    >>> from tensormesh.mesh.partition import partition_mesh
    >>> submeshes = partition_mesh(mesh, n_parts=4, method='coordinate')
    >>> for i, sub in enumerate(submeshes):
    ...     if sub is not None:
    ...         print(f"Part {i}: {sub.n_nodes} nodes, {sub.n_elements} elements")
    ...         # Access original node IDs for global assembly
    ...         global_ids = sub.point_data['orig_nid']
    """
    if not ghost_nodes:
        raise NotImplementedError("ghost_nodes=False is not yet supported. Element-based partitioning always implies ghost nodes.")

    # 1. Partition Elements
    if method == 'coordinate':
        # Fast path: RCB on element centroids — no adjacency graph needed
        centroids, target_types, cell_counts = _compute_element_centroids(mesh)
        element_labels = _rcb_partition_elements(centroids, n_parts)
    else:
        # Slow path: spectral / metis (requires element adjacency)
        element_labels = mesh.partition(n_parts, method)
        dim_key = getattr(mesh, 'max_dim', mesh.dim)
        target_types = mesh.dim2eletyp[dim_key]
        if isinstance(target_types, str):
            target_types = [target_types]
        cell_counts = [mesh.cells[k].shape[0] for k in target_types]

    # 2. Iterate and split
    submeshes = []

    labels_per_type = {}
    curr = 0
    for k, count in zip(target_types, cell_counts):
        labels_per_type[k] = element_labels[curr:curr+count]
        curr += count
        
    for p in range(n_parts):
        sub_cells_dict = {}
        sub_nodes_indices = []
        
        # Filter elements for this partition
        for k in target_types:
            labels = labels_per_type[k]
            mask = (labels == p)
            if not mask.any():
                continue
                
            cells = mesh.cells[k][mask] # [n_sub, n_nodes_per_elem]
            sub_cells_dict[k] = cells
            sub_nodes_indices.append(cells.reshape(-1))
            
        if not sub_cells_dict:
            # Empty partition
            submeshes.append(None)
            continue
            
        # Unique nodes required by this partition
        used_nodes = torch.cat(sub_nodes_indices).unique()
        used_nodes = torch.sort(used_nodes).values
        
        # Extract points
        sub_points = mesh.points[used_nodes]
        
        # Remap cells
        sub_cells_meshio = []
        for k, cells in sub_cells_dict.items():
            new_cells = torch.searchsorted(used_nodes, cells)
            sub_cells_meshio.append((k, new_cells.cpu().numpy()))
            
        # Create meshio mesh
        m_io = meshio.Mesh(
            points=sub_points.cpu().numpy(),
            cells=sub_cells_meshio
        )
        
        # Create TensorMesh
        submesh = mesh.__class__(m_io)
        submesh.to(mesh.device)
        
        # Add metadata
        submesh.point_data['orig_nid'] = used_nodes
        
        submeshes.append(submesh)
        
    return submeshes


import torch


def partite_by_node(edge_u, edge_v, inner_mask, return_outer=False, return_in2ou=False, return_ou2in=False):
    """
        partition the matrix A into four blocks
        according to the mask
        Parameters:
        -----------
            A: torch.sparse_coo_tensor of shape [n_node, n_node]
                the sparse matrix A
            mask: torch.Tensor of shape [n_node]
                the mask of the matrix A
        Returns:
        --------
            edge_inner_mask: torch.Tensor of shape [n_edge]
                the mask of the inner edges
            edge_outer_mask: torch.Tensor of shape [n_edge]
                the mask of the boundary edges
            edge_in2ou_mask: torch.Tensor of shape [n_edge]
                the mask of the edges from inner to boundary
            edge_ou2in_mask: torch.Tensor of shape [n_edge]
                the mask of the edges from boundary to inner
    """
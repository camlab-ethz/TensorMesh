from matplotlib.pyplot import connect
from sympy import N
import torch
import numpy as np
from functools import reduce
from typing import Optional, Iterator, Tuple, Dict
from .. import sparse 
from .. import element as E
from ..vmap import vmap


def cum_dict(d:Dict[str,int])->Dict[str,Tuple[int,int]]:
    """cumulative sum of the dictionary values

    Parameters
    ----------
    d : Dict[str,int]
        the dictionary to be cumulated
    Returns
    -------
    Dict[str, Tuple[int,int]]
        the dictionary with the cumulative sum of the dictionary values
    """
    keys = list(d.keys())
    values = list(d.values())
    cum_values     = np.zeros(len(values)+1)
    cum_values[1:] = np.cumsum(values)
    return {k:(int(v1),int(v2)) for k,v1, v2 in zip(keys, cum_values[:-1], cum_values[1:])}

def dense_connect(x:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        1D Tensor of shape [n_batch, n_node]
    Returns
    -------
    torch.Tensor
        2D Tensor of shape [2, n_batch * n_node^2]
    """
    dense          = lambda x: torch.stack(torch.meshgrid(x,x),-1) # [n]->[n^2, 2]
    parallel_dense = vmap(dense) # [n_batch, n]->[n_batch, n^2, 2]
    return parallel_dense(x).reshape(-1, 2).T # [2, n_batch * n**2]

def coalesce(edges:torch.Tensor, n_points:int)->torch.Tensor:
    """
    Parameters
    ----------
    edge : torch.Tensor
        2D Tensor of shape [2, n_edge]
    Returns
    -------
    torch.Tensor
        2D Tensor of shape [2, n_edge]
    """
    connections = torch.sparse_coo_tensor(
        edges, torch.ones(edges.shape[1], device=edges.device), size=(n_points, n_points)
    ).coalesce()
    edges = connections.indices()
    return edges

def facet_connect(facet:torch.Tensor, element_ids:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    facet : torch.Tensor
        2D Tensor of shape [n_element*n_facet, n_node]
    element_ids: torch.Tensor
        1D Tensor of shape [n_element*n_facet]
    Returns
    -------
    torch.Tensor
        2D Tensor of shape [2, n_edges]
    """
    assert facet.dim() == 2, f"the facet should be 3D, but got {facet.dim()}"
    assert facet.shape[0] == element_ids.shape[0], f"the first dimension of the facet should be the same as the element_ids, but got {facet.shape[0]} and {element_ids.shape[0]}"
    # TODO: add this genius idea to the paper and explain with an algorithm
 
    # make sure the facet node order is unique
    facet  = facet.sort(dim=-1).values
    unique_facet, inverse_indices, counts = facet.unique(
                                                dim=0, 
                                                return_counts=True,  
                                                return_inverse=True)
    # unique_facet : [n_unique_facet, n_node_per_facet]
    # inverse_indices : [n_element * n_facet_per_element]
    # counts : [n_unique_facet]
    
    # the count = 2 means the facet is shared by two elements, 
    # otherwise the facet is on the boundary of the domain
    assert counts.max() == 2, f"the maximum number of elements sharing a boundary is 2, but got {counts.max()}"
    connect_facet_mask = counts == 2                         # [n_unique_facet ]
    connect_facet_mask = connect_facet_mask[inverse_indices] # [n_element * n_facet_per_element] 
    # connect_facet_mask has 2*n_unique_facet True values

    # by sorting the inverse_indices, we can get the order like [0,0,1,1,2,2,3,3,...]
    connect_indices = inverse_indices[connect_facet_mask] # [2*n_unique_facet]
    connect_indices = torch.argsort(connect_indices)      # [2*n_unique_facet]

    connect_element_ids = element_ids[connect_facet_mask]           # [2*n_unique_facet]
    connect_element_ids = connect_element_ids[connect_indices]      # [2*n_unique_facet]

    connection  = connect_element_ids.reshape(-1, 2).T # [2, n_unique_facet]

    # add reverse direction
    connection = torch.cat([connection, torch.stack([connection[1], connection[0]])], -1) # [2, 2*n_unique_facet]
    return connection



def node_adjacency(elements:torch.Tensor | Iterator[torch.Tensor],
                   n_points:int
                   )->sparse.SparseMatrix:
    """get the node adjacency matrix, inside each element, the nodes are considered fully connected

    Parameters
    ----------
    elements: torch.Tensor or Iterator[torch.Tensor]
        2D Tensor of shape [n_element, n_node] or Iterator of 2D Tensor of shape [n_node]
    Returns
    -------
    SparseMatrix 
        the adjacency matrix of nodes :math:`[|\\mathcal V|,|\\mathcal V|]`, where :math:`|\\mathcal V|` is the number of nodes
    """
    dense_connect = lambda y: vmap(lambda x:torch.stack(torch.meshgrid(x,x),-1))(y).reshape(-1, 2).T

    if isinstance(elements, torch.Tensor):
        edges = dense_connect(elements)
    else:
        edges = torch.cat(list(map(dense_connect, elements)), -1)
    
    edges = coalesce(edges, n_points)

    adjacency = sparse.SparseMatrix(
        edata = torch.ones(edges.shape[1], device=edges.device),
        row   = edges[0],
        col   = edges[1],
        shape = (n_points, n_points)
    )

    return adjacency

def element_adjacency(elements:Dict[str,torch.Tensor])->sparse.SparseMatrix:
        """get the element adjacency matrix, the element are considered connected only if they share a boundary/facet
        
        Parameters
        ----------
        element_type : str or Iterable[str] or None
            the type of the elements, should be of same dimension
            if :obj:`None` is the :obj:`default_element_type`
            default : :obj:`None`

        Returns
        -------
        SparseMatrix 
            the adjacency matrix of elements :math:`[|\\mathcal C|,|\\mathcal C|]`, where :math:`|\\mathcal C|` is the number of elements
        """
        n_elements = sum([v.shape[0] for v in elements.values()])
        element_ids= cum_dict({k:v.shape[0] for k,v in elements.items()})

        facet_type2facet         = {}
        facet_type2element_ids   = {}
       
        for element_type, element_index in elements.items():
            element   = E.element_type2element(element_type)
            order     = E.element_type2order[element_type]
            start, end = element_ids[element_type]
            _element_ids = torch.arange(start, end, device=element_index.device).repeat_interleave(element.get_n_facet())
            if element.is_mix_facet:
                local_tri_facet, local_quad_facet = element.get_facet(order)
                global_tri_facet = element_index[:, local_tri_facet] # [n_element, n_tri_facet, n_node_per_facet]
                global_quad_facet= element_index[:, local_quad_facet]# [n_element, n_quad_facet, n_node_per_facet]
                global_tri_facet = global_tri_facet.reshape(-1, global_tri_facet.shape[-1]) # [n_element*n_tri_facet, n_node_per_facet]
                global_quad_facet= global_quad_facet.reshape(-1, global_quad_facet.shape[-1]) # [n_element*n_quad_facet, n_node_per_facet]
                if E.Triangle in facet_type2facet:
                    facet_type2facet[E.Triangle] = torch.cat([facet_type2facet[E.Triangle], global_tri_facet], 0)
                    facet_type2element_ids[E.Triangle] = torch.cat([facet_type2element_ids[E.Triangle], _element_ids], 0)
                else:
                    facet_type2facet[E.Triangle] = global_tri_facet
                    facet_type2element_ids[E.Triangle] = _element_ids
                if E.Quadrilateral in facet_type2facet:
                    facet_type2facet[E.Quadrilateral] = torch.cat([facet_type2facet[E.Quadrilateral], global_quad_facet], 0)
                    facet_type2element_ids[E.Quadrilateral] = torch.cat([facet_type2element_ids[E.Quadrilateral], _element_ids], 0)
                else:
                    facet_type2facet[E.Quadrilateral] = global_quad_facet
                    facet_type2element_ids[E.Quadrilateral] = _element_ids
                 
            else:
                local_facet     = element.get_facet(order) # [n_facet, n_node_per_facet]
                global_facet    = element_index[:, local_facet] # [n_element, n_facet, n_node_per_facet]
                global_facet    = global_facet.reshape(-1, global_facet.shape[-1]) # [n_element*n_facet, n_node_per_facet]
                facet_type      = element.get_facet_type() 
                assert not isinstance(facet_type, tuple), f"the facet type should not be a tuple, but got {facet_type}"
                if facet_type in facet_type2facet:
                    facet_type2facet[facet_type] = torch.cat([facet_type2facet[facet_type], global_facet], 0)
                    facet_type2element_ids[facet_type] = torch.cat([facet_type2element_ids[facet_type], _element_ids], 0)
                else:
                    facet_type2facet[facet_type] = global_facet
                    facet_type2element_ids[facet_type] = _element_ids

        edges = []
       
        for facet_type in facet_type2facet.keys():
            _edges = facet_connect(facet_type2facet[facet_type], facet_type2element_ids[facet_type])
            edges.append(_edges)
        edges = torch.cat(edges, -1)
        adjacency = sparse.SparseMatrix(
            edata = torch.ones(edges.shape[1], device=edges.device),
            row   = edges[0],
            col   = edges[1],
            shape = (n_elements, n_elements)
        )
        return adjacency
  
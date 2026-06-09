r"""Shared mesh-topology helpers for the assemblers.

The sparsity pattern of a Galerkin matrix is the set of unique
``(row, col)`` node pairs touched by any element. :func:`build_edges`
computes that pattern — together with the per-element scatter indices —
for both the square single-space case (:class:`~tensormesh.assemble.ElementAssembler`)
and the rectangular two-space case (the off-diagonal blocks of a mixed
multi-field form).
"""
from typing import Dict, Mapping, Tuple

import numpy as np
import scipy.sparse
import torch

__all__ = ["build_edges"]


def build_edges(
    conn_pairs: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""Deduplicate the ``(row, col)`` basis-pair indices of a bilinear form.

    For every element type, every element contributes all pairs of one
    row-connectivity entry and one column-connectivity entry. The unique
    pairs form the COO pattern (the "edges") of the assembled sparse
    matrix; the returned per-slot edge ids drive the
    :class:`~tensormesh.assemble.projector.ReduceProjector` scatter.

    Row and column connectivity may come from two *different* function
    spaces (e.g. the velocity–pressure coupling block of a Taylor–Hood
    form), so the pattern may be rectangular.

    Parameters
    ----------
    conn_pairs : Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
        Maps each element type to ``(row_conn, col_conn)`` integer tensors
        of shape ``[n_element, n_row_basis]`` / ``[n_element, n_col_basis]``
        (same ``n_element``), holding node indices in ``range(shape[0])``
        and ``range(shape[1])`` respectively.
    shape : Tuple[int, int]
        ``(n_row_nodes, n_col_nodes)`` bounds of the index space.

    Returns
    -------
    edges : torch.Tensor
        Long tensor of shape ``[2, n_edges]`` listing the unique
        ``(row, col)`` pairs in CSR (row-major) order.
    elem_eids : Dict[str, torch.Tensor]
        Maps each element type to a long tensor of shape
        ``[n_element * n_row_basis * n_col_basis]`` holding the edge id of
        every ``(element, i, j)`` slot, flattened in ``(e, i, j)`` order —
        exactly the flatten order used by
        ``ReduceProjector(from_shape=(n_element, n_row_basis, n_col_basis))``.
    """
    flat: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for element_type, (row_conn, col_conn) in conn_pairs.items():
        assert row_conn.shape[0] == col_conn.shape[0], (
            f"row/col connectivity of '{element_type}' disagree on n_element: "
            f"{row_conn.shape[0]} vs {col_conn.shape[0]}"
        )
        n_row_basis, n_col_basis = row_conn.shape[1], col_conn.shape[1]
        elem_u = row_conn[:, :, None].expand(-1, -1, n_col_basis).reshape(-1)
        elem_v = col_conn[:, None, :].expand(-1, n_row_basis, -1).reshape(-1)
        # .copy(): scipy's fancy indexing rejects the non-writeable views
        # .numpy() can return
        flat[element_type] = (
            elem_u.cpu().numpy().copy(),
            elem_v.cpu().numpy().copy(),
        )

    all_u = np.concatenate([u for u, _ in flat.values()])
    all_v = np.concatenate([v for _, v in flat.values()])

    tmp = scipy.sparse.coo_matrix((  # used to remove duplicated edges
        np.ones_like(all_u),  # data
        (all_u, all_v),       # (row, col)
    ), shape=shape).tocsr().tocoo()
    edge_u, edge_v = tmp.row, tmp.col
    num_edges = len(edge_u)
    eids_csr = scipy.sparse.coo_matrix((
        np.arange(num_edges), (edge_u, edge_v)
    ), shape=shape).tocsr()

    elem_eids = {
        element_type: torch.from_numpy(np.array(eids_csr[u, v]).ravel()).long()
        for element_type, (u, v) in flat.items()
    }
    edges = torch.from_numpy(np.stack([edge_u, edge_v], 0)).long()
    return edges, elem_eids

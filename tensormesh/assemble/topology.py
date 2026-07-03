r"""Shared mesh-topology helpers for the assemblers.

The sparsity pattern of a Galerkin matrix is the set of unique
``(row, col)`` node pairs touched by any element. :func:`build_edges`
computes that pattern — together with the per-element scatter indices —
for both the square single-space case (:class:`~tensormesh.assemble.ElementAssembler`)
and the rectangular two-space case (the off-diagonal blocks of a mixed
multi-field form).

:func:`lagrange_dofmap` builds the global DOF numbering of a Lagrange
space of **arbitrary order on a mesh of any (single) order** from the
mesh *topology* — vertices, unique edges, cell interiors — instead of
reusing mesh nodes. This is what decouples the field order from the
mesh order (generalized Taylor–Hood pairs, quadratic fields on linear
gmsh imports); :func:`lagrange_boundary_mask` marks its boundary DOFs
by facet incidence, without any geometric predicate.
"""
from typing import Dict, List, Mapping, NamedTuple, Tuple

import numpy as np
import scipy.sparse
import torch

from ..element import element_type2element

__all__ = ["build_edges", "lagrange_dofmap", "lagrange_boundary_mask", "LagrangeDofMap"]


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


class LagrangeDofMap(NamedTuple):
    r"""Global DOF numbering of a Lagrange space built from mesh topology.

    Attributes
    ----------
    conn : Dict[str, torch.Tensor]
        Per element type, the field-local connectivity ``[n_element, n_basis_f]``:
        slot ``i`` holds the global DOF of the field's ``i``-th reference
        basis function on that element.
    n_dofs : int
        Total number of scalar DOF carriers of the space.
    vertex_ids : torch.Tensor
        Mesh point ids of the vertex DOFs, sorted — ``[n_vertex_dofs]``.
        Vertex DOFs occupy the leading ``[0, n_vertex_dofs)`` block.
    ref_nodes : Dict[str, torch.Tensor]
        Per element type, the field's reference node coordinates
        ``[n_basis_f, D]`` (float64) — slot-aligned with ``conn``; used to
        evaluate interpolation tables and to push DOF coordinates through
        the isoparametric geometry map.
    """
    conn: Dict[str, torch.Tensor]
    n_dofs: int
    vertex_ids: torch.Tensor
    ref_nodes: Dict[str, torch.Tensor]


def _classify_reference_nodes(elem_cls, order: int):
    """Match each reference Lagrange node to (vertex | edge slot | interior).

    Returns a list of ``("vertex", local_vertex, 0)``, ``("edge", local_edge, j)``
    with ``j in 1..order-1`` counted from edge endpoint 0 towards endpoint 1,
    or ``("interior", running_index, 0)`` — derived purely by coordinate
    matching, so no assumption on the internal node ordering is needed.
    """
    ref = elem_cls.get_basis(order, torch.float64)          # [nb, D]
    verts = elem_cls.points.to(torch.float64)               # [n_vertex, D]
    edges = elem_cls.edge.tolist()                          # [n_le, 2]
    slots, n_interior = [], 0
    for x in ref:
        dist = (verts - x).norm(dim=1)
        if dist.min() < 1e-8:
            slots.append(("vertex", int(dist.argmin()), 0))
            continue
        hit = None
        for local_edge, (a, b) in enumerate(edges):
            va, vb = verts[a], verts[b]
            tv = vb - va
            t = float(torch.dot(x - va, tv) / torch.dot(tv, tv))
            if 1e-8 < t < 1 - 1e-8 and (va + t * tv - x).norm() < 1e-8:
                j = round(t * order)
                assert abs(t * order - j) < 1e-6 and 1 <= j <= order - 1, (
                    f"{elem_cls.__name__} order-{order} edge node at t={t} does "
                    f"not sit on the uniform lattice"
                )
                hit = ("edge", local_edge, j)
                break
        if hit is not None:
            slots.append(hit)
        else:
            slots.append(("interior", n_interior, 0))
            n_interior += 1
    return slots, n_interior


def lagrange_dofmap(elements: Mapping[str, torch.Tensor],
                    n_points: int,
                    order: int) -> LagrangeDofMap:
    r"""Build the DOF map of an order-``order`` Lagrange space on a mesh.

    Only the corner-vertex columns of ``elements`` are read (corner nodes
    come first in every TensorMesh connectivity), so the mesh may be of
    any order — the field order is fully decoupled from it. DOFs are laid
    out as ``[vertices | edges | cell interiors]``:

    * one DOF per mesh vertex used by the cells;
    * ``order - 1`` DOFs per unique edge, **oriented** from the lower to
      the higher global vertex id, with the element-local slot flipped
      whenever the element traverses the edge backwards — this is what
      makes the space :math:`C^0`-conforming for ``order >= 3``;
    * per-cell interior DOFs (2D cells; 1D cells reuse the edge slots).

    Scope: any order for 1D/2D elements; in 3D only spaces whose nodes sit
    on vertices and edges (e.g. P2 tetrahedra) — face DOFs need the
    orientation layer planned with the H(div)/H(curl) work and raise
    ``NotImplementedError`` here.

    Parameters
    ----------
    elements : Mapping[str, torch.Tensor]
        Mesh connectivity per element type, ``[n_element, n_basis_mesh]``.
    n_points : int
        Number of mesh points (bounds the vertex ids).
    order : int
        Polynomial order of the Lagrange space, ``>= 1``.
    """
    per_etype = {}
    for element_type, value in elements.items():
        elem_cls = element_type2element(element_type)
        slots, n_interior = _classify_reference_nodes(elem_cls, order)
        if elem_cls.dim == 3 and n_interior > 0:
            raise NotImplementedError(
                f"order-{order} Lagrange DOFs on '{element_type}' include "
                f"face/interior nodes; 3D face-DOF orientation is not "
                f"implemented yet (see ROADMAP) — 3D supports vertex+edge "
                f"spaces such as P2 tetrahedra"
            )
        corners = value[:, :elem_cls.n_vertex].long()        # [E, n_vertex] mesh ids
        per_etype[element_type] = (elem_cls, slots, n_interior, corners)

    # vertex DOFs: compact renumbering of the used corner vertices
    vertex_ids = torch.unique(torch.cat([c.reshape(-1) for *_ , c in per_etype.values()]))
    g2l = torch.full((n_points,), -1, dtype=torch.long)
    g2l[vertex_ids] = torch.arange(vertex_ids.shape[0], dtype=torch.long)
    n_vertex_dofs = vertex_ids.shape[0]

    # unique (undirected) edges across all element types
    edge_ids = {}
    n_global_edges = 0
    if order >= 2:
        chunks, sizes = [], []
        for element_type, (elem_cls, _, _, corners) in per_etype.items():
            pairs = corners[:, elem_cls.edge]                # [E, n_le, 2] mesh ids
            chunks.append(torch.sort(pairs, dim=-1).values.reshape(-1, 2))
            sizes.append((element_type, pairs.shape[0], pairs.shape[1]))
        cat = torch.cat(chunks, dim=0)
        unique_edges, inverse = torch.unique(cat, dim=0, return_inverse=True)
        n_global_edges = unique_edges.shape[0]
        offset = 0
        for element_type, n_element, n_le in sizes:
            edge_ids[element_type] = inverse[offset:offset + n_element * n_le] \
                .reshape(n_element, n_le)
            offset += n_element * n_le

    edge_base = n_vertex_dofs
    interior_base = n_vertex_dofs + n_global_edges * (order - 1)

    conn, ref_nodes = {}, {}
    for element_type, (elem_cls, slots, n_interior, corners) in per_etype.items():
        n_element = corners.shape[0]
        out = torch.empty(n_element, len(slots), dtype=torch.long)
        for i, (kind, a, j) in enumerate(slots):
            if kind == "vertex":
                out[:, i] = g2l[corners[:, a]]
            elif kind == "edge":
                ga = corners[:, elem_cls.edge[a, 0]]
                gb = corners[:, elem_cls.edge[a, 1]]
                jj = torch.where(ga > gb,
                                 torch.tensor(order - j, dtype=torch.long),
                                 torch.tensor(j, dtype=torch.long))
                out[:, i] = edge_base + edge_ids[element_type][:, a] * (order - 1) + (jj - 1)
            else:  # interior: per-cell, orientation-free
                out[:, i] = interior_base + torch.arange(n_element, dtype=torch.long) \
                    * n_interior + a
        conn[element_type] = out
        ref_nodes[element_type] = elem_cls.get_basis(order, torch.float64)
        interior_base += n_element * n_interior

    return LagrangeDofMap(conn=conn, n_dofs=interior_base,
                          vertex_ids=vertex_ids, ref_nodes=ref_nodes)


def lagrange_boundary_mask(conn: Mapping[str, torch.Tensor],
                           order: int,
                           n_dofs: int) -> torch.Tensor:
    r"""Boolean mask of the boundary DOFs of a Lagrange space — topological.

    A facet is on the boundary iff it is referenced by exactly one cell
    (the standard incidence criterion — no geometric tolerance and no
    ``is_boundary`` point data needed). All DOFs sitting on a boundary
    facet are marked, including edge/face nodes of higher-order spaces.

    Parameters
    ----------
    conn : Mapping[str, torch.Tensor]
        Field-local connectivity per element type (``LagrangeDofMap.conn``,
        or a mesh connectivity for mesh-node-carried spaces).
    order : int
        Polynomial order of the space (selects the facet node table).
    n_dofs : int
        Size of the returned mask.
    """
    keys, facet_nodes, widths = [], [], set()
    for element_type, value in conn.items():
        elem_cls = element_type2element(element_type)
        facet_local = elem_cls.get_facet(order)
        if isinstance(facet_local, tuple):
            raise NotImplementedError(
                f"mixed facet types of '{element_type}' are not supported by "
                f"the topological boundary detection yet"
            )
        n_corner = int((facet_local < elem_cls.n_vertex).sum(dim=1)[0])
        assert ((facet_local < elem_cls.n_vertex).sum(dim=1) == n_corner).all()
        widths.add(n_corner)
        corner_dofs = value[:, facet_local[:, :n_corner]]      # [E, n_facet, w]
        keys.append(torch.sort(corner_dofs, dim=-1).values.reshape(-1, n_corner))
        facet_nodes.append(value[:, facet_local].reshape(-1, facet_local.shape[1]))
    if len(widths) != 1:
        raise NotImplementedError(
            "meshes mixing facet arities (e.g. tetrahedra + hexahedra) are "
            "not supported by the topological boundary detection yet"
        )
    keys = torch.cat(keys, dim=0)                              # [n_all_facets, w]
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True,
                                      return_counts=True)
    is_boundary_facet = counts[inverse] == 1                   # [n_all_facets]
    mask = torch.zeros(n_dofs, dtype=torch.bool)
    boundary_nodes = torch.cat(facet_nodes, dim=0)[is_boundary_facet].reshape(-1)
    mask[boundary_nodes] = True
    return mask

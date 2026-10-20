"""Outward facet-normal helpers for 2D / 3D convex reference elements.

Internal utilities used by :class:`~tensormesh.Element` to populate
:meth:`~tensormesh.Element.get_outwards_facet_normal`. Not part of the public
API.
"""
import torch
from typing import Tuple


def _vnorm(vec: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize each row of ``vec`` to unit length.

    Parameters
    ----------
    vec : torch.Tensor
        2D tensor of shape ``(n_points, dim)``.
    dim : int, optional
        Dimension along which to compute the norm. Defaults to ``-1``.

    Returns
    -------
    torch.Tensor
        Row-normalized tensor with the same shape as ``vec``.
    """
    norm = torch.norm(vec, dim=dim, keepdim=True)
    return vec / norm


def _vdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise dot product of two 2D tensors.

    Parameters
    ----------
    a, b : torch.Tensor
        2D tensors of shape ``(n_points, dim)``.

    Returns
    -------
    torch.Tensor
        1D tensor of shape ``(n_points,)`` with ``a[i] · b[i]`` in entry ``i``.
    """
    return torch.einsum("ij,ij->i", a, b)


def _vrot90(vec: torch.Tensor) -> torch.Tensor:
    """Rotate each 2D vector 90 degrees counter-clockwise.

    Parameters
    ----------
    vec : torch.Tensor
        2D tensor of shape ``(n_points, 2)``.

    Returns
    -------
    torch.Tensor
        Rotated tensor of shape ``(n_points, 2)``.
    """
    assert vec.shape[-1] == 2
    return torch.stack([-vec[:, 1], vec[:, 0]], dim=1)


def outwards_normal_2d(points: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Compute outward unit normals for the edges of a 2D convex polygon.

    Parameters
    ----------
    points : torch.Tensor
        2D tensor of shape ``(n_points, 2)`` with the polygon's vertex
        coordinates.
    edges : torch.Tensor
        2D tensor of shape ``(n_edges, 2)`` whose rows are pairs of indices
        into ``points`` describing each edge.

    Returns
    -------
    torch.Tensor
        2D tensor of shape ``(n_edges, 2)`` containing the unit outward
        normal of each edge.
    """
    edge_coords = points[edges] # shape (n_edges, 2, 2)
    edge_vec    = edge_coords[:, 1] - edge_coords[:, 0] # shape (n_edges, 2)
    inner_point = points.mean(dim=0, keepdim=True) # shape (1, 2) 
    inner_vec   = inner_point - edge_coords[:,0] # shape (n_edges, 2)
    normals     = _vrot90(edge_vec) # shape (n_edges, 2)
    is_inwards  = _vdot(normals, inner_vec) > 0 # shape (n_edges, )
    normals[is_inwards] *= -1
    return _vnorm(normals)

def outwards_normal_3d(points: torch.Tensor, faces: Tuple[Tuple[int, ...], ...]) -> torch.Tensor:
    """Compute outward unit normals for the faces of a 3D convex polyhedron.

    Parameters
    ----------
    points : torch.Tensor
        2D tensor of shape ``(n_points, dim)`` with the polyhedron's vertex
        coordinates.
    faces : Tuple[Tuple[int, ...], ...]
        Per-face tuples of vertex indices into ``points``. Each face may
        carry a different number of vertices; only the first three are used
        to compute the normal.

    Returns
    -------
    torch.Tensor
        2D tensor of shape ``(n_faces, dim)`` containing the unit outward
        normal of each face.
    """
    tri_faces    = torch.tensor([face[:3] for face in faces]) # [n_faces, 3]
    faces_coords = points[tri_faces, :] # [n_faces, 3, dim]
    vec1 = faces_coords[:, 1] - faces_coords[:, 0] # [n_faces, dim]
    vec2 = faces_coords[:, 2] - faces_coords[:, 0] # [n_faces, dim]
    normals = torch.linalg.cross(vec1, vec2, dim=-1) # [n_faces, dim]
    mid_point  = faces_coords.mean(dim=1) # [n_faces, dim]
    inner_point= points.mean(dim=0, keepdim=True) # [1, dim]
    inner_vec  = inner_point - mid_point # [n_faces, dim]
    is_inwards = _vdot(inner_vec, normals) > 0 # [n_faces]
    normals[is_inwards] *= -1
    return _vnorm(normals)



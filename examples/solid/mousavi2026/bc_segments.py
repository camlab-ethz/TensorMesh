"""Boundary segmentation and node assignment.

Assigns boundary nodes to segments based on their angular position
on the boundary curve. Supports multi-segment BCs with validation.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

from boundary import SegmentBCs, BCGenerator


def compute_boundary_angle_parameter(points: np.ndarray,
                                      boundary_indices: np.ndarray,
                                      center: Tuple[float, float] = (0., 0.)) -> np.ndarray:
    """Compute normalized angle parameter for boundary nodes.

    Maps each boundary node to a value in [0, 1) based on its angle
    from the center point. 0 corresponds to the positive x-axis direction.

    Args:
        points: [n_points, 2] mesh coordinates
        boundary_indices: indices of boundary nodes
        center: center point for angle computation

    Returns:
        angle_params: [n_boundary] values in [0, 1)
    """
    bpts = points[boundary_indices]
    dx = bpts[:, 0] - center[0]
    dy = bpts[:, 1] - center[1]
    angles = np.arctan2(dy, dx)  # in [-pi, pi]
    # Normalize to [0, 1)
    params = (angles + np.pi) / (2 * np.pi)
    # Wrap to [0, 1)
    params = params % 1.0
    return params


def assign_nodes_to_segments(angle_params: np.ndarray,
                              segments: List[SegmentBCs],
                              boundary_indices: np.ndarray) -> List[SegmentBCs]:
    """Assign boundary nodes to segments based on angle parameter.

    Each boundary node is assigned to exactly one segment.
    Modifies segments in-place by setting their `indices` attribute.

    Args:
        angle_params: [n_boundary] values in [0, 1)
        segments: list of SegmentBCs with center/radius defining parametric range
        boundary_indices: global indices of boundary nodes

    Returns:
        segments with indices filled in
    """
    assigned = np.zeros(len(angle_params), dtype=bool)

    for seg in segments:
        mask = seg.contains(angle_params)
        # Don't double-assign: remove already-assigned nodes
        mask = mask & ~assigned
        seg.indices = boundary_indices[mask]
        assigned |= mask

    # Any unassigned nodes go to the largest segment
    if not assigned.all():
        largest = max(segments, key=lambda s: s.radius)
        largest.indices = np.union1d(
            largest.indices, boundary_indices[~assigned]
        )

    return segments


def evaluate_bc_values(segments: List[SegmentBCs],
                        points: np.ndarray,
                        ndims: int = 1):
    """Evaluate BC function values at assigned boundary nodes.

    For each segment and dimension, evaluates the BC functions (g, alpha)
    at the node coordinates and stores the values in the DimensionBC.

    Args:
        segments: list of SegmentBCs with indices already assigned
        points: [n_points, 2] all mesh coordinates
        ndims: number of BC dimensions (1 for scalar, 2 for vector)
    """
    for seg in segments:
        if seg.indices is None or len(seg.indices) == 0:
            continue
        seg_points = points[seg.indices]
        for d in range(ndims):
            dim_bc = seg.dims[d]
            dim_bc.values = {}
            for name, func in dim_bc.functions.items():
                dim_bc.values[name] = func(seg_points)


def draw_valid_bcs(bc_generator: BCGenerator,
                    min_non_dirichlet_length: float = 0.0,
                    max_non_dirichlet_length: float = 1.0,
                    ndims: int = 1,
                    max_attempts: int = 1000) -> List[SegmentBCs]:
    """Draw random BCs with validation constraints.

    Retries until:
    - All segments have length >= 0.1
    - Total non-Dirichlet length is within bounds

    Args:
        bc_generator: BCGenerator instance
        min_non_dirichlet_length: minimum total non-Dirichlet boundary fraction
        max_non_dirichlet_length: maximum total non-Dirichlet boundary fraction
        ndims: number of BC dimensions
        max_attempts: maximum number of retry attempts

    Returns:
        Valid list of SegmentBCs
    """
    for _ in range(max_attempts):
        bcs = bc_generator.draw()

        # Check all segments are long enough
        all_long_enough = all(seg.radius * 2 > 0.1 for seg in bcs)
        if not all_long_enough:
            continue

        # Check non-Dirichlet length constraints (per dimension)
        valid = True
        for d in range(ndims):
            non_dir_length = sum(
                seg.radius * 2 for seg in bcs
                if seg.dims[d].type in ['robin', 'neumann']
            )
            if non_dir_length < min_non_dirichlet_length:
                valid = False
                break
            if non_dir_length >= max_non_dirichlet_length:
                valid = False
                break
        if valid:
            return bcs

    raise RuntimeError(f"Could not draw valid BCs after {max_attempts} attempts")


def build_bc_masks_and_values(mesh_points: torch.Tensor,
                               boundary_mask: torch.Tensor,
                               segments: List[SegmentBCs],
                               ndims: int = 1,
                               center: Tuple[float, float] = (0., 0.),
                               ) -> dict:
    """Build Dirichlet/Neumann/Robin masks and values from segment BCs.

    This is the main entry point for converting segment-based BCs into
    the format needed by the Poisson solver.

    Args:
        mesh_points: [n_points, 2] tensor
        boundary_mask: [n_points] boolean tensor
        segments: list of SegmentBCs from BCGenerator.draw()
        ndims: number of BC dimensions
        center: center for angle computation

    Returns:
        dict with:
            'dirichlet_mask': [n_points] bool tensor
            'dirichlet_values': [n_points] tensor
            'neumann_mask': [n_points] bool tensor
            'neumann_values': [n_points] tensor
            'robin_mask': [n_points] bool tensor
            'robin_g_values': [n_points] tensor
            'robin_alpha_values': [n_points] tensor
            'segments': list of segments with indices and values filled
    """
    n_points = mesh_points.shape[0]
    points_np = mesh_points.detach().cpu().numpy()
    boundary_indices = torch.where(boundary_mask)[0].cpu().numpy()

    # Compute angle parameters for boundary nodes
    angle_params = compute_boundary_angle_parameter(points_np, boundary_indices, center)

    # Assign nodes to segments
    assign_nodes_to_segments(angle_params, segments, boundary_indices)

    # Evaluate BC values at assigned nodes
    evaluate_bc_values(segments, points_np, ndims)

    # Build masks and values for dimension 0 (scalar Poisson)
    # For vector problems, this would need to be done per-component
    dtype = mesh_points.dtype
    device = mesh_points.device

    dirichlet_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    dirichlet_values = torch.zeros(n_points, dtype=dtype, device=device)
    neumann_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    neumann_values = torch.zeros(n_points, dtype=dtype, device=device)
    robin_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
    robin_g_values = torch.zeros(n_points, dtype=dtype, device=device)
    robin_alpha_values = torch.zeros(n_points, dtype=dtype, device=device)

    for seg in segments:
        if seg.indices is None or len(seg.indices) == 0:
            continue
        idx = torch.tensor(seg.indices, dtype=torch.long, device=device)
        dim_bc = seg.dims[0]  # dimension 0 for scalar

        if dim_bc.type == 'dirichlet':
            dirichlet_mask[idx] = True
            dirichlet_values[idx] = torch.tensor(
                dim_bc.values['g'], dtype=dtype, device=device)
        elif dim_bc.type == 'neumann':
            neumann_mask[idx] = True
            neumann_values[idx] = torch.tensor(
                dim_bc.values['g'], dtype=dtype, device=device)
        elif dim_bc.type == 'robin':
            robin_mask[idx] = True
            robin_g_values[idx] = torch.tensor(
                dim_bc.values['g'], dtype=dtype, device=device)
            robin_alpha_values[idx] = torch.tensor(
                dim_bc.values['alpha'], dtype=dtype, device=device)

    return {
        'dirichlet_mask': dirichlet_mask,
        'dirichlet_values': dirichlet_values,
        'neumann_mask': neumann_mask,
        'neumann_values': neumann_values,
        'robin_mask': robin_mask,
        'robin_g_values': robin_g_values,
        'robin_alpha_values': robin_alpha_values,
        'segments': segments,
    }

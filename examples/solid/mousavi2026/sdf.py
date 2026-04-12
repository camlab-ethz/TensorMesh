"""Signed distance function computation.

Computes SDF at mesh nodes and on a regular grid, plus SDF gradient.
Ported from fenicsx-main/src/domain.py:200-235.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Optional, Tuple


def compute_sdf_at_nodes(mesh_points: np.ndarray,
                          boundary_points: np.ndarray,
                          boundary_node_indices: np.ndarray,
                          hole_boundary_points: Optional[Dict[str, np.ndarray]] = None,
                          ) -> np.ndarray:
    """Compute signed distance function at mesh nodes.

    SDF = min distance to any boundary (outer + holes).
    Boundary nodes get SDF = 0.

    Args:
        mesh_points: [n_points, 2] all mesh coordinates
        boundary_points: [n_boundary, 2] outer boundary curve points (dense)
        boundary_node_indices: indices of all boundary mesh nodes
        hole_boundary_points: dict of {'name': [n, 2] hole boundary points}

    Returns:
        sdf: [n_points] signed distance (positive inside domain)
    """
    d_ext = _min_distance_to_curve(mesh_points, boundary_points)

    if hole_boundary_points:
        d_holes = [_min_distance_to_curve(mesh_points, hp)
                    for hp in hole_boundary_points.values()]
        sdf = np.min(np.stack([d_ext] + d_holes), axis=0)
    else:
        sdf = d_ext

    sdf[boundary_node_indices] = 0.0
    return sdf


def compute_sdf_on_grid(mesh_points: np.ndarray,
                         boundary_points: np.ndarray,
                         hole_boundary_points: Optional[Dict[str, np.ndarray]] = None,
                         bbox_resolution: int = 256,
                         ) -> Tuple[list, np.ndarray, np.ndarray]:
    """Compute SDF on a regular grid and its gradient.

    Args:
        mesh_points: [n_points, 2] for bounding box computation
        boundary_points: [n_boundary, 2] outer boundary curve (dense)
        hole_boundary_points: dict of hole boundary points
        bbox_resolution: grid resolution per axis

    Returns:
        bbox_grid_arrays: list of 1D arrays [x_coords, y_coords]
        sdf_bbox_grid: [256, 256] SDF values on grid
        sdf_gradient: [n_points, 2] SDF gradient at mesh nodes
    """
    # Define grid in bounding box
    bbox = np.stack([mesh_points.min(axis=0), mesh_points.max(axis=0)], axis=1)
    bbox_grid_arrays = [np.linspace(bbox[i, 0], bbox[i, 1], bbox_resolution) for i in range(2)]
    bbox_grid = np.meshgrid(*bbox_grid_arrays)
    bbox_grid_points = np.stack(bbox_grid, axis=-1).reshape(-1, 2)

    # Distance to boundaries
    d_ext = _min_distance_to_curve(bbox_grid_points, boundary_points)
    if hole_boundary_points:
        d_holes = [_min_distance_to_curve(bbox_grid_points, hp)
                    for hp in hole_boundary_points.values()]
        df = np.min(np.stack([d_ext] + d_holes), axis=0)
    else:
        df = d_ext

    # Signed distance: positive inside, negative outside
    inside = _is_inside_curve(bbox_grid_points, boundary_points)
    if hole_boundary_points:
        for hp in hole_boundary_points.values():
            in_hole = _is_inside_curve(bbox_grid_points, hp)
            inside = inside & ~in_hole

    sdf_bbox_grid = np.where(inside, df, -df)
    sdf_bbox_grid = sdf_bbox_grid.reshape(bbox_resolution, bbox_resolution)

    # Interpolate SDF gradient to mesh nodes
    sdf_grad_grid = np.stack(
        np.gradient(sdf_bbox_grid, *bbox_grid_arrays, edge_order=1), axis=-1)
    # sdf_grad_grid shape: [256, 256, 2]

    grad_interpolator = RegularGridInterpolator(
        bbox_grid_arrays, sdf_grad_grid.swapaxes(0, 1), method='linear',
        bounds_error=False, fill_value=0.0)
    sdf_gradient = grad_interpolator(mesh_points)[:, ::-1].copy()

    return bbox_grid_arrays, sdf_bbox_grid, sdf_gradient


def _min_distance_to_curve(points: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    """Compute minimum distance from each point to the nearest curve point.

    Uses chunked computation to avoid memory issues with large arrays.

    Args:
        points: [n_points, 2]
        curve_points: [n_curve, 2]

    Returns:
        distances: [n_points]
    """
    n = points.shape[0]
    chunk_size = 5000
    distances = np.empty(n)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # [chunk, 1, 2] - [1, n_curve, 2] -> [chunk, n_curve]
        diff = points[i:end, np.newaxis, :] - curve_points[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        distances[i:end] = d.min(axis=1)

    return distances


def _is_inside_curve(points: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    """Check if points are inside a closed curve using winding number.

    Uses ray casting algorithm (crossing number test).

    Args:
        points: [n_points, 2]
        curve_points: [n_curve, 2] closed curve (first != last, wraps)

    Returns:
        inside: [n_points] boolean
    """
    n = points.shape[0]
    nc = curve_points.shape[0]

    # Close the curve
    p1 = curve_points
    p2 = np.roll(curve_points, -1, axis=0)

    inside = np.zeros(n, dtype=bool)

    # Process in chunks
    chunk_size = 2000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        px = points[i:end, 0:1]  # [chunk, 1]
        py = points[i:end, 1:2]

        x1 = p1[:, 0:1].T  # [1, nc]
        y1 = p1[:, 1:2].T
        x2 = p2[:, 0:1].T
        y2 = p2[:, 1:2].T

        # Ray casting: count crossings of horizontal ray from (px, py) to the right
        cond1 = (y1 <= py) & (y2 > py)  # upward crossing
        cond2 = (y1 > py) & (y2 <= py)  # downward crossing
        crossing = cond1 | cond2

        # x-coordinate of intersection
        t = (py - y1) / (y2 - y1 + 1e-30)
        x_intersect = x1 + t * (x2 - x1)

        # Count crossings to the right of the test point
        right_crossing = crossing & (x_intersect > px)
        count = right_crossing.sum(axis=1)

        inside[i:end] = (count % 2) == 1

    return inside

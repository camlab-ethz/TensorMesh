"""Geometry definitions for dataset paper.

Defines boundary curves for all geometries: boomerang, circles, polygons,
and smooth-joint curves (for holes in hollow domains).
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Boomerang
# ---------------------------------------------------------------------------

def get_boomerang_boundary(n_points: int = 500) -> np.ndarray:
    """Generate boomerang boundary points (transformed unit circle).

    Ported from fenicsx-main/src/geometry/geometries.py:45-53.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    y = 0.5 * (2 * x) ** 2 + y
    y = 0.7 * y
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x = (x - x_min) / (x_max - x_min) * 2 - 1
    y = (y - y_min) / (y_max - y_min) * 2 - 1
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Basic shapes
# ---------------------------------------------------------------------------

def circle_boundary(cx: float, cy: float, r: float, n_points: int = 200) -> np.ndarray:
    """Circle boundary points (CCW)."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def polygon_boundary(vertices: List[Tuple[float, float]], n_points_per_edge: int = 50) -> np.ndarray:
    """Polygon boundary points from vertices (closed, CCW)."""
    verts = np.array(vertices)
    n_v = len(verts)
    points = []
    for i in range(n_v):
        p0 = verts[i]
        p1 = verts[(i + 1) % n_v]
        t = np.linspace(0, 1, n_points_per_edge, endpoint=False)
        edge_pts = p0[np.newaxis, :] + t[:, np.newaxis] * (p1 - p0)[np.newaxis, :]
        points.append(edge_pts)
    return np.concatenate(points, axis=0)


def transform_points(pts: np.ndarray,
                     scale: Tuple[float, float] = (1., 1.),
                     center: Tuple[float, float] = (0., 0.),
                     angle: float = 0.) -> np.ndarray:
    """Scale, translate, and rotate points around their centroid."""
    centroid = pts.mean(axis=0)
    # Scale around centroid
    p = (pts - centroid) * np.array(scale) + centroid
    # Rotate around centroid
    if angle != 0:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        p = (p - centroid) @ R.T + centroid
    # Move centroid to target center
    p = p - p.mean(axis=0) + np.array(center)
    return p


# ---------------------------------------------------------------------------
# SmoothJoint2DCurve approximation
# ---------------------------------------------------------------------------

def smooth_joint_curve(features: List[Tuple[Tuple[float, float], float, float]],
                       n_points: int = 200) -> np.ndarray:
    """Approximate a SmoothJoint2DCurve from the original code.

    Creates a smooth closed curve through the anchor points with rounded corners.
    Uses cubic Bezier interpolation to create smooth connections.

    Args:
        features: list of (center_xy, radius, angle) tuples
            - center_xy: anchor point (x, y)
            - radius: corner rounding radius
            - angle: not used in simplified version

    Returns:
        [n_points, 2] numpy array
    """
    anchors = np.array([f[0] for f in features])
    radii = np.array([f[1] for f in features])
    n_anchors = len(anchors)

    # Generate smooth closed curve using Catmull-Rom spline through anchors
    # Pad with wrap-around for closed curve
    padded = np.concatenate([anchors[-1:], anchors, anchors[:1]], axis=0)
    padded_r = np.concatenate([radii[-1:], radii, radii[:1]])

    segments = []
    pts_per_segment = max(n_points // n_anchors, 10)

    for i in range(n_anchors):
        p0 = padded[i]
        p1 = padded[i + 1]
        p2 = padded[i + 2]

        # Direction vectors
        d_in = p1 - p0
        d_out = p2 - p1
        d_in_len = max(np.linalg.norm(d_in), 1e-10)
        d_out_len = max(np.linalg.norm(d_out), 1e-10)

        # Bezier control points for smooth corner at p1
        r = padded_r[i + 1]
        cp_in = p1 - r * d_in / d_in_len
        cp_out = p1 + r * d_out / d_out_len

        t = np.linspace(0, 1, pts_per_segment, endpoint=False)[:, np.newaxis]

        # Quadratic Bezier: B(t) = (1-t)^2 * cp_in + 2(1-t)t * p1 + t^2 * cp_out
        curve_pts = (1 - t) ** 2 * cp_in + 2 * (1 - t) * t * p1 + t ** 2 * cp_out
        segments.append(curve_pts)

    pts = np.concatenate(segments, axis=0)

    # Resample to exact n_points
    if len(pts) != n_points:
        cumlen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        total_len = cumlen[-1]
        target_s = np.linspace(0, total_len, n_points, endpoint=False)
        pts = np.column_stack([
            np.interp(target_s, cumlen, pts[:, 0]),
            np.interp(target_s, cumlen, pts[:, 1]),
        ])

    return pts


# ---------------------------------------------------------------------------
# Hollow geometry hole definitions (from fenicsx-main/src/configs.py)
# ---------------------------------------------------------------------------

def get_circlehollow_holes(n_points: int = 200) -> dict:
    """Holes for CircleHollow geometry.
    Purple: SmoothJoint2DCurve, Brown: rotated rectangle
    """
    purple = smooth_joint_curve([
        ((0., 0.), 0.2, 0.), ((.5, -.5), 0.2, 0.), ((.7, -.3), 0.1, 0.),
        ((.4, .4), 0.2, 0.), ((.3, .7), 0.2, 0.), ((-.3, .7), 0.1, 0.),
        ((-.5, .2), 0.2, 0.),
    ], n_points=n_points)

    brown = polygon_boundary([(-1, -1), (1, -1), (1, 1), (-1, 1)], n_points_per_edge=n_points // 4)
    brown = transform_points(brown, scale=(0.1, 0.2), center=(-0.3, -0.3), angle=np.pi / 4)

    return {'purple': purple, 'brown': brown}


def get_squarehollow_holes(n_points: int = 200) -> dict:
    """Holes for SquareHollow geometry.
    Purple: SmoothJoint2DCurve, Brown: rotated boomerang
    """
    purple = smooth_joint_curve([
        ((0., 0.), 0.2, 0.), ((.5, -.5), 0.2, 0.), ((.7, -.3), 0.1, 0.),
        ((.6, .6), 0.3, 0.), ((.0, .5), 0.2, 0.), ((-.3, .7), 0.1, 0.),
        ((-.5, .2), 0.2, 0.),
    ], n_points=n_points)

    brown = get_boomerang_boundary(n_points)
    brown = transform_points(brown, scale=(0.2, 0.2), center=(-0.3, -0.3), angle=np.pi / 2)

    return {'purple': purple, 'brown': brown}


def get_boomcircletri_holes(n_points: int = 200) -> dict:
    """Holes for BoomCircleTri geometry.
    Purple: circle r=0.2 at (0, -0.6)
    Pink: circle r=0.2 at (-0.4, 0)
    Brown: triangle scaled 0.1 at (0.2, -0.2)
    """
    purple = circle_boundary(0., -0.6, 0.2, n_points)
    pink = circle_boundary(-0.4, 0., 0.2, n_points)

    tri_verts = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.73)]
    brown = polygon_boundary(tri_verts, n_points_per_edge=n_points // 3)
    brown = transform_points(brown, scale=(0.1, 0.1), center=(0.2, -0.2))

    return {'purple': purple, 'pink': pink, 'brown': brown}

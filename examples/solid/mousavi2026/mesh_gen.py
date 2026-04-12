"""Mesh generation for dataset paper geometries.

Phase 1: Simple geometries (circle, square) using TensorMesh built-in generators.
Phase 3: Boomerang geometry using Gmsh geo API with custom boundary points.
Phase 5+: Hollow geometries (circlehollow, squarehollow, boomcircletri).
"""

import os
import torch
import numpy as np
import gmsh

from tensormesh import Mesh
from geometry import (get_boomerang_boundary, circle_boundary, polygon_boundary,
                      get_circlehollow_holes, get_squarehollow_holes,
                      get_boomcircletri_holes)


CACHE_DIR = ".gmsh_cache"


def create_mesh(shape: str, order: int = 1) -> Mesh:
    """Create a mesh for the given shape name.

    Args:
        shape: one of 'circle', 'square', 'boomerang',
               'circlehollow', 'squarehollow', 'boomcircletri'
        order: polynomial order (1 for Poisson, 2 for elasticity)

    Returns:
        TensorMesh Mesh object with boundary_mask set
    """
    if shape == 'circle':
        return _create_circle_mesh(order=order)
    elif shape == 'square':
        return _create_square_mesh(order=order)
    elif shape == 'boomerang':
        return _create_boomerang_mesh(order=order)
    elif shape == 'circlehollow':
        return _create_circlehollow_mesh(order=order)
    elif shape == 'squarehollow':
        return _create_squarehollow_mesh(order=order)
    elif shape == 'boomcircletri':
        return _create_boomcircletri_mesh(order=order)
    else:
        raise NotImplementedError(f"Shape '{shape}' not yet implemented.")


def _create_circle_mesh(order: int = 1) -> Mesh:
    """Circle r=1 centered at origin.
    Original: points_per_unit_length=80, resolution=2/80=0.025
    """
    chara_length = 0.025
    mesh = Mesh.gen_circle(
        chara_length=chara_length,
        order=order,
        element_type="tri",
        cx=0.0, cy=0.0, r=1.0,
    )
    return mesh


def _create_square_mesh(order: int = 1) -> Mesh:
    """Square [-1,1]^2.
    Original: points_per_unit_length=70, resolution=2/70≈0.0286
    """
    chara_length = 2.0 / 70.0
    mesh = Mesh.gen_rectangle(
        chara_length=chara_length,
        order=order,
        element_type="tri",
        left=-1.0, right=1.0,
        bottom=-1.0, top=1.0,
    )
    return mesh


def _create_circlehollow_mesh(order: int = 2) -> Mesh:
    """Circle with 2 holes (SmoothJoint + rotated rectangle).
    Original: points_per_unit_length=60, resolution=2/60≈0.033
    """
    resolution = 2.0 / 60.0
    outer = circle_boundary(0., 0., 1.0, n_points=400)
    holes = get_circlehollow_holes(n_points=200)
    cache_path = os.path.join(CACHE_DIR, f"circlehollow_{resolution}_{order}.msh")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        _generate_mesh_from_boundary(outer, cache_path, resolution, order,
                                      holes_dict=holes)
    mesh = Mesh.from_file(cache_path, reorder=True)
    _add_boundary_mask_from_lines(mesh)
    return mesh


def _create_squarehollow_mesh(order: int = 2) -> Mesh:
    """Square with 2 holes (SmoothJoint + rotated boomerang).
    Original: points_per_unit_length=60, resolution=2/60≈0.033
    """
    resolution = 2.0 / 60.0
    outer = polygon_boundary([(-1, -1), (1, -1), (1, 1), (-1, 1)], n_points_per_edge=100)
    holes = get_squarehollow_holes(n_points=200)
    cache_path = os.path.join(CACHE_DIR, f"squarehollow_{resolution}_{order}.msh")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        _generate_mesh_from_boundary(outer, cache_path, resolution, order,
                                      holes_dict=holes)
    mesh = Mesh.from_file(cache_path, reorder=True)
    _add_boundary_mask_from_lines(mesh)
    return mesh


def _create_boomcircletri_mesh(order: int = 2) -> Mesh:
    """Boomerang with 3 holes (2 circles + triangle).
    Original: points_per_unit_length=80, resolution=2/80=0.025
    """
    resolution = 0.025
    outer = get_boomerang_boundary(n_points=500)
    holes = get_boomcircletri_holes(n_points=200)
    cache_path = os.path.join(CACHE_DIR, f"boomcircletri_{resolution}_{order}.msh")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(cache_path):
        _generate_mesh_from_boundary(outer, cache_path, resolution, order,
                                      holes_dict=holes)
    mesh = Mesh.from_file(cache_path, reorder=True)
    _add_boundary_mask_from_lines(mesh)
    return mesh


def _create_boomerang_mesh(order: int = 1) -> Mesh:
    """Boomerang shape using Gmsh geo API.
    Original: points_per_unit_length=80, resolution=2/80=0.025
    """
    resolution = 0.025
    n_boundary_points = 500

    cache_path = os.path.join(CACHE_DIR, f"boomerang_{resolution}_{order}.msh")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(cache_path):
        boundary_pts = get_boomerang_boundary(n_boundary_points)
        _generate_mesh_from_boundary(
            boundary_pts, cache_path, resolution, order,
            boundary_margin=0.4)

    mesh = Mesh.from_file(cache_path, reorder=True)

    # Detect boundary nodes from line elements
    _add_boundary_mask_from_lines(mesh)

    return mesh


def _generate_mesh_from_boundary(boundary_pts: np.ndarray,
                                  cache_path: str,
                                  resolution: float,
                                  order: int,
                                  holes_dict: dict = None,
                                  boundary_margin: float = 0.4,
                                  hole_margin: float = 0.3):
    """Generate a Gmsh mesh from boundary points and optional hole boundaries.

    Args:
        boundary_pts: [n, 2] closed boundary (endpoint not included)
        cache_path: path to save .msh file
        resolution: base mesh size
        order: element polynomial order
        holes_dict: optional dict of {'name': [n, 2] boundary points}
        boundary_margin: distance from outer boundary for refinement transition
        hole_margin: distance from holes for refinement transition
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("custom_geometry")

    # Create outer boundary: points → lines → curve loop
    outer_tags, outer_loop = _add_curve_from_points(boundary_pts, closed=True)

    # Create hole boundaries
    hole_loops = []
    all_hole_tags = []
    if holes_dict:
        for name, hole_pts in holes_dict.items():
            h_tags, h_loop = _add_curve_from_points(hole_pts, closed=True)
            hole_loops.append(h_loop)
            all_hole_tags.extend(h_tags)

    # Create surface (with holes subtracted)
    surface = gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops)
    gmsh.model.geo.synchronize()

    # Mesh refinement fields
    thresholds = []

    # Refinement near outer boundary
    dist_outer = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(dist_outer, 'CurvesList', outer_tags)
    thresh_outer = gmsh.model.mesh.field.add('Threshold')
    gmsh.model.mesh.field.setNumber(thresh_outer, 'IField', dist_outer)
    gmsh.model.mesh.field.setNumber(thresh_outer, 'DistMin', -0.01)
    gmsh.model.mesh.field.setNumber(thresh_outer, 'DistMax', boundary_margin)
    gmsh.model.mesh.field.setNumber(thresh_outer, 'SizeMin', resolution / 3)
    gmsh.model.mesh.field.setNumber(thresh_outer, 'SizeMax', resolution)
    thresholds.append(thresh_outer)

    # Refinement near holes
    if all_hole_tags:
        dist_holes = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(dist_holes, 'CurvesList', all_hole_tags)
        thresh_holes = gmsh.model.mesh.field.add('Threshold')
        gmsh.model.mesh.field.setNumber(thresh_holes, 'IField', dist_holes)
        gmsh.model.mesh.field.setNumber(thresh_holes, 'DistMin', -0.01)
        gmsh.model.mesh.field.setNumber(thresh_holes, 'DistMax', hole_margin)
        gmsh.model.mesh.field.setNumber(thresh_holes, 'SizeMin', resolution / 3)
        gmsh.model.mesh.field.setNumber(thresh_holes, 'SizeMax', resolution)
        thresholds.append(thresh_holes)

    # Combine refinement fields
    if len(thresholds) > 1:
        min_field = gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(min_field, 'FieldsList', thresholds)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(thresholds[0])

    # Mesh options
    gmsh.option.setNumber('Mesh.Algorithm', 5)  # Delaunay
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeMin', resolution / 10)
    gmsh.option.setNumber('Mesh.MeshSizeMax', resolution)
    gmsh.option.setNumber('Mesh.ElementOrder', order)

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, 1, "domain")

    # Boundary physical group (for line elements in output)
    boundary_lines = gmsh.model.getBoundary([(2, surface)], oriented=False)
    line_group = gmsh.model.addPhysicalGroup(1, [l[1] for l in boundary_lines])
    gmsh.model.setPhysicalName(1, line_group, "boundary")

    # Generate and save
    gmsh.model.mesh.generate(2)
    gmsh.write(cache_path)
    gmsh.finalize()


def _add_curve_from_points(pts: np.ndarray, closed: bool = True):
    """Add a curve from sequential points to the Gmsh model.

    Args:
        pts: [n, 2] points (endpoint NOT included for closed curves)
        closed: if True, close the curve back to the first point

    Returns:
        (line_tags, loop_tag): list of line tags and the curve loop tag
    """
    point_tags = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0)
                  for p in pts]

    if closed:
        # Close the loop by connecting last point back to first
        point_tags_loop = point_tags + [point_tags[0]]
    else:
        point_tags_loop = point_tags

    line_tags = [gmsh.model.geo.addLine(point_tags_loop[i], point_tags_loop[i + 1])
                 for i in range(len(point_tags_loop) - 1)]

    loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
    return line_tags, loop_tag


def _add_boundary_mask_from_lines(mesh: Mesh):
    """Detect boundary nodes from line elements and register as point_data.

    For meshes loaded from Gmsh with physical groups on boundaries,
    the 'line' cell type contains the boundary edges.
    """
    cell_keys = list(mesh.cells.keys())
    # Look for any line element type (line, line3, line4, ...)
    line_key = [k for k in cell_keys if k.startswith('line')]
    if line_key:
        line_nodes = mesh.cells[line_key[0]].flatten().unique()
        is_boundary = torch.zeros(mesh.points.shape[0], dtype=torch.bool)
        is_boundary[line_nodes] = True
        mesh.register_point_data("is_boundary", is_boundary)
    else:
        # Fallback: extract from triangle edges
        tri_key = [k for k in cell_keys if 'triangle' in k]
        if not tri_key:
            raise RuntimeError("Cannot detect boundary: no line or triangle elements")
        tris = mesh.cells[tri_key[0]]
        edges = torch.cat([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], dim=0)
        # Sort each edge so (i,j) with i<j
        edges_sorted = torch.sort(edges, dim=1)[0]
        # Find edges that appear exactly once (boundary edges)
        edge_keys = edges_sorted[:, 0] * mesh.points.shape[0] + edges_sorted[:, 1]
        unique_keys, counts = edge_keys.unique(return_counts=True)
        boundary_edge_keys = unique_keys[counts == 1]
        # Extract boundary nodes
        is_boundary_edge = torch.isin(edge_keys, boundary_edge_keys)
        boundary_nodes = edges[is_boundary_edge].flatten().unique()
        is_boundary = torch.zeros(mesh.points.shape[0], dtype=torch.bool)
        is_boundary[boundary_nodes] = True
        mesh.register_point_data("is_boundary", is_boundary)

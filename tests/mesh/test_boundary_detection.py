"""Boundary detection — :meth:`tensormesh.Mesh.topological_boundary_mask`,
the ``boundary_mask`` fallback, and the masks the ``gen_*`` generators
register.

The generators used to build ``is_boundary`` by exact coordinate
comparison (``x == 0``, ``r == R``); on curved geometry gmsh places nodes
one ulp off the curve, so nodes were missed and their Dirichlet condition
silently dropped. The hollow generators additionally crashed outright
(``Unknown model face``) because ``occ.cut`` consumes its inputs.
"""
import meshio
import numpy as np
import pytest
import torch

from tensormesh import Mesh


# ------------------------------------------------------------------ #
# topological mask on hand-built meshes
# ------------------------------------------------------------------ #
def test_topological_mask_hand_built_fan():
    # unit square fanned around a centre node: 4 boundary nodes, 1 interior
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]], dtype=float)
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    mesh = Mesh(meshio.Mesh(points=pts, cells=[("triangle", cells)]))
    mask = mesh.topological_boundary_mask()
    assert mask.tolist() == [True, True, True, True, False]


def test_boundary_mask_falls_back_to_topology_and_caches():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]], dtype=float)
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    mesh = Mesh(meshio.Mesh(points=pts, cells=[("triangle", cells)]))
    assert "is_boundary" not in mesh.point_data
    mask = mesh.boundary_mask                       # used to raise
    assert mask.tolist() == [True, True, True, True, False]
    assert "is_boundary" in mesh.point_data         # cached for the next access
    assert mesh.boundary_mask is mesh.point_data["is_boundary"]


def test_topological_mask_includes_higher_order_edge_nodes():
    mesh = Mesh.gen_rectangle(chara_length=0.25, element_type="tri", order=2)
    mask = mesh.topological_boundary_mask()
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    geometric = (x == 0) | (x == 1) | (y == 0) | (y == 1)   # exact on a straight-sided square
    assert torch.equal(mask, geometric)


# ------------------------------------------------------------------ #
# generators: every boundary node is marked (curved geometry)
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("order", [1, 2])
def test_gen_circle_marks_every_boundary_node(order):
    mesh = Mesh.gen_circle(chara_length=0.06, cx=.5, cy=.5, r=.5, order=order)
    radius = ((mesh.points - 0.5) ** 2).sum(1).sqrt()
    on_circle = (radius - 0.5).abs() < 1e-8
    assert torch.equal(mesh.boundary_mask, on_circle)
    assert torch.equal(mesh.boundary_mask, mesh.topological_boundary_mask())
    # the old exact test missed nodes: make sure the fixture still exercises that
    assert (radius[on_circle] != 0.5).any()


def test_gen_sphere_marks_every_boundary_node():
    mesh = Mesh.gen_sphere(chara_length=0.4)
    radius = (mesh.points ** 2).sum(1).sqrt()
    assert torch.equal(mesh.boundary_mask, (radius - 1.0).abs() < 1e-8)


# ------------------------------------------------------------------ #
# hollow generators: no crash, inner/outer partition, nothing inside the hole
# ------------------------------------------------------------------ #
def _inside_hole_rectangle(mesh, tol=1e-9):
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    return (x > .25 + tol) & (x < .75 - tol) & (y > .25 + tol) & (y < .75 - tol)


def _inside_hole_cube(mesh, tol=1e-9):
    x, y, z = mesh.points.T
    return ((x > .25 + tol) & (x < .75 - tol) & (y > .25 + tol) & (y < .75 - tol)
            & (z > .25 + tol) & (z < .75 - tol))


def _inside_hole_radial(mesh, r_inner, tol=1e-9):
    return (mesh.points ** 2).sum(1).sqrt() < r_inner - tol


HOLLOW = [
    pytest.param(lambda: Mesh.gen_hollow_rectangle(chara_length=0.08, element_type="tri"),
                 _inside_hole_rectangle, "triangle", id="rect-tri"),
    pytest.param(lambda: Mesh.gen_hollow_rectangle(chara_length=0.08, element_type="quad"),
                 _inside_hole_rectangle, "quad", id="rect-quad"),
    pytest.param(lambda: Mesh.gen_hollow_rectangle(chara_length=0.08, element_type="tri", order=2),
                 _inside_hole_rectangle, "triangle6", id="rect-tri-o2"),
    pytest.param(lambda: Mesh.gen_hollow_circle(chara_length=0.15, element_type="tri"),
                 lambda m: _inside_hole_radial(m, 1.0), "triangle", id="circle-tri"),
    pytest.param(lambda: Mesh.gen_hollow_circle(chara_length=0.15, element_type="quad"),
                 lambda m: _inside_hole_radial(m, 1.0), "quad", id="circle-quad"),
    pytest.param(lambda: Mesh.gen_hollow_circle(chara_length=0.15, element_type="tri", order=2),
                 lambda m: _inside_hole_radial(m, 1.0), "triangle6", id="circle-tri-o2"),
    pytest.param(lambda: Mesh.gen_hollow_cube(chara_length=0.2),
                 _inside_hole_cube, "tetra", id="cube"),
    pytest.param(lambda: Mesh.gen_hollow_sphere(chara_length=0.4),
                 lambda m: _inside_hole_radial(m, 1.0), "tetra", id="sphere"),
]


@pytest.mark.parametrize("make, inside_hole, element_type", HOLLOW)
def test_hollow_generators(make, inside_hole, element_type):
    mesh = make()                                   # used to raise "Unknown model face"
    assert mesh.default_element_type == element_type
    pd = mesh.point_data
    inner, outer, boundary = pd["is_inner_boundary"], pd["is_outer_boundary"], pd["is_boundary"]
    assert torch.equal(boundary, mesh.topological_boundary_mask())
    assert inner.any() and outer.any()
    assert torch.equal(inner | outer, boundary)
    assert not (inner & outer).any()
    assert not inside_hole(mesh).any()              # the hole is really cut out
    for key in pd.keys():                           # every side mask lives on the boundary
        if key.endswith("_boundary"):
            assert not (pd[key] & ~boundary).any(), key


def test_hollow_rectangle_inner_sides_are_confined_to_the_hole():
    mesh = Mesh.gen_hollow_rectangle(chara_length=0.05, element_type="tri")
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    left = mesh.point_data["is_inner_left_boundary"]
    assert left.any()
    assert ((x[left] - 0.25).abs() < 1e-8).all()
    assert (y[left] >= 0.25 - 1e-8).all() and (y[left] <= 0.75 + 1e-8).all()
    # the outer sides are what they were
    assert ((x[mesh.point_data["is_outer_left_boundary"]]).abs() < 1e-8).all()


def test_hollow_cube_default_cache_depends_on_parameters():
    # regression: the default cache path was a fixed ".gmsh_cache/tmp.msh",
    # so every call after the first returned the first mesh
    coarse = Mesh.gen_hollow_cube(chara_length=0.4)
    fine = Mesh.gen_hollow_cube(chara_length=0.2)
    assert fine.n_points > coarse.n_points


def test_gen_L_reentrant_sides():
    mesh = Mesh.gen_L(chara_length=0.1, element_type="tri")
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    top = mesh.point_data["is_L_top_boundary"]
    right = mesh.point_data["is_L_right_boundary"]
    assert top.any() and right.any()
    assert (x[top] >= 0.5 - 1e-8).all() and ((y[top] - 0.5).abs() < 1e-8).all()
    assert (y[right] >= 0.5 - 1e-8).all() and ((x[right] - 0.5).abs() < 1e-8).all()
    assert torch.equal(mesh.boundary_mask, mesh.topological_boundary_mask())

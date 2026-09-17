"""Node-ordering validation — :meth:`tensormesh.Mesh.check_node_ordering`.

A Gmsh/VTK mesh of order >= 2 loaded without ``reorder=True`` keeps a
valid-looking connectivity whose edge nodes are attached to the wrong
edges: every cell is distorted, every downstream result is wrong, and
nothing raises. The constructor now detects this (each edge node must be
nearest to the chord position of *its own* edge) and raises ``ValueError``.
The hand-built meshes keep most of this file gmsh-free; the curved and 3D
cases use the generators.
"""
import os
import tempfile

import meshio
import numpy as np
import pytest
import torch

from tensormesh import Mesh, Triangle


def _square_triangle6(offset=(0.0, 0.0)):
    """Unit square split along the diagonal, P2 nodes, TensorMesh ordering.

    Triangle slots 3, 4, 5 are the midpoints of edges (1,2), (0,2), (0,1).
    """
    ox, oy = offset
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                    [.5, .5], [0, .5], [.5, 0], [1, .5], [.5, 1]], dtype=float)
    pts += np.array([ox, oy])
    cells = np.array([[0, 1, 2, 4, 5, 6],      # v=(0,1,2): mid(1,2)=4 mid(0,2)=5 mid(0,1)=6
                      [1, 3, 2, 8, 4, 7]])     # v=(1,3,2): mid(3,2)=8 mid(1,2)=4 mid(1,3)=7
    return pts, cells


def test_correct_ordering_passes():
    pts, cells = _square_triangle6()
    mesh = Mesh(meshio.Mesh(points=pts, cells=[("triangle6", cells)]))
    assert mesh.check_node_ordering() == {"triangle6": 0}


def test_gmsh_ordering_raises_and_reorder_fixes_it():
    pts, cells = _square_triangle6()
    gmsh_cells = Triangle.reorder(torch.from_numpy(cells), to_gmsh=True).numpy()
    assert not np.array_equal(gmsh_cells, cells)      # the permutation is not the identity
    raw = meshio.Mesh(points=pts, cells=[("triangle6", gmsh_cells)])

    with pytest.raises(ValueError, match="reorder=True"):
        Mesh(raw, reorder=False)

    mesh = Mesh(raw, reorder=True)
    assert torch.equal(mesh.cells["triangle6"], torch.from_numpy(cells))
    assert mesh.check_node_ordering() == {"triangle6": 0}


def test_linear_cells_are_not_checked():
    mesh = Mesh.gen_rectangle(chara_length=0.5, element_type="tri")
    assert mesh.check_node_ordering() == {}


def test_partial_violation_warns_instead_of_raising():
    # four cells, one corrupted: a distorted-cell diagnosis, not a permutation
    pts_a, cells_a = _square_triangle6()
    pts_b, cells_b = _square_triangle6(offset=(2.0, 0.0))
    pts = np.concatenate([pts_a, pts_b])
    cells = np.concatenate([cells_a, cells_b + len(pts_a)])
    cells[0, [3, 4]] = cells[0, [4, 3]]                 # swap two edge nodes of one cell
    raw = meshio.Mesh(points=pts, cells=[("triangle6", cells)])
    with pytest.warns(RuntimeWarning, match="1/4"):
        mesh = Mesh(raw)
    assert mesh.check_node_ordering(raise_on_error=False) == {"triangle6": 1}


@pytest.mark.parametrize("make", [
    pytest.param(lambda: Mesh.gen_rectangle(chara_length=0.25, element_type="quad", order=2), id="quad9"),
    pytest.param(lambda: Mesh.gen_circle(chara_length=0.3, cx=.5, cy=.5, r=.5, order=2), id="triangle6-curved"),
    pytest.param(lambda: Mesh.gen_cube(chara_length=0.5, order=2), id="tetra10"),
])
def test_generator_meshes_roundtrip_through_gmsh_ordering(make):
    mesh = make()
    exported = mesh.to_meshio(reorder=True)             # Gmsh/VTK ordering on disk
    with pytest.raises(ValueError, match="does not follow TensorMesh"):
        Mesh(exported, reorder=False)
    back = Mesh(exported, reorder=True)
    assert back.check_node_ordering() == {mesh.default_element_type: 0}
    assert torch.equal(back.cells[mesh.default_element_type], mesh.cells[mesh.default_element_type])


def test_gen_sphere_order2_is_reordered():
    # regression: gen_sphere loaded its .msh without reorder=True, so every
    # order-2 tetrahedron had its edge nodes on the wrong edges
    mesh = Mesh.gen_sphere(chara_length=0.5, order=2)
    assert mesh.default_element_type == "tetra10"
    assert mesh.check_node_ordering() == {"tetra10": 0}


def test_read_vtu_without_reorder_raises():
    # the exact scenario reported from an external Stokes pipeline: an
    # order-2 .vtu read back with reorder=False solved fine and was wrong
    mesh = Mesh.gen_circle(chara_length=0.3, cx=.5, cy=.5, r=.5, order=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "circle2.vtu")
        mesh.save(path)
        with pytest.raises(ValueError, match="reorder=True"):
            Mesh.read(path, reorder=False)
        again = Mesh.read(path, reorder=True)
    assert again.check_node_ordering() == {"triangle6": 0}

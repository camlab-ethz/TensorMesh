"""MeshGen CSG regression tests (CPU, gmsh required)."""
import signal

import pytest
import torch

from tensormesh import MeshGen


class _Deadline:
    """Fail the test if the block runs longer than ``seconds``.

    The issue-40 regression is a *hang* inside ``gmsh.model.mesh.generate``,
    so a plain assertion could never fire; SIGALRM turns the hang into a
    test failure instead of stalling the whole CI job.
    """

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        signal.alarm(0)

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("mesh generation exceeded the deadline (hang regression)")


def test_remove_sphere_terminates_and_meshes():
    """Issue #40: cube minus a small sphere used to hang in generate().

    Root cause: ``Mesh.MeshSizeFromCurvature`` was set to 1 (it is a
    target element count per 2*pi of curvature, not a boolean), which on
    a sphere smaller than the requested size field — combined with
    ``MeshSizeExtendFromBoundary=0`` — never converged.
    """
    gen = MeshGen(dimension=3, chara_length=1.0 / 10)
    gen.add_cube(0.0, 0.0, 0.0, 1.0, 2.0, 1.0)
    gen.remove_sphere(0.1, 0.2, 0.3, 0.05)

    with _Deadline(120):
        mesh = gen.gen().double()

    points = mesh.points
    n_points = points.shape[0]
    assert n_points > 500, f"suspiciously coarse mesh: {n_points} points"

    # The spherical cavity must actually be cut out: no mesh point may
    # lie strictly inside the removed sphere.
    center = torch.tensor([0.1, 0.2, 0.3], dtype=points.dtype)
    dist = torch.linalg.norm(points - center, dim=1)
    assert (dist > 0.05 - 1e-9).all(), "points found inside the removed sphere"

    # Points must exist ON the cavity surface (the hole is meshed, not lost).
    assert (dist < 0.05 + 1e-3).any(), "no mesh points on the sphere surface"


def test_remove_cube_still_fast_and_bounded():
    """The companion path from the same report stays healthy."""
    gen = MeshGen(dimension=3, chara_length=1.0 / 10)
    gen.add_cube(0.0, 0.0, 0.0, 1.0, 2.0, 1.0)
    gen.remove_cube(0.1, 0.2, 0.3, 0.05, 0.05, 0.05)

    with _Deadline(120):
        mesh = gen.gen().double()

    assert mesh.points.shape[0] > 500

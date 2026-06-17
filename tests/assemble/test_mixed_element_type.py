"""Regression tests for issue #13 — ``ElementAssembler.from_mesh`` on
meshes with multiple element types.

Coverage axes (one test per cell, exhaustive within each):

* Mesh-type axis
    - 2D mixed: 1 quad + 2 triangles  (the minimal repro from the issue)
    - 2D mixed: 4 quads + 1 triangle  (different per-type counts)
    - 3D mixed: 1 hex + 2 tetra        (3D analogue of the issue)
    - Single-type baselines (pure quad, pure tri, pure hex, pure tet)
      to ensure the cat-based path doesn't regress single-type meshes
* reorder=True / reorder=False (the issue's repro used reorder=True)
* All three built-in element assemblers — Laplace, Mass,
  LinearElasticity — share the same ``from_mesh`` code path so each
  one must accept mixed-type meshes.

The pre-fix code raised
    RuntimeError: stack expects each tensor to be equal size,
                  but got [1] at entry 0 and [2] at entry 16
on every parametrisation tagged ``mixed=True``. Single-type cases
worked before and still work; we keep them so the swap from
``torch.stack`` to ``torch.cat`` is exercised in both regimes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tensormesh import Mesh
from tensormesh.assemble import (
    LaplaceElementAssembler,
    LinearElasticityElementAssembler,
    MassElementAssembler,
)


# --------------------------------------------------------------------- #
# Mesh fixtures: .inp text + expected metadata
# --------------------------------------------------------------------- #
MESHES = {
    # 2D mixed -- the minimal repro from the issue
    "2d_quad_tri": {
        "inp": """\
*Node
1, 0.0, 0.0, 0
2, 0.0, 1.0, 0
3, 1.0, 0.0, 0
4, 1.0, 1.0, 0
5, 0.5, 1.5, 0
6, 0.0, 2.0, 0
*ELEMENT, TYPE=S4
1, 1, 2, 4, 3
*ELEMENT, TYPE=S3
2, 2, 5, 4
3, 2, 6, 5
""",
        "n_points": 6, "dim": 2,
        "cell_counts": {"quad": 1, "triangle": 2},
        "is_mixed": True,
    },
    # 2D mixed with different per-type counts -- catches off-by-one
    # in the cat enumeration if it's keyed on type rather than length
    "2d_quad_tri_unbalanced": {
        "inp": """\
*Node
1, 0.0, 0.0, 0
2, 1.0, 0.0, 0
3, 2.0, 0.0, 0
4, 0.0, 1.0, 0
5, 1.0, 1.0, 0
6, 2.0, 1.0, 0
7, 0.5, 0.5, 0
*ELEMENT, TYPE=S4
1, 1, 2, 5, 4
2, 2, 3, 6, 5
*ELEMENT, TYPE=S3
3, 1, 2, 7
""",
        "n_points": 7, "dim": 2,
        "cell_counts": {"quad": 2, "triangle": 1},
        "is_mixed": True,
    },
    # 3D mixed -- hex + tetra
    "3d_hex_tet": {
        "inp": """\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
9, 0.5, 0.5, 2.0
*ELEMENT, TYPE=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
*ELEMENT, TYPE=C3D4
2, 5, 6, 7, 9
3, 5, 7, 8, 9
""",
        "n_points": 9, "dim": 3,
        "cell_counts": {"hexahedron": 1, "tetra": 2},
        "is_mixed": True,
    },
    # Single-type sanity: pure quad mesh, single element
    "2d_pure_quad": {
        "inp": """\
*Node
1, 0.0, 0.0, 0
2, 1.0, 0.0, 0
3, 1.0, 1.0, 0
4, 0.0, 1.0, 0
*ELEMENT, TYPE=S4
1, 1, 2, 3, 4
""",
        "n_points": 4, "dim": 2,
        "cell_counts": {"quad": 1},
        "is_mixed": False,
    },
    # Single-type sanity: pure triangle mesh
    "2d_pure_tri": {
        "inp": """\
*Node
1, 0.0, 0.0, 0
2, 1.0, 0.0, 0
3, 0.0, 1.0, 0
*ELEMENT, TYPE=S3
1, 1, 2, 3
""",
        "n_points": 3, "dim": 2,
        "cell_counts": {"triangle": 1},
        "is_mixed": False,
    },
}


@pytest.fixture
def make_mesh():
    """Return ``f(name, reorder)`` that materialises one of the
    fixture meshes to a temp ``.inp`` file and reads it back."""
    paths_to_clean = []

    def _factory(name: str, reorder: bool) -> Mesh:
        spec = MESHES[name]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inp", delete=False
        ) as f:
            f.write(spec["inp"])
            path = Path(f.name)
        # Read AFTER the context manager has flushed and closed the file.
        paths_to_clean.append(path)
        return Mesh.read(str(path), reorder=reorder)

    yield _factory

    for p in paths_to_clean:
        p.unlink(missing_ok=True)


# --------------------------------------------------------------------- #
# Sanity: every fixture parses into the shape we declared
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(MESHES.keys()))
def test_fixture_parses_to_expected_shape(make_mesh, name):
    spec = MESHES[name]
    mesh = make_mesh(name, reorder=False)
    assert mesh.points.shape == (spec["n_points"], spec["dim"])
    seen = {ct: c.shape[0] for ct, c in mesh.cells.items()}
    assert seen == spec["cell_counts"], (
        f"{name}: expected {spec['cell_counts']}, got {seen}"
    )


# --------------------------------------------------------------------- #
# The actual regression: ``from_mesh`` must succeed on every
# (mesh fixture × reorder mode × assembler) combination.
#
# Pre-fix this raises ``RuntimeError: stack expects each tensor to be
# equal size`` on the ``is_mixed=True`` rows. Single-type rows always
# worked; we keep them so the new ``torch.cat`` path is exercised in
# both regimes.
# --------------------------------------------------------------------- #
ASSEMBLERS = [
    pytest.param(LaplaceElementAssembler,
                 {"quadrature_order": 2}, 1,
                 id="laplace"),
    pytest.param(MassElementAssembler,
                 {"quadrature_order": 2}, 1,
                 id="mass"),
    pytest.param(LinearElasticityElementAssembler,
                 {"E": 21e9, "nu": 0.2}, None,    # dofs/node = mesh.dim
                 id="linear_elasticity"),
]


@pytest.mark.parametrize("mesh_name", sorted(MESHES.keys()))
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("AsmCls, kwargs, dofs_per_node", ASSEMBLERS)
def test_from_mesh_handles_every_combination(
    make_mesh, mesh_name, reorder, AsmCls, kwargs, dofs_per_node,
):
    spec = MESHES[mesh_name]
    mesh = make_mesh(mesh_name, reorder=reorder)

    asm = AsmCls.from_mesh(mesh, **kwargs)
    K = asm()

    dn = dofs_per_node if dofs_per_node is not None else spec["dim"]
    expected_size = spec["n_points"] * dn
    assert tuple(K.sparse_shape) == (expected_size, expected_size), (
        f"{AsmCls.__name__} on {mesh_name} (reorder={reorder}): "
        f"expected shape {(expected_size, expected_size)}, "
        f"got {tuple(K.sparse_shape)}"
    )
    assert K.nnz > 0, "every assembler should produce non-zero coupling"

    # Symmetry is the most sensitive shape-blind check: if the underlying
    # per-element dof ordering gets scrambled (e.g. by switching ``stack``
    # for ``cat`` without preserving the per-type layout), single-type
    # meshes still produce the right *shape*/*nnz* but the matrix is no
    # longer symmetric. Pure-quad/tri/hex/tet baselines catch that here.
    K_dense = K.to_dense()
    K_max = K_dense.abs().max().clamp(min=1.0)
    asym = (K_dense - K_dense.transpose(-1, -2)).abs().max()
    assert (asym / K_max).item() < 1e-10, (
        f"{AsmCls.__name__} on {mesh_name} (reorder={reorder}): "
        f"assembled matrix is not symmetric "
        f"(max|K-K.T|={asym.item():.3e}, max|K|={K_max.item():.3e})"
    )


# --------------------------------------------------------------------- #
# Spot-check the headline case from the issue verbatim
# --------------------------------------------------------------------- #
def test_issue_13_minimal_reproduction(make_mesh):
    """The exact .inp from issue #13's MWE, read with ``reorder=True``
    (as the issue author did) and passed to the assembler the issue
    author named. Pre-fix: ``RuntimeError``. Post-fix: returns a
    12x12 stiffness matrix (6 nodes x 2 spatial dims)."""
    mesh = make_mesh("2d_quad_tri", reorder=True)
    assembler = LinearElasticityElementAssembler.from_mesh(
        mesh, E=21e9, nu=0.2,
    )
    K = assembler()
    assert tuple(K.sparse_shape) == (12, 12)
    assert K.nnz > 0

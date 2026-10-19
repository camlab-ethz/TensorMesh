"""Caching of geometric quantities in Transformation: reuse across assembly
calls, invalidation when the points change, and autograd through the points."""

import meshio
import numpy as np
import pytest
import torch

from tensormesh import ElementAssembler, FacetAssembler, Mesh, NodeAssembler
from tensormesh.element.element import small_inverse


def _square(n=4):
    """Unit square, n x n cells split into triangles, float64."""
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nid = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = nid[:-1, :-1].ravel(), nid[1:, :-1].ravel(), nid[1:, 1:].ravel(), nid[:-1, 1:].ravel()
    cells = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return Mesh(meshio.Mesh(np.stack([X.ravel(), Y.ravel()], 1), [("triangle", cells)]))


def _with_points(mesh, points):
    return Mesh(meshio.Mesh(points.detach().numpy(), [("triangle", mesh.cells["triangle"].numpy())]))


def _moved(points):
    """Stretch and shear: area x 2, and every element changes shape."""
    return points @ torch.tensor([[2.0, 0.3], [0.0, 1.0]], dtype=points.dtype)


def _count_shape_grad_evaluations(monkeypatch, assembler):
    element = assembler.transformation["triangle"].element
    original, calls = element.eval_shape_grad, []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(element, "eval_shape_grad", counting)
    return calls


class Laplace(ElementAssembler):
    def forward(self, gradu, gradv):
        return (gradu * gradv).sum()


class Ones(NodeAssembler):
    def forward(self, v):
        return v


class Divergence(NodeAssembler):
    def forward(self, gradv, x):
        return (gradv * x).sum()


class BoundaryOnes(FacetAssembler):
    def forward(self, v):
        return v


def test_assembly_reuses_the_cached_geometry(monkeypatch):
    asm = Laplace.from_mesh(_square())
    calls = _count_shape_grad_evaluations(monkeypatch, asm)
    first, second = asm(), asm()
    assert len(calls) == 1
    assert torch.equal(first.values, second.values)


def test_chunked_assembly_keeps_nothing_and_agrees_with_full():
    asm = Laplace.from_mesh(_square(), quadrature_order=4)
    trans = asm.transformation["triangle"]
    cold = asm(batch_size=1).to_dense()
    assert "_shape_grad" not in trans._buffers  # chunks stay within their memory budget
    full = asm().to_dense()
    warm = asm(batch_size=1).to_dense()  # chunks served from the cache
    assert torch.allclose(cold, full, rtol=1e-13, atol=1e-13)
    assert torch.allclose(warm, full, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("fill", ["assembly", "energy"])
def test_element_assembler_follows_new_points(fill):
    mesh = _square()
    moved = _moved(mesh.points)
    asm = Laplace.from_mesh(mesh)
    asm()
    if fill == "energy":
        asm.energy(func=lambda gradu: (gradu * gradu).sum(), point_data={"u": torch.zeros_like(mesh.points[:, 0])})
    expected = Laplace.from_mesh(_with_points(mesh, moved))().to_dense()
    assert torch.allclose(asm(points=moved).to_dense(), expected, rtol=1e-12, atol=1e-12)
    original = Laplace.from_mesh(mesh)().to_dense()
    assert torch.allclose(asm(points=mesh.points).to_dense(), original, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("batch_size", [-1, 1])
def test_node_assembler_follows_new_points(batch_size):
    mesh = _square()
    moved = _moved(mesh.points)
    asm = Divergence.from_mesh(mesh)
    asm(batch_size=-1)  # fills the cache with the original geometry
    expected = Divergence.from_mesh(_with_points(mesh, moved))(batch_size=-1)
    assert torch.allclose(asm(points=moved, batch_size=batch_size), expected, rtol=1e-12, atol=1e-14)


def test_compiled_node_assembler_follows_new_points():
    mesh = _square()
    moved = _moved(mesh.points)
    asm = Ones.from_mesh(mesh).compile()
    before = asm()
    after = asm(points=moved)
    expected = Ones.from_mesh(_with_points(mesh, moved))()
    assert torch.allclose(after, expected, rtol=1e-12, atol=1e-14)
    assert float(after.sum()) == pytest.approx(2 * float(before.sum()))  # the area doubled


def test_facet_assembler_follows_new_points():
    mesh = _square()
    moved = _moved(mesh.points)
    asm = BoundaryOnes.from_mesh(mesh)
    asm()  # facet geometry is cached on first use
    expected = BoundaryOnes.from_mesh(_with_points(mesh, moved))()
    assert torch.allclose(asm(points=moved), expected, rtol=1e-12, atol=1e-14)


def test_unchanged_points_keep_the_cache_and_in_place_edits_drop_it(monkeypatch):
    mesh = _square()
    points = mesh.points.clone()
    asm = Laplace.from_mesh(mesh)
    calls = _count_shape_grad_evaluations(monkeypatch, asm)
    asm(points=points)
    asm(points=points)
    assert len(calls) == 1
    points.copy_(_moved(points))
    K = asm(points=points)
    assert len(calls) == 2
    expected = Laplace.from_mesh(_with_points(mesh, points))().to_dense()
    assert torch.allclose(K.to_dense(), expected, rtol=1e-12, atol=1e-12)


def test_gradients_through_points_get_a_fresh_graph_every_call():
    """With points that require grad nothing geometric is served from the cache,
    so repeated forward/backward passes neither fail on a freed graph nor
    return gradients of an earlier geometry."""
    mesh = _square()
    asm = Laplace.from_mesh(mesh)
    area = lambda points: asm.energy(func=lambda x: torch.ones_like(x[0]), points=points)  # noqa: E731
    for stretch in (1.5, 2.0):
        s = torch.tensor(stretch, dtype=torch.float64, requires_grad=True)
        points = mesh.points * torch.stack([s, torch.ones_like(s)])
        value = area(points)
        value.backward()
        assert float(value.detach()) == pytest.approx(stretch)
        assert float(s.grad) == pytest.approx(1.0)  # d(area)/d(stretch) of the unit square
    points = mesh.points.clone().requires_grad_(True)
    for _ in range(2):  # the same leaf twice: the first graph is freed by then
        (grad,) = torch.autograd.grad(area(points), points)
        assert float(grad.sum()) == pytest.approx(0.0, abs=1e-12)  # translations keep the area
    K = asm(points=points)
    K.values.sum().backward()
    assert points.grad is not None


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_small_inverse_matches_linalg_inv(dim):
    g = torch.Generator().manual_seed(dim)
    A = torch.rand(7, 5, dim, dim, generator=g, dtype=torch.float64) + dim * torch.eye(dim, dtype=torch.float64)
    assert torch.allclose(small_inverse(A), torch.linalg.inv(A), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_small_inverse_gradients(dim):
    g = torch.Generator().manual_seed(10 + dim)
    A = torch.rand(4, dim, dim, generator=g, dtype=torch.float64) + dim * torch.eye(dim, dtype=torch.float64)
    assert torch.autograd.gradcheck(small_inverse, (A.requires_grad_(True),))


def test_small_inverse_rejects_singular_matrices():
    with pytest.raises(ValueError, match="singular"):
        small_inverse(torch.tensor([[[1.0, 2.0], [2.0, 4.0]]], dtype=torch.float64))

"""Data that varies per quadrature point (``quadrature_data``) in ElementAssembler,
NodeAssembler and ElementAssembler.energy."""

import itertools

import meshio
import pytest
import torch

from tensormesh import ElementAssembler, LinearElasticityElementAssembler, Mesh, NodeAssembler

f64 = torch.float64


def _square(element_type="triangle6", k=3):
    """Unit square, k x k cells, two triangles each, nodes from the reference element."""
    from tensormesh.element import element_type2element, element_type2order
    ref = element_type2element(element_type).get_basis(element_type2order[element_type], f64)
    h, eye, nodes = 1.0 / k, torch.eye(2, dtype=f64), []
    for i, j in itertools.product(range(k), repeat=2):
        o = torch.tensor([i, j], dtype=f64) * h
        nodes += [o + ref @ (h * eye), o + h + ref @ (-h * eye)]
    nodes = torch.cat(nodes)
    key, conn = torch.unique(torch.round(nodes * 1e9).long(), dim=0, return_inverse=True)
    points = torch.zeros(len(key), 2, dtype=f64).index_put_((conn,), nodes)
    return Mesh(meshio.Mesh(points.numpy(), [(element_type, conn.reshape(2 * k * k, -1).numpy())]))


class Diffusion(ElementAssembler):
    def forward(self, gradu, gradv, A):
        return gradu @ A @ gradv


class Elasticity(ElementAssembler):
    """Block stiffness from a per-point fourth-order tangent C_ijkl."""

    def forward(self, gradu, gradv, C):
        return torch.einsum("j,ijkl,l->ik", gradu, C, gradv)


class StressDivergence(NodeAssembler):
    def forward(self, gradv, sigma):
        return sigma @ gradv


class Weighted(ElementAssembler):
    def forward(self, u, v):
        return u * v


def _random_spd(*shape, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*shape, dim, dim, generator=g, dtype=f64)
    return a @ a.mT + dim * torch.eye(dim, dtype=f64)


def _manual_matrix(asm, element_type, local):
    """Scatter per-element blocks local[e, a, b, (i, j)] into a dense matrix."""
    elements = asm.elements[element_type]
    n, block = asm.n_points, (local.shape[-1] if local.dim() == 5 else 1)
    dense = torch.zeros(n * block, n * block, dtype=local.dtype)
    if local.dim() == 3:
        local = local[..., None, None]
    E, B = elements.shape
    rows = (elements[:, :, None] * block + torch.arange(block)).reshape(E, B * block)
    local = local.permute(0, 1, 3, 2, 4).reshape(E, B * block, B * block)
    dense.index_put_((rows[:, :, None].expand(-1, -1, B * block), rows[:, None, :].expand(-1, B * block, -1)),
                     local, accumulate=True)
    return dense


def test_element_assembler_matches_manual_assembly():
    mesh = _square()
    asm = Diffusion.from_mesh(mesh, quadrature_order=3)
    trans = asm.transformation["triangle6"]
    A = _random_spd(trans.n_elements, trans.n_quadrature)
    K = asm(quadrature_data={"A": A}).to_dense()
    G, W = trans.shape_grad, trans.JxW                                   # [E, Q, B, D], [E, Q]
    local = torch.einsum("eqad,eqdk,eqbk,eq->eab", G, A, G, W)
    assert torch.allclose(K, _manual_matrix(asm, "triangle6", local), atol=1e-12)


def test_constant_tangent_reproduces_linear_elasticity():
    mesh = _square("triangle")
    E, nu = 2.0, 0.3
    reference = LinearElasticityElementAssembler.from_mesh(mesh, E=E, nu=nu)()
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))   # as voigt_stiffness in 2D
    I = torch.eye(2, dtype=f64)
    C = lam * torch.einsum("ij,kl->ijkl", I, I) + mu * (torch.einsum("ik,jl->ijkl", I, I)
                                                       + torch.einsum("il,jk->ijkl", I, I))
    asm = Elasticity.from_mesh(mesh)
    n_e, n_q = asm.transformation["triangle"].n_elements, asm.transformation["triangle"].n_quadrature
    K = asm(quadrature_data={"C": C.expand(n_e, n_q, 2, 2, 2, 2)})
    assert torch.allclose(K.to_dense(), reference.to_dense(), atol=1e-12)


def test_node_assembler_matches_manual_assembly():
    mesh = _square()
    asm = StressDivergence.from_mesh(mesh, quadrature_order=3)
    trans = asm.transformation["triangle6"]
    sigma = _random_spd(trans.n_elements, trans.n_quadrature, seed=1)
    f = asm(quadrature_data={"sigma": sigma}, batch_size=-1).reshape(mesh.n_points, 2)
    local = torch.einsum("eqij,eqbj,eq->ebi", sigma, trans.shape_grad, trans.JxW)
    expected = torch.zeros(mesh.n_points, 2, dtype=f64).index_add_(
        0, asm.elements["triangle6"].reshape(-1), local.reshape(-1, 2))
    assert torch.allclose(f, expected, atol=1e-12)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_quadrature_batches_agree(batch_size):
    mesh = _square()
    element = Diffusion.from_mesh(mesh, quadrature_order=3)
    node = StressDivergence.from_mesh(mesh, quadrature_order=3)
    trans = element.transformation["triangle6"]
    A = _random_spd(trans.n_elements, trans.n_quadrature, seed=2)
    full = element(quadrature_data={"A": A}).to_dense()
    assert torch.allclose(element(quadrature_data={"A": A}, batch_size=batch_size).to_dense(), full, atol=1e-12)
    full = node(quadrature_data={"sigma": A}, batch_size=-1)
    assert torch.allclose(node(quadrature_data={"sigma": A}, batch_size=batch_size), full, atol=1e-12)


@pytest.mark.parametrize("batch_size", [-1, 2])
def test_energy_takes_quadrature_data(batch_size):
    mesh = _square()
    asm = Weighted.from_mesh(mesh, quadrature_order=3)
    trans = asm.transformation["triangle6"]
    g = torch.Generator().manual_seed(3)
    c = torch.rand(trans.n_elements, trans.n_quadrature, generator=g, dtype=f64, requires_grad=True)
    u = torch.sin(mesh.points.sum(1))

    def density(gradu, c):
        return c * (gradu * gradu).sum()

    energy = asm.energy(func=density, point_data={"u": u}, quadrature_data={"c": c}, batch_size=batch_size)
    grad_u = torch.einsum("eb,eqbd->eqd", u[asm.elements["triangle6"]], trans.shape_grad)
    expected = (c * (grad_u ** 2).sum(-1) * trans.JxW).sum()
    assert torch.allclose(energy, expected, atol=1e-12)
    # the old shape detection of per-point element_data agrees, also in batches
    assert torch.allclose(asm.energy(func=density, point_data={"u": u}, element_data={"c": c},
                                     batch_size=batch_size), expected, atol=1e-12)
    # and the energy is differentiable in the per-point data
    (dc,) = torch.autograd.grad(energy, c)
    assert torch.allclose(dc, (grad_u ** 2).sum(-1) * trans.JxW, atol=1e-12)


def test_wrong_shapes_and_clashing_keys_are_rejected():
    mesh = _square()
    asm = Diffusion.from_mesh(mesh, quadrature_order=3)
    trans = asm.transformation["triangle6"]
    with pytest.raises(ValueError, match="must have shape"):
        asm(quadrature_data={"A": _random_spd(trans.n_elements, trans.n_quadrature + 1)})
    with pytest.raises(ValueError, match="also passed"):
        asm(point_data={"A": torch.zeros(mesh.n_points, dtype=f64)},
            quadrature_data={"A": _random_spd(trans.n_elements, trans.n_quadrature)})
    with pytest.raises(ValueError, match="also passed"):
        StressDivergence.from_mesh(mesh)(quadrature_data={"x": torch.zeros(trans.n_elements, 3, dtype=f64)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    mesh = _square()
    asm = Diffusion.from_mesh(mesh, quadrature_order=3)
    trans = asm.transformation["triangle6"]
    A = _random_spd(trans.n_elements, trans.n_quadrature, seed=4)
    cpu = asm(quadrature_data={"A": A}).to_dense()
    gpu_asm = Diffusion.from_mesh(mesh.to("cuda"), quadrature_order=3)
    gpu = gpu_asm(quadrature_data={"A": A.cuda()}).to_dense()
    assert gpu.is_cuda
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-12)

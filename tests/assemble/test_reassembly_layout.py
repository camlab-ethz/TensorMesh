"""Reassembling on the same topology must keep ``layout_signature`` stable.

Regression tests for https://github.com/camlab-ethz/TensorMesh/issues/50:
vector-valued (multi-DOF-per-node) assembly used to expand fresh block
indices on every call, so a :class:`Condenser` reused across Picard /
Newton reassemblies asserted with "the layout of the matrix is changed"
on the second call.
"""
import torch

from tensormesh import Condenser, Mesh
from tensormesh.assemble import (
    LaplaceElementAssembler,
    LinearElasticityElementAssembler,
)


def _mesh():
    return Mesh.gen_rectangle(chara_length=0.3, element_type="tri").double()


def test_scalar_reassembly_keeps_layout_signature():
    asm = LaplaceElementAssembler.from_mesh(_mesh(), quadrature_order=2)
    K1 = asm()
    K2 = asm()
    assert K1.has_same_layout(K2)


def test_vector_reassembly_keeps_layout_signature():
    asm = LinearElasticityElementAssembler.from_mesh(
        _mesh(), quadrature_order=2, E=1.0, nu=0.3,
    )
    K1 = asm()
    K2 = asm()
    assert K1.has_same_layout(K2)
    assert torch.allclose(K1.to_dense(), K2.to_dense())


def test_condenser_reused_across_vector_reassembly():
    mesh = _mesh()
    n = mesh.points.shape[0]
    asm = LinearElasticityElementAssembler.from_mesh(
        mesh, quadrature_order=2, E=1.0, nu=0.3,
    )

    dirichlet_mask = mesh.boundary_mask.repeat_interleave(2)  # node-major DOFs
    condenser = Condenser(dirichlet_mask)
    f = torch.ones(2 * n, dtype=torch.float64)

    K1 = asm()
    K1_, f1_ = condenser(K1, f)
    u1 = condenser.recover(K1_.solve(f1_))

    K2 = asm()  # second assembly used to trip the layout assertion here
    K2_, f2_ = condenser(K2, f)
    u2 = condenser.recover(K2_.solve(f2_))

    assert torch.allclose(u1, u2)

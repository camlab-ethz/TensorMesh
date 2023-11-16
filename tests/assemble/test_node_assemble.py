# TODO

import sys 
import numpy as np
import torch
import meshio
sys.path.append("../..")

from torch_fem import ElementAssembler, NodeAssembler,  Mesh
from torch_fem import dot, mul
import skfem


class TestAssembler(NodeAssembler):
    def forward(self, u, x):
        """
            Parameters:
            -----------
                u: torch.tensor[..., n_basis]
                x: torch.tensor[..., n_basis, n_dim]

            Returns:
            --------
                y: torch.tensor[..., n_basis]
        """
        # return (torch.cos(x[...,0]) + torch.sin(x[...,1])) * u 
        return u

@skfem.LinearForm
def test_assembler(u, w):
    # return (np.cos(w.x[0]) + np.sin(w.x[1])) * u
    return u

def node_assemble(mesh):
    assambler = TestAssembler
    F_asm = assambler.from_mesh(mesh, quadrature_order=2)
    F = F_asm(mesh.points)

    if mesh.default_element_type.startswith("tri"):
        Mesh = skfem.MeshTri
    elif mesh.default_element_type.startswith("quad"):
        Mesh = skfem.MeshQuad
    elif mesh.default_element_type.startswith("tet"):
        Mesh = skfem.MeshTet
    else:
        raise NotImplementedError

    mesh_skfem = Mesh(mesh.points.T.cpu().numpy(), mesh.elements().T.numpy())

    skfem_assembler = test_assembler

    element = {
        "triangle": skfem.ElementTriP1(),
        "quad": skfem.ElementQuad1(),
        "tetra": skfem.ElementTetP1(),
    }[mesh.default_element_type]
    basis = skfem.InteriorBasis(mesh_skfem, element)

    F_skfem = skfem.asm(skfem_assembler, basis)

    np.testing.assert_allclose(F.numpy(), F_skfem, rtol=1e-5)


def test_node_assembler_tri1():
    node_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="tri"))
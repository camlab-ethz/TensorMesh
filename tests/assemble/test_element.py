# TODO
# add linear elasticity

import sys 
import numpy as np
import torch
import meshio
sys.path.append("../..")

from tensormesh import ElementAssembler, NodeAssembler,  Mesh
from pytest import mark
import skfem


class LaplaceAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class ProductAssembler(ElementAssembler):
    def forward(self, u, v):
        return u * v

@skfem.BilinearForm
def laplace_assembler(u, v, w):
    from skfem.helpers import dot,grad 
    # breakpoint()
    return dot(grad(u), grad(v))

@skfem.BilinearForm
def product_assembler(u, v, w):
    from skfem.helpers import dot,grad 
    return u * v

def element_assemble(mesh, model="laplace",
                     rtol:float = 1e-5,
                     atol:float = 1e-10):
    
    
    assert model in ["laplace", "product"]
    assambler = {
        "laplace": LaplaceAssembler,
        "product": ProductAssembler
    }[model]

    quadrature_order = 3 if mesh.default_element_type.startswith("quad") else 2
    K_asm = assambler.from_mesh(mesh, quadrature_order=quadrature_order)
    K = K_asm(mesh.points)
    
    K_scipy = K.to_scipy_coo()
    
    if mesh.default_element_type.startswith("tri"):
        Mesh = skfem.MeshTri
    elif mesh.default_element_type.startswith("quad"):
        Mesh = skfem.MeshQuad
    elif mesh.default_element_type.startswith("tet"):
        Mesh = skfem.MeshTet
    else:
        raise NotImplementedError()
    # breakpoint()
    points = mesh.points.T.cpu().numpy()
    if mesh.default_element_type == "quad":
        elements = mesh.elements().clone().T.numpy()
        elements[[2,3]] = elements[[3,2]]
    else:
        elements = mesh.elements().clone().T.numpy()

    mesh_skfem = Mesh(points, elements)

    skfem_assembler = {
        "laplace": laplace_assembler,
        "product": product_assembler
    }[model]

    element = {
        "triangle": skfem.ElementTriP1(),
        "quad": skfem.ElementQuad1(),
        "tetra": skfem.ElementTetP1(),
    }[mesh.default_element_type]
    basis = skfem.InteriorBasis(mesh_skfem, element)


    K_skfem = skfem_assembler.assemble(basis)
    # K_skfem = skfem.asm(skfem_assembler, basis)

    K_scipy_dense = K_scipy.toarray() # [n_node, n_node]
    K_skfem_dense = K_skfem.toarray()

   
    np.testing.assert_allclose(K_scipy_dense, K_skfem_dense, rtol=rtol, atol=atol)
    for K in [K_scipy_dense, K_skfem_dense]:
        min_lambda = np.linalg.eig(K)[0].min()
        if min_lambda < 0:
            np.testing.assert_allclose(min_lambda, 0, atol=1e-10)

@mark.parametrize("chara_length", [0.2, 0.1, 0.05, 0.02])
def test_tri1_1(chara_length:float):
    element_assemble(Mesh.gen_rectangle(chara_length=chara_length,  element_type="tri"), model="laplace")
    # element_assemble(Mesh.gen_rectangle(chara_length=0.2,  element_type="quad"), model="product")

# @mark.parametrize("refinement", [3, 4, 5])
def test_tri1_2(refinement=4):
    m = skfem.MeshTri().refined(refinement)
    mesh = meshio.Mesh(points = m.p.T, cells = [("triangle", m.t.T)])
    mesh = Mesh.from_meshio(mesh)
    # element_assemble(mesh, model="laplace")
    element_assemble(mesh, model="product", atol=1e-3)

def test_quad1():
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="laplace")
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="product")

def test_quad2():
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="laplace")
    element_assemble(Mesh.gen_rectangle(chara_length=0.1,  element_type="quad"), model="product")


def test_tet1():
    element_assemble(Mesh.gen_cube(chara_length=0.1), model="laplace")
    element_assemble(Mesh.gen_cube(chara_length=0.1), model="product")



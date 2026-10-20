import torch 
import sys 
from typing import Type
sys.path.append("../..")
from tensormesh.element.normal import outwards_normal_2d, outwards_normal_3d
from tensormesh.element import   Element,\
                                Triangle,\
                                Quadrilateral,\
                                Tetrahedron,\
                                Hexahedron,\
                                Prism,\
                                Pyramid

def _2d(element:Type[Element]):
    normals = element.get_outwards_facet_normal() # [n_facet, n_dim]

    assert torch.linalg.norm(normals, dim=1).allclose(torch.ones(normals.shape[0]))
    vertex  = element.points[element.vertex] # [n_vertex, n_dim]
    edge    = element.points[element.edge]   # [n_facet, n_ends, n_dim] 
    edge_vec = edge[:, 1] - edge[:, 0] # [n_facet, n_dim]
    assert (torch.vmap(torch.dot)(normals, edge_vec)).allclose(torch.zeros(normals.shape[0]))
    # outward: from the edge midpoint, the centroid lies against the normal
    inward = vertex.reshape(-1, vertex.shape[-1]).mean(dim=0) - edge.mean(dim=1) # [n_facet, n_dim]
    assert bool(((normals * inward).sum(-1) < 0).all())

def _3d(element:Type[Element]):
    normals = element.get_outwards_facet_normal() # [n_facet, n_dim]

    assert torch.linalg.norm(normals, dim=1).allclose(torch.ones(normals.shape[0]))
    vertex  = element.points[element.vertex] # [n_vertex, n_dim]
    for f in range(element.n_face):
        normal = normals[f]
        face   = element.points[torch.tensor(element.face[f])] # [n_vertex_per_facet, n_dim]
        n_vertex_per_face = face.shape[0]
        for i in range(n_vertex_per_face):
            for j in range(i+1, n_vertex_per_face):
                vec = face[i] - face[j]
                assert torch.dot(vec, normal).allclose(torch.zeros(1))
        # outward: from the face centroid, the cell centroid lies against the normal
        centroid = vertex.reshape(-1, vertex.shape[-1]).mean(dim=0)
        assert float(torch.dot(centroid - face.mean(dim=0), normal)) < 0

def test_tri():
    _2d(Triangle)
    
def test_quad():
    _2d(Quadrilateral)

def test_tet():
    _3d(Tetrahedron)

def test_hex():
    _3d(Hexahedron)

def test_pyr():
    _3d(Pyramid)

def test_pri():
    _3d(Prism)
import numpy as np
import torch 
import sys 
sys.path.append("../..")
from torch_fem import Mesh, MeshGen

def test_node_adjacency_tri():
    mesh = Mesh.gen_rectangle(chara_length=0.2) 
    node_adj = mesh.node_adjacency()
    

    n_node = mesh.points.shape[0]
    elements = mesh.elements()
    n_element = elements.shape[0]
    node_adj_gt = torch.zeros((n_node, n_node), dtype=torch.float)
    for element in elements:
        for u in element:
            for v in element:
                node_adj_gt[u, v] = 1
    
    assert torch.all(node_adj.to_dense() == node_adj_gt), f"f{node_adj.to_dense()} {node_adj_gt}"

def test_node_adjacency_mix():
    mesh_gen = MeshGen(element_type=None, chara_length=0.3, order=1)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh = mesh_gen.gen()
    node_adj = mesh.node_adjacency()
    n_node = mesh.points.shape[0]
    node_adj_gt = torch.zeros((n_node, n_node), dtype=torch.float)
    n_node = mesh.points.shape[0]
    for elements in mesh.elements().values():
        for element in elements:
            for u in element:
                for v in element:
                    node_adj_gt[u, v] = 1
    
    assert torch.all(node_adj.to_dense() == node_adj_gt), f"f{node_adj.to_dense()} {node_adj_gt}"
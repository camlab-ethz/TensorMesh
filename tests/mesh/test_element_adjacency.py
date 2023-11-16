import numpy as np
import torch 
import sys 
sys.path.append("../..")
from torch_fem import Mesh, MeshGen

def test_element_adjacency_tri():
    mesh = Mesh.gen_rectangle(chara_length=0.2) 
    ele_adj = mesh.element_adjacency()
    

    n_element  = mesh.elements().shape[0]
    ele_adj_gt = torch.zeros((n_element, n_element), dtype=torch.float)
    for i in range(n_element):
        for j in range(i):
            triangle_i, triangle_j = set(mesh.elements()[i].tolist()), set(mesh.elements()[j].tolist())
            if len(triangle_i.intersection(triangle_j)) == 2:
                ele_adj_gt[i, j] = 1
                ele_adj_gt[j, i] = 1
    
    assert torch.all(ele_adj.to_dense() == ele_adj_gt), f"f{ele_adj.to_dense()} {ele_adj_gt}"

def test_element_adjacency_mix():
    mesh_gen = MeshGen(element_type=None, chara_length=0.3, order=1)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh = mesh_gen.gen()
    ele_adj = mesh.element_adjacency()
    
    n_element  = sum([i.shape[0] for i in mesh.elements().values()])
    ele_adj_gt = torch.zeros((n_element, n_element), dtype=torch.float)
    ptr_i = 0
    for elements_i in mesh.elements().values():
        for e_i in range(elements_i.shape[0]):
            nodes_i = set(elements_i[e_i].tolist())
            ptr_j = 0 
            for elements_j in mesh.elements().values():
                for e_j in range(elements_j.shape[0]):
                    nodes_j = set(elements_j[e_j].tolist())
                    if len(nodes_i.intersection(nodes_j)) == 2:
                        ele_adj_gt[ptr_i, ptr_j]+= 1
                    ptr_j += 1
                    print(f"ptr_i {ptr_i} ptr_j {ptr_j}")
            ptr_i += 1



  
    assert torch.all(ele_adj.to_dense() == ele_adj_gt), f"f{ele_adj.to_dense()} {ele_adj_gt}"
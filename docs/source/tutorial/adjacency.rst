Adjacency 
=========

Node Adjacency
--------------

.. code-block:: python 

    import torch_fem
    import matplotlib.pyplot as plt

    mesh_gen = torch_fem.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh = mesh_gen.gen()

    adj = mesh.node_adjacency()

    torch_fem.draw_graph(adj, mesh.points.numpy())
    torch_fem.draw_mesh(mesh)

    plt.show()

.. image:: ../_static/node_adjacency.png
    :width: 400px
    :align: center

Element Adjacency
-----------------

.. code-block:: python

    import torch_fem
    import matplotlib.pyplot as plt

    mesh_gen = torch_fem.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh = mesh_gen.gen()

    adj = mesh.element_adjacency()

    centers = torch.cat([mesh.points[value].mean(axis=1) for value in mesh.elements().values()]).numpy() # [n_element, 2]
    torch_fem.draw_graph(adj, centers)
    torch_fem.draw_mesh(mesh)

    plt.show()

.. image:: ../_static/element_adjacency.png
    :width: 400px
    :align: center
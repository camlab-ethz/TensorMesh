tensormesh.mesh
===============

.. py:module:: tensormesh.mesh

Mesh
----

.. autoclass:: tensormesh.Mesh
    :members:
    :show-inheritance:
    :exclude-members: cells, cell_data, cell_sets, default_element_type, default_eletyp, dim, dim2eletyp, field_data, point_data, points


Graph algorithms
----------------

Helpers used internally by :class:`~tensormesh.distributed.DistributedMesh`
for race-free parallel assembly and domain decomposition.

.. autofunction:: tensormesh.mesh.graph_coloring

.. autofunction:: tensormesh.mesh.graph_partition

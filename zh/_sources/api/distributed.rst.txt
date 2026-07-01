tensormesh.distributed
======================

.. py:module:: tensormesh.distributed

Mesh partitioning and parallel assembly across multiple devices,
with integration into ``torch-sla``'s distributed sparse solver. See
:doc:`/example_gallery/distributed` for a worked walkthrough.

DistributedMesh
---------------

.. autoclass:: tensormesh.distributed.DistributedMesh
    :members:
    :show-inheritance:


Distributed assembly
--------------------

.. autofunction:: tensormesh.distributed.distributed_element_assemble

.. autofunction:: tensormesh.distributed.distributed_element_assemble_to_sparse

.. autofunction:: tensormesh.distributed.distributed_node_assemble

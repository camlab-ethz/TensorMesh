tensormesh.distributed
======================

.. py:module:: tensormesh.distributed

Mesh partitioning and parallel assembly across multiple devices,
with integration into ``torch-sla``'s distributed sparse solver. See
the :doc:`user guide chapter </user_guide/distributed>` for a worked
walkthrough.

DistributedMesh
---------------

.. autoclass:: tensormesh.distributed.DistributedMesh
    :members:
    :show-inheritance:


DSparseMatrix
-------------

.. autoclass:: tensormesh.distributed.DSparseMatrix
    :members: to_single, layout_signature
    :show-inheritance:


The ``@distributed`` decorator
------------------------------

.. autofunction:: tensormesh.distributed.distributed


Distributed assembly
--------------------

.. autofunction:: tensormesh.distributed.distributed_element_assemble

.. autofunction:: tensormesh.distributed.distributed_element_assemble_per_rank

.. autofunction:: tensormesh.distributed.distributed_element_assemble_to_sparse

.. autofunction:: tensormesh.distributed.distributed_node_assemble


Utilities
---------

.. autofunction:: tensormesh.distributed.broadcast_from_rank0

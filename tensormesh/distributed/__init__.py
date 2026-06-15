"""
Distributed FEM assembly for multi-GPU computation.

Provides mesh partitioning and parallel assembly across multiple devices,
with integration into torch-sla's distributed sparse solver.
"""

from .mesh import DistributedMesh
from .assemble import (
    distributed_element_assemble,
    distributed_element_assemble_to_sparse,
    distributed_element_assemble_per_rank,
    distributed_node_assemble,
)

__all__ = [
    'DistributedMesh',
    'distributed_element_assemble',
    'distributed_element_assemble_to_sparse',
    'distributed_element_assemble_per_rank',
    'distributed_node_assemble',
]

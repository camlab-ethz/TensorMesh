Distributed FEM
===============

When a problem no longer fits on one device — memory-wise or
time-wise — TensorMesh can partition the mesh, assemble each subdomain
on its own GPU (or CPU rank), and hand the result to ``torch-sla``'s
distributed sparse solver as one logical system. The whole pipeline
stays in the familiar shape:

.. code-block:: text

   Mesh → DistributedMesh → @distributed Assembler → DSparseMatrix → Condenser → torch_sla.solve

Everything lives in :mod:`tensormesh.distributed` (see the
:doc:`API reference </api/distributed>`), and the solver side is
``torch-sla``'s ``DSparseTensor`` machinery — domain decomposition with
owned-rows-plus-halo partitions, which upstream benchmarks have scaled
past 400M DOF on multi-GPU nodes.

.. caution::

   The distributed layer is functional end-to-end (assembly →
   boundary conditions → multi-GPU CG → verified against analytical
   solutions), but some contracts are **interim** and may change:
   the :class:`~tensormesh.Condenser` bridge currently round-trips
   through a gathered single-device matrix, and the distributed RHS
   assembly still returns the full-length vector on every rank.
   Worked examples with multi-GPU benchmark numbers will be added to
   the example gallery once they are validated on a multi-card
   cluster.


Partitioning the mesh
---------------------

:class:`~tensormesh.distributed.DistributedMesh` wraps the mesh
partitioner: it splits a global :class:`~tensormesh.Mesh` into
submeshes and assigns each to a device. Every submesh keeps an
``orig_nid`` entry in its ``point_data`` — the local-to-global node
map that the assembly layer uses to scatter into global coordinates.

.. code-block:: python

   from tensormesh import Mesh
   from tensormesh.distributed import DistributedMesh

   mesh = Mesh.gen_rectangle(chara_length=0.02)
   dmesh = DistributedMesh(mesh, num_partitions=4)   # defaults: one per GPU
   print(dmesh)          # partition sizes + devices
   sub0 = dmesh[0]       # a plain Mesh on its assigned device

Three partitioning methods are available via ``method=``:
``"coordinate"`` (recursive coordinate bisection — fast, the default),
``"spectral"`` (Fiedler-vector bisection), and ``"metis"`` (graph
partitioning, needs ``pymetis``). Build and partition on **CPU**
first and let the submeshes carry the device assignment — the
partitioners index on CPU.


Making any assembler distributed
--------------------------------

You do not write a second assembler for the distributed path. The
:func:`~tensormesh.distributed.distributed` class decorator turns any
existing :class:`~tensormesh.ElementAssembler` or
:class:`~tensormesh.NodeAssembler` subclass — built-in or your own —
into its distributed counterpart by swapping the two entry points:
``from_mesh`` accepts a ``DistributedMesh``, and ``__call__`` runs the
distributed assembly:

.. code-block:: python

   from tensormesh.assemble import LaplaceElementAssembler
   from tensormesh.distributed import distributed

   DistLaplace = distributed(LaplaceElementAssembler)
   K_dist = DistLaplace.from_mesh(dmesh)()     # -> DSparseMatrix

The weak form, quadrature setup, and element kernels are inherited
unchanged — the decorator only reroutes *where* the element loops run
and *how* the results are stitched together. Constructor arguments
still go through ``from_mesh(dmesh, **kwargs)`` and call-time data
(``point_data`` etc.) through the call, exactly like the single-device
contract.

Underneath, two execution modes cover the common deployments:

* **Single process, multiple devices.** Without an initialized
  ``torch.distributed`` process group, submeshes are assembled in
  parallel *threads*, one per device, and merged. This is the
  low-friction mode: no launcher, works in a notebook, and on a
  CPU-only machine it simply simulates the partition — useful for
  testing partition-dependent code before renting the GPUs.
* **One process per rank (SPMD).** Under ``torchrun``, each rank
  assembles **only its own submesh**; the resulting COO triples are
  routed to the rank that owns their row via a single
  ``all_to_all`` exchange (NCCL on CUDA, gloo on CPU), halo columns
  are discovered automatically, and each rank ends up with exactly
  its owned-rows-plus-halo slice. This is the mode that scales.

The free functions behind the decorator —
:func:`~tensormesh.distributed.distributed_element_assemble`,
:func:`~tensormesh.distributed.distributed_element_assemble_per_rank`,
:func:`~tensormesh.distributed.distributed_node_assemble` — are public
too, plus ``distributed_element_assemble_to_sparse`` which merges
everything back into a plain global
:class:`~tensormesh.sparse.SparseMatrix` (handy for verifying the
distributed result against the single-device one).


The distributed matrix
----------------------

Distributed assembly returns a
:class:`~tensormesh.distributed.DSparseMatrix` — a subclass of
``torch_sla.distributed.DSparseTensor``, the same way
:class:`~tensormesh.sparse.SparseMatrix` subclasses
``torch_sla.SparseTensor``. Because it *is* a ``DSparseTensor``,
every torch-sla API accepts it directly: ``@`` matvecs, arithmetic,
I/O, and — most importantly — the distributed solver. Two
FEM-specific additions:

* a **partition UUID** folded into ``layout_signature`` — generated
  on rank 0 and broadcast, so repeated assemblies on the same
  partition share condensation caches, while two independently
  partitioned matrices never do (even if their local layouts happen
  to coincide);
* ``to_single()`` — allgather the distributed matrix back into one
  :class:`~tensormesh.sparse.SparseMatrix`, for verification and
  small-scale debugging.


Boundary conditions and the distributed solve
---------------------------------------------

The unchanged :class:`~tensormesh.Condenser` dispatches on the matrix
type: handing it a ``DSparseMatrix`` routes through the distributed
bridge. Two practical rules in the current (interim) contract:

* build the Dirichlet mask on **CPU** (``mesh.boundary_mask.cpu()``) —
  the condensation buffers then stay on CPU and the bridge moves the
  condensed output back to the GPU itself;
* the solver consumes each rank's **owned slice** of the RHS and
  returns the owned slice of the solution — slice before, gather
  after.

.. code-block:: python

   import torch_sla
   from torch_sla import SolverConfig
   from torch_sla.distributed import gather_owned_to_global

   cond = Condenser(mesh.boundary_mask.cpu())
   K_inner, b_inner = cond(K_dist, b)

   owned = K_inner.partition.owned_nodes.long().to(device)
   with SolverConfig(method="cg", atol=1e-12, rtol=1e-10, maxiter=1000,
                     verbose=(rank == 0)):
       u_owned = torch_sla.solve(K_inner, b_inner.to(device)[owned])

   u_inner = gather_owned_to_global(owned, u_owned, b_inner.shape[0]).cpu()
   u = cond.recover(u_inner)

Note that the solve goes through the ``torch_sla.solve`` **free
function** — ``DSparseTensor`` deliberately has no instance
``.solve()`` — and works on ``DSparseMatrix`` precisely because of the
subclass relationship. Distributed solves are iterative
(preconditioned CG for the SPD systems FEM produces); see
:doc:`linear_solvers` for where that sits in the backend matrix.


Keeping ranks consistent
------------------------

Under SPMD every rank executes the whole script, so anything random —
a sampled coefficient field, a randomized load — would silently
diverge across ranks and the solver would converge to a
rank-dependent answer.
:func:`~tensormesh.distributed.broadcast_from_rank0` wraps the draw:

.. code-block:: python

   from tensormesh.distributed import broadcast_from_rank0

   a = broadcast_from_rank0(lambda: torch.empty((K, K)).uniform_(-1, 1))

Rank 0 evaluates the closure; every other rank receives the broadcast
tensor. In single-process runs it short-circuits to a plain call, so
the same script stays valid both ways.


A complete ``torchrun`` skeleton
--------------------------------

``examples/distributed/poisson_distributed_cuda.py`` is the reference
end-to-end script (it doubles as the integration test). The skeleton:

.. code-block:: python

   import os, torch, torch.distributed as dist
   from tensormesh import Mesh, Condenser
   from tensormesh.assemble import LaplaceElementAssembler
   from tensormesh.distributed import DistributedMesh, distributed

   def main():
       rank = int(os.environ["LOCAL_RANK"])
       world = int(os.environ["WORLD_SIZE"])
       dist.init_process_group(backend="nccl", rank=rank, world_size=world)
       torch.cuda.set_device(rank)
       device = torch.device(f"cuda:{rank}")
       try:
           mesh = Mesh.gen_rectangle(chara_length=0.02)        # CPU first
           dmesh = DistributedMesh(mesh, num_partitions=world,
                                   devices=[device] * world)
           mesh = mesh.to(device=device)

           K = distributed(LaplaceElementAssembler).from_mesh(dmesh)()
           # ...assemble b, condense, torch_sla.solve, recover (above)...
       finally:
           dist.destroy_process_group()

   if __name__ == "__main__":
       main()

launched with:

.. code-block:: bash

   torchrun --nproc-per-node=2 examples/distributed/poisson_distributed_cuda.py

Use ``backend="gloo"`` instead of ``"nccl"`` to run the same SPMD
pipeline on CPU ranks (e.g. for CI or laptops).


Related building blocks
-----------------------

* ``mesh.color()`` — greedy coloring of the element-adjacency graph,
  the classic prerequisite for race-free scatter on a single GPU.
* ``tensormesh.mesh.partition.partition_mesh`` — the partitioner
  behind ``DistributedMesh``, usable standalone.
* ``examples/distributed/`` — runnable scripts: the ``torchrun``
  Poisson above, partitioning / coloring visualizations, and
  assembly benchmarks.


What's next
-----------

* :doc:`linear_solvers` — the full solver-backend matrix, including
  where the distributed path takes over from single-device solves.
* :doc:`batched_workflows` — ``batch_size`` chunking for assemblies
  that fit on one device.
* The `torch-sla documentation <https://www.torchsla.com/>`_ —
  ``DSparseTensor``, ``Partition``, and the distributed solver
  internals.

"""Distributed multi-GPU Poisson via NCCL.

End-to-end demo of the new TensorMesh distributed surface:

* :class:`Mesh` → :class:`tensormesh.distributed.DistributedMesh` (geometric / METIS partition)
* ``@distributed(LaplaceElementAssembler)`` produces a
  :class:`tensormesh.sparse.DSparseMatrix` per rank (its local slice)
* :class:`tensormesh.Condenser` dispatches on the matrix type and
  returns a per-rank distributed system
* ``torch_sla.solve`` accepts the ``DSparseMatrix`` directly because
  ``isinstance(D, DSparseTensor)`` -- no manual unwrap

This file is both:

1. A reference example users can ``torchrun`` directly:

   .. code-block:: bash

      torchrun --nproc-per-node=2 examples/distributed/poisson_distributed_cuda.py

2. An integration test driven by
   :file:`tests/distributed/test_distributed_assemble_cuda.py`, which
   spawns workers via ``torch.multiprocessing`` instead of ``torchrun``.

The mesh size is intentionally modest (``chara_length=0.1`` → ~few
hundred DOFs) so the test path runs in seconds on autodl 2× A100; the
``torchrun`` path is fine on larger meshes too.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def run(rank: int, world: int, *, chara_length: float = 0.05,
        check_against_analytical: bool = True) -> dict:
    """Run the distributed Poisson solve on this rank.

    Returns a dict with diagnostic metrics so the caller (interactive
    user or integration test) can verify convergence + accuracy.
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from tensormesh import Mesh, Condenser
    from tensormesh.assemble import LaplaceElementAssembler, NodeAssembler
    from tensormesh.dataset import PoissonMultiFrequency
    from tensormesh.distributed import (
        DistributedMesh, broadcast_from_rank0,
        distributed, distributed_node_assemble,
    )
    from tensormesh.distributed import DSparseMatrix
    from torch_sla import SolverConfig, solve

    # ---- Mesh ---------------------------------------------------------
    # Build + partition on CPU first: the RCB / geometric partitioner
    # in tensormesh.mesh.partition indexes on CPU and cross-device
    # access trips ``indices should be either on cpu or on the same
    # device as the indexed tensor``. Move to CUDA after the
    # DistributedMesh is built (the per-rank submeshes inherit the
    # ``devices`` arg).
    mesh = Mesh.gen_rectangle(chara_length=chara_length)
    dmesh = DistributedMesh(
        mesh, num_partitions=world,
        devices=[device] * world,
    )
    mesh = mesh.to(device=device)

    # ---- Source term (PoissonMultiFrequency reference) ---------------
    # ``PoissonMultiFrequency.__init__`` samples its coefficient matrix
    # ``a`` from the local RNG -- benign in single-process use, fatal
    # under multi-rank: every rank would draw a *different* ``a``, the
    # source terms diverge and the distributed solve converges to a
    # rank-dependent (wrong) answer. The library class is intentionally
    # left distributed-naive; this example, being the distributed
    # entry point, is responsible for sharing the random draw.
    #
    # Pattern: sample on rank 0 inside ``broadcast_from_rank0``, hand
    # the resulting tensor in via the ``a=`` constructor argument that
    # the class already exposes. Single-process runs are a no-op (the
    # helper short-circuits to a plain call).
    K = 8
    a = broadcast_from_rank0(
        lambda: torch.empty((K, K)).uniform_(-1, 1)
    )
    equation = PoissonMultiFrequency(a=a)
    f_vals = equation.source_term(mesh.points, domain="rectangle")

    # ---- Distributed assembly via decorator --------------------------
    DistLaplace = distributed(LaplaceElementAssembler)
    K_dist = DistLaplace.from_mesh(dmesh)()        # DSparseMatrix
    K_dist = K_dist.cuda()                          # already cuda, no-op safe

    class FAssembler(NodeAssembler):
        def forward(self, v, f):
            return v * f

    # Right-hand side: each rank assembles its share (currently each
    # rank holds the full N-vector). FAssembler picks up ``f`` from
    # ``point_data``.
    b = distributed_node_assemble(
        FAssembler, dmesh, quadrature_order=2,
        point_data={"f": f_vals.to(device=device)},
    )

    # ---- Apply Dirichlet BC + distributed CG -------------------------
    # Condenser keeps its dirichlet_mask as a buffer on the device it
    # was constructed with. The Condenser._call_distributed bridge for
    # DSparseMatrix routes the cached single-device condensation logic
    # through CPU (the inner-edge mask gather is cross-device-unsafe);
    # construct the mask on CPU so the cached buffers stay on CPU and
    # the round-trip doesn't trip a cross-device index. The condensed
    # output is moved back to CUDA inside _call_distributed.
    cond = Condenser(mesh.boundary_mask.cpu())
    K_inner, b_inner = cond(K_dist, b)
    # K_inner is a fresh DSparseMatrix (round-trip via to_single in this
    # interim contract); torch_sla.solve accepts it directly because the
    # subclass passes the isinstance gate.
    # The current Condenser DSparseMatrix bridge round-trips through
    # ``to_single`` and returns a global ``b_inner`` vector (replicated
    # across ranks). torch_sla.solve expects the per-rank owned slice,
    # so we explicitly hand it the slice corresponding to the new
    # condensed partition and re-gather the result.
    from torch_sla.distributed import gather_owned_to_global
    owned_inner = K_inner.partition.owned_nodes.long().to(device)
    b_inner_owned = b_inner.to(device)[owned_inner]

    # DSparseTensor exposes no instance ``.solve(b)``; use the
    # ``torch_sla.solve`` free function. This works because
    # DSparseMatrix is-a DSparseTensor (subclass) -- the whole point
    # of the inheritance refactor.
    with SolverConfig(method="cg", atol=1e-12, rtol=1e-10, maxiter=1000,
                       verbose=(rank == 0)):
        u_inner_owned = solve(K_inner, b_inner_owned)
    # Allgather owned -> global so ``cond.recover`` (cpu-only path)
    # can splat into the full DOF vector.
    u_inner = gather_owned_to_global(
        owned_inner, u_inner_owned, b_inner.shape[0],
    ).cpu()
    u_dist = cond.recover(u_inner)

    # ---- Compare to analytical (broadcast to every rank for the dist
    # smoke; the result is identical across ranks because Condenser
    # currently round-trips via to_single).
    info = {"rank": rank, "world": world, "n_dof": int(mesh.n_points)}
    if check_against_analytical:
        u_analytical = equation.solution(mesh.points).to(device=device)
        u_dist_cuda = u_dist.to(device=device)
        rel_err = float(((u_dist_cuda - u_analytical).norm() /
                          u_analytical.norm()).item())
        info["rel_err_vs_analytical"] = rel_err
    return info


def main():
    """``torchrun`` entry point.

    Initialises NCCL using LOCAL_RANK/WORLD_SIZE that torchrun exports,
    runs the solve, prints diagnostics on rank 0.
    """
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    try:
        info = run(rank, world)
        if rank == 0:
            print(f"[distributed Poisson] rank={info['rank']} "
                  f"world={info['world']} N={info['n_dof']} "
                  f"rel_err={info.get('rel_err_vs_analytical', 'n/a'):.3e}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

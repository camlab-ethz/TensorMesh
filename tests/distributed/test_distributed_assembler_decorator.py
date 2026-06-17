"""End-to-end test for the ``@distributed`` Assembler class decorator.

Build a Laplace assembly two ways:
1. Single-device reference  (``LaplaceElementAssembler.from_mesh(mesh)()``)
2. Distributed via decorator (``distributed(LaplaceElementAssembler).from_mesh(dmesh)()``)

then matvec both with the same vector and assert the results match.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(rank, world, port, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from tensormesh import Mesh
        from tensormesh.assemble import LaplaceElementAssembler
        from tensormesh.distributed import DistributedMesh, distributed
        from tensormesh.distributed import DSparseMatrix

        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()

        dmesh = DistributedMesh(
            mesh, num_partitions=world,
            devices=[torch.device("cpu")] * world,
        )

        # --- Decorator usage --- #
        DistLaplace = distributed(LaplaceElementAssembler)
        K_dist = DistLaplace.from_mesh(dmesh)()

        # Type contract
        is_dsparse = isinstance(K_dist, DSparseMatrix)

        # Same seed across ranks -> identical x; both compute the same matvec.
        torch.manual_seed(0)
        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x

        owned = K_dist.partition.owned_nodes.long()
        x_owned = x[owned]
        y_dist_owned = K_dist @ x_owned

        from torch_sla.distributed import gather_owned_to_global
        y_dist_global = gather_owned_to_global(owned, y_dist_owned, mesh.n_points)
        max_diff = float((y_ref - y_dist_global).abs().max().item())
        q.put((rank, "OK", is_dsparse, max_diff))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:800]))
    finally:
        dist.destroy_process_group()


def test_distributed_decorator_matches_single_2procs():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, 31611, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == 2
    for rank, tag, *rest in results:
        assert tag == "OK", f"rank {rank}: {rest}"
        is_dsparse, max_diff = rest
        assert is_dsparse, f"rank {rank}: K_dist is not DSparseMatrix"
        assert max_diff < 1e-8, f"rank {rank}: matvec diff {max_diff}"

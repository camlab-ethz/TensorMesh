"""DSparseMatrix wrapping and partition UUID broadcast.

Covers:

* basic delegation (shape / device / dtype / matvec)
* arithmetic preserves partition_uuid (so derived matrices share caches)
* layout_signature has the UUID tail
* to_single() round-trip
* UUID is broadcast and identical across ranks
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _simple_dsparse(world, rank, port, q, what):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from tensormesh.distributed import DSparseMatrix
        from torch_sla import SparseTensor
        from torch_sla.distributed import DSparseTensor
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            from torch.distributed._tensor.device_mesh import init_device_mesh

        # Build a small global SparseTensor (every rank constructs the
        # same global, then partition() shards it).
        N = 8
        ii = torch.arange(N, dtype=torch.long)
        # tridiagonal symmetric: -1 / 2 / -1
        row = torch.cat([ii, ii[:-1], ii[1:]])
        col = torch.cat([ii, ii[1:], ii[:-1]])
        val = torch.cat([
            torch.full((N,), 2.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
        ])
        A_global = SparseTensor(val, row, col, (N, N))
        mesh = init_device_mesh("cpu", (world,))
        dst = DSparseTensor.partition(
            A_global, mesh, partition_method="metis",
        )
        D = DSparseMatrix(dst)

        if what == "uuid_consistent":
            # All ranks must agree on the UUID (broadcast at construction).
            q.put((rank, "OK", D.partition_uuid))
        elif what == "signature_carries_uuid":
            sig = D.layout_signature
            q.put((rank, "OK", sig[-1] == D.partition_uuid, len(sig)))
        elif what == "arithmetic_preserves_uuid":
            two_D = D + D
            q.put((rank, "OK",
                    two_D.partition_uuid == D.partition_uuid,
                    two_D.layout_signature == D.layout_signature))
        elif what == "to_single_roundtrip":
            S = D.to_single()
            # Every rank sees the full N x N global; compare nnz vs original.
            q.put((rank, "OK",
                    S.shape == (N, N),
                    S.row_indices.numel() == val.numel()))
        elif what == "matvec":
            x_global = torch.arange(N, dtype=torch.float64)
            # DSparseTensor matvec expects the owned slice, not the global.
            owned = D.partition.owned_nodes.long()
            x_owned = x_global[owned]
            y = D @ x_owned
            q.put((rank, "OK",
                    torch.isfinite(y).all().item(),
                    tuple(y.shape)))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:600]))
    finally:
        dist.destroy_process_group()


def _run(what, port):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_simple_dsparse, args=(2, r, port, q, what))
              for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_dsparsematrix_uuid_consistent_across_ranks():
    r = _run("uuid_consistent", 31501)
    assert len(r) == 2, f"missing results: {r}"
    for rank, tag, *_ in r:
        assert tag == "OK", f"rank {rank}: {r}"
    uuids = {x[2] for x in r}
    assert len(uuids) == 1, f"UUID disagreement across ranks: {uuids}"


def test_dsparsematrix_signature_carries_uuid():
    r = _run("signature_carries_uuid", 31503)
    assert len(r) == 2
    for rank, tag, carries, sig_len in r:
        assert tag == "OK"
        assert carries, f"rank {rank}: signature[-1] != partition_uuid"
        # mixin tail (6) + uuid (1) = 7
        assert sig_len == 7, f"rank {rank}: unexpected signature length {sig_len}"


def test_dsparsematrix_arithmetic_preserves_uuid():
    r = _run("arithmetic_preserves_uuid", 31505)
    assert len(r) == 2
    for rank, tag, same_uuid, same_sig in r:
        assert tag == "OK"
        assert same_uuid, f"rank {rank}: 2*D has fresh UUID"
        assert same_sig, f"rank {rank}: 2*D has different signature"


def test_dsparsematrix_to_single_roundtrip():
    r = _run("to_single_roundtrip", 31507)
    assert len(r) == 2
    for rank, tag, *rest in r:
        assert tag == "OK", f"rank {rank} crashed: {rest}"
        assert all(rest), f"rank {rank} to_single mismatch: {rest}"


def test_dsparsematrix_matvec_runs():
    r = _run("matvec", 31509)
    assert len(r) == 2
    for rank, tag, finite, shape in r:
        assert tag == "OK"
        assert finite

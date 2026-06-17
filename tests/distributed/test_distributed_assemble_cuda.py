"""NCCL + multi-GPU CUDA coverage of distributed assembly and solve.

Mirror of :file:`test_distributed_assemble.py` but with ``backend="nccl"``
and CUDA devices. ``pytestmark`` skips the whole module on machines
without 2+ visible GPUs, so CI / Mac / single-card tb16 see green
"skipped" while the autodl 2× A100 box (or any multi-GPU node) actually
runs these.

The Gloo-CPU tests already cover the data-flow correctness; this file
is the canary for NCCL-specific footguns (P2P ordering, allreduce
device placement, halo exchange on CUDA streams, UUID broadcast through
the NCCL backend).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="requires 2+ CUDA GPUs (NCCL multi-rank)",
)


# ─── Workers ────────────────────────────────────────────────────────


def _matvec_worker(rank, world, port, q):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    try:
        from tensormesh import Mesh
        from tensormesh.assemble import LaplaceElementAssembler
        from tensormesh.distributed import DistributedMesh, distributed
        from tensormesh.distributed import DSparseMatrix
        from torch_sla.distributed import gather_owned_to_global, DSparseTensor

        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()

        dmesh = DistributedMesh(
            mesh, num_partitions=world,
            devices=[torch.device(f"cuda:{rank}")] * world,
        )

        K_dist = distributed(LaplaceElementAssembler).from_mesh(dmesh)()
        K_dist = K_dist.cuda()

        # Subclass invariant: DSparseMatrix is-a DSparseTensor so the
        # whole torch-sla free-function surface accepts it.
        is_subclass = isinstance(K_dist, DSparseTensor)

        torch.manual_seed(0)
        x = torch.randn(mesh.n_points, dtype=K_ref.dtype, device=f"cuda:{rank}")
        y_ref = K_ref.cuda() @ x

        owned = K_dist.partition.owned_nodes.long().to(f"cuda:{rank}")
        x_owned = x[owned]
        y_dist_owned = K_dist @ x_owned

        y_dist_global = gather_owned_to_global(owned, y_dist_owned, mesh.n_points)
        max_diff = float((y_ref - y_dist_global).abs().max().item())
        q.put((rank, "OK", is_subclass, max_diff))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:1000]))
    finally:
        dist.destroy_process_group()


def _uuid_worker(rank, world, port, q):
    """Confirm UUID broadcast is consistent across ranks under NCCL.

    Independent code path from Gloo because rank-0 → others broadcast
    uses NCCL's send/recv collective which has different device-buffer
    requirements (must live on a CUDA device).
    """
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    try:
        from tensormesh.distributed import DSparseMatrix
        from torch_sla import SparseTensor
        from torch_sla.distributed import DSparseTensor
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            from torch.distributed._tensor.device_mesh import init_device_mesh

        N = 8
        ii = torch.arange(N, dtype=torch.long)
        row = torch.cat([ii, ii[:-1], ii[1:]])
        col = torch.cat([ii, ii[1:], ii[:-1]])
        val = torch.cat([
            torch.full((N,), 2.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
        ])
        A = SparseTensor(val, row, col, (N, N))
        mesh = init_device_mesh("cuda", (world,))
        dst = DSparseTensor.partition(A, mesh, partition_method="metis").cuda()
        D = DSparseMatrix(dst)
        q.put((rank, "OK", D.partition_uuid))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:600]))
    finally:
        dist.destroy_process_group()


def _solve_worker(rank, world, port, q):
    """End-to-end distributed solve through the torch-sla free function.

    The whole point of the inheritance refactor was that
    ``solve(K_dist, b)`` from ``torch_sla`` should accept a DSparseMatrix
    without unwrapping. This test exercises that gate.
    """
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    try:
        from tensormesh.distributed import DSparseMatrix
        from torch_sla import SparseTensor, SolverConfig, solve
        from torch_sla.distributed import DSparseTensor, gather_owned_to_global
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            from torch.distributed._tensor.device_mesh import init_device_mesh

        # Make a small SPD tridiagonal so CG converges in a handful of iters.
        N = 32
        ii = torch.arange(N, dtype=torch.long)
        row = torch.cat([ii, ii[:-1], ii[1:]])
        col = torch.cat([ii, ii[1:], ii[:-1]])
        val = torch.cat([
            torch.full((N,), 2.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
            torch.full((N - 1,), -1.0, dtype=torch.float64),
        ])
        A_global = SparseTensor(val, row, col, (N, N))
        mesh = init_device_mesh("cuda", (world,))
        dst = DSparseTensor.partition(A_global, mesh, partition_method="metis").cuda()
        D = DSparseMatrix(dst)

        # Reference solve on rank 0 only (we just need a known x to test
        # the distributed result against).
        torch.manual_seed(123)
        x_ref = torch.randn(N, dtype=torch.float64, device=f"cuda:{rank}")
        b_global = (A_global.cuda()) @ x_ref

        owned = D.partition.owned_nodes.long().to(f"cuda:{rank}")
        b_owned = b_global[owned]

        # The free function should accept the subclass directly.
        with SolverConfig(method="cg", atol=1e-12, rtol=1e-10, maxiter=200):
            x_owned = solve(D, b_owned)

        x_global = gather_owned_to_global(owned, x_owned, N)
        rel = float(((x_global - x_ref).norm() / x_ref.norm()).item())
        q.put((rank, "OK", rel))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:1200]))
    finally:
        dist.destroy_process_group()


# ─── Test drivers ───────────────────────────────────────────────────


def _run(target, port, world=2):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    results = []
    while not q.empty():
        results.append(q.get())
    return results


def test_nccl_assembler_decorator_matches_single():
    r = _run(_matvec_worker, 35511)
    assert len(r) == 2, f"missing rank results: {r}"
    for rank, tag, *rest in r:
        assert tag == "OK", f"rank {rank}: {rest}"
        is_subclass, max_diff = rest
        assert is_subclass, "K_dist must be isinstance(DSparseTensor)"
        assert max_diff < 1e-8, f"rank {rank} matvec diff = {max_diff}"


def test_nccl_uuid_broadcast_consistent():
    r = _run(_uuid_worker, 35513)
    assert len(r) == 2
    for rank, tag, *_rest in r:
        assert tag == "OK", f"rank {rank}: {_rest}"
    uuids = {x[2] for x in r}
    assert len(uuids) == 1, f"UUID diverged across NCCL ranks: {uuids}"


def test_nccl_torch_sla_solve_accepts_dsparsematrix():
    """The point of the subclass refactor: torch_sla.solve(D, b)
    works without unwrapping ``D._t``."""
    r = _run(_solve_worker, 35515)
    assert len(r) == 2
    for rank, tag, *rest in r:
        assert tag == "OK", f"rank {rank}: {rest}"
        (rel,) = rest
        assert rel < 1e-6, f"rank {rank} rel error = {rel}"


# ─── Integration test: example file ─────────────────────────────────


def _poisson_example_worker(rank, world, port, q):
    """Exercise the published distributed Poisson example so it stays
    runnable as the API evolves."""
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    try:
        # Import the example as a module so any refactor of its
        # internals shows up as a test failure rather than silent
        # documentation rot.
        from examples.distributed import poisson_distributed_cuda as ex
        info = ex.run(rank, world, chara_length=0.05)
        q.put((rank, "OK", info))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:1500]))
    finally:
        dist.destroy_process_group()


def test_nccl_poisson_example_runs():
    """End-to-end test for ``examples/distributed/poisson_distributed_cuda.py``.

    Asserts FEM-level accuracy: the distributed Poisson solve on a
    mesh with chara_length=0.05 (N≈500 DOFs) converges to the
    analytical PoissonMultiFrequency reference within ~10% rel-err,
    matching the single-device baseline (~6%).

    Was a smoke test ("no NaN") when first checked in -- the loose
    bound was masking a real bug: ``PoissonMultiFrequency.__init__``
    samples its coefficient matrix from the local RNG, so ranks saw
    different source terms unless seeded; the example now does
    ``torch.manual_seed(0)`` before construction so every rank
    builds the same source. With the seed fix this test catches any
    future regression in the Condenser dispatch path numerically.
    """
    r = _run(_poisson_example_worker, 35517)
    assert len(r) == 2
    for rank, tag, *rest in r:
        assert tag == "OK", f"rank {rank}: {rest}"
        (info,) = rest
        rel = info["rel_err_vs_analytical"]
        assert rel == rel, f"rank {rank} rel_err is NaN"
        # FEM accuracy bound: chara_length=0.05 + PoissonMultiFrequency
        # gives single-device rel_err ~6%; allow 10% slack.
        assert rel < 0.10, (
            f"rank {rank} poisson example rel_err = {rel:.3e}; "
            "distributed solve has drifted from the single-device baseline"
        )

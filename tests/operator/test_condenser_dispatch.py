"""Condenser polymorphic dispatch: SparseMatrix vs DSparseMatrix path."""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_condenser_with_sparsematrix_single_device():
    """Sanity: SparseMatrix path still works after the layout_hash ->
    layout_signature switch."""
    from tensormesh import Mesh, Condenser
    from tensormesh.assemble import LaplaceElementAssembler, const_node_assembler

    mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
    K = LaplaceElementAssembler.from_mesh(mesh)()
    ConstLoad = const_node_assembler()
    f = ConstLoad.from_mesh(mesh)()

    cond = Condenser(mesh.boundary_mask)
    K_c, f_c = cond(K, f)
    assert K_c.shape[0] == K_c.shape[1]
    assert K_c.shape[0] == (~mesh.boundary_mask).sum().item()
    assert f_c.shape[0] == K_c.shape[0]

    # Cached layout_signature should hit on a second call with the same K
    K_c2, f_c2 = cond(K, f)
    assert K_c2.shape == K_c.shape


def _dsparse_condense_worker(rank, world, port, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from tensormesh import Mesh, Condenser
        from tensormesh.assemble import LaplaceElementAssembler, const_node_assembler
        from tensormesh.distributed import (
            DistributedMesh, distributed_element_assemble,
        )
        from tensormesh.distributed import DSparseMatrix

        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(
            mesh, num_partitions=world,
            devices=[torch.device("cpu")] * world,
        )

        dst = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2,
        )
        K_dist = DSparseMatrix(dst)

        ConstLoad = const_node_assembler()
        f_ref = ConstLoad.from_mesh(mesh)()

        cond = Condenser(mesh.boundary_mask)

        # The DSparseMatrix branch should emit a UserWarning about the
        # to_single() round-trip but still return (K_inner_dist, f_inner).
        with pytest.warns(UserWarning, match="DSparseMatrix"):
            K_inner, f_inner = cond(K_dist, f_ref)

        # Output type contract: K_inner should be a DSparseMatrix again
        is_dsparse = isinstance(K_inner, DSparseMatrix)
        # f_inner is a global single-device tensor in the current round-trip
        # implementation; we just check shapes.
        n_inner = (~mesh.boundary_mask).sum().item()
        q.put((rank, "OK", is_dsparse,
                K_inner.shape == (n_inner, n_inner),
                f_inner.shape[0] == n_inner))
    except Exception as e:
        import traceback
        q.put((rank, "ERR", str(e), traceback.format_exc()[:800]))
    finally:
        dist.destroy_process_group()


def test_condenser_dispatches_to_distributed_path_2procs():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_dsparse_condense_worker,
                          args=(r, 2, 31621, q)) for r in range(2)]
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
        is_dsparse, K_shape_ok, f_shape_ok = rest
        assert is_dsparse, f"rank {rank}: K_inner not DSparseMatrix"
        assert K_shape_ok and f_shape_ok, f"rank {rank}: shape mismatch"

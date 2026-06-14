"""Tests for distributed (multi-device) Galerkin assembly.

All tests use CPU fallback mode so they can run without multiple GPUs.
"""

import sys
sys.path.append("../..")

import torch
import pytest

from tensormesh import Mesh
from tensormesh.assemble import (
    LaplaceElementAssembler,
    MassElementAssembler,
    const_node_assembler,
)
from tensormesh.distributed import (
    DistributedMesh,
    distributed_element_assemble,
    distributed_element_assemble_to_sparse,
    distributed_node_assemble,
)


# ─── Helpers ────────────────────────────────────────────────────────

def _cpu_devices(n):
    return [torch.device('cpu')] * n


# ─── DistributedMesh ────────────────────────────────────────────────

class TestDistributedMesh:
    """Tests for DistributedMesh partitioning."""

    def test_partition_creates_submeshes(self):
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        assert dmesh.num_partitions == 2
        assert dmesh.n_global_points == mesh.n_points
        assert len(dmesh.submeshes) == 2

        # Each submesh should have orig_nid mapping
        for sub in dmesh.submeshes:
            if sub is not None:
                assert 'orig_nid' in list(sub.point_data.keys())
                orig_nid = sub.point_data['orig_nid']
                assert orig_nid.max() < mesh.n_points

    def test_partition_covers_all_points(self):
        """All global points should appear in at least one partition."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=3, devices=_cpu_devices(3))

        all_global_ids = set()
        for sub in dmesh.submeshes:
            if sub is not None:
                ids = sub.point_data['orig_nid'].tolist()
                all_global_ids.update(ids)

        assert all_global_ids == set(range(mesh.n_points))

    def test_repr(self):
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))
        r = repr(dmesh)
        assert "DistributedMesh" in r
        assert "partitions=2" in r


# ─── Element Assembly ───────────────────────────────────────────────

class TestDistributedElementAssembly:
    """Verify distributed element assembly matches single-device assembly."""

    def test_laplace_matvec_2_partitions(self):
        """K_dist @ x should match K_ref @ x for Laplace."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        # Reference: single-device assembly
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()

        # Distributed assembly → DSparseTensor
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_laplace_matvec_4_partitions(self):
        """Test with more partitions than 2."""
        mesh = Mesh.gen_rectangle(chara_length=0.15, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=4, devices=_cpu_devices(4))

        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_mass_matvec(self):
        """Mass matrix distributed assembly."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = MassElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble(
            MassElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_to_sparse_matrix(self):
        """distributed_element_assemble_to_sparse returns a SparseMatrix."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble_to_sparse(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        from tensormesh.sparse import SparseMatrix
        assert isinstance(K_dist, SparseMatrix)
        assert K_dist.shape == K_ref.shape

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8)

    def test_quad_mesh(self):
        """Test with quadrilateral mesh."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="quad")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=3)()
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=3
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"


# ─── Node Assembly ──────────────────────────────────────────────────

class TestDistributedNodeAssembly:
    """Verify distributed node assembly matches single-device assembly."""

    def test_const_node_assembler(self):
        """Constant load vector: distributed vs single."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        # const_node_assembler() returns a CLASS, not an instance
        ConstLoad = const_node_assembler()
        asm_ref = ConstLoad.from_mesh(mesh)
        f_ref = asm_ref()

        # Distributed
        f_dist = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)

        assert torch.allclose(f_ref.cpu(), f_dist, atol=1e-8), \
            f"Max diff: {(f_ref.cpu() - f_dist).abs().max().item():.2e}"


# ─── Distributed Solve (integration test) ───────────────────────────

class TestDistributedSolve:
    """End-to-end distributed solve test."""

    def test_poisson_solve(self):
        """Solve Poisson equation: -Δu = 1, u|∂Ω = 0, compare solutions."""
        from tensormesh import Condenser

        mesh = Mesh.gen_rectangle(chara_length=0.15, element_type="tri")
        boundary_mask = mesh.boundary_mask

        # --- Reference (single-device) ---
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        ConstLoad = const_node_assembler()
        f_ref = ConstLoad.from_mesh(mesh)()

        condenser = Condenser(boundary_mask)
        K_c, f_c = condenser(K_ref, f_ref)
        u_ref = condenser.recover(K_c.solve(f_c))

        # --- Distributed assembly → SparseMatrix solve ---
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))
        K_sparse = distributed_element_assemble_to_sparse(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )
        f_dist = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)

        condenser2 = Condenser(boundary_mask)
        K_c2, f_c2 = condenser2(K_sparse, f_dist)
        u_dist = condenser2.recover(K_c2.solve(f_c2))

        assert torch.allclose(u_ref, u_dist, atol=1e-6), \
            f"Max diff: {(u_ref - u_dist).abs().max().item():.2e}"


def _multiproc_assemble_worker(rank, world, port, q):
    """Per-rank worker for the multi-process distributed-assembly smoke test."""
    import os
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=world, devices=_cpu_devices(world))
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        # Reference assembly (every rank can do it; small mesh).
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()

        # Same global x on every rank.
        torch.manual_seed(0)
        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x

        # DSparseTensor matvec lives in Shard(0) space -- pass this
        # rank's owned slice, not the global vector.
        partition = K_dist._spec.placement.partition
        owned = partition.owned_nodes.long()
        x_owned = x[owned]
        y_dist = K_dist @ x_owned
        # Allgather across ranks
        idx_pad = torch.zeros(mesh.n_points, dtype=torch.long)
        idx_pad[: owned.numel()] = owned
        val_pad = torch.zeros(mesh.n_points, dtype=y_dist.dtype)
        val_pad[: y_dist.numel()] = y_dist
        sz_local = torch.tensor([owned.numel()], dtype=torch.long)
        all_sz = [torch.zeros_like(sz_local) for _ in range(world)]
        all_idx = [torch.zeros_like(idx_pad) for _ in range(world)]
        all_val = [torch.zeros_like(val_pad) for _ in range(world)]
        dist.all_gather(all_sz, sz_local)
        dist.all_gather(all_idx, idx_pad)
        dist.all_gather(all_val, val_pad)
        y_global = torch.zeros(mesh.n_points, dtype=y_dist.dtype)
        for sz, ix, vl in zip(all_sz, all_idx, all_val):
            n = int(sz.item())
            y_global[ix[:n]] = vl[:n]

        rel = (y_global - y_ref).abs().max() / y_ref.abs().max()
        q.put({"rank": rank, "rel_err": float(rel)})
    finally:
        dist.destroy_process_group()


def test_multiproc_distributed_assemble_matvec_2procs():
    """End-to-end multi-process: assemble via TensorMesh, partition via
    DSparseTensor, matvec across ranks, gather and compare to reference."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    import torch.multiprocessing as mp
    world = 2
    port = 29770
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_multiproc_assemble_worker,
                          args=(rank, world, port, q))
             for rank in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world, f"expected {world} rank results, got {len(results)}"
    for r in results:
        assert r["rel_err"] < 1e-8, (
            f"rank {r['rank']}: matvec rel err {r['rel_err']:.2e} too large")


def _multiproc_solve_worker(rank, world, port, q):
    """Per-rank worker: distributed assemble + distributed CG solve.

    End-to-end test of the TensorMesh -> torch-sla integration:

      1. TensorMesh assembles the **Mass** matrix on each rank's submesh
         (Mass is SPD and non-singular; Laplace stiffness has a constant
         null space which makes CG drift along the constant mode, a
         separate orthogonality issue out of scope here).
      2. The result is wrapped as a DSparseTensor partitioned across the
         live torch.distributed world.
      3. We pick a known x_ref, compute b = M @ x_ref via the
         single-process reference Mass matrix, then run distributed CG
         in Shard(0) space (each rank holds its owned slice of x and b),
         allgather the result and verify x_dist == x_ref.
    """
    import os
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from torch_sla.distributed.solve import cg_shard

        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=world, devices=_cpu_devices(world))
        M_dist = distributed_element_assemble(
            MassElementAssembler, dmesh, quadrature_order=2
        )
        N = M_dist.shape[0]

        # Known reference solution; same seed -> same x_ref on every rank.
        torch.manual_seed(123)
        x_ref = torch.randn(N, dtype=M_dist.dtype)

        # b = M @ x_ref via the single-process reference assembly
        # (clean RHS, isolates the distributed *solve* path).
        M_ref_single = MassElementAssembler.from_mesh(mesh)()
        b_global = M_ref_single @ x_ref

        # Slice b to this rank's owned rows.
        partition = M_dist._spec.placement.partition
        owned = partition.owned_nodes.long()
        b_owned = b_global[owned]

        # Distributed CG in Shard(0) space (raw tensor in, raw tensor out).
        x_owned = cg_shard(
            M_dist, b_owned,
            M_apply=lambda r: r,
            atol=1e-12, rtol=1e-10, maxiter=2000, verbose=False,
        )

        # Distributed residual.
        r_owned = b_owned - M_dist @ x_owned
        # Norm via all_reduce
        r_sq = (r_owned * r_owned).sum()
        b_sq = (b_owned * b_owned).sum()
        dist.all_reduce(r_sq); dist.all_reduce(b_sq)
        rel_residual = float((r_sq.sqrt() / b_sq.sqrt()).item())

        # Allgather x_owned -> x_global, compare to x_ref.
        idx_pad = torch.zeros(N, dtype=torch.long); idx_pad[:owned.numel()] = owned
        val_pad = torch.zeros(N, dtype=x_owned.dtype); val_pad[:x_owned.numel()] = x_owned
        sz = torch.tensor([owned.numel()], dtype=torch.long)
        all_sz = [torch.zeros_like(sz) for _ in range(world)]
        all_idx = [torch.zeros_like(idx_pad) for _ in range(world)]
        all_val = [torch.zeros_like(val_pad) for _ in range(world)]
        dist.all_gather(all_sz, sz)
        dist.all_gather(all_idx, idx_pad)
        dist.all_gather(all_val, val_pad)
        x_global = torch.zeros_like(x_ref)
        for s, i, v in zip(all_sz, all_idx, all_val):
            n = int(s.item())
            x_global[i[:n]] = v[:n]

        rel_to_xref = float(((x_global - x_ref).norm() / x_ref.norm()).item())

        q.put({
            "rank": rank,
            "rel_residual": rel_residual,
            "rel_to_xref": rel_to_xref,
        })
    finally:
        dist.destroy_process_group()


def test_multiproc_distributed_solve_2procs():
    """End-to-end: assemble + distributed CG solve on 2 ranks vs scipy ref."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    import torch.multiprocessing as mp
    world = 2
    port = 29772
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_multiproc_solve_worker,
                          args=(rank, world, port, q))
             for rank in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world, f"expected {world} rank results, got {len(results)}"
    for r in results:
        assert r["rel_residual"] < 1e-8, (
            f"rank {r['rank']}: distributed residual {r['rel_residual']:.2e} too high")
        assert r["rel_to_xref"] < 1e-6, (
            f"rank {r['rank']}: solution drift vs x_ref {r['rel_to_xref']:.2e}")


def test_multiproc_distributed_solve_4procs():
    """Same as 2-proc but world=4: more partitions, more halo exchange."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    import torch.multiprocessing as mp
    world = 4
    port = 29774
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_multiproc_solve_worker,
                          args=(rank, world, port, q))
             for rank in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world, f"expected {world} rank results, got {len(results)}"
    for r in results:
        assert r["rel_residual"] < 1e-8
        assert r["rel_to_xref"] < 1e-6


def _multiproc_poisson_dirichlet_worker(rank, world, port, q):
    """Distributed Poisson with Dirichlet BC via tensormesh.Condenser.

    Exercises the full physics path: assemble + condense out Dirichlet
    DOFs + distributed solve on the reduced system + recover full u.
    """
    import os
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from tensormesh import Condenser

        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        boundary_mask = mesh.boundary_mask

        # Single-process reference (every rank rebuilds; small mesh).
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        ConstLoad = const_node_assembler()
        f_ref = ConstLoad.from_mesh(mesh)()
        cond_ref = Condenser(boundary_mask)
        K_c_ref, f_c_ref = cond_ref(K_ref, f_ref)
        u_ref = cond_ref.recover(K_c_ref.solve(f_c_ref))

        # Distributed assembly -> SparseMatrix path (multi-rank). Gives
        # the same global SparseMatrix on every rank as in single-process.
        dmesh = DistributedMesh(mesh, num_partitions=world, devices=_cpu_devices(world))
        K_sparse = distributed_element_assemble_to_sparse(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )
        f_dist = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)
        cond_dist = Condenser(boundary_mask)
        K_c_dist, f_c_dist = cond_dist(K_sparse, f_dist)
        u_dist = cond_dist.recover(K_c_dist.solve(f_c_dist))

        diff = (u_ref - u_dist).abs().max().item()
        q.put({"rank": rank, "max_diff": diff})
    finally:
        dist.destroy_process_group()


def test_multiproc_poisson_dirichlet_2procs():
    """End-to-end Poisson with Dirichlet BC across 2 ranks vs single-process."""
    import torch.distributed as dist
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    import torch.multiprocessing as mp
    world = 2
    port = 29776
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_multiproc_poisson_dirichlet_worker,
                          args=(rank, world, port, q))
             for rank in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world
    for r in results:
        assert r["max_diff"] < 1e-6, (
            f"rank {r['rank']}: max u diff vs single-proc {r['max_diff']:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Distributed assembly for multi-GPU FEM computation.

Provides functions to assemble element matrices and node vectors in parallel
across multiple GPUs, then combine results for distributed solving via torch-sla.

Strategy:
  Phase 1 (sequential): create assemblers + pre-warm all lazy properties
           (CUDA lazy init and Transformation caching are NOT thread-safe)
  Phase 2 (parallel threads): run asm() on each device concurrently
           (pure computation, all lazy state already cached)
  Phase 3 (sequential): collect results and merge COO
"""

import threading
import torch
from typing import Type, Optional, Dict, List, Tuple

from ..assemble import ElementAssembler, NodeAssembler
from ..sparse import SparseMatrix

from .mesh import DistributedMesh

try:
    from torch_sla import DSparseTensor, SparseTensor
    HAS_DSPARSE = True
except ImportError:
    HAS_DSPARSE = False

import torch.distributed as dist


# ─── True distributed assembly: each rank assembles ONLY its submesh ──

def _compute_partition_ids(dmesh: "DistributedMesh") -> torch.Tensor:
    """Deterministic mapping global_node_id -> owning rank (lowest rank wins).

    For each rank ``r`` (lowest first), claim all nodes in submesh ``r``'s
    ``orig_nid`` that aren't yet claimed. Identical on every rank (no
    communication needed).
    """
    N_global = dmesh.n_global_points
    world = dmesh.num_partitions
    partition_ids = torch.full((N_global,), world, dtype=torch.long)
    for r in range(world):
        sub = dmesh.submeshes[r]
        if sub is None:
            continue
        orig = sub.point_data['orig_nid'].cpu()
        cur = partition_ids[orig]
        unset = cur > r
        partition_ids[orig[unset]] = r
    assert (partition_ids < world).all(), \
        f"some nodes unassigned to any rank"
    return partition_ids


def _all_to_all_sparse_coo(rows: torch.Tensor, cols: torch.Tensor,
                            vals: torch.Tensor, dest_ranks: torch.Tensor,
                            world: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-to-all exchange of sparse (row, col, val) triples.

    Each rank sends each triple to its ``dest_rank``. After exchange,
    every rank has all triples destined for it (across all senders).
    Works on gloo (CPU) and NCCL (CUDA); torch.distributed must be
    initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("torch.distributed must be initialised")

    # Sort triples by destination rank so we can pack per-dest segments.
    order = torch.argsort(dest_ranks)
    rows_s = rows[order].contiguous()
    cols_s = cols[order].contiguous()
    vals_s = vals[order].contiguous()
    dests_s = dest_ranks[order]

    # Per-destination send counts.
    send_counts = torch.bincount(dests_s, minlength=world)

    # Exchange counts so each rank knows how much to receive from each peer.
    recv_counts = torch.zeros(world, dtype=torch.long,
                               device=send_counts.device)
    dist.all_to_all_single(recv_counts, send_counts)

    send_split = send_counts.cpu().tolist()
    recv_split = recv_counts.cpu().tolist()
    total_recv = int(recv_counts.sum().item())

    # Triples may be on CUDA (NCCL) or CPU (gloo). Inputs must be
    # contiguous + same dtype on every rank.
    recv_rows = torch.empty(total_recv, dtype=rows_s.dtype, device=rows_s.device)
    recv_cols = torch.empty(total_recv, dtype=cols_s.dtype, device=cols_s.device)
    recv_vals = torch.empty(total_recv, dtype=vals_s.dtype, device=vals_s.device)

    dist.all_to_all_single(recv_rows, rows_s, recv_split, send_split)
    dist.all_to_all_single(recv_cols, cols_s, recv_split, send_split)
    dist.all_to_all_single(recv_vals, vals_s, recv_split, send_split)

    return recv_rows, recv_cols, recv_vals


def _assemble_my_submesh(
    assembler_cls: Type[ElementAssembler],
    dmesh: "DistributedMesh",
    rank: int,
    quadrature_order: int,
    project: str,
    assembler_kwargs: dict,
    call_kwargs: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble ONLY submesh[rank] -> (vals, row_global, col_global) on CPU.

    No threading -- this rank does its own (one) submesh and that's it.
    Returns empty triple if this rank has no submesh.
    """
    sub = dmesh.submeshes[rank]
    if sub is None:
        return (torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long))

    device = dmesh.devices[rank]
    orig_nid = sub.point_data['orig_nid'].clone()
    sub.to(device)

    asm = assembler_cls.from_mesh(
        sub, quadrature_order=quadrature_order,
        project=project, **assembler_kwargs,
    )
    _warmup_assembler(asm)

    K_local: SparseMatrix = asm(**call_kwargs)
    row_global = orig_nid[K_local.row.cpu()]
    col_global = orig_nid[K_local.col.cpu()]
    vals = K_local.edata.cpu()

    sub.cpu()
    return vals, row_global, col_global


def distributed_element_assemble_per_rank(
    assembler_cls: Type[ElementAssembler],
    dmesh: "DistributedMesh",
    rank: int,
    quadrature_order: int = 2,
    project: str = 'reduce',
    call_kwargs: Optional[dict] = None,
    **assembler_kwargs,
) -> "DSparseTensor":
    """**True** distributed assembly: each rank does ONLY its submesh.

    Pipeline (every rank in lockstep, torch.distributed must be init'd):

      1. Compute partition_ids deterministically from dmesh (lowest rank
         wins for shared boundary nodes). No communication.
      2. This rank assembles submesh[rank] -> local triples in *global*
         coords. No threading, no work for other ranks' submeshes.
      3. For each local triple, route it to the rank that owns its row.
         all_to_all_single exchange (NCCL on CUDA, gloo on CPU).
      4. Coalesce incoming triples (sum duplicate (row,col) pairs from
         elements assembled by different ranks that touch the same row).
      5. Discover halo columns: distinct columns in received triples that
         aren't owned by this rank.
      6. Build Partition + remap to local coords + DSparseTensor.from_sparse_local.

    Total compute is roughly the same as single-process assembly divided
    across ranks, instead of duplicated.

    Caller is responsible for initialising torch.distributed before
    calling. The (rank, world) is taken from dist.

    Returns a row-shard DSparseTensor with each rank holding only its
    owned-row contributions.
    """
    if not HAS_DSPARSE:
        raise ImportError(
            "torch-sla with DSparseTensor support is required.\n"
            "Install with: pip install torch-sla>=0.2.0"
        )
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "distributed_element_assemble_per_rank() requires "
            "torch.distributed.init_process_group() first."
        )
    if call_kwargs is None:
        call_kwargs = {}

    world = dist.get_world_size()
    assert dmesh.num_partitions == world, (
        f"DistributedMesh has {dmesh.num_partitions} partitions but "
        f"torch.distributed world_size is {world}")
    assert rank == dist.get_rank(), (
        f"rank arg {rank} != dist.get_rank() {dist.get_rank()}")

    N_global = dmesh.n_global_points

    # 1. Partition ids (deterministic from dmesh)
    partition_ids = _compute_partition_ids(dmesh)

    # 2. Assemble my submesh
    vals, g_row, g_col = _assemble_my_submesh(
        assembler_cls, dmesh, rank,
        quadrature_order=quadrature_order, project=project,
        assembler_kwargs=assembler_kwargs, call_kwargs=call_kwargs,
    )

    # 3. Route each triple to its row's owning rank
    dest = partition_ids[g_row]
    # Move to current device + cast to types all_to_all_single needs
    cur_device = vals.device
    if dist.get_backend() == "nccl":
        # NCCL needs CUDA tensors. The mesh assembly returns CPU triples;
        # promote here.
        cur_device = torch.device(f"cuda:{rank}")
        g_row = g_row.cuda(cur_device)
        g_col = g_col.cuda(cur_device)
        vals = vals.cuda(cur_device)
        dest = dest.cuda(cur_device)

    recv_rows, recv_cols, recv_vals = _all_to_all_sparse_coo(
        g_row.long(), g_col.long(), vals, dest.long(), world,
    )

    # 4. Coalesce duplicate (row, col) pairs from incoming triples.
    # Linearise into a single key for unique() + scatter_add.
    if recv_rows.numel() == 0:
        # This rank received nothing -- empty owned rows. Edge case for
        # very imbalanced partitions; build an empty SparseTensor.
        unique_rows = recv_rows.new_empty(0)
        unique_cols = recv_cols.new_empty(0)
        unique_vals = recv_vals.new_empty(0)
    else:
        keys = recv_rows * N_global + recv_cols
        uk, inv = torch.unique(keys, return_inverse=True)
        unique_vals = torch.zeros(uk.numel(), dtype=recv_vals.dtype,
                                   device=recv_vals.device)
        unique_vals.scatter_add_(0, inv, recv_vals)
        unique_rows = (uk // N_global).long()
        unique_cols = (uk % N_global).long()

    # Pull back to CPU for the partition / SparseTensor construction
    # (Partition struct lives on CPU; the local tensor can be cuda'd later
    # by the caller via D.cuda()).
    unique_rows_cpu = unique_rows.cpu()
    unique_cols_cpu = unique_cols.cpu()
    unique_vals_cpu = unique_vals.cpu()

    # 5. Build the Partition from partition_ids + the column adjacency
    # we see locally. ``build_partition`` does halo discovery off
    # (row, col, partition_ids) -- we use the coalesced unique pattern
    # as the local view, which after coalesce only references columns
    # this rank actually needs (either owned or halo).
    from torch_sla.partition import build_partition
    partition = build_partition(
        unique_rows_cpu, unique_cols_cpu, N_global, partition_ids, rank,
    )

    # 6. Map to local coords + build local SparseTensor + DSparseTensor
    g2l = partition.global_to_local
    local_rows = g2l[unique_rows_cpu]
    local_cols = g2l[unique_cols_cpu]
    num_local = int(partition.local_to_global.numel())
    local_st = SparseTensor(
        unique_vals_cpu, local_rows, local_cols, (num_local, num_local),
    )

    # Build a device_mesh of size=world on appropriate backend.
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        from torch.distributed._tensor.device_mesh import init_device_mesh
    backend_device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    device_mesh = init_device_mesh(backend_device, (world,))

    D = DSparseTensor.from_sparse_local(
        local_st, device_mesh, partition, global_shape=(N_global, N_global),
    )
    # If we're on NCCL, the user expects the matrix on this rank's GPU
    # (just like distributed_element_assemble's CUDA story). Push the
    # local tensor over once here so callers don't have to remember.
    if dist.get_backend() == "nccl":
        D = D.cuda()
    return D


def _build_dsparse(global_values, global_row, global_col, N, dmesh):
    """Wrap the merged global COO as a torch-sla DSparseTensor.

    Two paths:

    * ``torch.distributed`` is **initialised** -> partition the global
      ``SparseTensor`` across the live mesh; each rank gets its own
      shard. This is the real distributed-solve path.
    * Not initialised -> return a single-rank DSparseTensor that holds
      the whole matrix locally. Useful for single-process drivers
      (CPU multi-thread assembly + sequential solve) and for unit
      tests; matvec / solve still work via the standard API.

    Either way the caller gets a ``DSparseTensor`` whose ``@`` and
    ``solve`` methods compose with the rest of torch-sla.
    """
    A_global = SparseTensor(global_values, global_row, global_col, (N, N))
    coords = dmesh.global_mesh.points.cpu()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Build (or look up) a device mesh sized to the running world.
        try:
            # torch >= 2.2
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            # torch 2.0-2.1 keeps it under the private namespace
            from torch.distributed._tensor.device_mesh import init_device_mesh
        world = torch.distributed.get_world_size()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = init_device_mesh(device, (world,))
        return DSparseTensor.partition(
            A_global, mesh,
            partition_method="rcb", coords=coords,
        )
    # Single-process simulator: mesh=None gives rank=0, world=1 so the
    # DSparseTensor holds the entire matrix. The partition still uses
    # ``dmesh.num_partitions`` as a label for downstream inspection.
    return DSparseTensor.partition(
        A_global, mesh=None,
        partition_method="rcb", coords=coords,
    )


# ─── Warmup helpers ─────────────────────────────────────────────────

def _warmup_assembler(asm: ElementAssembler):
    """Force-evaluate all lazy cached properties in Transformation.

    Transformation stores shape_val, shape_grad, JxW etc. as lazy
    ``@property`` that compute-on-first-access and cache in _buffers.
    If these are first triggered inside a thread, PyTorch's internal
    lazy init (CUDA, vmap, etc.) can race.  Calling them here in the
    main thread makes the subsequent threaded asm() purely arithmetic.
    """
    for element_type in asm.element_types:
        trans = asm.transformation[element_type]
        # Access lazy properties to force caching
        _ = trans.shape_val
        _ = trans.shape_grad
        _ = trans.JxW


# ─── Core implementation ────────────────────────────────────────────

def _element_assemble_all(
    assembler_cls: Type[ElementAssembler],
    dmesh: DistributedMesh,
    quadrature_order: int,
    project: str,
    assembler_kwargs: dict,
    call_kwargs: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare assemblers sequentially, run assembly in parallel threads.

    Returns merged global COO (values, row, col) on CPU.
    """
    n = dmesh.num_partitions
    assemblers = []  # (asm, orig_nid_cpu, submesh) or None

    # ── Phase 1: sequential setup + warmup (main thread) ──
    for i in range(n):
        submesh = dmesh.submeshes[i]
        if submesh is None:
            assemblers.append(None)
            continue

        device = dmesh.devices[i]
        orig_nid = submesh.point_data['orig_nid'].clone()
        submesh.to(device)

        asm = assembler_cls.from_mesh(
            submesh, quadrature_order=quadrature_order,
            project=project, **assembler_kwargs,
        )
        _warmup_assembler(asm)
        assemblers.append((asm, orig_nid, submesh))

    # ── Phase 2: parallel assembly (threads) ──
    results: List[Optional[Tuple]] = [None] * n
    errors: List[Tuple[int, Exception]] = []

    def _worker(i):
        try:
            asm, orig_nid, submesh = assemblers[i]
            device = asm.device
            if device.type == 'cuda':
                torch.cuda.set_device(device)

            K_local: SparseMatrix = asm(**call_kwargs)

            global_row = orig_nid[K_local.row.cpu()]
            global_col = orig_nid[K_local.col.cpu()]
            values = K_local.edata.cpu()
            results[i] = (values, global_row, global_col)
        except Exception as e:
            errors.append((i, e))

    threads = []
    for i in range(n):
        if assemblers[i] is None:
            results[i] = (
                torch.tensor([], dtype=torch.float64),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )
            continue
        t = threading.Thread(target=_worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # ── Phase 3: collect & merge ──
    # Move submeshes back to CPU
    for item in assemblers:
        if item is not None:
            item[2].cpu()

    if errors:
        msgs = [f"Partition {pid}: {e}" for pid, e in errors]
        raise RuntimeError(
            f"Distributed assembly failed on {len(errors)} partition(s):\n"
            + "\n".join(msgs)
        )

    all_values, all_row, all_col = [], [], []
    for r in results:
        assert r is not None
        vals, row, col = r
        if vals.numel() > 0:
            all_values.append(vals)
            all_row.append(row)
            all_col.append(col)

    if not all_values:
        raise RuntimeError("All partitions produced empty assemblies")

    return torch.cat(all_values), torch.cat(all_row), torch.cat(all_col)


def _node_assemble_all(
    assembler_cls: Type[NodeAssembler],
    dmesh: DistributedMesh,
    quadrature_order: int,
    project: str,
    assembler_kwargs: dict,
    call_kwargs: dict,
) -> torch.Tensor:
    """Prepare node assemblers sequentially, run in parallel threads.

    Returns global vector on CPU.
    """
    n = dmesh.num_partitions
    assemblers = []      # (asm, orig_nid, submesh, local_call_kwargs) or None

    # ── Phase 1: sequential setup + warmup ──
    for i in range(n):
        submesh = dmesh.submeshes[i]
        if submesh is None:
            assemblers.append(None)
            continue

        device = dmesh.devices[i]
        orig_nid = submesh.point_data['orig_nid'].clone()
        submesh.to(device)

        asm = assembler_cls.from_mesh(
            submesh, quadrature_order=quadrature_order,
            project=project, **assembler_kwargs,
        )
        # Warmup lazy properties
        for et in asm.element_types:
            trans = asm.transformation[et]
            _ = trans.shape_val
            _ = trans.shape_grad
            _ = trans.JxW

        # Remap global point_data → local
        lkw = dict(call_kwargs)
        if 'point_data' in lkw and lkw['point_data'] is not None:
            local_pd = {}
            for k, v in lkw['point_data'].items():
                local_pd[k] = v[orig_nid].to(device)
            lkw['point_data'] = local_pd

        assemblers.append((asm, orig_nid, submesh, lkw))

    # ── Phase 2: parallel assembly ──
    node_results: List[Optional[torch.Tensor]] = [None] * n
    errors: List[Tuple[int, Exception]] = []

    def _worker(i):
        try:
            asm, _, submesh, lkw = assemblers[i]
            device = asm.device
            if device.type == 'cuda':
                torch.cuda.set_device(device)
            node_results[i] = asm(**lkw).cpu()
        except Exception as e:
            errors.append((i, e))

    threads = []
    for i in range(n):
        if assemblers[i] is None:
            continue
        t = threading.Thread(target=_worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Move submeshes back
    for item in assemblers:
        if item is not None:
            item[2].cpu()

    if errors:
        msgs = [f"Partition {pid}: {e}" for pid, e in errors]
        raise RuntimeError(
            f"Distributed node assembly failed on {len(errors)} partition(s):\n"
            + "\n".join(msgs)
        )

    # ── Phase 3: gather into global vector ──
    first_i = next(i for i in range(n) if node_results[i] is not None)
    first_result = node_results[first_i]
    local_n_points = assemblers[first_i][2].n_points
    dof_per_point = first_result.shape[0] // local_n_points

    N = dmesh.n_global_points
    f_global = torch.zeros(N * dof_per_point, dtype=first_result.dtype)

    for i in range(n):
        if assemblers[i] is None or node_results[i] is None:
            continue
        orig_nid = assemblers[i][1]
        f_local = node_results[i]

        if dof_per_point == 1:
            f_global.scatter_add_(0, orig_nid, f_local)
        else:
            f_local_2d = f_local.view(-1, dof_per_point)
            for h in range(dof_per_point):
                idx = orig_nid * dof_per_point + h
                f_global.scatter_add_(0, idx, f_local_2d[:, h])

    return f_global


# ─── Public API ─────────────────────────────────────────────────────

def distributed_element_assemble(
    assembler_cls: Type[ElementAssembler],
    dmesh: DistributedMesh,
    quadrature_order: int = 2,
    project: str = 'reduce',
    call_kwargs: Optional[dict] = None,
    **assembler_kwargs,
) -> "DSparseTensor":
    """Assemble element matrix in parallel across multiple devices.

    Assemblers are created sequentially (for CUDA thread-safety), then
    assembly computation runs in parallel threads on separate GPUs.

    Parameters
    ----------
    assembler_cls : Type[ElementAssembler]
        The assembler class (e.g., ``LaplaceElementAssembler``).
    dmesh : DistributedMesh
        Partitioned mesh with device assignments.
    quadrature_order : int, optional
        Quadrature order for integration. Default: 2.
    project : str, optional
        Projection method: ``'reduce'`` or ``'sparse'``. Default: ``'reduce'``.
    call_kwargs : dict, optional
        Extra keyword arguments passed to ``assembler.__call__()``
        (e.g., ``point_data``, ``scalar_data``).
    **assembler_kwargs
        Extra keyword arguments passed to ``assembler_cls.from_mesh()``.

    Returns
    -------
    DSparseTensor
        Distributed sparse matrix ready for distributed solve.
    """
    if not HAS_DSPARSE:
        raise ImportError(
            "torch-sla with DSparseTensor support is required.\n"
            "Install with: pip install torch-sla>=0.2.0"
        )

    if call_kwargs is None:
        call_kwargs = {}

    global_values, global_row, global_col = _element_assemble_all(
        assembler_cls, dmesh, quadrature_order, project, assembler_kwargs, call_kwargs,
    )
    N = dmesh.n_global_points

    if global_values.dim() > 1:
        global_values, global_row, global_col, N = _expand_block_coo(
            global_values, global_row, global_col, N
        )

    return _build_dsparse(global_values, global_row, global_col, N, dmesh)


def distributed_element_assemble_to_sparse(
    assembler_cls: Type[ElementAssembler],
    dmesh: DistributedMesh,
    quadrature_order: int = 2,
    project: str = 'reduce',
    call_kwargs: Optional[dict] = None,
    **assembler_kwargs,
) -> SparseMatrix:
    """Assemble element matrix in parallel, returning a global SparseMatrix.

    Same as :func:`distributed_element_assemble` but returns a standard
    :class:`~tensormesh.sparse.SparseMatrix` instead of torch-sla's
    ``DSparseTensor``.
    """
    if call_kwargs is None:
        call_kwargs = {}

    global_values, global_row, global_col = _element_assemble_all(
        assembler_cls, dmesh, quadrature_order, project, assembler_kwargs, call_kwargs,
    )
    N = dmesh.n_global_points

    if global_values.dim() > 1:
        return SparseMatrix.from_block_coo(
            global_values, global_row, global_col, shape=(N, N)
        )

    return SparseMatrix(global_values, global_row, global_col, shape=(N, N))


def distributed_node_assemble(
    assembler_cls: Type[NodeAssembler],
    dmesh: DistributedMesh,
    quadrature_order: int = 2,
    project: str = 'reduce',
    point_data: Optional[Dict[str, torch.Tensor]] = None,
    call_kwargs: Optional[dict] = None,
    **assembler_kwargs,
) -> torch.Tensor:
    """Assemble node vector (RHS) in parallel across multiple devices."""
    if call_kwargs is None:
        call_kwargs = {}
    if point_data is not None:
        call_kwargs['point_data'] = point_data

    return _node_assemble_all(
        assembler_cls, dmesh, quadrature_order, project, assembler_kwargs, call_kwargs,
    )


def _expand_block_coo(
    values: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    n_points: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expand block COO [nnz, dof, dof] to scalar COO."""
    nnz = values.shape[0]
    dof = values.shape[1]
    N = n_points * dof

    block_offsets_i = torch.arange(dof, device=values.device)
    block_offsets_j = torch.arange(dof, device=values.device)

    row_expanded = (row[:, None, None] * dof + block_offsets_i[None, :, None]).expand(nnz, dof, dof)
    col_expanded = (col[:, None, None] * dof + block_offsets_j[None, None, :]).expand(nnz, dof, dof)

    return values.reshape(-1), row_expanded.reshape(-1), col_expanded.reshape(-1), N

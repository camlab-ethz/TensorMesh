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
        from torch.distributed.device_mesh import init_device_mesh
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

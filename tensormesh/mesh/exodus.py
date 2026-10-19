r"""Exodus II mesh I/O — element blocks, node sets and side sets.

Exodus II (``.e`` / ``.exo``) is the mesh format of MOOSE, Cubit, Sierra and
the SEACAS tools. Besides the geometry it carries the named regions that
boundary conditions are written against:

* **element blocks** → ``cell_data["block_id"][element_type]``;
* **node sets** → :attr:`tensormesh.Mesh.point_sets` (point indices);
* **side sets** → :attr:`tensormesh.Mesh.side_sets` (``(cell, local facet)``
  pairs per element type).

The file is read and written directly with :mod:`netCDF4` (the backend that
``meshio`` uses for Exodus as well). ``meshio``'s own Exodus reader is not
used because it drops side sets and element-block ids, rejects type names
MOOSE writes (``TETRA4``, ``EDGE2``, ``PYRAMID5``, ``WEDGE18``), and keeps
the Exodus numbering of ``HEX27``, which differs from Gmsh/VTK (see
``_EXODUS_EDGE_FACE_NODES``).

Node ordering is converted **geometrically**: every Exodus node is located
on the TensorMesh reference element (corner vertices through
:meth:`tensormesh.Element.reorder`, higher-order nodes as the centroid of the
corners they are defined by) and matched to the TensorMesh reference nodes.
Side numbers are matched to TensorMesh facets the same way, so neither table
depends on TensorMesh's internal node or facet numbering.

Only the mesh is read: nodal/element variables and time steps are ignored.
"""
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import meshio

from .. import element as E
from .mesh import Mesh, local_facets

__all__ = ["read_exodus", "write_exodus", "is_exodus", "EXODUS_EXTENSIONS"]

#: File extensions recognised as Exodus II by :meth:`tensormesh.Mesh.read` /
#: :meth:`tensormesh.Mesh.save`.
EXODUS_EXTENSIONS = (".e", ".exo", ".ex2", ".exii", ".gen", ".g")

# Exodus ``elem_type`` (upper-cased) -> TensorMesh element type string.
_EXODUS_TO_TENSORMESH = {
    "EDGE": "line", "EDGE2": "line", "BAR": "line", "BAR2": "line",
    "BEAM": "line", "BEAM2": "line", "TRUSS": "line", "TRUSS2": "line",
    "EDGE3": "line3", "BAR3": "line3", "BEAM3": "line3", "TRUSS3": "line3",
    "TRI": "triangle", "TRI3": "triangle", "TRIANGLE": "triangle",
    "TRI6": "triangle6",
    "QUAD": "quad", "QUAD4": "quad",
    "QUAD9": "quad9",
    "TET": "tetra", "TET4": "tetra", "TETRA": "tetra", "TETRA4": "tetra",
    "TET10": "tetra10", "TETRA10": "tetra10",
    "HEX": "hexahedron", "HEX8": "hexahedron", "HEXAHEDRON": "hexahedron",
    "HEX27": "hexahedron27",
    "WEDGE": "wedge", "WEDGE6": "wedge",
    "WEDGE18": "wedge18",
    "PYRAMID": "pyramid", "PYRAMID5": "pyramid",
    "PYRAMID14": "pyramid14",
}

# TensorMesh element type string -> Exodus ``elem_type`` written by ``write_exodus``.
_TENSORMESH_TO_EXODUS = {
    "line": "EDGE2", "line3": "EDGE3",
    "triangle": "TRI3", "triangle6": "TRI6",
    "quad": "QUAD4", "quad9": "QUAD9",
    "tetra": "TETRA4", "tetra10": "TETRA10",
    "hexahedron": "HEX8", "hexahedron27": "HEX27",
    "wedge": "WEDGE6", "wedge18": "WEDGE18",
    "pyramid": "PYRAMID5", "pyramid14": "PYRAMID14",
}

# Higher-order Exodus nodes, in file order after the corners, each given by
# the (0-based) Exodus corners whose centroid it sits at. From the Exodus II
# manual; checked against MOOSE/libMesh output.
_EXODUS_EDGE_FACE_NODES = {
    "line3": [(0, 1)],
    "triangle6": [(0, 1), (1, 2), (2, 0)],
    "quad9": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1, 2, 3)],
    "tetra10": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "hexahedron27": [
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 1, 2, 3, 4, 5, 6, 7),                   # 20: volume centroid
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 3, 7, 4),   # 21-23: z-, z+, x- faces
        (1, 2, 6, 5), (0, 1, 5, 4), (2, 3, 7, 6),   # 24-26: x+, y-, y+ faces
    ],
    "wedge18": [
        (0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (5, 3),
        (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5),
    ],
    "pyramid14": [
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4),
        (0, 1, 2, 3),
    ],
}

# Exodus side number (1-based) -> the (0-based) Exodus corners of that side.
_EXODUS_SIDES = {
    "line": {1: (0,), 2: (1,)},
    "triangle": {1: (0, 1), 2: (1, 2), 3: (2, 0)},
    "quad": {1: (0, 1), 2: (1, 2), 3: (2, 3), 4: (3, 0)},
    "tetra": {1: (0, 1, 3), 2: (1, 2, 3), 3: (0, 3, 2), 4: (0, 2, 1)},
    "hexahedron": {
        1: (0, 1, 5, 4), 2: (1, 2, 6, 5), 3: (2, 3, 7, 6),
        4: (0, 4, 7, 3), 5: (0, 3, 2, 1), 6: (4, 5, 6, 7),
    },
    "wedge": {1: (0, 1, 4, 3), 2: (1, 2, 5, 4), 3: (0, 3, 5, 2), 4: (0, 2, 1), 5: (3, 4, 5)},
    "pyramid": {1: (0, 1, 4), 2: (1, 2, 4), 3: (2, 3, 4), 4: (3, 0, 4), 5: (0, 3, 2, 1)},
}

_TOL = 1e-8


def is_exodus(file_name: str, file_format: Optional[str] = None) -> bool:
    """Whether ``file_name`` / ``file_format`` designates an Exodus II file."""
    if file_format is not None:
        return file_format.lower() in ("exodus", "exodusii", "exodus2")
    return str(file_name).lower().endswith(EXODUS_EXTENSIONS)


def _shape(element_type: str) -> str:
    """Linear element type string of the shape of ``element_type``."""
    return {
        E.Line: "line", E.Triangle: "triangle", E.Quadrilateral: "quad",
        E.Tetrahedron: "tetra", E.Hexahedron: "hexahedron", E.Prism: "wedge",
        E.Pyramid: "pyramid",
    }[E.element_type2element(element_type)]


def _require_netcdf4():
    try:
        import netCDF4
    except ImportError as err:
        raise ImportError(
            "Exodus II I/O requires the netCDF4 package: "
            "pip install netCDF4  (or pip install \"tensormesh-fem[exodus]\")"
        ) from err
    return netCDF4


@lru_cache(maxsize=None)
def _reference_nodes(element_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference coordinates of the Exodus nodes and of the TensorMesh nodes.

    Returns ``(exodus_ref [n, D], tensormesh_ref [n, D])``, both in float64.
    """
    if _shape(element_type) == "line":
        # TensorMesh has no Line basis table; its ordering is the Gmsh one:
        # both ends, then the interior nodes — identical to Exodus.
        n = E.element_type2order[element_type] + 1
        ref = torch.tensor([[0.0], [1.0]] + [[0.5]] * (n - 2), dtype=torch.float64)
        return ref, ref
    element = E.element_type2element(element_type)
    order = E.element_type2order[element_type]
    linear = element.get_basis(1, torch.float64)                   # [n_vertex, D]
    n_vertex = linear.shape[0]
    # Exodus numbers the corners like Gmsh/VTK; reorder() maps them to TensorMesh.
    internal_to_exodus = element.get_gmsh_permutation(n_vertex)
    corners = torch.empty_like(linear)
    corners[internal_to_exodus] = linear
    extra = [corners[list(c)].mean(0) for c in _EXODUS_EDGE_FACE_NODES.get(element_type, [])]
    exodus_ref = torch.cat([corners] + [x[None] for x in extra]) if extra else corners
    return exodus_ref, element.get_basis(order, torch.float64)


@lru_cache(maxsize=None)
def _exodus_to_tensormesh_permutation(element_type: str) -> torch.Tensor:
    """``perm`` such that ``tensormesh_conn = exodus_conn[:, perm]``."""
    exodus_ref, tensormesh_ref = _reference_nodes(element_type)
    if exodus_ref.shape != tensormesh_ref.shape:
        raise NotImplementedError(
            f"no Exodus node numbering for TensorMesh element type '{element_type}'"
        )
    dist = torch.cdist(tensormesh_ref, exodus_ref)
    perm = dist.argmin(dim=1)
    assert bool((dist.min(dim=1).values < _TOL).all()) and perm.unique().numel() == perm.numel(), \
        f"Exodus and TensorMesh reference nodes of '{element_type}' do not match"
    return perm


@lru_cache(maxsize=None)
def _exodus_side_to_facet(element_type: str) -> torch.Tensor:
    """``table[s]`` is the TensorMesh local facet of Exodus side ``s`` (1-based)."""
    exodus_ref, tensormesh_ref = _reference_nodes(element_type)
    sides = _EXODUS_SIDES[_shape(element_type)]
    facets = local_facets(element_type)
    table = torch.full((max(sides) + 1,), -1, dtype=torch.long)
    for side, corners in sides.items():
        side_points = exodus_ref[list(corners)]                    # [n_corner, D]
        hits = [
            f for f, nodes in enumerate(facets)
            if bool((torch.cdist(side_points, tensormesh_ref[nodes]).min(dim=1).values < _TOL).all())
        ]
        assert len(hits) == 1, f"Exodus side {side} of '{element_type}' matches facets {hits}"
        table[side] = hits[0]
    return table


def _names(nc, variable: str, ids: np.ndarray) -> List[str]:
    """Entity names, falling back to the entity id for unnamed entities."""
    names = [""] * len(ids)
    if variable in nc.variables:
        rows = np.asarray(nc.variables[variable][:], dtype="S1")
        names = [b"".join(row).split(b"\0", 1)[0].decode().strip() for row in rows]
    return [name if name else str(int(i)) for name, i in zip(names, ids)]


def _ids(nc, prefix: str, count: int) -> np.ndarray:
    key = f"{prefix}_prop1"
    if key in nc.variables:
        return np.asarray(nc.variables[key][:]).astype(np.int64)
    return np.arange(1, count + 1, dtype=np.int64)


def _dim(nc, name: str) -> int:
    return len(nc.dimensions[name]) if name in nc.dimensions else 0


def read_exodus(file_name: str) -> Mesh:
    r"""Read an Exodus II mesh with its element blocks, node sets and side sets.

    Every Exodus element block becomes cells of the corresponding TensorMesh
    element type, with the connectivity converted to TensorMesh node
    ordering; blocks of one element type are concatenated in file order and
    their Exodus block ids are kept in ``cell_data["block_id"][element_type]``.
    Node sets land in :attr:`~tensormesh.Mesh.point_sets` and side sets in
    :attr:`~tensormesh.Mesh.side_sets`, keyed by their Exodus name (or by
    their id, as a string, when unnamed).

    Parameters
    ----------
    file_name : str
        Path to the Exodus II file.

    Returns
    -------
    tensormesh.Mesh
        The mesh, already in TensorMesh node ordering.

    Raises
    ------
    NotImplementedError
        For element types without a TensorMesh Lagrange equivalent
        (serendipity ``QUAD8`` / ``HEX20`` / ``WEDGE15`` / ``PYRAMID13``,
        bubble-enriched ``TRI7`` / ``TET14``, shells, polyhedra).
    """
    netCDF4 = _require_netcdf4()
    with netCDF4.Dataset(file_name) as nc:
        nc.set_auto_mask(False)
        var = nc.variables
        n_dim = _dim(nc, "num_dim")
        if "coord" in var:
            points = np.asarray(var["coord"][:], dtype=np.float64).T
        else:
            points = np.stack([np.asarray(var[f"coord{a}"][:], dtype=np.float64)
                               for a in "xyz"[:n_dim]], axis=1)

        # ---- element blocks -> cells grouped by TensorMesh element type ----
        n_block = _dim(nc, "num_el_blk")
        block_ids = _ids(nc, "eb", n_block)
        conn: Dict[str, List[np.ndarray]] = {}
        block_id: Dict[str, List[np.ndarray]] = {}
        owner: List[Tuple[str, np.ndarray]] = []    # per block: (type, local cell indices)
        for b in range(n_block):
            key = f"connect{b + 1}"
            if key not in var:                        # empty block
                continue
            exodus_type = str(var[key].getncattr("elem_type")).strip().upper()
            if exodus_type not in _EXODUS_TO_TENSORMESH:
                raise NotImplementedError(
                    f"Exodus element type '{exodus_type}' (block {int(block_ids[b])}) has no "
                    f"TensorMesh equivalent; supported: {', '.join(_TENSORMESH_TO_EXODUS.values())}"
                )
            element_type = _EXODUS_TO_TENSORMESH[exodus_type]
            data = np.asarray(var[key][:], dtype=np.int64) - 1
            perm = _exodus_to_tensormesh_permutation(element_type).numpy()
            if data.shape[1] != perm.shape[0]:
                raise ValueError(f"block {int(block_ids[b])}: '{exodus_type}' cells have "
                                 f"{data.shape[1]} nodes, expected {perm.shape[0]}")
            offset = sum(len(c) for c in conn.get(element_type, []))
            conn.setdefault(element_type, []).append(data[:, perm])
            block_id.setdefault(element_type, []).append(np.full(len(data), block_ids[b]))
            owner.append((element_type, offset + np.arange(len(data))))
        if not conn:
            raise ValueError(f"{file_name} contains no elements")

        # Exodus element ids run through the blocks in order.
        types = list(conn)
        elem_type = np.concatenate([np.full(len(i), types.index(t)) for t, i in owner])
        elem_local = np.concatenate([i for _, i in owner])

        # ---- node sets ----
        n_node_set = _dim(nc, "num_node_sets")
        point_sets = {}
        for s, name in enumerate(_names(nc, "ns_names", _ids(nc, "ns", n_node_set))):
            key = f"node_ns{s + 1}"
            nodes = np.asarray(var[key][:], dtype=np.int64) - 1 if key in var else np.zeros(0, np.int64)
            point_sets[name] = np.unique(nodes)

        # ---- side sets ----
        n_side_set = _dim(nc, "num_side_sets")
        side_sets: Dict[str, Dict[str, torch.Tensor]] = {}
        for s, name in enumerate(_names(nc, "ss_names", _ids(nc, "ss", n_side_set))):
            if f"elem_ss{s + 1}" in var:
                elems = np.asarray(var[f"elem_ss{s + 1}"][:], dtype=np.int64) - 1
                sides = np.asarray(var[f"side_ss{s + 1}"][:], dtype=np.int64)
            else:
                elems = sides = np.zeros(0, np.int64)
            side_sets[name] = {}
            for t, element_type in enumerate(types):
                sel = elem_type[elems] == t
                if not sel.any():
                    continue
                table = _exodus_side_to_facet(element_type)
                side = torch.from_numpy(sides[sel])
                if bool(((side < 1) | (side >= len(table))).any()):
                    raise ValueError(f"side set '{name}': invalid side number for '{element_type}'")
                side_sets[name][element_type] = torch.stack(
                    [torch.from_numpy(elem_local[elems[sel]]), table[side]], dim=1)

    m = meshio.Mesh(
        points=points,
        cells=[(t, np.concatenate(conn[t])) for t in types],
        cell_data={"block_id": [np.concatenate(block_id[t]) for t in types]},
        point_sets=point_sets,
    )
    mesh = Mesh(m)
    for name, facets in side_sets.items():
        mesh.register_side_set(name, facets)
    return mesh


def write_exodus(mesh: Mesh, file_name: str, title: Optional[str] = None) -> None:
    r"""Write ``mesh`` as an Exodus II file, with node sets and side sets.

    Each distinct ``(element type, block_id)`` pair becomes one element block
    (one block per element type when ``cell_data["block_id"]`` is absent).
    :attr:`~tensormesh.Mesh.point_sets` are written as node sets and
    :attr:`~tensormesh.Mesh.side_sets` as side sets. Point, cell and field
    data are not written.

    Parameters
    ----------
    mesh : tensormesh.Mesh
        The mesh to write.
    file_name : str
        Output path.
    title : str, optional
        Exodus database title; defaults to ``file_name``.
    """
    netCDF4 = _require_netcdf4()
    points = mesh.points.detach().cpu().double().numpy()
    n_point, n_dim = points.shape
    len_name = 256

    blocks = []           # (exodus type, block id, exodus connectivity, local cell indices)
    ids_used = set()
    for element_type, cells in mesh.cells.items():
        if element_type not in _TENSORMESH_TO_EXODUS:
            raise NotImplementedError(f"cannot write '{element_type}' cells to Exodus")
        perm = _exodus_to_tensormesh_permutation(element_type)
        cells = cells.detach().cpu()[:, torch.argsort(perm)].numpy() + 1
        ids = None
        if "block_id" in mesh.cell_data and element_type in mesh.cell_data["block_id"]:
            ids = mesh.cell_data["block_id"][element_type].detach().cpu().numpy().astype(np.int64)
        for b in (np.unique(ids) if ids is not None else [None]):
            local = np.arange(len(cells)) if b is None else np.flatnonzero(ids == b)
            blocks.append([_TENSORMESH_TO_EXODUS[element_type], b, cells[local], element_type, local])
            if b is not None:
                ids_used.add(int(b))
    next_id = max(ids_used, default=0) + 1
    for block in blocks:                      # number the blocks that carry no id
        if block[1] is None:
            block[1], next_id = next_id, next_id + 1

    # Exodus element id of every (element type, local cell index).
    global_id: Dict[str, np.ndarray] = {t: np.zeros(len(c), np.int64) for t, c in mesh.cells.items()}
    start = 1
    for _, _, _, element_type, local in blocks:
        global_id[element_type][local] = start + np.arange(len(local))
        start += len(local)

    def put_names(name, dim, names):
        chars = b"".join(n.encode()[:len_name - 1].ljust(len_name, b"\0") for n in names)
        nc.createVariable(name, "S1", (dim, "len_name"))[:] = \
            np.frombuffer(chars, dtype="S1").reshape(len(names), len_name)

    with netCDF4.Dataset(file_name, "w", format="NETCDF3_64BIT_OFFSET") as nc, \
            warnings.catch_warnings():
        # netCDF4 <= 1.7 reshapes 2D arrays in place on write, which NumPy 2.5 deprecates
        warnings.filterwarnings("ignore", "Setting the shape on a NumPy array", DeprecationWarning)
        nc.setncatts({
            "api_version": np.float32(8.11), "version": np.float32(8.11),
            "floating_point_word_size": np.int32(8), "file_size": np.int32(1),
            "maximum_name_length": np.int32(32), "int64_status": np.int32(0),
            "title": title if title is not None else str(file_name),
        })
        nc.createDimension("len_name", len_name)
        nc.createDimension("time_step", None)
        nc.createDimension("num_dim", n_dim)
        nc.createDimension("num_nodes", n_point)
        nc.createDimension("num_elem", start - 1)
        nc.createDimension("num_el_blk", len(blocks))
        nc.createVariable("time_whole", "f8", ("time_step",))
        put_names("coor_names", "num_dim", list("XYZ"[:n_dim]))
        for a, axis in enumerate("xyz"[:n_dim]):
            nc.createVariable(f"coord{axis}", "f8", ("num_nodes",))[:] = points[:, a]

        nc.createVariable("eb_status", "i4", ("num_el_blk",))[:] = 1
        prop = nc.createVariable("eb_prop1", "i4", ("num_el_blk",))
        prop.setncattr("name", "ID")
        prop[:] = [b[1] for b in blocks]
        put_names("eb_names", "num_el_blk", [""] * len(blocks))
        for i, (exodus_type, _, cells, _, _) in enumerate(blocks, start=1):
            nc.createDimension(f"num_el_in_blk{i}", len(cells))
            nc.createDimension(f"num_nod_per_el{i}", cells.shape[1])
            v = nc.createVariable(f"connect{i}", "i4", (f"num_el_in_blk{i}", f"num_nod_per_el{i}"))
            v.setncattr("elem_type", exodus_type)
            v[:] = cells

        if len(mesh.point_sets) > 0:
            names = list(mesh.point_sets.keys())
            nc.createDimension("num_node_sets", len(names))
            nc.createVariable("ns_status", "i4", ("num_node_sets",))[:] = 1
            prop = nc.createVariable("ns_prop1", "i4", ("num_node_sets",))
            prop.setncattr("name", "ID")
            prop[:] = np.arange(1, len(names) + 1)
            put_names("ns_names", "num_node_sets", names)
            for i, name in enumerate(names, start=1):
                nodes = mesh.point_sets[name].detach().cpu().numpy() + 1
                if len(nodes) == 0:
                    continue
                nc.createDimension(f"num_nod_ns{i}", len(nodes))
                nc.createVariable(f"node_ns{i}", "i4", (f"num_nod_ns{i}",))[:] = nodes

        if len(mesh.side_sets) > 0:
            names = list(mesh.side_sets.keys())
            nc.createDimension("num_side_sets", len(names))
            nc.createVariable("ss_status", "i4", ("num_side_sets",))[:] = 1
            prop = nc.createVariable("ss_prop1", "i4", ("num_side_sets",))
            prop.setncattr("name", "ID")
            prop[:] = np.arange(1, len(names) + 1)
            put_names("ss_names", "num_side_sets", names)
            for i, name in enumerate(names, start=1):
                elems, sides = [], []
                for element_type, pairs in mesh.side_sets[name].items():
                    pairs = pairs.detach().cpu()
                    facet_to_side = torch.argsort(_exodus_side_to_facet(element_type)[1:]) + 1
                    elems.append(global_id[element_type][pairs[:, 0].numpy()])
                    sides.append(facet_to_side[pairs[:, 1]].numpy())
                if not elems or sum(len(e) for e in elems) == 0:
                    continue
                elems, sides = np.concatenate(elems), np.concatenate(sides)
                order = np.argsort(elems, kind="stable")
                nc.createDimension(f"num_side_ss{i}", len(elems))
                nc.createVariable(f"elem_ss{i}", "i4", (f"num_side_ss{i}",))[:] = elems[order]
                nc.createVariable(f"side_ss{i}", "i4", (f"num_side_ss{i}",))[:] = sides[order]

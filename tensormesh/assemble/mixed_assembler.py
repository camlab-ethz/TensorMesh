r"""Mixed (multi-field / block) assembly over several Lagrange spaces.

:class:`MixedElementAssembler` assembles a bilinear form that couples
several Lagrange fields living on the *same* mesh — e.g. Taylor–Hood
P2–P1 for Stokes — into one block :class:`~tensormesh.sparse.SparseMatrix`,
without any manual block/offset bookkeeping. Each field is declared with
a :class:`Field` (trial/test argument names, polynomial order, number of
components); ``forward`` is then written once, as the scalar integrand of
the full bilinear form:

.. code-block:: python

    class StokesAssembler(MixedElementAssembler):
        fields = [
            Field(trial="u", test="v", order=2, components=2),
            Field(trial="p", test="q", order=1),
        ]

        def __post_init__(self, mu=1.0):
            self.mu = mu

        def forward(self, gradu, p, gradv, q):
            return self.mu * (gradu * gradv).sum() \
                 - p * gradv.diagonal().sum() \
                 - q * gradu.diagonal().sum()

    mesh = Mesh.gen_rectangle(order=2)            # triangle6 (P2) mesh
    asm  = StokesAssembler.from_mesh(mesh, mu=1.0)
    K    = asm()                                  # the whole saddle-point system
    lay  = asm.layout                             # block-DOF helpers

Conventions
-----------
* **Trial arguments index columns, test arguments index rows** (the
  standard :math:`a(u, v)` convention). Note this is the opposite letter
  convention from :class:`~tensormesh.ElementAssembler`, where
  the ``u`` argument is the test (row) function — here each role is
  explicit in the :class:`Field` declaration, so nothing is implicit.
* A field with ``components == 1`` passes its value as a 0-d scalar and
  its gradient as ``[D]`` (exactly like ``ElementAssembler``); a field
  with ``components == c > 1`` passes its value as ``[c]`` and its
  gradient (Jacobian) as ``[c, D]``.
* The integrand must be **bilinear**: linear in the trial tuple and in
  the test tuple. Block :math:`(\alpha, \beta)` is extracted by
  evaluating ``forward`` with one-hot basis functions for trial field
  :math:`\alpha` and test field :math:`\beta` and zeros for every other
  field, which is only valid for bilinear integrands. A constant
  (field-independent) term is detected and rejected; a term that is
  linear in only one side is *not* detectable — avoid it.
* ``point_data`` entries (including the coordinate ``x``) are
  interpolated with the **mesh-order** basis; ``field_data`` entries
  (values living on a *field's* DOFs, e.g. a previous Picard iterate)
  are interpolated with that field's own basis.
* The sparsity pattern contains every block whose trial *and* test
  arguments appear in the ``forward`` signature. Changing the signature
  (or assembling with a different ``func=``) therefore changes the
  pattern — build a fresh :class:`~tensormesh.Condenser` in
  that case.

Field orders are decoupled from the mesh order (generalized pairs, e.g.
P3–P2 Taylor–Hood, or quadratic fields on a linear gmsh import): fields
of order 1 or the mesh order carry their DOFs on mesh nodes (corner
nodes come first in every TensorMesh connectivity); any other order gets
a **topological DOF map** (:func:`~tensormesh.assemble.topology.lagrange_dofmap`)
with vertex, oriented-edge and cell-interior carriers. The geometry map
always stays isoparametric with the mesh order, so sub- and
super-parametric fields remain correct on curved elements (mind that a
low-order geometry on a *curved* domain limits the attainable
convergence rate of higher-order fields near the boundary). 3D spaces
whose nodes sit on faces (e.g. P3 tetrahedra) await the face-orientation
layer and raise ``NotImplementedError``.
"""
import inspect
import math
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .projector import ReduceProjector
from .topology import build_edges, lagrange_boundary_mask, lagrange_dofmap
from ..nn import BufferDict
from ..element import (
    Transformation,
    element_type2dimension,
    element_type2element,
    element_type2order,
)
from ..sparse import SparseMatrix
from ..mesh import Mesh
from ..vmap import vmap

__all__ = ["Field", "BlockLayout", "MixedElementAssembler"]


class Field:
    r"""Declaration of one Lagrange field of a mixed bilinear form.

    Parameters
    ----------
    trial : str
        Name of the trial-function argument in ``forward``; its gradient
        is available as ``"grad" + trial``. Trial arguments index matrix
        **columns**.
    test : str
        Name of the test-function argument in ``forward``; its gradient
        is available as ``"grad" + test``. Test arguments index matrix
        **rows**.
    order : int, optional
        Polynomial order of the field's Lagrange space — independent of
        the mesh order. Orders ``1`` and the mesh order carry their DOFs
        on mesh nodes; any other order uses a topological DOF map (see
        the module docstring for the 3D face-DOF limitation). Default ``1``.
    components : int, optional
        Number of vector components :math:`c`. With ``c == 1`` the field
        is scalar (value ``[]``, gradient ``[D]``); with ``c > 1`` the
        value is ``[c]`` and the gradient ``[c, D]``. Default ``1``.

    Examples
    --------
    .. code-block:: python

        Field(trial="u", test="v", order=2, components=2)   # P2 velocity
        Field(trial="p", test="q", order=1)                 # P1 pressure
    """

    def __init__(self, trial: str, test: str, order: int = 1, components: int = 1):
        for role, name in (("trial", trial), ("test", test)):
            if not (isinstance(name, str) and name.isidentifier()):
                raise ValueError(f"Field {role} name {name!r} is not a valid identifier")
            if name == "x":
                raise ValueError(f"Field {role} name 'x' collides with the coordinate argument")
            if name.startswith("grad"):
                raise ValueError(
                    f"Field {role} name {name!r} must not start with 'grad' "
                    f"(gradient arguments are derived as 'grad' + name)"
                )
        if trial == test:
            raise ValueError(f"Field trial and test names must differ, both are {trial!r}")
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"Field order must be a positive integer, got {order!r}")
        if not (isinstance(components, int) and components >= 1):
            raise ValueError(f"Field components must be a positive integer, got {components!r}")
        self.trial = trial
        self.test = test
        self.order = order
        self.components = components

    def __repr__(self):
        return (f"Field(trial={self.trial!r}, test={self.test!r}, "
                f"order={self.order}, components={self.components})")


class BlockLayout:
    r"""Block-DOF layout helpers of a :class:`MixedElementAssembler`.

    The global DOF vector stacks the fields in declaration order; within
    a field the layout is node-major:

    .. math::

        \mathrm{dof} = \mathrm{offset}_f + n_{\mathrm{local}} \cdot c_f + \mathrm{comp}.

    Access it as ``assembler.layout``. All tensors are read through the
    assembler's buffers, so the layout follows ``.to()`` / ``.double()``.
    """

    def __init__(self, assembler: "MixedElementAssembler"):
        self._asm = assembler
        self._boundary_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # basic queries
    # ------------------------------------------------------------------ #
    @property
    def names(self) -> List[str]:
        """Trial names of the fields, in declaration (block) order."""
        return [f.trial for f in self._asm.fields]

    @property
    def n_dofs(self) -> int:
        """Total number of DOFs :math:`N = \\sum_f n_f\\, c_f`."""
        return self._asm.n_dofs

    @property
    def offsets(self) -> Dict[str, int]:
        """First global DOF of each field, keyed by trial name."""
        return dict(self._asm._offsets)

    def field(self, name: str) -> Field:
        """The :class:`Field` declared with trial name ``name``."""
        return self._asm._field(name)

    def n_nodes(self, name: str) -> int:
        """Number of nodes (scalar DOF carriers) of field ``name``."""
        self.field(name)
        return self._asm.field_n_nodes[name]

    def node_ids(self, name: str) -> torch.Tensor:
        """Mesh point ids of the field's nodes, sorted — ``[n_f]`` long.

        Only fields whose DOFs live on mesh nodes (order 1 or the mesh
        order) have this; generalized-order fields carry DOFs on
        vertices/edges/interiors and raise here — use :meth:`points`,
        :meth:`boundary_mask` or ``dof_mask(where=...)`` instead.
        """
        if self._asm.field_kind[name] != "nodes":
            raise ValueError(
                f"field {name!r} (order {self.field(name).order}) has topological "
                f"DOF carriers, not mesh nodes — use points()/boundary_mask()/"
                f"dof_mask(where=...) instead"
            )
        return self._asm.field_node_ids[name]

    def points(self, name: str) -> torch.Tensor:
        """Coordinates of the field's nodes — ``[n_f, D]``.

        For generalized-order fields the coordinates are the images of the
        field's reference nodes under the isoparametric (mesh-order)
        geometry map, so they are exact on curved elements too.
        """
        if self._asm.field_kind[name] == "nodes":
            return self._asm._points[self._asm.field_node_ids[name]]
        pts = self._asm._points
        out = torch.zeros(self.n_nodes(name), pts.shape[1],
                          dtype=pts.dtype, device=pts.device)
        for element_type in self._asm.element_types:
            interp = self._interp_mesh_to_field(name, element_type)          # [nb_f, nb_mesh]
            conn_f = self._asm.field_conn[f"{name}__{element_type}"]
            coords = torch.einsum("fb,ebd->efd", interp,
                                  pts[self._asm.elements[element_type]])
            out[conn_f] = coords
        return out

    def _interp_mesh_to_field(self, name: str, element_type: str) -> torch.Tensor:
        """Mesh-order shape values at the field's reference nodes — ``[nb_f, nb_mesh]``."""
        elem_cls = element_type2element(element_type)
        pts = self._asm._points
        ref = self._asm.field_ref_nodes[f"{name}__{element_type}"] \
            .to(dtype=pts.dtype, device=pts.device)
        return elem_cls.eval_shape_val(ref, self._asm.mesh_order)

    # ------------------------------------------------------------------ #
    # data movement between mesh points, field nodes and the DOF vector
    # ------------------------------------------------------------------ #
    def restrict(self, name: str, point_data: torch.Tensor) -> torch.Tensor:
        """Interpolate mesh-point data ``[n_points, ...]`` onto the field's nodes ``[n_f, ...]``.

        A plain gather for node-carried fields; FE interpolation with the
        mesh-order basis for generalized-order fields.
        """
        n_points = self._asm.n_points
        assert point_data.shape[0] == n_points, (
            f"point_data must have shape [{n_points}, ...], got {list(point_data.shape)}"
        )
        if self._asm.field_kind[name] == "nodes":
            return point_data[self._asm.field_node_ids[name]]
        out = torch.zeros((self.n_nodes(name), *point_data.shape[1:]),
                          dtype=point_data.dtype, device=point_data.device)
        for element_type in self._asm.element_types:
            interp = self._interp_mesh_to_field(name, element_type)          # [nb_f, nb_mesh]
            conn_f = self._asm.field_conn[f"{name}__{element_type}"]
            vals = torch.einsum("fb,eb...->ef...", interp.to(point_data.dtype),
                                point_data[self._asm.elements[element_type]])
            out[conn_f] = vals
        return out

    def prolong(self, name: str, values: torch.Tensor) -> torch.Tensor:
        r"""Interpolate field-node values ``[n_f, ...]`` to **all** mesh points ``[n_points, ...]``.

        For a field of mesh order this is the identity; for an order-1
        field on a higher-order mesh the values are FE-interpolated to the
        midside/interior nodes (useful e.g. to plot a P1 pressure on a P2
        mesh).
        """
        field = self.field(name)
        n_f = self.n_nodes(name)
        assert values.shape[0] == n_f, (
            f"values must have shape [{n_f}, ...], got {list(values.shape)}"
        )
        if field.order == self._asm.mesh_order:
            return values
        out = torch.zeros((self._asm.n_points, *values.shape[1:]),
                          dtype=values.dtype, device=values.device)
        for element_type in self._asm.element_types:
            elem_cls = element_type2element(element_type)
            basis_pts = elem_cls.get_basis(self._asm.mesh_order) \
                                .type(values.dtype).to(values.device)      # [nb_mesh, D]
            interp = elem_cls.eval_shape_val(basis_pts, field.order)        # [nb_mesh, nb_f]
            conn_f = self._asm.field_conn[f"{name}__{element_type}"]        # [E, nb_f]
            vals_e = torch.einsum("mb,eb...->em...", interp, values[conn_f])
            out[self._asm.elements[element_type]] = vals_e
        return out

    def split(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Split a DOF vector ``[N, ...]`` into per-field tensors.

        Returns ``{name: [n_f, c_f, ...]}``; the component axis is
        squeezed away for scalar fields (``c_f == 1``).
        """
        assert x.shape[0] == self.n_dofs, (
            f"x must have shape [{self.n_dofs}, ...], got {list(x.shape)}"
        )
        out = {}
        for f in self._asm.fields:
            n_f, c = self.n_nodes(f.trial), f.components
            off = self._asm._offsets[f.trial]
            seg = x[off:off + n_f * c].reshape(n_f, c, *x.shape[1:])
            out[f.trial] = seg.squeeze(1) if c == 1 else seg
        return out

    def cat(self, fields: Optional[Mapping[str, Union[torch.Tensor, float]]] = None,
            **kwargs: Union[torch.Tensor, float]) -> torch.Tensor:
        r"""Concatenate per-field values into one flat DOF vector ``[N]`` (inverse of :meth:`split`).

        Every field must be given, either as a tensor of shape
        ``[n_f, c_f]`` / ``[n_f * c_f]`` (or ``[n_f]`` for scalar fields)
        or as a python scalar that is broadcast (e.g. ``p=0.0``).
        """
        given = dict(fields or {})
        given.update(kwargs)
        names = set(self.names)
        if set(given) != names:
            raise ValueError(f"cat() needs exactly the fields {sorted(names)}, got {sorted(given)}")
        ref = next((v for v in given.values() if isinstance(v, torch.Tensor)), None)
        dtype = ref.dtype if ref is not None else self._asm._points.dtype
        device = ref.device if ref is not None else self._asm._points.device
        parts = []
        for f in self._asm.fields:
            n_f, c = self.n_nodes(f.trial), f.components
            v = given[f.trial]
            if not isinstance(v, torch.Tensor):
                v = torch.full((n_f * c,), float(v), dtype=dtype, device=device)
            if v.numel() != n_f * c:
                raise ValueError(
                    f"field {f.trial!r} expects {n_f * c} values "
                    f"([{n_f}, {c}] or flat), got shape {list(v.shape)}"
                )
            parts.append(v.reshape(-1).to(dtype=dtype, device=device))
        return torch.cat(parts)

    # ------------------------------------------------------------------ #
    # DOF addressing (boundary conditions)
    # ------------------------------------------------------------------ #
    def dof_mask(self, name: str, node_mask: Optional[torch.Tensor] = None,
                 component: Optional[int] = None,
                 where: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                 ) -> torch.Tensor:
        r"""Boolean mask over **all** ``N`` DOFs selecting (part of) one field.

        Parameters
        ----------
        name : str
            Trial name of the field.
        node_mask : torch.Tensor, optional
            Boolean mask over the field's nodes ``[n_f]``, or over the
            mesh points ``[n_points]`` (restricted to the field's nodes
            automatically — node-carried fields only). ``None`` selects
            every node of the field.
        component : int, optional
            Restrict to a single component; ``None`` selects all.
        where : Callable, optional
            Coordinate predicate: called with the field's node coordinates
            ``[n_f, D]`` (see :meth:`points`), must return a bool ``[n_f]``
            mask. Works for every field kind; mutually exclusive with
            ``node_mask``. Example: ``where=lambda x: x[:, 1] > 1 - 1e-6``.
        """
        f, n_f, c = self.field(name), self.n_nodes(name), self.field(name).components
        off = self._asm._offsets[name]
        if where is not None:
            assert node_mask is None, "pass either node_mask or where, not both"
            node_sel = where(self.points(name))
            assert isinstance(node_sel, torch.Tensor) and node_sel.shape == (n_f,), (
                f"where must return a bool mask of shape [{n_f}]"
            )
            node_sel = node_sel.bool()
        elif node_mask is None:
            node_sel = torch.ones(n_f, dtype=torch.bool, device=self._asm._points.device)
        elif node_mask.shape[0] == n_f:
            node_sel = node_mask
        elif node_mask.shape[0] == self._asm.n_points:
            node_sel = node_mask[self.node_ids(name)]
        else:
            raise ValueError(
                f"node_mask must have shape [{n_f}] (field nodes) or "
                f"[{self._asm.n_points}] (mesh points, node-carried fields only), "
                f"got {list(node_mask.shape)}"
            )
        block = torch.zeros(n_f, c, dtype=torch.bool, device=node_sel.device)
        if component is None:
            block[:] = node_sel[:, None]
        else:
            assert 0 <= component < c, f"component must be in [0, {c}), got {component}"
            block[:, component] = node_sel
        mask = torch.zeros(self.n_dofs, dtype=torch.bool, device=node_sel.device)
        mask[off:off + n_f * c] = block.reshape(-1)
        return mask

    def boundary_mask(self, name: str, component: Optional[int] = None) -> torch.Tensor:
        r"""``dof_mask`` of the field's boundary DOFs, detected **topologically**.

        A facet is on the boundary iff exactly one cell references it, so
        no ``is_boundary`` point data or geometric tolerance is needed and
        edge/interior DOFs of higher-order fields are handled correctly.
        The per-field result is cached after the first call.
        """
        if name not in self._boundary_cache:
            f = self.field(name)
            conn = {element_type: self._asm.field_conn[f"{name}__{element_type}"]
                    for element_type in self._asm.element_types}
            self._boundary_cache[name] = lagrange_boundary_mask(
                conn, f.order, self.n_nodes(name))
        node_sel = self._boundary_cache[name].to(self._asm._points.device)
        return self.dof_mask(name, node_mask=node_sel, component=component)

    def dof_index(self, name: str, node: Union[int, torch.Tensor],
                  component: int = 0) -> torch.Tensor:
        r"""Global DOF index(es) of field ``name`` at mesh point id(s) ``node``.

        Node-carried fields only (order 1 or the mesh order); raises if a
        requested mesh point carries no DOF of this field (e.g. a midside
        node for an order-1 field). For generalized-order fields select
        DOFs via :meth:`dof_mask` / :meth:`boundary_mask` instead.
        """
        if self._asm.field_kind[name] != "nodes":
            raise ValueError(
                f"field {name!r} has topological DOF carriers, not mesh nodes — "
                f"select its DOFs via dof_mask(where=...) or boundary_mask()"
            )
        f, c = self.field(name), self.field(name).components
        assert 0 <= component < c, f"component must be in [0, {c}), got {component}"
        node = torch.as_tensor(node, dtype=torch.long,
                               device=self._asm._points.device)
        local = self._asm.field_g2l[name][node]
        if (local < 0).any():
            bad = node[local < 0] if node.dim() else node
            raise ValueError(f"mesh point(s) {bad.tolist() if local.dim() else int(bad)} "
                             f"carry no DOF of field {name!r}")
        return self._asm._offsets[name] + local * c + component

    def __repr__(self):
        rows = ", ".join(
            f"{f.trial}: [{self._asm._offsets[f.trial]}, "
            f"{self._asm._offsets[f.trial] + self.n_nodes(f.trial) * f.components})"
            for f in self._asm.fields
        )
        return f"BlockLayout(n_dofs={self.n_dofs}, {rows})"


class MixedElementAssembler(nn.Module):
    r"""Assemble a multi-field bilinear form into one block sparse matrix.

    Declare the fields as the class attribute ``fields`` (a list of
    :class:`Field`), override :meth:`forward` with the scalar integrand
    of the bilinear form, and build with :meth:`from_mesh`. See the
    module docstring for the conventions (trial → columns, test → rows;
    vector fields pass tensor-valued arguments; the integrand must be
    bilinear).

    Besides the field arguments, ``forward`` may take the same data
    arguments as :class:`~tensormesh.ElementAssembler`: ``x``
    (coordinates), any ``point_data`` key and its ``grad{key}``,
    ``element_data`` keys and ``scalar_data`` keys — plus ``field_data``
    keys (and their ``grad{key}``), which live on a field's DOFs and are
    interpolated with that field's own basis. ``point_data`` is
    interpolated with the mesh-order basis.

    The assembled matrix is square of size
    :math:`N = \sum_f n_f c_f` with the field blocks laid out in
    declaration order (see :class:`~tensormesh.BlockLayout`); use ``assembler.layout``
    to build boundary-condition masks and to split/concatenate DOF
    vectors. The unchanged :class:`~tensormesh.Condenser`
    applies on top.

    Notes
    -----
    * Only the ``ReduceProjector`` scatter backend is supported (the
      ``SparseProjector`` is float32-only, see the complex-FEM ROADMAP item).
    * ``energy`` / ``from_assembler`` are not provided for mixed forms.
    * Load vectors are assembled space-aware by :meth:`assemble_vector`
      (override :meth:`forward_vector` or pass ``func=``): same argument
      dispatch, linear in the test functions, same block DOF layout.
    """

    fields: List[Field] = []

    __autodoc__ = [
        "__call__",
        "forward",
        "forward_vector",
        "__post_init__",
        "from_mesh",
        "assemble_vector",
        "layout",
    ]

    def __init__(self, topology: dict, *args, **kwargs):
        super().__init__()
        self._validate_fields()

        self.transformation: nn.ModuleDict = topology["transformation"]
        self.elements: BufferDict = topology["elements"]
        self.field_node_ids: BufferDict = topology["field_node_ids"]
        self.field_g2l: BufferDict = topology["field_g2l"]
        self.field_conn: BufferDict = topology["field_conn"]
        self.field_ref_nodes: BufferDict = topology["field_ref_nodes"]
        self.ref_val: BufferDict = topology["ref_val"]
        self.ref_grad: BufferDict = topology["ref_grad"]
        self.geom_ref_grad: BufferDict = topology["geom_ref_grad"]
        self.pair_projector: nn.ModuleDict = topology["pair_projector"]
        self.pair_rows: BufferDict = topology["pair_rows"]
        self.pair_cols: BufferDict = topology["pair_cols"]

        self.element_types: List[str] = list(self.elements.keys())
        self.dimension: int = element_type2dimension[self.element_types[0]]
        self.mesh_order: int = topology["mesh_order"]
        self.n_points: int = topology["n_points"]
        self.field_kind: Dict[str, str] = topology["field_kind"]        # "nodes" | "dofmap"
        self.field_n_nodes: Dict[str, int] = topology["field_n_nodes"]

        self._offsets: Dict[str, int] = {}
        offset = 0
        for f in self.fields:
            self._offsets[f.trial] = offset
            offset += self.field_n_nodes[f.trial] * f.components
        self.n_dofs: int = offset

        self._layout = BlockLayout(self)
        self.__post_init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # declaration handling
    # ------------------------------------------------------------------ #
    def _validate_fields(self):
        cls = type(self)
        if not self.fields or not all(isinstance(f, Field) for f in self.fields):
            raise ValueError(
                f"{cls.__name__}.fields must be a non-empty list of Field declarations"
            )
        names = [n for f in self.fields for n in (f.trial, f.test)]
        if len(set(names)) != len(names):
            raise ValueError(f"{cls.__name__}.fields trial/test names must all differ, got {names}")

    def _field(self, name: str) -> Field:
        for f in self.fields:
            if f.trial == name:
                return f
        raise KeyError(f"no field with trial name {name!r}; fields are {self.layout.names}")

    @property
    def layout(self) -> BlockLayout:
        """Block-DOF layout helpers (offsets, masks, split/cat, ...)."""
        return self._layout

    @property
    def _points(self) -> torch.Tensor:
        return next(iter(self.transformation.values())).points  # type: ignore

    @property
    def device(self) -> torch.device:
        r"""Device on which the assembler's buffers live."""
        return next(iter(self.transformation.values())).device  # type: ignore

    @property
    def dtype(self) -> torch.dtype:
        r"""Floating dtype of the assembler's buffers (``float32`` or ``float64``)."""
        return next(iter(self.transformation.values())).dtype  # type: ignore

    def type(self, dtype: torch.dtype):
        if dtype == torch.float64:
            self.double()
        elif dtype == torch.float32:
            self.float()
        else:
            raise Exception(f"the dtype {dtype} is not supported")
        return self

    def forward(self, *args):
        r"""Scalar integrand of the bilinear form at one quadrature point.

        Override in subclasses. Arguments are requested by name: the
        trial/test names declared in ``fields`` (values), their
        ``grad``-prefixed gradients, ``x``, and any ``point_data`` /
        ``element_data`` / ``scalar_data`` key. Must return a 0-d tensor.
        """
        raise NotImplementedError("forward is not implemented")

    def forward_vector(self, *args):
        r"""Scalar integrand of the **linear** form assembled by :meth:`assemble_vector`.

        Override in subclasses (or pass ``func=`` to
        :meth:`assemble_vector`). Same argument dispatch as
        :meth:`forward`, but only **test** arguments (and data) may
        appear — e.g. a Stokes body force::

            def forward_vector(self, v, x):
                return x[1] * v[0]      # f = (y, 0)

        Must return a 0-d tensor, linear in the test functions.
        """
        raise NotImplementedError("forward_vector is not implemented")

    def __post_init__(self):
        r"""Override this function to store parameters after the initialization."""
        pass

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_mesh(cls, mesh: Mesh, quadrature_order: Optional[int] = None,
                  *args, **kwargs):
        r"""Build a :class:`MixedElementAssembler` from a :class:`~tensormesh.Mesh`.

        Parameters
        ----------
        mesh : tensormesh.Mesh
            Source mesh; its element order is the geometry (and maximum
            field) order.
        quadrature_order : int, optional
            Degree of exactness of the quadrature rule. Defaults to
            ``2 * max(field.order)``, which integrates every product of
            two field values/gradients exactly on affine elements.
        *args, **kwargs
            Additional arguments forwarded to ``__post_init__``.
        """
        # fields are validated again in __init__; check early for clear errors
        if not cls.fields or not all(isinstance(f, Field) for f in cls.fields):
            raise ValueError(
                f"{cls.__name__}.fields must be a non-empty list of Field declarations"
            )

        points: torch.Tensor = mesh.points  # type: ignore
        elements = mesh.elements()  # type: ignore
        n_points: int = points.shape[0]
        if isinstance(elements, torch.Tensor):
            elements = {mesh.default_element_type: elements}
        elements = {k: v.long() for k, v in elements.items()}

        orders = {element_type: element_type2order[element_type] for element_type in elements}
        if len(set(orders.values())) != 1:
            raise ValueError(f"mesh mixes element orders {orders}; this is not supported")
        mesh_order = next(iter(orders.values()))
        if quadrature_order is None:
            quadrature_order = 2 * max(f.order for f in cls.fields)
            if quadrature_order > 7:
                raise ValueError(
                    f"the default quadrature degree {quadrature_order} (= 2 * max "
                    f"field order) exceeds the tabulated maximum of 7 — pass "
                    f"quadrature_order <= 7 explicitly (products of two highest-"
                    f"order fields then integrate inexactly) or extend the "
                    f"quadrature tables"
                )

        # ---- per-field DOF carriers and field-local connectivity ---- #
        # orders 1 / mesh_order ride on mesh nodes ("nodes" kind, keeps the
        # mesh-point numbering); any other order gets a topological DOF map.
        field_kind: Dict[str, str] = {}
        field_n_nodes: Dict[str, int] = {}
        field_node_ids: Dict[str, torch.Tensor] = {}
        field_g2l: Dict[str, torch.Tensor] = {}
        field_conn: Dict[str, torch.Tensor] = {}
        field_ref_nodes: Dict[str, torch.Tensor] = {}
        for f in cls.fields:
            if f.order == mesh_order:
                field_kind[f.trial] = "nodes"
                node_ids = torch.arange(n_points, dtype=torch.long)
                g2l = torch.arange(n_points, dtype=torch.long)
                conn = {element_type: value for element_type, value in elements.items()}
            elif f.order == 1:  # corner vertices, shared across etypes
                field_kind[f.trial] = "nodes"
                corners = {
                    element_type: value[:, :element_type2element(element_type).n_vertex]
                    for element_type, value in elements.items()
                }
                node_ids = torch.unique(torch.cat([v.reshape(-1) for v in corners.values()]))
                g2l = torch.full((n_points,), -1, dtype=torch.long)
                g2l[node_ids] = torch.arange(node_ids.shape[0], dtype=torch.long)
                conn = {element_type: g2l[value].contiguous()
                        for element_type, value in corners.items()}
            else:  # generalized pair: topological DOF map (vertices/edges/interiors)
                field_kind[f.trial] = "dofmap"
                dofmap = lagrange_dofmap(elements, n_points, f.order)
                conn = dofmap.conn
                field_n_nodes[f.trial] = dofmap.n_dofs
                for element_type, ref in dofmap.ref_nodes.items():
                    field_ref_nodes[f"{f.trial}__{element_type}"] = ref.to(points.dtype)
            if field_kind[f.trial] == "nodes":
                field_node_ids[f.trial] = node_ids
                field_g2l[f.trial] = g2l
                field_n_nodes[f.trial] = node_ids.shape[0]
            for element_type, value in conn.items():
                field_conn[f"{f.trial}__{element_type}"] = value

        # ---- geometry (mesh order) and per-field reference tables ---- #
        transformations: Dict[str, Transformation] = {}
        ref_val: Dict[str, torch.Tensor] = {}
        ref_grad: Dict[str, torch.Tensor] = {}
        geom_ref_grad: Dict[str, torch.Tensor] = {}
        for element_type, value in elements.items():
            trans = Transformation(
                points=points,
                elements=value,
                element_type=element_type,
                quadrature_order=quadrature_order,
            )
            transformations[element_type] = trans
            _, q = trans.quadrature  # [n_q, D] in points.dtype
            elem_cls = element_type2element(element_type)
            geom_ref_grad[element_type] = \
                elem_cls.get_basis_grad_fns(mesh_order, q.dtype, q.device).map(q)  # [n_q, D, nb_mesh]
            for f in cls.fields:
                key = f"{f.trial}__{element_type}"
                ref_val[key] = elem_cls.eval_shape_val(q, f.order)                      # [n_q, nb_f]
                ref_grad[key] = elem_cls.get_basis_grad_fns(f.order, q.dtype, q.device).map(q)

        # ---- per-(test, trial) pair: edge pattern, scatter, expanded COO ---- #
        pair_projector: Dict[str, ReduceProjector] = {}
        pair_rows: Dict[str, torch.Tensor] = {}
        pair_cols: Dict[str, torch.Tensor] = {}
        offsets: Dict[str, int] = {}
        offset = 0
        for f in cls.fields:
            offsets[f.trial] = offset
            offset += field_n_nodes[f.trial] * f.components
        for beta in cls.fields:          # test  -> rows
            for alpha in cls.fields:     # trial -> columns
                pair_key = f"{beta.trial}__{alpha.trial}"
                edges, eids = build_edges(
                    {
                        element_type: (
                            field_conn[f"{beta.trial}__{element_type}"],
                            field_conn[f"{alpha.trial}__{element_type}"],
                        )
                        for element_type in elements
                    },
                    shape=(field_n_nodes[beta.trial],
                           field_n_nodes[alpha.trial]),
                )
                num_edges = edges.shape[1]
                for element_type in elements:
                    n_element = elements[element_type].shape[0]
                    nb_beta = field_conn[f"{beta.trial}__{element_type}"].shape[1]
                    nb_alpha = field_conn[f"{alpha.trial}__{element_type}"].shape[1]
                    pair_projector[f"{pair_key}__{element_type}"] = ReduceProjector(
                        indices=eids[element_type],
                        from_shape=(n_element, nb_beta, nb_alpha),
                        to_shape=(num_edges,),
                    )
                c_b, c_a = beta.components, alpha.components
                arange_b = torch.arange(c_b, dtype=torch.long)
                arange_a = torch.arange(c_a, dtype=torch.long)
                rows = offsets[beta.trial] + edges[0][:, None, None] * c_b + arange_b[None, :, None]
                cols = offsets[alpha.trial] + edges[1][:, None, None] * c_a + arange_a[None, None, :]
                pair_rows[pair_key] = rows.expand(-1, c_b, c_a).reshape(-1)
                pair_cols[pair_key] = cols.expand(-1, c_b, c_a).reshape(-1)

        topology = {
            "transformation": nn.ModuleDict(transformations),
            "elements": BufferDict(elements),
            "field_node_ids": BufferDict(field_node_ids),
            "field_g2l": BufferDict(field_g2l),
            "field_conn": BufferDict(field_conn),
            "field_ref_nodes": BufferDict(field_ref_nodes),
            "ref_val": BufferDict(ref_val),
            "ref_grad": BufferDict(ref_grad),
            "geom_ref_grad": BufferDict(geom_ref_grad),
            "pair_projector": nn.ModuleDict(pair_projector),
            "pair_rows": BufferDict(pair_rows),
            "pair_cols": BufferDict(pair_cols),
            "mesh_order": mesh_order,
            "n_points": n_points,
            "field_kind": field_kind,
            "field_n_nodes": field_n_nodes,
        }
        assembler = cls(topology, *args, **kwargs)
        assembler = assembler.type(mesh.dtype).to(mesh.device)
        return assembler

    # ------------------------------------------------------------------ #
    # signature handling
    # ------------------------------------------------------------------ #
    def _classify_params(self, fn: Callable,
                         point_data: Mapping[str, torch.Tensor],
                         element_data: Mapping[str, Mapping[str, torch.Tensor]],
                         scalar_data: Mapping[str, torch.Tensor],
                         field_data: Mapping[str, Tuple[str, torch.Tensor]]):
        roles: Dict[str, Tuple[str, Field]] = {}
        for f in self.fields:
            roles[f.trial] = ("trial_val", f)
            roles["grad" + f.trial] = ("trial_grad", f)
            roles[f.test] = ("test_val", f)
            roles["grad" + f.test] = ("test_grad", f)

        data_keys = set(point_data) | set(element_data) | set(scalar_data) | set(field_data)
        collisions = sorted(set(roles) & data_keys)
        if collisions:
            raise ValueError(
                f"data key(s) {collisions} collide with the field trial/test "
                f"argument names — rename the data or the fields"
            )
        ambiguous = sorted(set(field_data) & set(point_data))
        if ambiguous:
            raise ValueError(
                f"key(s) {ambiguous} appear in both point_data and field_data"
            )

        params = []
        for key in inspect.signature(fn).parameters:
            if key in roles:
                kind, f = roles[key]
                params.append((key, kind, f))
            elif key in element_data:
                params.append((key, "element", None))
            elif key in scalar_data:
                params.append((key, "scalar", None))
            elif key in field_data:
                params.append((key, "fielddata", None))
            elif key in point_data:
                params.append((key, "point", None))
            elif key.startswith("grad") and key[4:] in field_data:
                params.append((key, "gradfielddata", None))
            elif key.startswith("grad") and key[4:] in point_data:
                params.append((key, "gradpoint", None))
            else:
                raise ValueError(
                    f"{key!r} is not supported — valid names are the field "
                    f"arguments {sorted(roles)} or keys provided by "
                    f"point_data, element_data, scalar_data or field_data"
                )
        return params

    @staticmethod
    def _executed_pairs(fields: List[Field], params) -> List[Tuple[Field, Field]]:
        has_trial = {f.trial for _, kind, f in params if kind in ("trial_val", "trial_grad")}
        has_test = {f.trial for _, kind, f in params if kind in ("test_val", "test_grad")}
        return [
            (alpha, beta)
            for beta in fields
            for alpha in fields
            if alpha.trial in has_trial and beta.trial in has_test
        ]

    def _check_bilinear(self, fn: Callable, params, data_args, dtype, device,
                        form: str = "bilinear"):
        """Evaluate ``fn`` with every field argument zero; nonzero ⇒ not (bi)linear."""
        D = self.dimension
        args = []
        for key, kind, f in params:
            if kind in ("trial_val", "test_val"):
                shape = () if f.components == 1 else (f.components,)
                args.append(torch.zeros(shape, dtype=dtype, device=device))
            elif kind in ("trial_grad", "test_grad"):
                shape = (D,) if f.components == 1 else (f.components, D)
                args.append(torch.zeros(shape, dtype=dtype, device=device))
            elif kind == "scalar":
                args.append(data_args[key])
            elif kind == "element":
                args.append(data_args[key][0])
            else:  # point / gradpoint / fielddata / gradfielddata: [E, Q, ...]
                args.append(data_args[key][0, 0])
        out = fn(*args)
        if not isinstance(out, torch.Tensor) or out.dim() != 0:
            raise ValueError(
                "the mixed forward must return a 0-d scalar integrand, "
                f"got {out.shape if isinstance(out, torch.Tensor) else type(out)}"
            )
        if not (out == 0).all():
            raise ValueError(
                f"the mixed integrand is not {form}: it is nonzero when every "
                f"field argument is zero (constant term detected)"
            )

    # ------------------------------------------------------------------ #
    # one pass = one (trial field, test field) block on one element type;
    # with alpha=None it evaluates a LINEAR form (test side only)
    # ------------------------------------------------------------------ #
    def _run_pass(self, fn: Callable, params, alpha: Optional[Field], beta: Field,
                  tables_val, tables_grad, data_args, zeros, eyes):
        AX_E, AX_Q, AX_I, AX_J, AX_B, AX_A = range(6)
        raw: List[torch.Tensor] = []
        dims: List[List[Optional[int]]] = []
        builders: List[Callable] = []

        def add_raw(t, axes: Dict[int, int]) -> int:
            d: List[Optional[int]] = [None] * 6
            for ax, v in axes.items():
                d[ax] = v
            raw.append(t)
            dims.append(d)
            return len(raw) - 1

        eye_idx = {"trial": None, "test": None}

        def add_field_arg(side, kind, f, table, basis_ax, comp_ax):
            grad = kind.endswith("grad")
            if grad:
                k = add_raw(table, {AX_E: 0, AX_Q: 0, basis_ax: 0})
            else:
                k = add_raw(table, {AX_Q: 0, basis_ax: 0})
            if f.components > 1:
                if eye_idx[side] is None:
                    eye_idx[side] = add_raw(eyes[side], {comp_ax: 0})
                m = eye_idx[side]
                if grad:
                    builders.append(lambda r, k=k, m=m: r[m][:, None] * r[k][None, :])
                else:
                    builders.append(lambda r, k=k, m=m: r[m] * r[k])
            else:
                builders.append(lambda r, k=k: r[k])

        for key, kind, f in params:
            if kind in ("trial_val", "trial_grad"):
                if f is alpha:
                    add_field_arg("trial", kind, f, tables_grad[f.trial] if kind == "trial_grad"
                                  else tables_val[f.trial], AX_J, AX_A)
                else:
                    builders.append(lambda r, z=zeros[(f.trial, kind)]: z)
            elif kind in ("test_val", "test_grad"):
                if f is beta:
                    add_field_arg("test", kind, f, tables_grad[f.trial] if kind == "test_grad"
                                  else tables_val[f.trial], AX_I, AX_B)
                else:
                    builders.append(lambda r, z=zeros[(f.trial, kind)]: z)
            elif kind == "scalar":
                builders.append(lambda r, t=data_args[key]: t)
            elif kind == "element":
                k = add_raw(data_args[key], {AX_E: 0})
                builders.append(lambda r, k=k: r[k])
            else:  # point / gradpoint / fielddata / gradfielddata: [E, Q, ...]
                k = add_raw(data_args[key], {AX_E: 0, AX_Q: 0})
                builders.append(lambda r, k=k: r[k])

        def inner(*r):
            return fn(*[b(r) for b in builders])

        has_e = any(d[AX_E] is not None for d in dims)
        # a vmap layer exists iff some raw input carries that axis: q/i are
        # always carried by the test tables, j/a only in bilinear passes
        # (alpha is not None), b only for vector-valued test fields
        layers = [ax for ax in (AX_E, AX_Q, AX_I, AX_J, AX_B, AX_A)
                  if any(d[ax] is not None for d in dims)]
        parallel = inner
        for ax in reversed(layers):  # wrap innermost first
            parallel = vmap(parallel, in_dims=tuple(d[ax] for d in dims))

        out = parallel(*raw)
        if out.dim() != len(layers):
            raise ValueError(
                "the mixed forward must return a 0-d scalar integrand, got a "
                f"tensor with {out.dim() - len(layers)} extra dimension(s)"
            )
        if alpha is not None:  # bilinear block: [..., i, j, b, a]
            if alpha.components == 1:
                out = out.unsqueeze(-1)
            if beta.components == 1:
                out = out.unsqueeze(-2)
        elif beta.components == 1:  # linear pass: [..., i, b]
            out = out.unsqueeze(-1)
        return out, has_e

    @staticmethod
    def _integrate_pair(batch_integral, jxw, use_element_parallel):
        # [E?, Q, i, j, b, a] (bilinear) or [E?, Q, i, b] (linear) -> drop Q
        if use_element_parallel:
            return torch.einsum("eqi...,eq->ei...", batch_integral, jxw)
        return torch.einsum("qi...,eq->ei...", batch_integral, jxw)

    # ------------------------------------------------------------------ #
    # shared assembly plumbing
    # ------------------------------------------------------------------ #
    def _normalize_inputs(self, points, point_data, element_data, scalar_data, field_data):
        """Validate/normalize the ``__call__``/``assemble_vector`` inputs."""
        assert isinstance(point_data, dict) or point_data is None, (
            f"point_data should be a dict, but got {type(point_data)}. "
            f"Please pass in extra parameters using key-value pairs"
        )
        if point_data is None:
            point_data = {}

        if element_data is None:
            element_data = {}
        else:
            if not isinstance(next(iter(element_data.values())), dict):
                assert len(self.element_types) == 1
                element_type = self.element_types[0]
                element_data = {key: {element_type: value} for key, value in element_data.items()}  # type: ignore
            for key in element_data:
                for element_type in self.element_types:
                    assert element_data[key][element_type].shape[0] == self.elements[element_type].shape[0], (
                        f"the shape of {key} should be "
                        f"[{self.elements[element_type].shape[0]}, ...], but got "
                        f"{element_data[key][element_type].shape[0]}"
                    )

        if scalar_data is None:
            scalar_data = {}
        else:
            scalar_data = {k: torch.tensor(v) for k, v in scalar_data.items()}

        if field_data is None:
            field_data = {}
        else:
            checked: Dict[str, Tuple[str, torch.Tensor]] = {}
            for key, spec in field_data.items():
                assert isinstance(spec, tuple) and len(spec) == 2, (
                    f"field_data[{key!r}] must be a (field_name, values) tuple"
                )
                fname, values = spec
                f = self._field(fname)
                n_f = self.field_n_nodes[fname]
                assert values.shape[0] == n_f, (
                    f"field_data[{key!r}]: values must have shape [{n_f}, ...] "
                    f"(nodes of field {fname!r}), got {list(values.shape)}"
                )
                checked[key] = (fname, values)
            field_data = checked

        if points is None:
            points = self._points
        else:
            for element_type in self.element_types:
                assert points.shape[1] == self.transformation[element_type].dim, (
                    f"the dimension of points should be "
                    f"{self.transformation[element_type].dim}, but got {points.shape[1]}"
                )
                self.transformation[element_type].update_points(points)  # type: ignore

        point_data["x"] = points  # type: ignore

        self.type(points.dtype).to(points.device)

        for key, value in point_data.items():
            assert value.shape[0] == points.shape[0], (
                f"the shape of {key} should be [n_point, ...], but got {value.shape}"
            )
        return points, point_data, element_data, scalar_data, field_data

    def _pass_context(self, params, field_data, dtype, device):
        """Zero constants, one-hot eyes and the basis-table requirements of ``params``."""
        D = self.dimension
        zeros = {}
        for _, kind, f in params:
            if f is None:
                continue
            if kind.endswith("grad"):
                shape = (D,) if f.components == 1 else (f.components, D)
            else:
                shape = () if f.components == 1 else (f.components,)
            zeros[(f.trial, kind)] = torch.zeros(shape, dtype=dtype, device=device)
        eyes_by_field = {
            f.trial: torch.eye(f.components, dtype=dtype, device=device)
            for f in self.fields if f.components > 1
        }
        needs_val = {f.trial for _, kind, f in params if kind in ("trial_val", "test_val")}
        needs_grad = {f.trial for _, kind, f in params if kind in ("trial_grad", "test_grad")}
        needs_val.update(field_data[key][0] for key, kind, _ in params if kind == "fielddata")
        needs_grad.update(field_data[key[4:]][0] for key, kind, _ in params if kind == "gradfielddata")
        return zeros, eyes_by_field, needs_val, needs_grad

    def _batches(self, params, point_data, element_data, scalar_data, field_data,
                 needs_val, needs_grad, batch_size):
        """Yield ``(element_type, jxw, tables_val, tables_grad, data_args)`` per quadrature batch.

        One yield per (element type, quadrature batch): the isoparametric
        geometry, the per-field physical basis tables and every data
        argument of ``params`` interpolated at the batch's quadrature points.
        """
        point_keys = [key for key, kind, _ in params if kind == "point"]
        gradpoint_keys = [key for key, kind, _ in params if kind == "gradpoint"]
        fielddata_keys = [key for key, kind, _ in params if kind == "fielddata"]
        gradfielddata_keys = [key for key, kind, _ in params if kind == "gradfielddata"]

        for element_type in self.element_types:
            trans: Transformation = self.transformation[element_type]  # type: ignore
            n_quadrature = trans.n_quadrature
            if batch_size in (-1, None):
                n_batch, n_batch_size = 1, n_quadrature
            else:
                n_batch_size = batch_size
                n_batch = math.ceil(n_quadrature / batch_size)

            elements: torch.Tensor = self.elements[element_type]
            ele_point_data = {k: v[elements] for k, v in point_data.items()}
            ele_field_data = {
                key: values[self.field_conn[f"{fname}__{element_type}"]]
                for key, (fname, values) in field_data.items()
            }  # {key: [E, nb_f, ...]}
            element_coords = trans.element_coords  # [E, nb_mesh, D]

            for i in range(n_batch):
                qs = i * n_batch_size
                w, _ = trans.batch_quadrature(qs, n_batch_size)  # [Qb], [Qb, D]
                qb = w.shape[0]

                # geometry: isoparametric (mesh-order) jacobian, shared by all fields
                ref_g_geom = self.geom_ref_grad[element_type][qs:qs + qb]  # [Qb, D, nb_mesh]
                jacobian = torch.einsum("ebj,qib->eqij", element_coords, ref_g_geom)
                inv_jacobian = torch.inverse(jacobian)
                jxw = torch.einsum("q,eq->eq", w, torch.linalg.det(jacobian).abs())

                tables_val, tables_grad = {}, {}
                for f in self.fields:
                    key = f"{f.trial}__{element_type}"
                    if f.trial in needs_val:
                        tables_val[f.trial] = self.ref_val[key][qs:qs + qb]  # [Qb, nb_f]
                    if f.trial in needs_grad:
                        tables_grad[f.trial] = torch.einsum(
                            "qib,eqji->eqbj", self.ref_grad[key][qs:qs + qb], inv_jacobian
                        )  # [E, Qb, nb_f, D]

                data_args: Dict[str, torch.Tensor] = {}
                if point_keys or gradpoint_keys:
                    sv_mesh = trans.batch_shape_val(qs, n_batch_size)  # [Qb, nb_mesh]
                    for key in point_keys:
                        data_args[key] = torch.einsum("eb...,qb->eq...", ele_point_data[key], sv_mesh)
                    if gradpoint_keys:
                        sg_mesh = torch.einsum("qib,eqji->eqbj", ref_g_geom, inv_jacobian)
                        for key in gradpoint_keys:
                            data_args[key] = torch.einsum(
                                "eb...,eqbd->eq...d", ele_point_data[key[4:]], sg_mesh
                            )
                for key in fielddata_keys:  # interpolated with the field's own basis
                    fname = field_data[key][0]
                    data_args[key] = torch.einsum(
                        "eb...,qb->eq...", ele_field_data[key], tables_val[fname]
                    )
                for key in gradfielddata_keys:
                    fname = field_data[key[4:]][0]
                    data_args[key] = torch.einsum(
                        "eb...,eqbd->eq...d", ele_field_data[key[4:]], tables_grad[fname]
                    )
                for key, kind, _ in params:
                    if kind == "element":
                        data_args[key] = element_data[key][element_type]  # type: ignore
                    elif kind == "scalar":
                        data_args[key] = scalar_data[key]  # type: ignore

                yield element_type, jxw, tables_val, tables_grad, data_args

    # ------------------------------------------------------------------ #
    # assembly
    # ------------------------------------------------------------------ #
    def __call__(self, points: Optional[torch.Tensor] = None,
                 func: Optional[Callable] = None,
                 point_data: Optional[Mapping[str, torch.Tensor]] = None,
                 element_data: Optional[Union[Mapping[str, Mapping[str, torch.Tensor]],
                                              Mapping[str, torch.Tensor]]] = None,
                 scalar_data: Optional[Mapping[str, torch.Tensor]] = None,
                 batch_size: int = -1,
                 field_data: Optional[Mapping[str, Tuple[str, torch.Tensor]]] = None
                 ) -> SparseMatrix:
        r"""Assemble the mixed bilinear form into the global block sparse matrix.

        The signature mirrors :meth:`ElementAssembler.__call__`; see the
        class docstring for the mixed-form conventions.

        Parameters
        ----------
        field_data : Mapping[str, Tuple[str, torch.Tensor]], optional
            Data living on a *field's* DOFs rather than on mesh points:
            ``{"w": ("u", values)}`` with ``values`` of shape
            ``[n_f, c_f]`` (or ``[n_f]`` for scalar fields). The key (and
            its ``grad``-prefixed form) becomes a ``forward`` argument,
            interpolated with that field's own basis — e.g. a previous
            Picard iterate of a generalized-order velocity field.

        Returns
        -------
        SparseMatrix
            Square sparse matrix of shape :math:`[N, N]` with
            :math:`N = \sum_f n_f c_f` (see :class:`~tensormesh.BlockLayout`).
        """
        points, point_data, element_data, scalar_data, field_data = \
            self._normalize_inputs(points, point_data, element_data, scalar_data, field_data)
        dtype, device = points.dtype, points.device

        fn = self.forward if func is None else func
        params = self._classify_params(fn, point_data, element_data, scalar_data, field_data)
        executed = self._executed_pairs(self.fields, params)
        zeros, eyes_by_field, needs_val, needs_grad = \
            self._pass_context(params, field_data, dtype, device)

        acc: Dict[Tuple[str, str, str], Optional[torch.Tensor]] = {}
        checked_bilinear = False
        for element_type, jxw, tables_val, tables_grad, data_args in self._batches(
                params, point_data, element_data, scalar_data, field_data,
                needs_val, needs_grad, batch_size):
            if not checked_bilinear:
                self._check_bilinear(fn, params, data_args, dtype, device)
                checked_bilinear = True
            for alpha, beta in executed:
                eyes = {"trial": eyes_by_field.get(alpha.trial),
                        "test": eyes_by_field.get(beta.trial)}
                out, has_e = self._run_pass(
                    fn, params, alpha, beta,
                    tables_val, tables_grad, data_args, zeros, eyes,
                )
                batch_integral = self._integrate_pair(out, jxw, has_e)  # [E, i, j, b, a]
                key = (alpha.trial, beta.trial, element_type)
                acc[key] = batch_integral if acc.get(key) is None else acc[key] + batch_integral

        pass_vals: Dict[Tuple[str, str], torch.Tensor] = {}
        for alpha, beta in executed:
            total = None
            for element_type in self.element_types:
                proj = self.pair_projector[f"{beta.trial}__{alpha.trial}__{element_type}"]
                projected = proj(acc[(alpha.trial, beta.trial, element_type)])  # [n_pair_edges, c_b, c_a]
                total = projected if total is None else total + projected
            pass_vals[(alpha.trial, beta.trial)] = total

        vals = []
        pattern = []
        for beta in self.fields:
            for alpha in self.fields:
                key = (alpha.trial, beta.trial)
                if key not in pass_vals:
                    continue
                vals.append(pass_vals[key].reshape(-1))
                pattern.append(f"{beta.trial}__{alpha.trial}")
        rows, cols = self._coo_pattern(tuple(pattern), device)
        return SparseMatrix(
            torch.cat(vals), rows, cols,
            shape=(self.n_dofs, self.n_dofs),
        )

    def _coo_pattern(self, pattern: Tuple[str, ...], device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenated global (rows, cols) of the executed blocks — cached.

        ``SparseMatrix.layout_signature`` is sequence-identity
        (``data_ptr`` + version), so downstream pattern caches like
        :class:`~tensormesh.Condenser` only hit when repeated
        assemblies hand over the *same* index tensors. Re-concatenating
        per call would allocate fresh tensors and defeat that, so the
        concatenation is cached per executed-block pattern (and rebuilt
        on a device change).
        """
        if not hasattr(self, "_coo_cache"):
            self._coo_cache: Dict[Tuple[str, ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        cached = self._coo_cache.get(pattern)
        if cached is None or cached[0].device != device:
            rows = torch.cat([self.pair_rows[pair_key] for pair_key in pattern])
            cols = torch.cat([self.pair_cols[pair_key] for pair_key in pattern])
            self._coo_cache[pattern] = (rows, cols)
        return self._coo_cache[pattern]

    def assemble_vector(self, points: Optional[torch.Tensor] = None,
                        func: Optional[Callable] = None,
                        point_data: Optional[Mapping[str, torch.Tensor]] = None,
                        element_data: Optional[Union[Mapping[str, Mapping[str, torch.Tensor]],
                                                     Mapping[str, torch.Tensor]]] = None,
                        scalar_data: Optional[Mapping[str, torch.Tensor]] = None,
                        batch_size: int = -1,
                        field_data: Optional[Mapping[str, Tuple[str, torch.Tensor]]] = None
                        ) -> torch.Tensor:
        r"""Assemble a **linear** form into the global block load vector.

        The space-aware counterpart of a load-vector assembler: the
        integrand (:meth:`forward_vector`, or ``func=``) is written like
        :meth:`forward` but may reference only **test** arguments plus
        data — e.g. :math:`\int f \cdot v` for the Stokes momentum
        equation. It is evaluated with one-hot test basis functions per
        field and scattered into the same block DOF layout as
        ``__call__``, so the result pairs directly with the assembled
        matrix and the :class:`~tensormesh.Condenser`. Fields
        whose test arguments do not appear contribute a zero segment.

        Works for every field kind, including generalized-order fields
        (this supersedes the interim ``b = M @ f(points)`` interpolation
        recipe). ``field_data`` is supported — e.g. the previous time
        step of a velocity living on its own field DOFs.

        Returns
        -------
        torch.Tensor
            Dense load vector of shape :math:`[N]` with
            :math:`N = \sum_f n_f c_f` (see :class:`~tensormesh.BlockLayout`).
        """
        points, point_data, element_data, scalar_data, field_data = \
            self._normalize_inputs(points, point_data, element_data, scalar_data, field_data)
        dtype, device = points.dtype, points.device

        fn = self.forward_vector if func is None else func
        params = self._classify_params(fn, point_data, element_data, scalar_data, field_data)
        trial_used = [key for key, kind, _ in params if kind in ("trial_val", "trial_grad")]
        if trial_used:
            raise ValueError(
                f"assemble_vector assembles a form linear in the test functions; "
                f"trial argument(s) {trial_used} are not allowed — use __call__ "
                f"for bilinear forms"
            )
        has_test = {f.trial for _, kind, f in params if kind in ("test_val", "test_grad")}
        executed = [f for f in self.fields if f.trial in has_test]
        if not executed:
            raise ValueError(
                "the linear form references no test argument — nothing to assemble"
            )
        zeros, eyes_by_field, needs_val, needs_grad = \
            self._pass_context(params, field_data, dtype, device)

        acc: Dict[Tuple[str, str], Optional[torch.Tensor]] = {}
        checked_linear = False
        for element_type, jxw, tables_val, tables_grad, data_args in self._batches(
                params, point_data, element_data, scalar_data, field_data,
                needs_val, needs_grad, batch_size):
            if not checked_linear:
                self._check_bilinear(fn, params, data_args, dtype, device, form="linear")
                checked_linear = True
            for beta in executed:
                eyes = {"trial": None, "test": eyes_by_field.get(beta.trial)}
                out, has_e = self._run_pass(
                    fn, params, None, beta,
                    tables_val, tables_grad, data_args, zeros, eyes,
                )
                batch_integral = self._integrate_pair(out, jxw, has_e)  # [E, i, b]
                key = (beta.trial, element_type)
                acc[key] = batch_integral if acc.get(key) is None else acc[key] + batch_integral

        out_vec = torch.zeros(self.n_dofs, dtype=dtype, device=device)
        for beta in executed:
            n_f, c = self.field_n_nodes[beta.trial], beta.components
            seg = torch.zeros(n_f, c, dtype=dtype, device=device)
            for element_type in self.element_types:
                conn = self.field_conn[f"{beta.trial}__{element_type}"]  # [E, nb_f]
                seg.index_add_(0, conn.reshape(-1),
                               acc[(beta.trial, element_type)].reshape(-1, c))
            off = self._offsets[beta.trial]
            out_vec[off:off + n_f * c] = seg.reshape(-1)
        return out_vec

    def __str__(self):
        fields = ", ".join(repr(f) for f in self.fields)
        return (
            f"{self.__class__.__name__}(\n"
            f"    element_types: {self.element_types}\n"
            f"    fields: [{fields}]\n"
            f"    n_dofs: {self.n_dofs}\n"
            f")"
        )

    def __repr__(self):
        return str(self)

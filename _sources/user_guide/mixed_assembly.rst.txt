Mixed Assembly
==============

Many PDE systems couple several unknown fields in one weak form:
velocity and pressure in Stokes / Navier-Stokes, displacement and
pressure in poroelasticity, velocity, pressure **and** temperature in
Boussinesq convection. Discretizing them stably usually requires
*different* function spaces per field — the classic example is the
Taylor-Hood pair (quadratic velocity, linear pressure), which satisfies
the inf-sup (LBB) condition that equal-order pairs violate.

:class:`~tensormesh.MixedElementAssembler` assembles such multi-field
bilinear forms into **one block sparse matrix**. You declare the fields
once, write the scalar integrand of the coupled weak form as a single
``forward``, and everything downstream — boundary conditions via
:class:`~tensormesh.Condenser`, ``.solve()``, autograd — works on the
block system exactly as it does for a single field:

.. code-block:: python

   from tensormesh import Condenser, Field, Mesh, MixedElementAssembler

   class StokesAssembler(MixedElementAssembler):
       fields = [
           Field(trial="u", test="v", order=2, components=2),  # P2 velocity
           Field(trial="p", test="q", order=1),                # P1 pressure
       ]

       def __post_init__(self, mu=1.0):
           self.mu = mu

       def forward(self, gradu, p, gradv, q):
           return self.mu * (gradu * gradv).sum() \
               - p * gradv.diagonal().sum() \
               - q * gradu.diagonal().sum()

   mesh = Mesh.gen_rectangle(chara_length=0.05, order=2).double()
   assembler = StokesAssembler.from_mesh(mesh, mu=1.0)
   K = assembler()          # SparseMatrix over all velocity + pressure DOFs

The rest of this page walks through the pieces: field declarations, the
integrand contract, the block DOF layout, load vectors, the data
channels, and *generalized order pairs* — fields whose polynomial order
differs from the mesh order.


.. _mixed-fields:

Declaring fields
----------------

``fields`` is a class attribute: a list of :class:`~tensormesh.Field`
declarations, one per unknown. Each field names its **trial** and
**test** function (the argument names your ``forward`` will use), its
polynomial ``order``, and its number of vector ``components``:

.. code-block:: python

   fields = [
       Field(trial="u", test="v", order=2, components=2),
       Field(trial="p", test="q", order=1),
   ]

Three conventions to remember:

* **Trial indexes columns, test indexes rows.** The block
  :math:`K_{\beta\alpha}` couples trial field :math:`\alpha` (columns)
  to test field :math:`\beta` (rows). For a symmetric weak form you
  never notice; for non-symmetric couplings (convection, divergence
  constraints) the declaration makes the orientation explicit.
* **Names are global.** All trial and test names must be distinct,
  valid identifiers, must not be ``"x"`` (reserved for coordinates) and
  must not start with ``"grad"`` (gradients are derived as
  ``"grad" + name``).
* **Order is per-field** and independent of the mesh order — see
  :ref:`mixed-generalized-pairs`.

.. _mixed-integrand:

The scalar integrand
--------------------

``forward`` returns the **scalar** integrand of the coupled bilinear
form at one quadrature point — the same by-name dispatch as
:class:`~tensormesh.ElementAssembler` (see :ref:`forms-contract`), but
the field arguments are now *tensor-valued*:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Argument
     - Per-point shape
     - Meaning
   * - scalar field value (``p``, ``q``)
     - 0-d ``[]``
     - the field at the quadrature point
   * - scalar field gradient (``gradp``)
     - ``[D]``
     - physical-space gradient
   * - vector field value (``u``, ``v``)
     - ``[c]``
     - all :math:`c` components together
   * - vector field gradient (``gradu``)
     - ``[c, D]``
     - the Jacobian: row :math:`i` is :math:`\nabla u_i`

so ``gradu.diagonal().sum()`` is :math:`\nabla\cdot u`,
``(gradu * gradv).sum()`` is :math:`\nabla u : \nabla v`, and
``(gradu @ w).dot(v)`` is the convection term
:math:`(w\cdot\nabla)u\cdot v`. Only the arguments you name are
computed: a pass that touches neither field of a block is skipped, so
absent couplings (e.g. the zero pressure-pressure block of Stokes) cost
nothing beyond their sparsity-pattern slot.

.. important::

   The integrand must be **bilinear**: every term must contain exactly
   one trial factor and one test factor. Internally each
   :math:`(\alpha, \beta)` block is extracted by evaluating the
   integrand with the *other* fields zeroed — an identity for bilinear
   forms, silently wrong otherwise. The assembler guards this with a
   cheap zero-evaluation check and raises if your integrand has a
   constant or non-bilinear part. Nonlinear weak forms are handled the
   usual FEM way: linearize (Picard / Newton) and feed the previous
   iterate in as *data*, e.g. the lagged convection velocity ``w``
   below.

A Picard-linearized Navier-Stokes step, for example, adds one data
argument to the Stokes integrand (see
:doc:`/example_gallery/fluid/cavity` for the full script):

.. code-block:: python

   def forward(self, gradu, p, v, gradv, q, w):
       convection = self.rho * (gradu @ w).dot(v)
       diffusion = self.mu * (gradu * gradv).sum()
       return convection + diffusion \
           - p * gradv.diagonal().sum() \
           - q * gradu.diagonal().sum()


Building and calling
--------------------

:meth:`~tensormesh.MixedElementAssembler.from_mesh` builds the
assembler; extra positional/keyword arguments go to your
``__post_init__`` (physical parameters), exactly like the single-field
assemblers:

.. code-block:: python

   assembler = StokesAssembler.from_mesh(mesh, mu=1.0)
   K = assembler()                       # -> SparseMatrix  [N, N]

The default quadrature degree is ``2 * max(field orders)`` — exact for
the products of two highest-order basis functions on affine elements.
Pass ``quadrature_order=`` to override (e.g. for data-weighted or
convective terms); the shipped tables go up to degree 7.

The result is a square :class:`~tensormesh.sparse.SparseMatrix` of size
:math:`N = \sum_f n_f\, c_f`. Repeated calls (Picard iterations, time
steps) reuse the same cached index tensors, so a
:class:`~tensormesh.Condenser` built once keeps its condensation layout
across calls.

.. note::

   Mixed assembly is currently **single-device**: the
   :func:`~tensormesh.distributed.distributed` decorator
   (:doc:`distributed`) covers the single-field assemblers only.
   Distributing the block layout across ranks needs a global DOF
   numbering layer that does not exist yet — it is a planned ROADMAP
   item.


.. _mixed-layout:

The block DOF layout
--------------------

The global DOF vector stacks the fields in declaration order, node-major
within each field:

.. math::

   \mathrm{dof} \;=\; \mathrm{offset}_f + n_{\mathrm{local}}\cdot c_f + \mathrm{comp}.

You never index this by hand — ``assembler.layout`` (a
:class:`~tensormesh.BlockLayout`) provides the vocabulary:

.. code-block:: python

   layout = assembler.layout

   layout.n_dofs                      # total N
   layout.names                       # ["u", "p"]
   layout.n_nodes("u")                # nodes of the P2 velocity space
   layout.points("u")                 # their coordinates  [n_u, D]

   # masks and indices (all in global block-DOF numbering)
   layout.dof_mask("u")                          # every u DOF
   layout.dof_mask("u", node_mask, component=0)  # u_x on selected nodes
   layout.dof_mask("p", where=lambda x: x[:, 0] < 1e-6)   # coordinate predicate
   layout.boundary_mask("u")                     # topological boundary DOFs
   layout.dof_index("p", int(layout.node_ids("p")[0]))    # a single DOF (pressure pin)

   # moving between the block vector and per-field tensors
   fields = layout.split(sol)         # {"u": [n_u, 2], "p": [n_p]}
   sol = layout.cat(u=vel, p=0.0)     # assemble a block vector
   p_on_mesh = layout.prolong("p", fields["p"])   # P1 -> all mesh points
   f_on_p = layout.restrict("p", point_values)    # mesh points -> P1 nodes

Boundary conditions then go through the **unchanged**
:class:`~tensormesh.Condenser` — a mixed system is just a bigger DOF
vector with a boolean mask:

.. code-block:: python

   bc_mask = layout.dof_mask("u", mesh.boundary_mask)     # no-slip walls
   bc_mask[layout.dof_index("p", int(layout.node_ids("p")[0]))] = True

   bc_val = torch.zeros(layout.n_dofs, dtype=torch.float64)
   bc_val[layout.dof_mask("u", is_lid, component=0)] = 1.0

   condenser = Condenser(bc_mask, bc_val[bc_mask])
   K_, f_ = condenser(K, f)
   sol = condenser.recover(K_.solve(f_))

``boundary_mask`` detects the boundary **topologically** (a facet
referenced by exactly one cell), so it also works for meshes without
``is_boundary`` point data — e.g. gmsh imports — and for
generalized-order fields whose DOFs are not mesh nodes.


.. _mixed-vectors:

Load vectors
------------

:meth:`~tensormesh.MixedElementAssembler.assemble_vector` is the linear
form :math:`l(v)` of the mixed space: same dispatch, **test arguments
only**, output a dense ``[N]`` vector in the same block layout. Override
``forward_vector`` for the "natural" RHS of the problem, or pass a
one-off ``func=``:

.. code-block:: python

   class TransientNS(MixedElementAssembler):
       fields = [...]

       def forward(self, u, gradu, p, v, gradv, q, w):
           ...  # backward-Euler matrix side

       def forward_vector(self, v, uprev):
           return self.rho / self.dt * uprev.dot(v)

   f = assembler.assemble_vector(field_data={"uprev": ("u", u_prev)})

   # one-off linear forms: e.g. L2-project the vorticity of a P2
   # velocity onto the P1 pressure space
   omega_rhs = assembler.assemble_vector(
       func=lambda q, gradw: (gradw[1, 0] - gradw[0, 1]) * q,
       field_data={"w": ("u", velocity)},
   )

Because the load is integrated **in each field's own space**, sources on
a generalized-order field are consistent by construction — no manual
interpolation between spaces.


.. _mixed-data:

Data channels
-------------

All the single-field channels work in ``forward`` /
``forward_vector`` — ``x``, ``point_data`` (interpolated with the
mesh-order basis, ``grad{key}`` available), ``element_data``,
``scalar_data`` — plus one new channel:

``field_data={"name": (field, values)}``
   Data living on a **field's own DOFs**, interpolated with that
   field's basis (``grad{name}`` available too). This is the channel
   for Picard / time stepping on generalized-order fields: the previous
   iterate of a topological P2 velocity has no mesh-point
   representation, so it must ride on its own DOF vector:

   .. code-block:: python

      w = layout.split(sol)["u"]                    # [n_u, c] on the P2 DOFs
      K = assembler(field_data={"w": ("u", w)})

   On a mesh whose order equals the field order the two channels agree —
   ``point_data`` is then the simpler spelling.


.. _mixed-generalized-pairs:

Generalized order pairs
-----------------------

Field orders are **not** tied to the mesh order. Besides order 1 and
the mesh order (whose DOFs sit on mesh nodes), any other order builds a
*topological Lagrange DOF map*: one DOF per vertex, :math:`k-1` per
unique (oriented) edge, plus cell interiors. In practice this means:

* **Taylor-Hood on a linear gmsh mesh.** A ``Field(order=2)`` on an
  imported P1 mesh creates the quadratic space topologically — no
  re-meshing at order 2. The :doc:`cylinder flow
  </example_gallery/fluid/cylinder_flow>` and :doc:`flow past obstacles
  </example_gallery/fluid/flow_obstacles>` examples run exactly this
  way.
* **Higher pairs.** P3-P2 and beyond work the same; edge DOFs are
  stored on *oriented* edges so adjacent elements agree and the spaces
  stay :math:`C^0`-conforming at any order.
* **Geometry stays isoparametric at the mesh order.** A P2 field on a
  curved (order-2) mesh integrates on the curved geometry even when
  paired with a P1 pressure; sub- and super-parametric fields are both
  correct.

Current limits, checked with explicit errors: in 3D the topological map
covers vertex + edge DOFs (tetra P2 works; orders needing face DOFs
raise ``NotImplementedError``), and the quadrature tables stop at
degree 7 — about P3-P2 in practice.

For fields with topological DOFs, ``layout.node_ids`` (a mesh-node
concept) is unavailable; use ``layout.points(name)``,
``layout.boundary_mask(name)`` and ``layout.dof_mask(name, where=...)``,
which work for every field kind, and move data with
``layout.prolong`` / ``layout.restrict`` (isoparametric interpolation).


Worked examples
---------------

The fluid example gallery is built entirely on mixed assembly, in
increasing order of complexity:

* :doc:`/example_gallery/fluid/stokes_taylor_hood` — manufactured
  Stokes solution with a convergence study; the best starting point.
* :doc:`/example_gallery/fluid/cavity` — steady Navier-Stokes (Picard),
  2D and 3D.
* :doc:`/example_gallery/fluid/cylinder_flow` — transient
  Navier-Stokes with ``forward_vector`` and ``field_data`` on a P1
  gmsh mesh.
* :doc:`/example_gallery/fluid/rayleigh_benard` — a **three-field**
  Boussinesq system (velocity, pressure, temperature) with an
  off-diagonal buoyancy block.

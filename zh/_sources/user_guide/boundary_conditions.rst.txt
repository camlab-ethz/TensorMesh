Boundary Conditions
===================

Once you have a stiffness matrix ``K`` and a load vector ``b`` from
:doc:`forms`, you still need to enforce the problem's boundary
conditions. In TensorMesh today this is done via *static condensation*
of Dirichlet DOFs — a single call to :class:`~tensormesh.Condenser`
takes a full system and returns a reduced system on the interior
DOFs, ready to solve.

Neumann conditions don't need a separate operator: they appear
naturally in the weak form as a boundary integral, assembled with
:class:`~tensormesh.FacetAssembler`.

The rest of this chapter walks through both, plus the per-component
masks needed for vector-valued problems (linear elasticity, Stokes,
…), time-varying boundary data, and the gotchas that come up most
often in practice.


Why static condensation
-----------------------

Splitting the DOFs into "inner" (free) and "outer" (Dirichlet-fixed)
turns the linear system

.. math::

   \begin{bmatrix}
       K_{ii} & K_{io} \\
       K_{oi} & K_{oo}
   \end{bmatrix}
   \begin{bmatrix} u_i \\ u_o \end{bmatrix}
   =
   \begin{bmatrix} b_i \\ b_o \end{bmatrix}

into a smaller solve on just the interior block:

.. math::

   K_{ii} \, u_i \;=\; b_i - K_{io} \, u_o,
   \qquad u_o \text{ prescribed.}

Condensation avoids Lagrange multipliers, keeps an SPD operator SPD,
and is differentiable end-to-end — the boundary values flow through
the right-hand-side correction and pick up gradients in the usual
way.


The Condenser API
-----------------

Construct a :class:`~tensormesh.Condenser` from a boolean mask over
all DOFs:

.. code-block:: python

   from tensormesh import Condenser

   condenser = Condenser(mesh.boundary_mask)              # zero values (default)
   condenser = Condenser(mesh.boundary_mask, values)      # prescribed values

* ``dirichlet_mask: [n_dof]`` — bool tensor; ``True`` where the DOF is
  fixed. For scalar problems this is just ``mesh.boundary_mask``
  (length ``n_points``); for vector-valued problems see
  :ref:`bc-vector-valued` below.
* ``dirichlet_value`` — optional 1D tensor giving the prescribed
  values. May be ``[n_dof]`` (the constructor slices it down to the
  boundary entries) or already-sliced ``[n_outer_dof]``. Defaults to
  zeros.

Three methods cover the common workflow:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - What it does
   * - ``condenser(K, b)``
     - Condense both at once. Returns ``(K_inner, b_inner)``. **Run
       this first** — it caches the sparsity layout for subsequent
       calls.
   * - ``condenser.condense_rhs(b_new)``
     - Re-condense only the right-hand side, reusing the cached
       layout. Useful when ``K`` is fixed but ``b`` changes
       (transient problems, batched RHS).
   * - ``condenser.recover(u_inner)``
     - Glue the interior solution and the prescribed boundary values
       back together into a full ``[n_dof]`` vector.
   * - ``condenser.restrict(f)`` / ``condenser.prolong(f_inner)``
     - BC-value-free linear projections (drop / scatter with zero on
       the boundary). The ones to use when the quantity being lifted
       should be zero on the boundary regardless of the Dirichlet
       value — e.g. per-stage slopes in
       :class:`~tensormesh.ode.ImplicitLinearRungeKutta`. See
       :ref:`ti-boundaries` for the integrator-class pattern.

For time-dependent BCs, ``condenser.update_dirichlet(new_values)``
swaps in a new boundary-value vector (accepting either the full
``[n_dof]`` form or the already-sliced ``[n_outer_dof]`` form, just
like the constructor). The cached sparsity layout is independent of
the prescribed values, so this call is cheap.

The first ``condenser(K, b)`` call computes and caches the inner /
outer index split for ``K``'s sparsity pattern; subsequent calls on
matrices with the same layout reuse that work.

.. note::

   :class:`~tensormesh.Condenser` is a :class:`torch.nn.Module`, and
   every tensor it owns (the boolean mask, the prescribed values, and
   the lazily computed index buffers) is registered as a buffer. So

   .. code-block:: python

      condenser = Condenser(mesh.boundary_mask).to(mesh.device)

   moves the operator alongside ``K``, ``b``, and the mesh. If you
   forget, ``__call__`` and :meth:`~tensormesh.Condenser.condense_rhs`
   still auto-promote to the input matrix's device/dtype on every
   call — convenient, but the per-call ``.to(...)`` traffic shows up
   in profiles for tight transient loops.


Homogeneous Dirichlet
---------------------

The bread-and-butter case — zero values on the boundary — is exactly
what the :doc:`../getting_started/quickstart` does:

.. code-block:: python

   condenser = Condenser(mesh.boundary_mask)
   K_, b_ = condenser(K, b)
   u_inner = K_.solve(b_)
   u = condenser.recover(u_inner)

Three lines, including the solve. The recovered ``u`` has zeros at
boundary DOFs and the FEM solution everywhere else.


Non-homogeneous Dirichlet
-------------------------

Prescribe per-DOF values by passing them to the constructor. Two
shapes are accepted — choose whichever is more convenient:

.. code-block:: python

   # Option A: full-length [n_dof] vector (Condenser slices internally)
   values = torch.zeros(mesh.n_points, dtype=mesh.dtype)
   x = mesh.points[:, 0]
   values[(x == 1.0) & mesh.boundary_mask] = 1.0       # u = 1 on right edge
   condenser = Condenser(mesh.boundary_mask, values)

   # Option B: pre-sliced [n_outer_dof] vector
   sliced = values[mesh.boundary_mask]
   condenser = Condenser(mesh.boundary_mask, sliced)

The condensed RHS picks up the term :math:`-K_{io}\, u_o`
automatically, so the rest of the pipeline is unchanged.

.. tip::

   ``Mesh.gen_rectangle`` / ``gen_cube`` place boundary nodes at
   exactly integer coordinates, so ``x == 1.0`` works. For meshes
   loaded from Gmsh / VTK files, prefer

   .. code-block:: python

      mask = torch.isclose(x, torch.tensor(1.0, dtype=mesh.dtype)) & mesh.boundary_mask

   so a 1e-16 perturbation from the mesher doesn't silently drop a
   row of nodes.


.. _bc-vector-valued:

Vector-valued problems (elasticity, Stokes)
-------------------------------------------

For an elasticity problem, each node carries ``dim`` displacement
components, and assemblers from
:class:`~tensormesh.LinearElasticityElementAssembler` produce a
global system of size ``n_dof = n_points * dim``. Two extra steps
turn ``mesh.boundary_mask`` (shape ``[n_points]``) into a DOF-level
Dirichlet mask:

.. code-block:: python

   dim = mesh.points.shape[1]

   # Per-(node, component) mask, then flatten to per-DOF
   bc_mask = torch.zeros(mesh.n_points, dim, dtype=torch.bool)

   # Clamp the bottom edge in all directions, leave top free
   y = mesh.points[:, 1]
   bc_mask[(y == 0.0) & mesh.boundary_mask, :] = True

   # Pin x-displacement on the right edge (roller BC), y free
   x = mesh.points[:, 0]
   bc_mask[(x == 1.0) & mesh.boundary_mask, 0] = True

   condenser = Condenser(bc_mask.flatten())          # [n_points * dim]

The same flatten-trick applies to prescribed values: build a
``[n_points, dim]`` tensor of target displacements, then ``.flatten()``
before handing it to the constructor. The convention is that DOFs
are interleaved per node (``[u_x_0, u_y_0, u_x_1, u_y_1, ...]``),
matching the layout of TensorMesh's elasticity assemblers.


Time-varying boundary values
----------------------------

Build the condenser once, then push new values per timestep without
rebuilding the layout:

.. code-block:: python

   condenser = Condenser(mesh.boundary_mask, initial_values)
   K_, _ = condenser(K, torch.zeros(mesh.n_points, dtype=mesh.dtype))   # caches the split

   for t in time_steps:
       condenser.update_dirichlet(values_at(t))            # cheap setter
       b_new   = build_rhs_for(t)
       b_inner = condenser.condense_rhs(b_new)             # reuses cached K_io
       u       = condenser.recover(K_.solve(b_inner))

:meth:`~tensormesh.Condenser.condense_rhs` requires that
``__call__`` has been invoked at least once on the same matrix
layout — that's what populates the cached :math:`K_{io}` block it
applies. Call it before the loop, even if you discard the first
solve.


Combining Dirichlet with Neumann
--------------------------------

Non-zero traction or flux on a portion of :math:`\partial\Omega` is
a boundary integral in the right-hand side of the weak form:

.. math::

   l(v) \;=\; \int_{\Omega} f \cdot v \, \mathrm{d}\Omega
   \;+\; \int_{\Gamma_N} g \cdot v \, \mathrm{d}S .

There is no Condenser involvement — natural BCs are handled
entirely by the weak form. Assemble the surface integral with a
:class:`~tensormesh.FacetAssembler` subclass and add the result to
your load vector before condensing:

.. code-block:: python

   import torch
   from tensormesh import Mesh, Condenser, FacetAssembler
   from tensormesh.assemble import LaplaceElementAssembler

   mesh = Mesh.gen_rectangle(chara_length=0.1)
   x    = mesh.points[:, 0]

   # ----------------------- volume bilinear form ----------------------------
   K = LaplaceElementAssembler.from_mesh(mesh)()

   # ----------------------- Neumann flux on the right edge ------------------
   right_edge = (x == 1.0) & mesh.boundary_mask         # [n_points] bool

   class FluxAssembler(FacetAssembler):
       def forward(self, v):
           return v                                     # ∫_{Γ_N} g·v  with g=1

   b_neumann = FluxAssembler.from_mesh(mesh, boundary_mask=right_edge)()

   # Optional volume load f = 1
   from tensormesh.assemble import const_node_assembler
   b_volume = const_node_assembler(1.0).from_mesh(mesh)()

   b = b_volume + b_neumann

   # ----------------------- Dirichlet on the *complement* ------------------
   dirichlet = mesh.boundary_mask & ~right_edge         # bottom/top/left edges
   condenser = Condenser(dirichlet)

   K_, b_ = condenser(K, b)
   u      = condenser.recover(K_.solve(b_))

Two points worth highlighting:

* ``FacetAssembler.from_mesh(mesh, boundary_mask=right_edge)`` accepts
  the same kind of point-level boolean mask the :class:`~tensormesh.Condenser`
  uses, so the *same* logical "right edge" tensor drives both the
  Neumann integral and the choice of *which* nodes stay free.
* The Dirichlet mask passed to the :class:`~tensormesh.Condenser` is the
  boundary **excluding** the Neumann region — otherwise the prescribed
  values would overwrite the contribution from the flux integral.


Mixed regions and component-wise BCs
------------------------------------

For separate Dirichlet and Neumann patches (or per-component mixes
in elasticity), build masks from coordinates and combine them with
boolean algebra:

.. code-block:: python

   x, y = mesh.points[:, 0], mesh.points[:, 1]
   left  = (x == 0.0) & mesh.boundary_mask
   right = (x == 1.0) & mesh.boundary_mask

   # Scalar problem: fix left to zero, ramp on right
   dirichlet_mask = left | right
   values = torch.zeros(mesh.n_points, dtype=mesh.dtype)
   values[right] = 1.0
   condenser = Condenser(dirichlet_mask, values)

For meshes loaded from Gmsh with named physical groups, those
groups land in ``mesh.point_data`` as boolean masks; you can read
them by key (e.g. ``mesh.point_data["clamped_face"]``) instead of
building masks from coordinates by hand.


Gotchas
-------

* **Sparsity layout is cached on the first call.** The
  :class:`~tensormesh.Condenser` remembers the row/col split for the
  matrix you first hand it via ``__call__``; passing a matrix with a
  different sparsity pattern triggers an assertion. If your geometry
  or element type changes, build a new :class:`~tensormesh.Condenser`.
* **Vector-valued masks must be flattened.** A
  ``[n_points, dim]`` boolean tensor is not a valid ``dirichlet_mask`` —
  call ``.flatten()`` before passing it. Likewise, prescribed values
  for elasticity should be a flattened ``[n_points * dim]`` (or
  ``[n_outer_dof]``) vector.
* **Coordinate masks on float-perturbed meshes.** Exact comparison
  (``x == 1.0``) works on TensorMesh's built-in generators but can
  fail on imported meshes; use ``torch.isclose`` when in doubt.
* **Dirichlet wins on overlapping regions.** If a node is both in
  the Dirichlet mask and in the support of a Neumann facet integral,
  the Neumann contribution to that row is discarded by the
  condensation (the row is removed). Exclude the Neumann patch from
  the Dirichlet mask when you mean both to apply.
* **Same mask must drive the solve and the recover.** Don't mutate
  ``mesh.boundary_mask`` between condensing and recovering — the
  :class:`~tensormesh.Condenser` references the same buffer it was given at
  construction time.


What's not built in today
-------------------------

The :mod:`tensormesh.operator` module ships exactly one operator —
:class:`~tensormesh.Condenser` for Dirichlet via static condensation.
A few neighbouring techniques are *not* first-class operators:

* **Lagrange multipliers** for Dirichlet — an alternative to
  condensation that keeps the original DOF layout but produces a
  saddle-point system. Workable by hand if you build the augmented
  matrix yourself, but no helper class.
* **Periodic boundary conditions** — typically wired up by
  identifying matched DOFs and merging the corresponding rows /
  columns in the assembled matrix. No helper.
* **Robin (mixed) conditions** as a top-level operator — but the
  bilinear-form contribution :math:`\int_{\Gamma} \alpha\, u\, v\,
  \mathrm{d}S` is a one-line
  :class:`~tensormesh.FacetAssembler` subclass; just add it to
  ``K`` before condensing.

Penalty contact, surface tension, pressure loads, and similar
energy-based boundary terms *are* first-class — see
:class:`~tensormesh.assemble.ContactAssembler` (a
:class:`~tensormesh.FacetAssembler` subclass with an
``element_energy`` hook for the surface energy density). The
example gallery has worked recipes for the cases that need them.


What's next
-----------

* :doc:`linear_solvers` — solve the condensed system with the right
  backend.
* :doc:`time_integration` — transient problems where the same
  condenser is reused per step.
* :doc:`differentiability` — taking gradients through the boundary
  values themselves (inverse / control problems).
* :doc:`../getting_started/quickstart` — the full pipeline including
  homogeneous BCs.

Lid-Driven Cavity
=================

The lid-driven cavity is the textbook benchmark for incompressible
Navier-Stokes solvers. The physics is simple — a box of fluid, the top
wall slides at unit speed, no-slip on the other walls, no body forces —
but at moderate Reynolds numbers it already exhibits a primary vortex
plus secondary corner eddies that any reasonable solver must reproduce.
Two scripts in ``examples/fluid/cavity/`` run the steady-state problem at
:math:`\mathrm{Re} = 100`: ``cavity.py`` on a triangulated unit square
and ``cavity_3d.py`` on a tetrahedral unit cube. They share a single
**dimension-generic** ``NavierStokesAssembler``, so the 3D case is
essentially a mesh swap.

.. caution::

   TensorMesh does not yet natively support **mixed-element function
   spaces** for coupled multi-physics — there is no built-in Taylor-Hood
   (e.g. P2 velocity / P1 pressure) velocity-pressure pairing that would
   satisfy the inf-sup (LBB) condition out of the box. These examples
   therefore use **equal-order P1-P1** for every field and recover
   stability with **SUPG/PSPG stabilization**, which is why the weak form
   below carries the extra residual-based ``tau`` terms. Native support
   for mixed elements and tighter coupled-multiphysics workflows is on the
   roadmap and will be added incrementally; until then, treat the
   ``NavierStokesAssembler`` here as an example-grade pattern rather than a
   stable library API.


Problem
-------

The strong form of the steady incompressible Navier-Stokes equations is

.. math::

   \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u}
   \quad \text{in } \Omega,
   \qquad
   \nabla \cdot \mathbf{u} \;=\; 0,

with :math:`\Omega = (0, 1)^d`, :math:`\mu = 1/\mathrm{Re}`,
:math:`\rho = 1`, and boundary conditions

* top lid (:math:`y = 1`): :math:`\mathbf{u} = (1, 0, \dots)` (moving wall)
* other walls: :math:`\mathbf{u} = \mathbf{0}` (no-slip)
* one node: :math:`p = 0` (pressure pin to fix the null space).


Weak form and stabilization
---------------------------

The standard mixed weak form treats velocity and pressure together.
Because the scripts use **equal-order** P1-P1 elements for both, the
discrete LBB condition is violated and the naive Galerkin formulation has
spurious pressure modes. The fix is **SUPG/PSPG stabilization**: add a
residual-based perturbation to both the momentum and the continuity
equations, scaled by a mesh-size-dependent parameter :math:`\tau`.

Denote by
:math:`\mathbf{w} = \mathbf{w}^{n}` the  velocity at the previous Picard iterate. The SUPG/PSPG-stabilized weak form seeks :math:`\mathbf{u} \in (H^1_0(\Omega))^d + \mathbf{u}_b` and :math:`p \in L^2(\Omega)/\mathbb{R}`
such that for all :math:`\mathbf{v} \in (H^1_0(\Omega))^d, q \in L^2(\Omega)/\mathbb{R}`,

.. math::

   \int_\Omega \rho\,(\mathbf{w}\cdot\nabla)\mathbf{u}\cdot\mathbf{v}\,\mathrm{d}x
   + \int_\Omega \mu\,\nabla\mathbf{u} : \nabla\mathbf{v}\,\mathrm{d}x
   - \int_\Omega p\,(\nabla\cdot\mathbf{v})\,\mathrm{d}x \;+
   \int_\Omega q\,(\nabla\cdot\mathbf{u})\,\mathrm{d}x \;=\; 0 .

Here, :math:`\mathbf{u}_b` incorporates the non-homogeneous velocity boundary conditions; :math:`L^2(\Omega)/\mathbb{R}` is the space modulo constants (since pressure is only determined up to an additive constant). 

For equal-order P1-P1 this Galerkin form is unstable, so we append element-wise **SUPG/PSPG** terms built from the strong momentum residual

.. math::

   \mathbf{R}(\mathbf{u}, p) \;=\;
   \rho\,(\mathbf{w}\cdot\nabla)\mathbf{u} + \nabla p - \mu\,\Delta\mathbf{u}.

Note that the viscous term :math:`\mu\,\Delta\mathbf{u}` vanishes on P1 elements. The stabilized discretization uses P1-P1 elements for :math:`\mathbf{u}` and :math:`p`, and adds

.. math::

   \underbrace{\sum_e \int_{\Omega_e}
       \tau\,(\mathbf{w}\cdot\nabla)\mathbf{v}\cdot\mathbf{R}\,\mathrm{d}x}_{\text{SUPG}}
   \;+\;
   \underbrace{\sum_e \int_{\Omega_e}
       \tau\,\nabla q\cdot\mathbf{R}\,\mathrm{d}x}_{\text{PSPG}}

to the aforementioned weak form, with :math:`\tau` being a stabilization parameter.

The implementation is a custom
:class:`~tensormesh.ElementAssembler` defined locally in the example. Its
``forward`` returns the :math:`(d{+}1) \times (d{+}1)` block coupling one
test node to one trial node — :math:`d` velocity components plus pressure
— which the assembler stamps into a block-COO sparse matrix exactly as
the built-in vector-valued assemblers do. The block is built from four
named sub-blocks rather than entry-by-entry, so each physical term is
visible:

.. code-block:: python
   :caption: examples/fluid/cavity/cavity.py (essence)

   class NavierStokesAssembler(ElementAssembler):
       def __post_init__(self, rho=1.0, mu=0.01, tau=0.1):
           self.rho, self.mu, self.tau = rho, mu, tau

       def forward(self, u, v, gradu, gradv, w_prev):
           dim = gradu.shape[0]
           eye = torch.eye(dim, dtype=gradu.dtype, device=gradu.device)

           # velocity-velocity: convection + diffusion + SUPG (diagonal in components)
           convection = self.rho * torch.dot(w_prev, gradv) * u
           diffusion  = self.mu * torch.dot(gradu, gradv)
           supg       = self.rho * torch.dot(w_prev, gradv) * self.tau * torch.dot(w_prev, gradu)
           A_uu = (convection + diffusion + supg) * eye               # [dim, dim]

           # pressure gradient in momentum (+ PSPG); divergence in continuity (+ PSPG)
           B_up = -v * gradu + self.tau * torch.dot(w_prev, gradu) * gradv          # [dim]
           B_pu =  u * gradv + self.tau * self.rho * torch.dot(w_prev, gradv) * gradu  # [dim]
           C_pp =  self.tau * torch.dot(gradv, gradu)                  # PSPG pressure Laplacian

           top    = torch.cat([A_uu, B_up.unsqueeze(1)], dim=1)        # [dim, dim+1]
           bottom = torch.cat([B_pu, C_pp.reshape(1)]).unsqueeze(0)    # [1, dim+1]
           return torch.cat([top, bottom], dim=0)                      # [dim+1, dim+1]

The four named sub-blocks map one-to-one onto the terms of the stabilized
weak form above:

* ``A_uu`` — convection :math:`\rho\,(\mathbf{w}\cdot\nabla)\mathbf{u}\cdot\mathbf{v}`,
  diffusion :math:`\mu\,\nabla\mathbf{u}:\nabla\mathbf{v}`, and the SUPG
  convection term :math:`\tau\,(\mathbf{w}\cdot\nabla\mathbf{v})\,\rho\,(\mathbf{w}\cdot\nabla)\mathbf{u}`
  (diagonal in the components, hence ``* eye``);
* ``B_up`` — pressure gradient :math:`-p\,\nabla\cdot\mathbf{v}` and its SUPG
  counterpart :math:`\tau\,(\mathbf{w}\cdot\nabla\mathbf{v})\cdot\nabla p`;
* ``B_pu`` — divergence :math:`q\,\nabla\cdot\mathbf{u}` and the PSPG
  convection term :math:`\tau\,\nabla q\cdot\rho\,(\mathbf{w}\cdot\nabla)\mathbf{u}`;
* ``C_pp`` — the PSPG pressure-Laplacian :math:`\tau\,\nabla p\cdot\nabla q`.

Because ``dim`` is read from ``gradu.shape[0]``, the same ``forward``
produces a :math:`3{\times}3` block in 2D and a :math:`4{\times}4` block
in 3D with no changes.


Picard iteration
----------------

The convection term :math:`(\mathbf{u}\cdot\nabla)\mathbf{u}` is
linearized by passing the previous-iterate velocity
:math:`\mathbf{w}^{n}` as ``w_prev`` to the assembler:

.. math::

   \rho\, (\mathbf{w}^{n} \cdot \nabla)\mathbf{u}^{n+1}
   \;=\; -\nabla p^{n+1} + \mu\, \Delta \mathbf{u}^{n+1}.

This converges geometrically for moderate Reynolds numbers; the script
iterates until ``||u_new - u_full|| / ||u_new|| < 1e-4``, typically ~8
iterations at Re=100.

.. code-block:: python

   assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
   condenser = Condenser(bc_mask, bc_val)

   for i in range(max_iter):
       w_prev = u_full.reshape(-1, n_dof)[:, :2]   # previous velocity
       K = assembler(points, point_data={"w_prev": w_prev})
       f = torch.zeros(n_points * n_dof, dtype=torch.float64)
       K_, f_ = condenser(K, f)
       u_new  = condenser.recover(K_.solve(f_))

       diff = torch.norm(u_new - u_full) / (torch.norm(u_new) + 1e-8)
       u_full = u_new
       if diff < tol:
           break

A few details that matter:

* **DOF layout.** The unknowns are interleaved node-major:
  ``[u_0, v_0, p_0, u_1, v_1, p_1, …]`` in 2D (``[u, v, w, p]`` per node
  in 3D). The script's small ``component_dofs(n_points, n_dof, comp)``
  helper turns "component ``comp`` at every node" into the flat global
  indices the :class:`~tensormesh.Condenser` mask expects.
* **Pressure pin.** ``bc_mask[n_dof - 1] = True`` clamps pressure to zero
  at node 0 — required because pressure is only determined up to an
  additive constant in incompressible flow.
* ``w_prev`` is passed by name in ``point_data`` and arrives in
  ``forward`` as the matching keyword argument; see
  :doc:`../../user_guide/forms` for the dispatch contract.

.. figure:: /_static/fluid/cavity_results.png
   :alt: Lid-driven cavity speed magnitude and pressure at Re=100
   :width: 100%

   Output of ``cavity.py`` at Re = 100. Left: speed magnitude
   :math:`\|u\|` — the moving lid drags fluid into the upper right,
   sweeping it down the right wall and forming the primary vortex.
   Right: pressure field, with the characteristic high-pressure spot in
   the upper-right corner where the lid stagnates against the wall.


Going to 3D — ``cavity_3d.py``
------------------------------

``cavity_3d.py`` solves the same physics on a unit cube. Because the
``NavierStokesAssembler`` is dimension-generic, the differences are
mechanical:

* ``Mesh.gen_cube(chara_length=…)`` (tetrahedral) replaces
  ``Mesh.gen_rectangle``.
* The DOF layout is per-node ``[u, v, w, p]`` (``n_dof = 4``) instead of
  ``[u, v, p]``; ``w_prev`` now slices the first three components.
* The output is volumetric: ``cavity_3d.vtu`` for ParaView, plus a
  cross-section slice at :math:`z = 0.5` rendered via PyVista as
  ``cavity_3d.png``.

Everything else — the Picard loop, the SUPG/PSPG stamp, the pressure
pin, the use of :class:`~tensormesh.Condenser` — is unchanged:

.. code-block:: python
   :caption: examples/fluid/cavity/cavity_3d.py (essence)

   mesh = Mesh.gen_cube(chara_length=chara_length).double()
   n_dof = 4                                          # [u, v, w, p] per node
   assembler = NavierStokesAssembler.from_mesh(mesh, rho=rho, mu=mu, tau=tau)
   condenser = Condenser(bc_mask, bc_val)

   for i in range(max_iter):
       w_prev = u_full.reshape(-1, n_dof)[:, :3]      # 3D velocity
       K = assembler(mesh.points, point_data={"w_prev": w_prev})
       f = torch.zeros(n_points * n_dof, dtype=torch.float64)
       K_, f_ = condenser(K, f)
       u_full = condenser.recover(K_.solve(f_))

The 4-DOFs-per-node layout makes the linear system large quickly — at
``chara_length=0.05`` it is on the order of a million unknowns — so a GPU
backend (``backend="cudss"`` or ``"pytorch"``, see
:doc:`../../user_guide/linear_solvers`) is recommended once you go beyond
a few thousand nodes.

.. figure:: /_static/fluid/cavity_3d.png
   :alt: 3D lid-driven cavity speed and pressure on the z=0.5 mid-plane
   :width: 100%

   Output of ``cavity_3d.py``: speed magnitude (left) and pressure
   (right) on the :math:`z=0.5` mid-plane slice through the cube. The
   flow pattern matches the 2D solution near the lid but decays toward
   the front and back walls, so the mid-plane shows weaker recirculation
   than the strict-2D case. Full 3D fields are written to
   ``cavity_3d.vtu`` for ParaView.


Running it
----------

.. code-block:: bash

   cd examples/fluid/cavity
   python cavity.py        # 2D, writes cavity_results.png
   python cavity_3d.py     # 3D, writes cavity_3d.vtu and cavity_3d.png

The console reports the relative residual at each Picard step plus the
final convergence message. For 2D, compare the stream-function contours
qualitatively against the Ghia / Ghia / Shin reference (1982) for
Re=100 — the primary vortex center should match to within a few percent.
For 3D, open ``cavity_3d.vtu`` in ParaView for full volumetric inspection
(streamlines, iso-surfaces); the PNG is a quick sanity check.


What's next
-----------

* :doc:`cylinder_flow` — adds transient time stepping.
* :doc:`taylor_green` — the same family of assemblers, now used for a
  quantitative convergence study against an exact solution.
* :doc:`../../user_guide/forms` — argument-dispatch contract for
  vector-valued ``forward`` returns.
* :doc:`../../user_guide/linear_solvers` — backend choice for the larger
  3D systems.

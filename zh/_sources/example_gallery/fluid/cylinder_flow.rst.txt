Cylinder Flow (Vortex Shedding)
================================

The transient counterpart to the steady cavity examples. The
script ``examples/fluid/cylinder_flow/cylinder_flow.py`` runs the
classic *flow past a circular cylinder* benchmark — a long
rectangular channel with a small cylinder near the inlet. At
:math:`\mathrm{Re} = 100` the wake becomes unstable, vortices
shed alternately from the top and bottom of the cylinder, and a
**von Kármán vortex street** propagates downstream. The geometry
and parameters follow the DFG 2D benchmark (Schäfer & Turek,
1996).

Discretization: **Taylor-Hood P2-P1** via
:class:`~tensormesh.MixedElementAssembler`, backward Euler in time
with Picard sub-iterations. Two things make this script the
"advanced" mixed example:

* the gmsh/:class:`~tensormesh.MeshGen` channel mesh is **linear**
  — the quadratic velocity space is generated *topologically* on
  top of it (one DOF per unique edge), no order-2 re-meshing;
* the load vector and the vorticity post-processing both go through
  :meth:`~tensormesh.MixedElementAssembler.assemble_vector`, with
  the lagged velocities riding on the P2 field's own DOFs via
  ``field_data``.


Problem
-------

The transient incompressible Navier-Stokes system,

.. math::

   \rho\, \frac{\partial \mathbf{u}}{\partial t}
   + \rho\, (\mathbf{u} \cdot \nabla)\mathbf{u}
   \;=\; -\nabla p + \mu\, \Delta \mathbf{u},
   \qquad \nabla \cdot \mathbf{u} = 0,

on the channel :math:`\Omega = [0, 2.2] \times [0, 0.41]` with a
circular cylinder of radius :math:`r = 0.05` centered at
:math:`(0.2, 0.2)`. Boundary conditions:

* inlet (:math:`x = 0`): parabolic profile
  :math:`u_x(y) = 4\,U_\text{max}\, y\, (H - y) / H^2`,
  :math:`u_y = 0`,
* walls and cylinder surface: no-slip,
* outlet (:math:`x = 2.2`): "do-nothing" (natural), which also
  fixes the pressure gauge — **no pressure pin is needed**,

with :math:`U_\text{max} = 1.5` giving :math:`\bar{U} = 1`,
:math:`D = 0.1`, :math:`\rho = 1`, :math:`\mu = 10^{-3}`, and
hence :math:`\mathrm{Re} = \rho \bar{U} D / \mu = 100`.


Time integration: backward Euler + Picard
-----------------------------------------

The transient term is discretized with backward Euler — implicit,
unconditionally stable. Each timestep solves

.. math::

   \rho\, \frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t}
   + \rho\, (\mathbf{w} \cdot \nabla)\mathbf{u}^{n+1}
   \;=\; -\nabla p^{n+1} + \mu\, \Delta \mathbf{u}^{n+1},

with the advecting velocity :math:`\mathbf{w}` updated by one or
two Picard sub-iterations per step. Matrix side and load side are
the two halves of one assembler:

.. code-block:: python
   :caption: examples/fluid/cylinder_flow/cylinder_flow.py (essence)

   class NavierStokesTransientAssembler(MixedElementAssembler):
       fields = [
           Field(trial="u", test="v", order=2, components=2),
           Field(trial="p", test="q", order=1),
       ]

       def __post_init__(self, rho=1.0, mu=0.01, dt=1e-3):
           self.rho, self.mu, self.dt = rho, mu, dt

       def forward(self, u, gradu, p, v, gradv, q, w):
           mass = self.rho / self.dt * u.dot(v)
           convection = self.rho * (gradu @ w).dot(v)
           diffusion = self.mu * (gradu * gradv).sum()
           return mass + convection + diffusion \
               - p * gradv.diagonal().sum() \
               - q * gradu.diagonal().sum()

       def forward_vector(self, v, uprev):
           return self.rho / self.dt * uprev.dot(v)

The time loop is then three lines of assembly per Picard pass —
note the ``field_data`` channel carrying the lagged P2 velocities:

.. code-block:: python

   for step in range(n_steps):
       u_prev = layout.split(sol)["u"]        # [n_u, 2] on the P2 DOFs
       for _ in range(picard_iter):
           w = layout.split(u_iter)["u"]
           K = assembler(field_data={"w": ("u", w)})
           f = assembler.assemble_vector(field_data={"uprev": ("u", u_prev)})
           K_, f_ = condenser(K, f)
           u_iter = condenser.recover(K_.solve(f_))
       sol = u_iter


Boundary conditions on the topological P2 space
-----------------------------------------------

MeshGen meshes carry no ``is_boundary`` point data, and half the
velocity nodes are edge midpoints that are not mesh points at all.
Both problems disappear with the layout's topological helpers —
``boundary_mask`` classifies DOFs by facet incidence, and
``points("u")`` gives every velocity node a coordinate:

.. code-block:: python

   x_u = layout.points("u")
   is_boundary = layout.split(layout.boundary_mask("u"))["u"][:, 0]
   is_inlet = is_boundary & (x_u[:, 0] <= eps)
   is_outlet = x_u[:, 0] >= length - eps
   no_slip = is_boundary & ~is_inlet & ~is_outlet   # walls + cylinder

   bc_mask = layout.dof_mask("u", node_mask=is_inlet | no_slip)
   y_in = x_u[is_inlet, 1]
   bc_val[layout.dof_mask("u", node_mask=is_inlet, component=0)] = \
       4.0 * u_max * y_in * (height - y_in) / (height * height)

The outlet is left free (do-nothing), which anchors the pressure —
the saddle-point system is non-singular without a pin.


Post-processing: vorticity as a mixed load vector
-------------------------------------------------

At every saved frame the script recovers the vorticity
:math:`\omega = \partial_x u_y - \partial_y u_x` by :math:`L^2`
projection onto the P1 pressure space. The projection RHS
:math:`\int \omega_h\, q\,\mathrm{d}x` uses the **exact P2
gradient** of the velocity — a one-line ``func=`` linear form,
with the velocity passed via ``field_data``:

.. code-block:: python

   omega_rhs = assembler.assemble_vector(
       func=lambda q, gradw: (gradw[1, 0] - gradw[0, 1]) * q,
       field_data={"w": ("u", velocity)},
   )
   omega = m_mat.solve(layout.split(omega_rhs)["p"])   # P1 mass matrix

The von Kármán street is the qualitative signature to look for:
once the wake destabilizes, vortices shed alternately from the top
and bottom of the cylinder and convect downstream at roughly the
mean inlet velocity.


Output and rendering
--------------------

* **Frame sequence.** Every ``save_every`` steps the script renders
  a three-panel PNG (vorticity, speed, pressure) into ``frames/``
  via ``mesh.plot``.
* **MP4 rendering.** The companion script
  ``examples/fluid/cylinder_flow/render_video.py`` stitches the
  ``frames/*.png`` sequence into ``vortex_street.mp4`` with an
  ``ffmpeg`` concat pass.
* **Final snapshot.** ``cylinder_flow_final.png`` is the same
  three-panel figure for the last step.

.. raw:: html

   <video controls loop muted preload="metadata"
          width="100%" style="max-width: 900px; display: block; margin: 1em auto;">
     <source src="../../_static/fluid/vortex_street.mp4" type="video/mp4">
     Your browser does not support the HTML5 video tag.
   </video>

*Output of* ``cylinder_flow.py`` *(rendered to MP4 by*
``render_video.py``\ *): the developed von Kármán vortex street
behind the cylinder. After an initial symmetric phase, a small
asymmetry triggers periodic shedding; vortices alternate sign
and convect downstream at roughly the inlet velocity.*


Running it
----------

.. code-block:: bash

   cd examples/fluid/cylinder_flow
   python cylinder_flow.py            # writes frames/*.png + cylinder_flow_final.png
   python render_video.py             # stitches frames/ into vortex_street.mp4

The transient run is the longest in the gallery — the default
configuration is several thousand timesteps. Reduce ``n_steps`` or
coarsen the mesh for a quick smoke test.


What's next
-----------

* :doc:`cavity` — the steady cousin with the same weak form minus
  the time terms.
* :doc:`taylor_green` — a transient problem with an exact
  solution, for verifying accuracy.
* :doc:`flow_obstacles` — steady flow through a more complex
  channel geometry.
* :doc:`../../user_guide/mixed_assembly` — ``forward_vector``,
  ``field_data``, and generalized order pairs in detail.

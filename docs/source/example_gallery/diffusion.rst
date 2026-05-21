Diffusion
=========

Two transient examples in ``examples/diffusion/``: a linear heat
equation with implicit Euler time stepping (``heat/heat.py``) and a
nonlinear phase-field problem solved by Newton's method at each
timestep (``allen-cahn/ac.py``). Together they exercise both the
linear time-stepping pattern from
:doc:`../user_guide/time_integration` and the nonlinear / Newton
pattern from :doc:`../user_guide/linear_solvers`.


2D heat equation — ``heat/heat.py``
-----------------------------------

The strong form is

.. math::

   u_t \;=\; D^2\, \Delta u
   \quad \text{in } \Omega,
   \qquad u = 0 \text{ on } \partial\Omega,

on the unit square with a multi-frequency Fourier-series initial
condition from
:class:`~tensormesh.dataset.HeatMultiFrequency`.

Backward (implicit) Euler turns each step into a linear solve:

.. math::

   (M + \Delta t\, D^2\, A)\, U^{n+1} \;=\; M\, U^{n},

where :math:`M` is the mass matrix and :math:`A` is the Laplace
stiffness. The script assembles both once and reuses them across
steps:

.. code-block:: python
   :caption: examples/diffusion/heat/heat.py

   mesh = Mesh.gen_rectangle(chara_length=0.02, order=2,
                             element_type="tri")
   dataset = HeatMultiFrequency(d=16)

   class AAssembler(ElementAssembler):
       def forward(self, gradu, gradv): return gradu @ gradv
   class MAssembler(ElementAssembler):
       def forward(self, u, v):         return u * v

   M = MAssembler.from_mesh(mesh, quadrature_order=2)()
   A = AAssembler.from_mesh(mesh, quadrature_order=2)()
   condenser = Condenser(mesh.boundary_mask)

   dt, D, n = 5e-5, 1.0, 100
   K  = M + dt * D * D * A
   K_ = condenser(K)[0]                  # condense once

   U = dataset.initial_condition(mesh.points)
   Us = [U]
   for _ in range(n - 1):
       F  = M @ U                        # explicit RHS
       F_ = condenser.condense_rhs(F)
       U_ = K_.solve(F_)
       U  = condenser.recover(U_)
       Us.append(U)

A few choices reflect the user-guide patterns:

* **Quadratic triangles** (``order=2, element_type="tri"``) — heat
  diffusion is smooth on the interior, so order-2 elements give a
  much sharper match to the analytical solution at the same mesh
  size.
* **One assembly, many solves.** ``K`` and ``K_`` are built before
  the loop. Each step only does a matrix-vector ``M @ U``, a
  condense-RHS, and a solve — so the bulk of the cost is the
  per-step linear solve.
* **Comparison with truth.**
  :meth:`~tensormesh.dataset.HeatMultiFrequency.solution` returns the analytical
  solution at each time, and ``mesh.plot({"prediction": Us,
  "ground truth": Us_gt}, save_path="heat.mp4", dt=dt)`` writes a
  side-by-side animation.

The same scheme expressed with TensorMesh's
:class:`~tensormesh.ode.builtin.ImplicitLinearEuler` integrator ships
alongside as ``heat/heat_ode.py`` — same problem, with ``Condenser``
wired through the three boundary-condition hooks instead of the manual
loop. The two scripts produce identical snapshots to machine precision;
:doc:`../user_guide/time_integration` walks through the integrator
version, and this script is the lower-level "by hand" one.

.. raw:: html

   <video controls loop muted preload="metadata"
          width="100%" style="max-width: 720px; display: block; margin: 1em auto;">
     <source src="../_static/diffusion/heat.mp4" type="video/mp4">
     Your browser does not support the HTML5 video tag.
   </video>

*Output of* ``heat.py``: *FEM prediction (left) vs analytical truth
(right) over the time window. The two stay visually identical to
the eye, with the maximum nodal error decaying smoothly as the
backward-Euler step damps the highest Fourier mode.*


Allen-Cahn phase field — ``allen-cahn/ac.py``
---------------------------------------------

The Allen-Cahn equation is the textbook nonlinear phase-field
model:

.. math::

   u_t \;=\; \Delta u + \varepsilon^2\, u\, (1 - u^2),

with natural (no-flux) boundary conditions and a smooth initial
condition that evolves toward the binary phases :math:`u \in \{-1, 1\}`.
With :math:`\varepsilon = 220` and :math:`\Delta t = 10^{-6}`, the
nonlinearity is stiff enough that an explicit integrator would
require infeasibly small steps; the script uses **implicit Euler
plus Newton iteration**.

At each time step Newton solves the residual

.. math::

   R(c) \;=\;
   \frac{c - c_\text{old}}{\Delta t}\, v
   \;+\; D(c)\, \nabla v \cdot \nabla c
   \;-\; f(c)\, v
   \;\xrightarrow{!}\; 0,

with Jacobian :math:`K = \partial R / \partial c` linearized around
the current iterate. Both are written as TensorMesh assemblers so
the pattern is the same as a linear solve:

.. code-block:: python
   :caption: examples/diffusion/allen-cahn/ac.py (essence)

   class KAssembler(ElementAssembler):
       def forward(self, u, v, gradu, gradv, c, gradc, cold):
           return -1.0 * (
               (1.0/dt) * (u * v)
               + self.dD(c) * u * (gradv @ gradc)
               + self.D(c) * (gradu @ gradv)
               - self.df(c) * (u * v)
           )

   class RAssembler(NodeAssembler):
       def forward(self, v, gradv, c, gradc, cold):
           cdot = (c - cold) / dt
           return cdot * v + self.D(c) * (gradv @ gradc) - self.f(c) * v

   for step in range(steps):
       c = cold
       while True:
           pd = {"c": c, "cold": cold}
           K  = K_asm(mesh.points, point_data=pd)
           R  = R_asm(mesh.points, point_data=pd)
           K_, R_ = condenser(K, R)
           dC_ = K_.solve(R_)
           c   = c + condenser.recover(dC_)
           if torch.linalg.norm(R_) < 1e-10:
               break
       cold = c

Two things that make this easy to write:

* **Per-iterate parameters via** ``point_data``. The current
  iterate ``c`` and the previous timestep's solution ``cold`` are
  passed by name into both ``forward`` methods — exactly the
  argument-dispatch contract documented in
  :doc:`../user_guide/forms`. There is no per-quadrature-point
  Python loop.
* **Reassemble each iteration.** Because ``K`` depends on ``c``
  (through ``D(c)``, ``df(c)``), it has to be rebuilt every Newton
  iteration. That is intentional — assembly is cheap on the GPU,
  and the alternative (computing analytical Jacobians by hand)
  would require pages of additional code.

If you would rather hand the whole step to a packaged driver,
``allen-cahn/ac_torch_sla.py`` solves the *same* problem with
``torch-sla``'s ``nonlinear_solve`` (Newton / Picard / Anderson with
Armijo line search; see :doc:`../user_guide/linear_solvers`). The
per-step ``while`` loop above collapses to one call:

.. code-block:: python
   :caption: examples/diffusion/allen-cahn/ac_torch_sla.py (essence)

   from torch_sla import nonlinear_solve

   def residual_fn(c, cold):                       # F(c) = 0 is the BE step
       return R_asm(mesh.points, point_data={"c": c, "cold": cold})

   def jacobian_fn(c, cold):                        # KAssembler gives K = -J,
       K = K_asm(mesh.points, point_data={"c": c, "cold": cold})
       return (-K.values, K.row, K.col, K.shape)    # so hand back J = -K

   for step in range(steps):
       cold = nonlinear_solve(residual_fn, cold, cold, jacobian_fn=jacobian_fn,
                              method="newton", linear_solver="auto")

Two points worth noting. First, ``nonlinear_solve`` can build the
Jacobian by autograd, but here we pass the FEM consistent tangent
explicitly as ``jacobian_fn`` — it is already assembled, so this is
both cheaper and exact. Second, the sign: ``KAssembler`` assembles the
*negative* tangent :math:`K = -\partial R/\partial c`, so ``jacobian_fn``
returns ``-K.values`` to hand back the true Jacobian :math:`J`, which
``nonlinear_solve`` then steps with :math:`J\,\delta u = -R`. The two
scripts agree to round-off, and ``nonlinear_solve`` additionally gives a
correct adjoint backward through the converged solution.

.. raw:: html

   <video controls loop muted preload="metadata"
          width="100%" style="max-width: 640px; display: block; margin: 1em auto;">
     <source src="../_static/diffusion/allen-cahn.mp4" type="video/mp4">
     Your browser does not support the HTML5 video tag.
   </video>

*Output of* ``ac.py``: *the order parameter* :math:`\varphi` *evolving
on the unit square. Starting from random noise it rapidly coarsens
into the two equilibrium phases (* :math:`\varphi=\pm1` *), with
diffuse interfaces of width set by* :math:`\varepsilon`. *Each frame
is one resolved Newton solve.*


Running the examples
--------------------

.. code-block:: bash

   cd examples/diffusion/heat
   python heat.py            # manual loop -> heat.mp4
   python heat_ode.py        # integrator hooks -> heat_ode.mp4

   cd ../allen-cahn
   python ac.py              # hand-written Newton loop -> Allen-Cahn.mp4
   python ac_torch_sla.py    # via torch_sla.nonlinear_solve -> Allen-Cahn-torch-sla.mp4


What's next
-----------

* :doc:`../user_guide/time_integration` — the same heat problem,
  rewritten in terms of :class:`~tensormesh.ode.builtin.ImplicitLinearEuler`.
* :doc:`../user_guide/linear_solvers` —
  ``SparseMatrix.nonlinear_solve`` for a packaged Newton / Picard /
  Anderson loop with adjoint backward.
* :doc:`wave` — the next transient PDE up: hyperbolic, explicit
  central differences.
* :doc:`dataset` — batched heat solves for ML training data.

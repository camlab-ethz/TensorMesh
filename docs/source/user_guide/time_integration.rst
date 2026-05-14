Time Integration
================

Transient problems — heat, wave, transient elasticity, phase-field
dynamics — extend the static FEM pipeline with a time-stepping loop.
Once the mass and stiffness matrices have been assembled, the
semi-discrete weak form is just a system of ODEs in the nodal values

.. math::

   M\,\dot u(t) \;=\; A(t)\,u(t) \;+\; B(t),
   \qquad u(0) = u_0,

and the job of this chapter is to advance ``u`` from one time level to
the next, accurately and stably, while keeping everything
differentiable.

TensorMesh exposes two complementary styles:

1. **A manual time-stepping loop.** You assemble the time-stepped
   operator yourself, call :class:`~tensormesh.Condenser` once for the
   boundary conditions, and write a Python ``for`` loop that does one
   linear solve per step. Short, explicit, and the path of least
   resistance for a one-off problem — especially one with Dirichlet
   boundaries that need static condensation.

2. **The integrator classes in** :mod:`tensormesh.ode`. You override
   ``forward(t, u)`` (explicit) or ``forward_M`` / ``forward_A`` /
   ``forward_B`` (linear-implicit), and call ``step(t, u, dt)``.
   Useful when you want a generic transient driver that lets you
   swap one scheme for another without rewriting the loop, and ideal
   for problems whose state lives in a dense vector with no boundary
   surgery required.

Both styles compose with :mod:`torch.autograd`, so gradients flow back
through every step into initial conditions, material parameters, and
boundary data alike (see :doc:`differentiability`).


.. _the-integrators:

The integrator catalogue
------------------------

:mod:`tensormesh.ode` ships three concrete schemes plus two extensible
base classes:

.. list-table::
   :header-rows: 1
   :widths: 28 8 22 42

   * - Class
     - Order
     - Form
     - Use case
   * - :class:`~tensormesh.ode.ExplicitEuler`
     - 1
     - :math:`\dot u = f(t, u)`
     - Cheap explicit RHS; fine for non-stiff problems below the CFL.
   * - :class:`~tensormesh.ode.ImplicitLinearEuler`
     - 1
     - :math:`M\dot u = A u + B`
     - Heat / diffusion / stiff systems. Unconditionally stable.
   * - :class:`~tensormesh.ode.MidPointLinearEuler`
     - 2
     - :math:`M\dot u = A u + B`
     - Same family, second-order accurate (trapezoidal rule).
   * - :class:`~tensormesh.ode.ExplicitRungeKutta`
     - s-stage
     - :math:`\dot u = f(t, u)`
     - Base class — supply your own Butcher tableau ``(a, b)``.
   * - :class:`~tensormesh.ode.ImplicitLinearRungeKutta`
     - s-stage
     - :math:`M\dot u = A u + B`
     - Same, for linear-implicit schemes.

For the explicit family you override ``forward(t, u)`` to return the
right-hand side :math:`f(t, u)`. For the linear-implicit family you
override three methods that return the operators at the current time:

.. code-block:: python

   class MyScheme(ImplicitLinearEuler):
       def forward_M(self, t):  return M_matrix    # SparseMatrix, Tensor, or scalar
       def forward_A(self, t):  return -K_matrix   # SparseMatrix, Tensor, or scalar
       def forward_B(self, t):  return 0.0         # Tensor or scalar

A scalar return is lifted to that multiple of the identity (or to a
constant vector for ``B``), so you can leave any of the three at their
defaults of ``1``, ``1``, ``0``. Each call to ``step(t0, u0, dt)``
returns the new ``u`` advanced by one time step; ``u`` must be 1D
(``[D]``), so flatten vector-valued problems before stepping.


.. _ti-scalar:

Worked example 1: a scalar ODE
------------------------------

The integrator classes work on plain ODEs the same way they work on
FEM systems — you just leave the operators as scalars. Take

.. math::

   \dot u(t) \;=\; -\lambda\,u(t),
   \qquad u(0) = 1,
   \qquad \lambda = \pi^{2},

whose exact solution is :math:`u(t) = e^{-\lambda t}`. Both
:class:`~tensormesh.ode.ImplicitLinearEuler` (first order) and
:class:`~tensormesh.ode.MidPointLinearEuler` (second order) accept a
scalar ``M = 1`` and ``A = -\lambda``, so the driver is two short
classes:

.. code-block:: python

   import torch
   from tensormesh.ode import ImplicitLinearEuler, MidPointLinearEuler

   class ScalarIE(ImplicitLinearEuler):
       def __init__(self, lam):
           super().__init__()
           self.lam = lam
       def forward_M(self, t): return 1.0
       def forward_A(self, t): return -self.lam
       def forward_B(self, t): return 0.0

   class ScalarMP(MidPointLinearEuler):
       def __init__(self, lam):
           super().__init__()
           self.lam = lam
       def forward_M(self, t): return 1.0
       def forward_A(self, t): return -self.lam
       def forward_B(self, t): return 0.0

   lam = torch.pi ** 2
   u = torch.ones(1, dtype=torch.float64)
   dt = 1e-3
   integrator = ScalarMP(lam)
   for k in range(50):
       u = integrator.step(k * dt, u, dt)

Running the same problem at decreasing :math:`\Delta t` and measuring
the error at :math:`T = 0.05` produces the textbook order-1 and
order-2 slopes:

.. figure:: /_static/user_guide/time_integration/convergence.png
   :alt: Temporal convergence on the scalar test ODE.
   :align: center
   :width: 80%

   Endpoint error vs ``dt`` for ImplicitLinearEuler and
   MidPointLinearEuler on :math:`\dot u = -\pi^{2}\,u`. Dashed
   reference lines have slopes 1 and 2.

The practical reading: spending a higher-order method buys you orders
of magnitude in accuracy at the *same* step size — at ``dt = 5e-3``
the midpoint rule already beats backward Euler by four orders of
magnitude.


.. _ti-heat-manual:

Worked example 2: 2D heat equation
----------------------------------

For an FEM problem with Dirichlet boundaries, the manual loop is the
cleanest pattern. Solve

.. math::

   \frac{\partial u}{\partial t} \;=\; \kappa\,\Delta u
   \quad \text{in } \Omega = (0, 1)^{2},
   \qquad u = 0 \text{ on } \partial\Omega,
   \qquad u(x, y, 0) = \sin(\pi x)\,\sin(\pi y).

The semi-discrete form is :math:`M\dot u = -\kappa\,K\,u`, and one
backward-Euler step gives the linear system

.. math::

   (M + \Delta t\,\kappa\,K)\,u^{n+1} \;=\; M\,u^{n}.

The full driver assembles ``M`` and ``K`` once, builds the
time-stepped operator once, condenses it once, then loops:

.. code-block:: python

   import torch
   from tensormesh import (Mesh, ElementAssembler,
                           MassElementAssembler, LaplaceElementAssembler,
                           Condenser)

   mesh = Mesh.gen_rectangle(chara_length=0.025, order=1)
   M = MassElementAssembler.from_mesh(mesh)().double()
   K = LaplaceElementAssembler.from_mesh(mesh)().double()

   kappa = 1.0
   dt    = 5e-4

   # Build the time-stepped operator once and condense it once.
   A = M + dt * kappa * K
   condenser = Condenser(mesh.boundary_mask)
   A_in, _ = condenser(A, torch.zeros(mesh.n_points, dtype=torch.float64))

   # Initial condition (already zero on the Dirichlet boundary).
   x, y = mesh.points.double()[:, 0], mesh.points.double()[:, 1]
   u = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

   # Time stepping: each iteration is one mass-vector product,
   # one RHS condensation, one back-substitution, one recovery.
   snapshots = [u]
   for _ in range(100):
       f       = M @ u
       f_in    = condenser.condense_rhs(f)
       u_in    = A_in.solve(f_in)
       u       = condenser.recover(u_in)
       snapshots.append(u)

The two ingredients that make this loop fast are (a) factorising the
time-stepped operator only once — the per-step solve is a back-substitution
through ``A_in`` — and (b) reusing the cached condenser layout via
:meth:`~tensormesh.Condenser.condense_rhs`. Together they turn what
looks like an order-:math:`N^{3}` problem into an order-:math:`N`
inner loop.

The solution decays exponentially as :math:`e^{-2\pi^{2}t}`:

.. figure:: /_static/user_guide/time_integration/heat_snapshots.png
   :alt: 2D heat equation snapshots at three times.
   :align: center
   :width: 95%

   Snapshots of :math:`u(x, y, t)` at :math:`t = 0`, :math:`t = T/2`,
   and :math:`t = T` with :math:`T = 0.05`, backward Euler,
   :math:`\Delta t = 5\!\times\!10^{-4}`, characteristic mesh size
   :math:`h = 0.025`.

For an animated version, see the rendered ``heat.mp4`` in the
:doc:`example gallery <../example_gallery/index>` or under
``examples/diffusion/heat/`` in the source tree.


.. _ti-stability:

Stability: why "implicit" matters
---------------------------------

Forward (explicit) Euler is the simplest possible time scheme, but for
stiff problems — anything diffusive at meaningful mesh resolution — it
imposes a *CFL constraint*: the step size has to shrink with the
square of the mesh size, :math:`\Delta t \lesssim h^{2}/\lambda_{\max}`.
Backward (implicit) Euler has no such restriction. The plot below
takes the same heat problem on a coarse :math:`h = 0.1` mesh, adds a
small high-frequency perturbation to the initial condition, and runs
three schemes:

.. figure:: /_static/user_guide/time_integration/stability.png
   :alt: Forward vs backward Euler stability on the 2D heat equation.
   :align: center
   :width: 80%

   ``max|u|`` over time. Forward Euler at a safe ``dt`` (green) and
   backward Euler at a large ``dt`` (blue) both decay smoothly; forward
   Euler at the same large ``dt`` (red) amplifies the high-frequency
   mode by roughly :math:`6 \times` every step and blows up.

The takeaway: if the problem is stiff, pay the per-step linear solve
and use an implicit scheme. Forward Euler is for problems where
:math:`\Delta t \le h^{2}/\lambda_{\max}` is *cheap enough*, which in
practice means hyperbolic problems (wave, transient elasticity) at
their natural CFL — not parabolic ones.


.. _ti-custom-tableau:

Custom Butcher tableaux
-----------------------

The two ``RungeKutta`` base classes let you supply any Butcher
tableau. To use the classical fourth-order explicit Runge-Kutta on
:math:`\dot u = f(t, u)`:

.. code-block:: python

   import torch
   from tensormesh.ode import ExplicitRungeKutta

   a = torch.tensor([[0.,  0.,  0., 0.],
                     [0.5, 0.,  0., 0.],
                     [0.,  0.5, 0., 0.],
                     [0.,  0.,  1., 0.]])
   b = torch.tensor([1/6, 1/3, 1/3, 1/6])

   class MyRK4(ExplicitRungeKutta):
       def forward(self, t, u):
           return f(t, u)            # your problem-specific RHS

   integrator = MyRK4(a, b)
   for k in range(n_steps):
       u = integrator.step(k * dt, u, dt)

The class verifies that ``a`` is lower-triangular and that
:math:`\sum_i b_i = 1` (to within floating-point tolerance), so you
catch transcription mistakes early. The same pattern with
:class:`~tensormesh.ode.ImplicitLinearRungeKutta` gives you diagonally-implicit
or fully-implicit schemes — supply a non-zero diagonal and the class
will assemble and solve the block stage system for you.


.. _ti-boundaries:

Composing with boundary conditions
----------------------------------

Static condensation via :class:`~tensormesh.Condenser` (see
:doc:`boundary_conditions`) is the recommended way to enforce
Dirichlet conditions in TensorMesh. In a manual time loop the pattern
is:

* call ``condenser(A, _)`` *once*, before the loop, to factorise the
  time-stepped operator on the interior DOFs;
* call ``condenser.condense_rhs(f)`` *every step* to project the new
  RHS down to the interior;
* call ``condenser.recover(u_in)`` after every solve to glue boundary
  values back in.

For time-varying boundary data, swap in the new values between steps
with :meth:`~tensormesh.Condenser.update_dirichlet` — the sparsity
layout is cached and survives the update, so this call is cheap.

A note on the integrator classes: condensation re-sizes the linear
system from ``D`` (all DOFs) to ``D_inner`` (interior DOFs), and the
current ``step()`` implementation in :mod:`tensormesh.ode` assumes the
pre-/post-solve hooks preserve dimension. For problems that need
Dirichlet boundaries, prefer the manual loop pattern above; reserve
the integrator classes for ODE-shaped problems and for FEM problems
where boundary conditions can be expressed without reducing the system
size (e.g. Robin, weak penalty, or strong row-replacement).


Differentiability
-----------------

Every step is built on a differentiable solve (see
:doc:`linear_solvers`), so back-propagation through a transient
simulation is a no-op:

.. code-block:: python

   kappa = torch.tensor(1.0, requires_grad=True)
   u     = run_heat_solver(mesh, kappa, dt=5e-4, n_steps=100)
   loss  = (u - u_target).pow(2).sum()
   loss.backward()
   print(kappa.grad)            # gradient through 100 implicit solves

This is what makes transient inverse problems, parameter identification,
and gradient-based PDE design straightforward in TensorMesh — see
:doc:`differentiability` for the longer story.


What's next
-----------

* :doc:`linear_solvers` — the solver behind every step.
* :doc:`differentiability` — backprop through a transient solve to
  optimise parameters or initial conditions.
* :doc:`batched_workflows` — batched initial conditions or parameters
  with the same transient driver.
* :doc:`../example_gallery/index` — heat, wave, and transient
  elasticity demos with rendered animations.

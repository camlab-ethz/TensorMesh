Taylor-Green Vortex (Convergence Study)
========================================

The Taylor-Green vortex is the canonical incompressible-flow
benchmark with a known exact solution: a periodic decaying vortex
on the square :math:`[0, 2\pi]^2` whose velocity and pressure can
be written down in closed form. The script
``examples/fluid/taylor_green/taylor_green.py`` uses it to drive
a quantitative **h-convergence study** of the transient
Navier-Stokes solver — the only example in :doc:`index` whose
purpose is verification rather than visualization.


Exact solution
--------------

For viscosity :math:`\nu` and time :math:`t`,

.. math::

   u(x, y, t) &= -\cos(x)\, \sin(y)\, e^{-2\nu t}, \\
   v(x, y, t) &= \phantom{-}\sin(x)\, \cos(y)\, e^{-2\nu t}, \\
   p(x, y, t) &= -\tfrac14 \bigl(\cos(2x) + \cos(2y)\bigr)\, e^{-4\nu t}.

The velocity decays exponentially with rate :math:`2\nu` (twice
the kinematic viscosity); the pressure decays at rate
:math:`4\nu`. Substitution verifies that
:math:`(u, v, p)` satisfies the incompressible Navier-Stokes
equations exactly.

The script imposes Dirichlet velocities on every boundary node
equal to the exact solution at the current time. This sidesteps
the need for periodic boundaries and isolates the spatial /
temporal discretization error.


Solver
------

The transient ``NavierStokesTransientAssembler`` from
:doc:`cylinder_flow` is reused verbatim — implicit Euler in time,
SUPG/PSPG with adaptive :math:`\tau` in space. The only changes
are:

* domain :math:`[0, 2\pi]^2` instead of the cylinder channel,
* Dirichlet velocity from the exact solution rather than a
  parabolic inlet,
* every step writes :math:`L^2` errors against the analytical
  velocity and pressure to a results file.

.. code-block:: python
   :caption: examples/fluid/taylor_green/taylor_green.py (essence)

   def exact_velocity(points, t, nu):
       x, y = points[:, 0], points[:, 1]
       decay = math.exp(-2.0 * nu * t)
       u = -torch.cos(x) * torch.sin(y) * decay
       v =  torch.sin(x) * torch.cos(y) * decay
       return torch.stack([u, v], dim=1)

   for n_grid in [10, 20, 40, 80]:
       mesh = Mesh.gen_rectangle(left=0, right=2*math.pi,
                                 bottom=0, top=2*math.pi,
                                 chara_length=2*math.pi/n_grid)
       # …assemble, set Dirichlet to exact velocity, time-step…
       err_u_l2 = mass_weighted_norm(u_fem - u_exact)
       err_p_l2 = mass_weighted_norm(p_fem - p_exact)


h-convergence
-------------

The script sweeps the mesh size :math:`h = 2\pi / n_\text{grid}`
through a list of values and records the :math:`L^2` velocity and
pressure errors at a fixed final time. For P1-P1 elements with
SUPG/PSPG one expects:

* :math:`\|\mathbf{u}_h - \mathbf{u}\|_{L^2} = \mathcal{O}(h^2)`
* :math:`\|p_h - p\|_{L^2} = \mathcal{O}(h)`

The console output is a table:

.. code-block:: text

      n_grid     h       err_u_L2    rate_u    err_p_L2    rate_p
   ---------------------------------------------------------------
        10    0.628    1.3e-1                  4.5e-1
        20    0.314    3.4e-2     1.9          2.3e-1     1.0
        40    0.157    8.7e-3     2.0          1.2e-1     1.0
        80    0.079    2.2e-3     2.0          5.8e-2     1.0

(numbers approximate, will vary slightly with the chosen
:math:`\nu` and :math:`\Delta t`). The plot
``taylor_green_convergence.png`` shows the same data on a log-log
axis with reference :math:`h^2` and :math:`h^1` slopes.

If the observed rates differ from the expected
:math:`\mathcal{O}(h^2)` / :math:`\mathcal{O}(h)`, something in
the solver is wrong — most often a subtle bug in the
stabilization or in the boundary-condition imposition.


Mass-weighted error norm
------------------------

The script computes the discrete :math:`L^2` norm via the mass
matrix:

.. math::

   \|e_h\|_{L^2}^2
   \;=\;
   e_h^T\, M\, e_h
   \;=\;
   \sum_K \int_K e_h^2 \,\mathrm{d}\Omega,

assembled by :class:`~tensormesh.MassElementAssembler`. This is the
right norm for FEM error analysis (it integrates over the elements,
weighted by the basis), unlike the simple Euclidean norm of the
nodal error vector.


Output
------

* **Console table** — convergence rates at every refinement step.
* **``taylor_green_convergence.png``** — log-log error plot vs
  mesh size with reference slopes.
* **``taylor_green_results.png``** — three-panel snapshot of the
  finest mesh: speed, pressure, velocity-error magnitude.
* **``taylor_green.mp4``** (optional) — animation of the decaying
  vortex.

.. figure:: /_static/fluid/taylor_green_vortex_final.png
   :alt: Taylor-Green vortex at t=0.5 — vorticity + streamlines, speed + velocity vectors
   :width: 100%

   Output of ``taylor_green.py`` at :math:`t=0.5`. Left: vorticity
   field with overlaid streamlines — the canonical periodic
   :math:`2\times2` pattern of alternating-sign vortices. Right:
   speed magnitude with velocity vectors, showing the characteristic
   "saddle" structure between adjacent rolls. The amplitude has
   decayed by :math:`\exp(-2\nu t)` from the analytical initial
   condition, which the script's convergence study uses as
   ground truth.


Running it
----------

.. code-block:: bash

   cd examples/fluid/taylor_green
   python taylor_green.py      # writes convergence + results pngs


What's next
-----------

* :doc:`cavity` and :doc:`cylinder_flow` — the qualitative NS
  solvers whose convergence this example verifies.
* :doc:`../../user_guide/elements_and_quadrature` — switch to
  higher-order elements (``order=2``) and the expected rates
  shift to :math:`h^3` / :math:`h^2`.
* :doc:`../../user_guide/time_integration` — a higher-order time
  integrator from :mod:`tensormesh.ode` would reduce the time
  error component.

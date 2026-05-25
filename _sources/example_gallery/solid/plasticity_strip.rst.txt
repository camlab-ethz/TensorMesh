Plasticity (J2 with Isotropic Hardening)
==========================================

A plane-strain plasticity problem: a :math:`1.0 \times 0.2` m
steel strip is stretched 10 % then unloaded, exhibiting the
characteristic permanent (plastic) deformation that the elastic
solver in :doc:`cantilever_beam` cannot capture. The script
``examples/solid/plasticity_strip/plasticity_strip.py`` uses J2
flow theory with linear isotropic hardening, integrated in a
**variational constitutive update** style — the constitutive
response is encoded as an algorithmic potential, the displacement
update is one L-BFGS pass, and history variables are advanced once
per converged step.

The constitutive model is dimension-generic, so the same
:class:`~tensormesh.assemble.J2Plasticity` runs unchanged in 3D —
see `Going to 3D`_ at the end of this page.


Problem
-------

Small-strain elastoplasticity. The Cauchy stress is the elastic
response of the **elastic** part of the strain:

.. math::

   \boldsymbol{\sigma} \;=\;
   \mathbb{C} : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p),

with yield surface

.. math::

   f(\boldsymbol{\sigma}, \alpha)
   \;=\;
   \|\,\mathrm{dev}\, \boldsymbol{\sigma}\,\|
   - \sqrt{\tfrac{2}{3}} \bigl(\sigma_y + H\, \alpha\bigr),

evolution :math:`\dot{\boldsymbol{\varepsilon}}^p = \dot\gamma\, \mathbf{n}`,
:math:`\dot\alpha = \sqrt{\tfrac{2}{3}}\, \dot\gamma`. Material:
:math:`E = 200` GPa, :math:`\nu = 0.3`,
:math:`\sigma_y = 250` MPa, :math:`H = 1` GPa.

Boundary conditions:

* :math:`x = 0`: roller (:math:`u_x = 0`),
* corner :math:`(0, 0)`: full pin (:math:`u_x = u_y = 0`),
* :math:`x = 1`: prescribed :math:`u_x` ramped from 0 to 0.10 m
  (10 steps), then back down to 0 (10 more steps).


Variational constitutive update
-------------------------------

Rather than implementing an explicit "radial return" map per
quadrature point, the script defines an **algorithmic potential**
:math:`\Pi^\text{alg}(\boldsymbol{\varepsilon},
\boldsymbol{\varepsilon}^p_n, \alpha_n)`
whose stationarity in the plastic variables recovers the standard
return-mapping equations. The displacement update at each step is
then a single minimization of the integrated potential:

.. math::

   \mathbf{u}^{n+1}
   \;=\;
   \arg\min_{\mathbf{u}}
   \int_\Omega \Pi^\text{alg}\bigl(\boldsymbol{\varepsilon}(\mathbf{u}),\,
   \boldsymbol{\varepsilon}^p_n,\, \alpha_n\bigr)
   \,\mathrm{d}\Omega,

with autograd providing both the gradient (residual) and, via
L-BFGS' implicit Hessian approximation, an effective tangent.
After convergence, the history variables
:math:`(\boldsymbol{\varepsilon}^p, \alpha)` are advanced once
to the new converged values.

The potential is implemented in
:class:`~tensormesh.assemble.J2Plasticity`, a built-in shipping
in :mod:`tensormesh.assemble`. From the user's side it looks like
any other custom assembler:

.. code-block:: python
   :caption: examples/solid/plasticity_strip/plasticity_strip.py (essence)

   from tensormesh.assemble import J2Plasticity
   from tensormesh.material import IsotropicMaterial

   steel = IsotropicMaterial("Steel_Hardening",
                             E=200e9, nu=0.3, rho=7850,
                             sigma_y=250e6, H=1e9)
   model = J2Plasticity.from_mesh(mesh, material=steel)

   u = torch.zeros_like(mesh.points, requires_grad=True)
   optimizer = optim.LBFGS([u], lr=1.0, max_iter=50,
                           history_size=50,
                           line_search_fn="strong_wolfe")

   for step in range(1, n_total_steps + 1):
       target_disp = piecewise_load_schedule(step)

       def closure():
           optimizer.zero_grad()
           u_active = apply_bcs(u, target_disp)
           element_data = {
               "eps_p_n": {et: h["eps_p"] for et, h in model.history.items()},
               "alpha_n": {et: h["alpha"] for et, h in model.history.items()},
           }
           E = model.energy(point_data={"displacement": u_active},
                            element_data=element_data)
           E.backward()
           return E

       optimizer.step(closure)
       model.advance_history(...)        # commit eps_p, alpha


History variables
-----------------

The plastic strain tensor :math:`\boldsymbol{\varepsilon}^p` and
the equivalent plastic strain :math:`\alpha` are stored
**per-element / per-quadrature-point**, accessed through
``model.history``. They are passed into the closure as
``element_data`` (per-quadrature, per-element tensors), which
TensorMesh dispatches as kwargs to the underlying ``forward``
method exactly the same way ``point_data`` is dispatched (see
:doc:`../../user_guide/forms`).

After each load step's L-BFGS converges, the script calls
``model.advance_history(u_active)`` which evaluates the converged
:math:`\boldsymbol{\varepsilon}^p, \alpha` from the current
displacement and overwrites the stored history.


Output
------

The script renders a multi-panel animation as ``plasticity_strip.mp4``:

* deformed mesh, painted by equivalent plastic strain
  :math:`\alpha`,
* force-displacement curve at the loaded edge, traced through the
  load / unload cycle.

The force-displacement curve is the canonical signature of
plasticity: linear elastic loading until yield, a hardening branch
above :math:`\sigma_y`, then a linear elastic unloading branch
back to a non-zero residual displacement at zero load.

.. raw:: html

   <video controls loop muted preload="metadata"
          width="100%" style="max-width: 900px; display: block; margin: 1em auto;">
     <source src="../../_static/solid_mechanics/plasticity_strip.mp4" type="video/mp4">
     Your browser does not support the HTML5 video tag.
   </video>

*Output of* ``plasticity_strip.py``\ *: the strip deforms under the
prescribed elongation (left panel, colored by equivalent plastic
strain* :math:`\alpha`\ *) while the force-displacement curve at the
loaded edge (right panel) traces the load / unload cycle — elastic
loading to yield, the hardening branch, then elastic unloading back
to a permanent residual displacement at zero load.*


Going to 3D
-----------

:class:`~tensormesh.assemble.J2Plasticity` is dimension-generic, so
the same constitutive model, variational-update pattern, and
history-variable bookkeeping carry over to 3D unchanged. The script
``examples/solid/plasticity_3d/plasticity_3d.py`` runs J2 flow on a
:math:`0.5 \times 0.5 \times 0.5` m steel cube, stretched 40 % in
the :math:`x` direction over 50 loading steps and unloaded over 50
more. Only the *driver* differs from the 2D strip:

.. code-block:: python
   :caption: examples/solid/plasticity_3d/plasticity_3d.py (essence)

   mesh = gen_cube(chara_length=0.04,
                   left=0, right=0.5, bottom=0, top=0.5,
                   front=0, back=0.5)
   steel = IsotropicMaterial("Steel_Hardening",
                             E=200e9, nu=0.3, sigma_y=250e6, H=1e9)
   model = J2Plasticity.from_mesh(mesh, material=steel)
   # …same closure / advance_history loop as the 2D strip, 3D BCs…

What actually changes is scale, not the model:

* **3D mesh.** ``gen_cube`` produces a tetrahedral cube with on the
  order of 5–10 k DOFs (vs. ~3 k in the 2D strip); each L-BFGS
  iteration integrates over many more quadrature points.
* **More load steps.** 50 loading + 50 unloading = 100 steps,
  reaching 40 % strain — far enough that the hardening branch
  dominates. At 40 % nominal strain the small-strain assumption is
  being stretched (literally): the example is a useful regression
  test of the J2 model, but do not trust absolute stress numbers
  far from the elastic regime.
* **Output.** ``plasticity_3d_combined.mp4`` shows the deformed
  cube colored by **von Mises stress** plus a force-displacement
  curve at the loaded face.

The 3D run is L-BFGS-bound — most wall-clock time is the
per-iteration energy evaluation and autograd backward. Moving the
optimization to a GPU (``mesh.to("cuda")``, ``u.to("cuda")``) helps
significantly, and the assembler's ``batch_size`` argument chunks
the per-quadrature integrand if memory becomes tight (see
:doc:`../../user_guide/batched_workflows`).

.. raw:: html

   <video controls loop muted preload="metadata"
          width="100%" style="max-width: 900px; display: block; margin: 1em auto;">
     <source src="../../_static/solid_mechanics/plasticity_3d_combined.mp4" type="video/mp4">
     Your browser does not support the HTML5 video tag.
   </video>

*Output of* ``plasticity_3d.py``\ *: the steel cube stretched to
40 % strain (left panel, colored by von Mises stress) and the
force-displacement curve at the loaded face (right panel) over the
50-step load / 50-step unload cycle — the 3D counterpart of the
strip animation above.*

.. code-block:: bash

   cd examples/solid/plasticity_3d
   python plasticity_3d.py     # writes plasticity_3d_combined.mp4


Running it
----------

.. code-block:: bash

   cd examples/solid/plasticity_strip
   python plasticity_strip.py     # writes plasticity_strip.mp4

Console output reports per-step force, displacement, and maximum
plastic strain.


What's next
-----------

* :doc:`hyperelastic_beam` — the same L-BFGS-with-history pattern
  but for a *path-independent* (hyperelastic) energy.
* :doc:`hertzian_contact` — another non-quadratic energy driver,
  this time a contact penalty.
* :doc:`../../user_guide/batched_workflows` — chunking the
  per-quadrature integrand for the larger 3D problem.

Phase-Field Fracture
====================

Brittle fracture as energy minimization. A damage field
:math:`\alpha \in [0, 1]` smears the crack over a length :math:`\ell`.
Where the crack starts, the path it takes and when it runs then all follow
from minimizing one energy [Francfort1998]_ [Bourdin2000]_. The script
``examples/solid/phase_field_fracture/phase_field_fracture.py`` runs the
single-edge-notched tension (SENT) benchmark [Miehe2010]_ in plane strain.

The new ingredient over :doc:`plasticity_strip` is an **inequality
constraint**: damage cannot heal, :math:`\alpha_n \ge \alpha_{n-1}`. Every
load step is therefore a bound-constrained minimization. The script solves it
by alternate minimization, and both subproblems use ``vi_solve`` from
`torch-sla <https://www.torchsla.com/>`_, a Newton solve with bounds.


Model
-----

The energy of a displacement :math:`\mathbf{u}` and a damage field
:math:`\alpha` is

.. math::

   \mathcal{E}(\mathbf{u}, \alpha)
   = \int_\Omega \Bigl[ g(\alpha)\, \psi_D(\boldsymbol{\varepsilon})
     + \psi_R(\boldsymbol{\varepsilon}) \Bigr] \mathrm{d}\Omega
   + \frac{G_c}{c_w} \int_\Omega \Bigl[ \frac{w(\alpha)}{\ell}
     + \ell\, |\nabla\alpha|^2 \Bigr] \mathrm{d}\Omega,
   \qquad g(\alpha) = (1 - \alpha)^2 + g_0,

with :math:`\boldsymbol{\varepsilon} = \tfrac12(\nabla\mathbf{u} + \nabla\mathbf{u}^T)`
and a small residual stiffness :math:`g_0`.

* **Crack model.** AT1 takes :math:`w(\alpha) = \alpha` and
  :math:`c_w = 8/3`. It has an elastic phase: no damage forms below a
  critical stress [Pham2011]_. AT2 takes :math:`w(\alpha) = \alpha^2` and
  :math:`c_w = 2`, and damages at any load. Choose with ``--model``.
* **Energy split** [Amor2009]_. Only the tensile volumetric part and the
  deviatoric part are degraded, so compression does not crack:

  .. math::

     \psi_D = \tfrac{\kappa}{2} \langle \operatorname{tr}\boldsymbol{\varepsilon} \rangle_+^2
             + \mu\, |\operatorname{dev}\boldsymbol{\varepsilon}|^2,
     \qquad
     \psi_R = \tfrac{\kappa}{2} \langle \operatorname{tr}\boldsymbol{\varepsilon} \rangle_-^2.

The script writes the two energy densities once, in ``PhaseFieldModel``.
:meth:`~tensormesh.ElementAssembler.energy` integrates them, and
autograd turns the integrals into the residuals
:math:`\partial\mathcal{E}/\partial\mathbf{u}` and
:math:`\partial\mathcal{E}/\partial\alpha`.


Irreversibility as bounds
-------------------------

At load step :math:`n` the damage is sought in :math:`[\alpha_{n-1}, 1]`.
A minimizer under these bounds satisfies, node by node,

.. math::

   \alpha_i = \alpha_{n-1,i} \;\Rightarrow\; \partial_i \mathcal{E} \ge 0,
   \qquad
   \alpha_{n-1,i} < \alpha_i < 1 \;\Rightarrow\; \partial_i \mathcal{E} = 0,
   \qquad
   \alpha_i = 1 \;\Rightarrow\; \partial_i \mathcal{E} \le 0.

Either the residual vanishes, or the node rests on a bound that the residual
pushes against. This is a variational inequality, and it is exactly what
``torch_sla.vi_solve(residual, x0, lower=..., upper=..., jacobian_fn=...)``
solves. The method is reduced-space active-set Newton, the algorithm of
PETSc's ``SNESVINEWTONRSLS`` [Benson2006]_. Each Newton step solves for the
nodes off their bounds, then updates which nodes rest on a bound.

Prescribed values are bounds as well: ``lower == upper`` fixes a component.
The displacement Dirichlet conditions are imposed this way, so the
displacement solve needs no :class:`~tensormesh.Condenser`. A pre-crack can be
prescribed the same way, as :math:`\alpha = 1` held on a line of nodes (the
``alpha_fixed`` argument). The benchmark below uses a slit in the mesh
instead, so the notch faces are free from the start.


Alternate minimization
----------------------

Each load step alternates between the two fields until they agree:

#. **Displacement**, with :math:`\alpha` fixed. :math:`\mathcal{E}` is convex
   in :math:`\mathbf{u}` but only piecewise quadratic, since the split switches
   with the sign of :math:`\operatorname{tr}\boldsymbol{\varepsilon}`. Newton
   with a line search takes a few steps.
#. **Damage**, with :math:`\mathbf{u}` fixed. For AT1 and AT2,
   :math:`\mathcal{E}` is quadratic in :math:`\alpha`, so this is a
   bound-constrained quadratic program. Its Hessian does not depend on
   :math:`\alpha` and is assembled once.
#. **Stop** when the displacement residual on the free degrees of freedom,
   evaluated with the new damage, is below :math:`10^{-4}` of the reaction
   forces.

.. literalinclude:: ../../../../examples/solid/phase_field_fracture/phase_field_fracture.py
   :language: python
   :pyobject: StaggeredPhaseField.solve_step

The two Jacobians are bilinear forms. As in every TensorMesh form,
``forward`` describes one quadrature point and one pair of basis functions.
The point data ``disp`` and ``alpha`` arrive as ``graddisp`` and ``alpha``:

.. literalinclude:: ../../../../examples/solid/phase_field_fracture/phase_field_fracture.py
   :language: python
   :pyobject: DisplacementTangent

.. literalinclude:: ../../../../examples/solid/phase_field_fracture/phase_field_fracture.py
   :language: python
   :pyobject: PhaseFieldHessian


Single-edge-notched tension
---------------------------

* **Geometry**: the unit square :math:`[0, 1]^2` mm, with a slit along
  :math:`y = 0.5` from the left edge to the centre. The elements above the
  slit have their own copies of its nodes, so its faces separate freely.
* **Material**: :math:`E = 210` kN/mm², :math:`\nu = 0.3`,
  :math:`G_c = 2.7 \times 10^{-3}` kN/mm, :math:`\ell = 0.015` mm, AT1,
  :math:`g_0 = 10^{-6}`.
* **Loading**: bottom clamped; top :math:`u_x = 0`, :math:`u_y = \delta`, in
  increments of :math:`10^{-4}` mm.
* **Mesh**: union-jack triangles, built with NumPy. The diagonals alternate
  like a checkerboard, so the mesh is mirror-symmetric about the crack line and
  favours no crack direction. :math:`h = \ell / 2` in a band around the crack
  path, coarser above and below (about 6,400 nodes).

.. figure:: /_static/solid_mechanics/phase_field_fracture.png
   :alt: Damage at the slit tip at peak load, the broken specimen, and the load-displacement curve
   :width: 100%

   Output of ``phase_field_fracture.py``. Left: the damage around the slit tip
   (white line) at peak load. Middle: the broken specimen, with displacements
   magnified. Right: the reaction force on the top edge.

The response is linear elastic until :math:`\delta \approx 3.9 \times
10^{-3}` mm (:math:`F \approx 0.55` kN), when damage first appears at the slit
tip: AT1 has an elastic phase. A process zone then grows at the tip while the load keeps rising, to a
peak of 0.87 kN at :math:`\delta = 6.2 \times 10^{-3}` mm. In the next load
step the crack runs straight through the ligament and the reaction drops to
zero. Under displacement control this propagation is unstable, so it happens
within a single step.

Elastic steps take one or two staggered iterations, and steps with a growing
process zone up to about thirty. The step in which the crack runs takes 138,
out of about 320 for the whole run. With ``--model at2`` damage grows from
the first step and the response softens before the peak (0.73 kN at
:math:`\delta = 5.9 \times 10^{-3}` mm), but the crack again runs through in
one step.


Verification
------------

``tests/assemble/test_phase_field_fracture_example.py`` checks:

* **Hessians**: the assembled Hessians against central differences of the
  autograd residuals.
* **Homogeneous solution**: a strip in uniaxial tension with AT1 stays
  undamaged below the critical strain
  :math:`\varepsilon_c = \sqrt{3 G_c (1 - \nu^2) / (8 E \ell)}`. Above it, the
  damage is :math:`\alpha = 1 - (\varepsilon_c / \varepsilon)^2`, and the
  reaction force is :math:`g(\alpha)\, E \varepsilon / (1 - \nu^2)` times the
  width.
* **Irreversibility**: damage never decreases, including after an unloading
  step.
* **Prescribed damage**: nodes held at :math:`\alpha = 1` through
  ``alpha_fixed`` stay there exactly.
* **Coarse SENT run**: the crack starts at the slit tip and runs straight along
  :math:`y = 0.5` to the right edge. It dissipates :math:`G_c` times its
  length, up to the effective-toughness factor of the mesh (see below).


Running it
----------

.. code-block:: bash

   cd examples/solid/phase_field_fracture
   python phase_field_fracture.py                       # AT1, ell = 0.015 mm
   python phase_field_fracture.py --model at2
   python phase_field_fracture.py --ell 0.04 --h 0.02   # coarse and quick

The default run takes about a minute on one CPU node (AT2: about one and a
half), and the coarse one a few seconds.
``--device cuda`` assembles on the GPU. The Newton systems then use cuDSS if
torch-sla has it, and SciPy on the host otherwise.


Notes
-----

* **Resolution.** :math:`h` must resolve :math:`\ell` (here
  :math:`h = \ell/2`). The dissipated energy then exceeds :math:`G_c` times
  the crack length by a mesh-dependent factor, because a fully broken band is
  at least one element wide. Here the 0.5 mm crack dissipates
  :math:`0.61\, G_c`, a factor of 1.22.
* **The length scale is a material parameter.** With AT1, :math:`\ell` also
  sets the strength, :math:`\sigma_c = \sqrt{3 G_c E' / (8 \ell)}` with
  :math:`E' = E / (1 - \nu^2)` in plane strain [Tanne2018]_. Changing
  :math:`\ell` changes when a crack nucleates, not only how smooth it is.
* **Solver.** Alternate minimization is robust but slow while a crack runs.
  Over-relaxing the damage update [Farrell2017]_ helps: ``--relaxation 1.6``
  gives the same result here with 15–20 % fewer iterations, and saves more
  when a crack grows over many load steps. Composite and monolithic Newton
  schemes can need fewer iterations still, and they impose the bounds with the
  same ``vi_solve`` call.


What's next
-----------

* :doc:`plasticity_strip`: history variables updated once per step, the
  other way to make a solid model path-dependent.
* :doc:`../../user_guide/forms`: the dispatch-by-name rules behind
  ``forward`` and :meth:`~tensormesh.ElementAssembler.energy`.


.. [Francfort1998] G. A. Francfort and J.-J. Marigo, *Revisiting brittle
   fracture as an energy minimization problem*, Journal of the Mechanics and
   Physics of Solids, 46(8):1319–1342, 1998.

.. [Bourdin2000] B. Bourdin, G. A. Francfort and J.-J. Marigo, *Numerical
   experiments in revisited brittle fracture*, Journal of the Mechanics and
   Physics of Solids, 48(4):797–826, 2000.

.. [Miehe2010] C. Miehe, M. Hofacker and F. Welschinger, *A phase field model
   for rate-independent crack propagation: Robust algorithmic implementation
   based on operator splits*, Computer Methods in Applied Mechanics and
   Engineering, 199(45–48):2765–2778, 2010.

.. [Pham2011] K. Pham, H. Amor, J.-J. Marigo and C. Maurini, *Gradient damage
   models and their use to approximate brittle fracture*, International Journal
   of Damage Mechanics, 20(4):618–652, 2011.

.. [Amor2009] H. Amor, J.-J. Marigo and C. Maurini, *Regularized formulation of
   the variational brittle fracture with unilateral contact: Numerical
   experiments*, Journal of the Mechanics and Physics of Solids,
   57(8):1209–1229, 2009.

.. [Benson2006] S. J. Benson and T. S. Munson, *Flexible complementarity
   solvers for large-scale applications*, Optimization Methods and Software,
   21(1):155–168, 2006.

.. [Tanne2018] E. Tanné, T. Li, B. Bourdin, J.-J. Marigo and C. Maurini,
   *Crack nucleation in variational phase-field models of brittle fracture*,
   Journal of the Mechanics and Physics of Solids, 110:80–99, 2018.

.. [Farrell2017] P. E. Farrell and C. Maurini, *Linear and nonlinear solvers
   for variational phase-field models of brittle fracture*, International
   Journal for Numerical Methods in Engineering, 109(5):648–667, 2017.

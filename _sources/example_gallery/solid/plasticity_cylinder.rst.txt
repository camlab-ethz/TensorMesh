Plasticity with Newton's Method
===============================

The elastic-plastic thick-walled cylinder [Hill1950]_: a long tube under
internal pressure, loaded until the whole wall flows. The script
``examples/solid/plasticity_cylinder/plasticity_cylinder.py`` solves it in
plane strain with von Mises (J2) plasticity and checks the result against the
closed-form elastic response and the limit pressure.

The new ingredient over :doc:`plasticity_strip` is the **consistent tangent**.
There, every load step minimizes an incremental energy with L-BFGS. Here,
every load step is solved by Newton's method with the derivative of the
stress update itself, so the iterations converge quadratically: three to five
per plastic load step. The plastic state stays at the quadrature points and
reaches the assemblers as ``quadrature_data``.


Model
-----

Small strains, :math:`\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^e + \boldsymbol{\varepsilon}^p`,
isotropic elasticity with shear modulus :math:`\mu` and bulk modulus
:math:`\kappa`, the von Mises yield function with linear isotropic hardening,

.. math::

   f(\boldsymbol{\sigma}, p) = q(\boldsymbol{\sigma}) - (\sigma_y + H p),
   \qquad q = \sqrt{\tfrac32\, \mathbf{s} : \mathbf{s}},

(:math:`\mathbf{s}` the stress deviator, :math:`p` the equivalent plastic
strain, :math:`H = 0` perfect plasticity) and associated flow. Plane strain
keeps :math:`\varepsilon_{zz} = 0`, so the strain and stress are carried as
:math:`3 \times 3` tensors: :math:`\sigma_{zz}` is not zero, and neither is
:math:`\varepsilon^p_{zz}`.

**Radial return.** In a load step the total strain :math:`\boldsymbol{\varepsilon}`
at a quadrature point is given, and the state at the start of the step is
:math:`(\boldsymbol{\varepsilon}^p_n, p_n)`. The trial stress
:math:`\boldsymbol{\sigma}^{tr} = \mathbb{C} : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n)`
has the deviator :math:`\mathbf{s}^{tr}` and the von Mises stress
:math:`q^{tr}`. If :math:`f^{tr} = q^{tr} - (\sigma_y + H p_n) \le 0` the step
is elastic. Otherwise the deviator is scaled back onto the yield surface:

.. math::

   \Delta p = \frac{f^{tr}}{3\mu + H}, \qquad
   \boldsymbol{\sigma} = \boldsymbol{\sigma}^{tr} - 2\mu\, \Delta p\, \sqrt{\tfrac32}\, \mathbf{N},
   \qquad \mathbf{N} = \frac{\mathbf{s}^{tr}}{\|\mathbf{s}^{tr}\|}.

**Consistent tangent** [Simo1985]_ [deSouzaNeto2008]_. Differentiating this
update, not the continuum relation, gives the moduli Newton needs:

.. math::

   \frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}
   = \kappa\, \mathbf{1} \otimes \mathbf{1}
   + 2\mu \Bigl(1 - \frac{3\mu\, \Delta p}{q^{tr}}\Bigr) \mathbb{I}_{dev}
   + 6\mu^2 \Bigl(\frac{\Delta p}{q^{tr}} - \frac{1}{3\mu + H}\Bigr) \mathbf{N} \otimes \mathbf{N}

on plastic points, and the elastic moduli elsewhere. ``radial_return``
computes the stress, the new state and these moduli for all quadrature points
at once:

.. literalinclude:: ../../../../examples/solid/plasticity_cylinder/plasticity_cylinder.py
   :language: python
   :pyobject: radial_return

The test suite compares the tangent with a forward-mode autodiff derivative
of the stress update, ``torch.func.jvp``. That derivative is also the way to
get the tangent of a model without a closed form.


Quadrature-point data
---------------------

The stress and the moduli are arrays over the quadrature points,
``[n_elements, n_quadrature, ...]``. Passed as ``quadrature_data``, an entry
reaches ``forward`` as its value at the current point, under its own name:

.. literalinclude:: ../../../../examples/solid/plasticity_cylinder/plasticity_cylinder.py
   :language: python
   :pyobject: InternalForce

.. literalinclude:: ../../../../examples/solid/plasticity_cylinder/plasticity_cylinder.py
   :language: python
   :pyobject: Tangent

The internal-force vector is ``InternalForce(quadrature_data={"sigma": sigma})``
and the stiffness matrix ``Tangent(quadrature_data={"C": C})``, with the
in-plane parts of the :math:`3 \times 3` tensors. The pressure acts along the
outward unit normal of the inner wall, which ``forward`` of a
:class:`~tensormesh.FacetAssembler` receives as ``n``:

.. literalinclude:: ../../../../examples/solid/plasticity_cylinder/plasticity_cylinder.py
   :language: python
   :pyobject: Pressure


Newton's method
---------------

Each load step iterates until the residual on the free degrees of freedom is
below :math:`10^{-10}` of the applied or reaction forces. The first iteration
starts from the converged state of the previous step and carries the change of
the prescribed displacements through the whole body (a linear predictor,
through :meth:`~tensormesh.Condenser.update_dirichlet`); the later iterations
keep the prescribed values. The plastic state is committed once the step has
converged, so every iteration returns from the same state.

.. literalinclude:: ../../../../examples/solid/plasticity_cylinder/plasticity_cylinder.py
   :language: python
   :pyobject: J2NewtonSolver.solve_step


The thick-walled cylinder
-------------------------

* **Geometry**: a quarter of the tube, :math:`a = 100` mm,
  :math:`b = 200` mm, plane strain, with :math:`u_y = 0` on the
  :math:`x` axis and :math:`u_x = 0` on the :math:`y` axis.
* **Material**: :math:`E = 200` GPa, :math:`\nu = 0.3`,
  :math:`\sigma_y = 250` MPa, perfect plasticity.
* **Loading**: the inner wall is pushed out to six times the first-yield
  displacement in 30 steps. Under this displacement control the pressure is
  the reaction on the inner wall, projected on the nodal forces of a unit
  pressure. ``--control pressure`` raises the pressure instead, which stops
  short of the limit.
* **Mesh**: six-node triangles on a polar grid, built with NumPy. The radial
  spacing grows with :math:`r` and the mid-edge nodes of the arcs lie on the
  circles (825 nodes).

.. figure:: /_static/solid_mechanics/plasticity_cylinder.png
   :alt: Plastic strain in the quarter tube, pressure against inner-wall displacement, and Newton residuals
   :width: 100%

   Output of ``plasticity_cylinder.py``. Left: the equivalent plastic strain
   at the end of loading. Middle: the pressure against the inner-wall
   displacement, with the Lamé line, the first-yield point and the limit
   pressure. Right: the Newton residuals of an elastic step, the first
   plastic step, a half-plastic step and the last one.

The response follows the Lamé line until the inner wall yields at
:math:`p_e = 108.1` MPa. The plastic zone then spreads outward, and the
pressure rises more and more slowly. Once the zone reaches the outer wall the
tube flows at a nearly constant pressure: the last step carries 199.7 MPa,
within 0.2 % of the limit pressure :math:`\tfrac{2}{\sqrt3}\, \sigma_y \ln(b/a) = 200.1`
MPa, approached from below. Elastic steps take one Newton update and plastic
steps three to five, with the residual roughly squared from one to the next.


Verification
------------

``tests/assemble/test_plasticity_cylinder_example.py`` checks:

* **Return map**: plastic points end on the yield surface, elastic points
  keep the elastic stress, the plastic flow is deviatoric, and the tangent
  equals a forward-mode autodiff derivative of the stress update.
* **Stiffness matrix**: the assembled tangent against central differences of
  the residual.
* **Elastic range**: the inner-wall displacement against the Lamé solution,

  .. math::

     \frac{p}{u_r(a)} = \frac{E\, (b^2 - a^2)}{(1 + \nu)\, a\, \bigl[(1 - 2\nu)\, a^2 + b^2\bigr]},

  under pressure and displacement control.
* **First yield**: no plastic point below the first-yield displacement, and
  only in the innermost elements just above it.
* **Collapse**: the pressure of a perfectly plastic tube rises to the limit
  pressure within 1 %, from below, and Newton converges in at most six updates
  per load step.
* **Devices**: CUDA against CPU, when a GPU is present.


Running it
----------

.. code-block:: bash

   cd examples/solid/plasticity_cylinder
   python plasticity_cylinder.py                       # displacement control
   python plasticity_cylinder.py --control pressure    # up to 0.98 of the limit pressure
   python plasticity_cylinder.py --hardening 2000      # H = 2 GPa

The default run takes a few seconds on a CPU. ``--device cuda`` assembles on
the GPU; the Newton systems then use cuDSS if torch-sla has it, and SciPy on
the host otherwise.


Notes
-----

* **Why displacement control.** Under pressure control a perfectly plastic
  tube cannot be loaded past the limit pressure, and close to it the tangent
  becomes singular. Prescribing the inner-wall displacement follows the
  plateau.
* **Elements.** Plastic flow preserves volume, and linear (constant-strain)
  triangles lock under this constraint in plane strain, overestimating the
  collapse load [Nagtegaal1974]_. The quadratic triangles used here reach the
  limit pressure within 0.2 %.
* **The first iteration.** Moving only the boundary nodes to their new
  prescribed positions before the first solve strains only the layer of
  elements next to them, and can make it yield spuriously. The linear predictor
  spreads the change through the body first.


What's next
-----------

* :doc:`plasticity_strip`: the same J2 model as an incremental energy,
  minimized with L-BFGS, with :class:`~tensormesh.assemble.J2Plasticity`.
* :doc:`../../user_guide/forms`: ``quadrature_data``, ``n`` and the other
  names ``forward`` can take.


.. [Hill1950] R. Hill, *The Mathematical Theory of Plasticity*, Oxford
   University Press, 1950.

.. [Simo1985] J. C. Simo and R. L. Taylor, *Consistent tangent operators for
   rate-independent elastoplasticity*, Computer Methods in Applied Mechanics
   and Engineering, 48(1):101–118, 1985.

.. [deSouzaNeto2008] E. A. de Souza Neto, D. Perić and D. R. J. Owen,
   *Computational Methods for Plasticity: Theory and Applications*, Wiley,
   2008.

.. [Nagtegaal1974] J. C. Nagtegaal, D. M. Parks and J. R. Rice, *On
   numerically accurate finite element solutions in the fully plastic range*,
   Computer Methods in Applied Mechanics and Engineering, 4(2):153–177, 1974.

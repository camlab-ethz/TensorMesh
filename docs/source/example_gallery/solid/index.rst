Solid Mechanics
===============

Six worked solid-mechanics examples in ``examples/solid/``, built
as a progressive ladder — each rung adds exactly one new concept on
top of the previous one:

#. **Cantilever Beam** — linear elasticity, one direct solve. The
   baseline.
#. **Hyperelastic Beam** — finite strain (Neo-Hookean); introduces
   the L-BFGS energy-minimization recipe.
#. **Hertzian Contact** — adds a constraint (contact penalty) and a
   closed-form verification, reusing the same L-BFGS recipe.
#. **Plasticity (J2)** — adds path-dependence: per-quadrature
   history variables and a variational constitutive update, in 2D
   and 3D.
#. **Geomechanics (Drucker-Prager)** — adds pressure-dependent
   yield, reusing the J2 history-variable pattern in an
   example-local constitutive driver for soils and weak rock.
#. **Geomechanics (elastic footing)** — solves a small
   boundary-value problem for footing settlement using the direct
   linear-elasticity workflow.

Together they cover the two solver patterns TensorMesh uses for
solid problems:

* **Direct linear solve** for small-strain linear elasticity
  (``cantilever_beam``, ``elastic_footing``).
* **L-BFGS energy minimization** for nonlinear problems where the
  potential energy is well-defined — hyperelasticity, contact, and
  plasticity (``hyperelastic_beam``, ``hertzian_contact``,
  ``plasticity_strip``).

The order below mirrors solver complexity.

.. grid:: 1 2 2 2
   :gutter: 4

   .. grid-item-card:: Cantilever Beam
      :link: cantilever_beam
      :link-type: doc
      :img-top: /_static/solid_mechanics/cantilever_steel.png

      Linear elasticity, steel cantilever with a tip load — the simplest end-to-end recipe.

   .. grid-item-card:: Hyperelastic Beam
      :link: hyperelastic_beam
      :link-type: doc
      :img-top: /_static/solid_mechanics/hyperelastic_rubber.png

      Rubber beam under torsion, compressible Neo-Hookean, L-BFGS load stepping.

   .. grid-item-card:: Hertzian Contact
      :link: hertzian_contact
      :link-type: doc
      :img-top: /_static/solid_mechanics/hertzian_contact.png

      Penalty contact between a circular indenter and an elastic block, checked against the Hertz solution.

   .. grid-item-card:: Plasticity (J2)
      :link: plasticity_strip
      :link-type: doc
      :img-top: /_static/solid_mechanics/plasticity_strip.gif

      Plane-strain J2 plasticity with isotropic hardening, load / unload cycle, plus a 3D cube.

   .. grid-item-card:: Geomechanics: Drucker-Prager
      :link: drucker_prager_triaxial
      :link-type: doc
      :img-top: /_static/solid_mechanics/drucker_prager_triaxial.png

      Pressure-dependent Drucker-Prager plasticity in a small triaxial-compression driver.

   .. grid-item-card:: Geomechanics: elastic footing
      :link: elastic_footing
      :link-type: doc
      :img-top: /_static/solid_mechanics/elastic_footing.png

      Linear-elastic soil block under a centered strip footing, with settlement
      contours and a reaction/load-balance sanity check.


.. toctree::
   :hidden:
   :maxdepth: 1

   cantilever_beam
   hyperelastic_beam
   hertzian_contact
   plasticity_strip
   drucker_prager_triaxial
   elastic_footing

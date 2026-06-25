tensormesh.material
===================

.. py:module:: tensormesh.material

Isotropic linear-elastic material model and a small set of preset
instances used throughout the solid-mechanics examples.

IsotropicMaterial
-----------------

.. autoclass:: tensormesh.material.IsotropicMaterial
    :members:
    :show-inheritance:


Preset materials
----------------

The module also ships ready-to-use :class:`~tensormesh.material.IsotropicMaterial`
instances with literature values. Import directly:

.. code-block:: python

   from tensormesh.material import Steel, Aluminum, Rubber, Glass

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 28

   * - Preset
     - Young's modulus :math:`E`
     - Poisson's ratio :math:`\nu`
     - Density :math:`\rho`
     - Notes
   * - ``Steel``
     - 210 GPa
     - 0.30
     - 7850 kg/m³
     - :math:`\sigma_y = 250` MPa
   * - ``Aluminum``
     - 70 GPa
     - 0.33
     - 2700 kg/m³
     - :math:`\sigma_y = 100` MPa, :math:`H = 700` MPa
   * - ``Rubber``
     - 10 MPa
     - 0.48
     - 1100 kg/m³
     - near-incompressible
   * - ``Glass``
     - 70 GPa
     - 0.20
     - 2500 kg/m³
     -


FrictionalMaterial
------------------

Pressure-dependent (Drucker-Prager / Mohr-Coulomb) soil and weak-rock material
consumed by :class:`~tensormesh.assemble.DruckerPragerPlasticity` and the
:func:`~tensormesh.functional.plastic.drucker_prager_return_mapping` primitive. Only
``E``, ``nu``, ``cohesion`` and ``friction_angle`` are required; a
``dilatancy_angle`` of ``None`` selects associated plastic flow.

.. autoclass:: tensormesh.material.FrictionalMaterial
    :members:
    :show-inheritance:

Named presets are loaded from a small CSV table shipped as package data:

.. code-block:: python

   from tensormesh.material import FrictionalMaterial

   print(FrictionalMaterial.preset_names())
   soil = FrictionalMaterial.from_preset("DenseSand")

.. note::

   The frictional-material presets below are *illustrative example presets* for
   reproducing the geomechanics examples, not design-grade soil/rock parameters.

.. list-table::
   :header-rows: 1
   :widths: 20 14 10 16 16 16

   * - Preset
     - :math:`E`
     - :math:`\nu`
     - cohesion
     - friction angle
     - dilatancy angle
   * - ``DenseSand``
     - 50 MPa
     - 0.30
     - 5 kPa
     - 35°
     - 5°
   * - ``LooseSand``
     - 25 MPa
     - 0.30
     - 1 kPa
     - 30°
     - 0°
   * - ``SoftClay``
     - 10 MPa
     - 0.35
     - 15 kPa
     - 22°
     - 0°
   * - ``StiffClay``
     - 30 MPa
     - 0.32
     - 40 kPa
     - 25°
     - 0°
   * - ``WeatheredRock``
     - 300 MPa
     - 0.25
     - 100 kPa
     - 38°
     - 5°

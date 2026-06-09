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

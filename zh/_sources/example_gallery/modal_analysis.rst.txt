Modal Analysis
==============

Two examples that solve **generalized eigenproblems** instead of driven
systems — one photonic, one structural. Both follow the same recipe:
assemble two (or three) matrices with the builtin assemblers, hand them
to a shift-invert sparse eigensolver, and read physical frequencies /
propagation constants off the spectrum:

.. code-block:: text

   mesh -> assemble (K, M, ...) -> shift-invert eigsh -> modes + frequencies

Each validates against a closed-form reference (Marcatili's slab
approximation, Euler-Bernoulli beam theory).


Rectangular waveguide modes (photonics)
---------------------------------------

``examples/wave/wave_guide_mode/waveguide_modes.py`` computes the guided
modes of a rectangular dielectric waveguide from its 2D cross-section.
In the weakly-guiding scalar approximation the transverse field
:math:`E(x,y)` and propagation constant :math:`\beta` satisfy

.. math::

   \big(K - k_0^2 M_\varepsilon\big)\,E \;=\; -\beta^2\, M\,E,

built from three builtins — stiffness
(:class:`~tensormesh.LaplaceElementAssembler`), mass
(:class:`~tensormesh.MassElementAssembler`), and the index-weighted mass
:math:`M_\varepsilon = \int n^2\,\phi_i\phi_j`
(:class:`~tensormesh.assemble.ScaledMassElementAssembler`). A mode is
guided when its effective index :math:`n_{\text{eff}} = \beta/k_0`
falls between the cladding and core indices; the default geometry
(:math:`18 \times 14` µm core, :math:`n = 1.450/1.444`,
:math:`\lambda_0 = 1.55` µm) carries seven guided modes, and the
computed effective indices agree with Marcatili's separable slab
approximation to :math:`\sim 10^{-4}`.

.. figure:: /_static/wave/waveguide_modes.png
   :alt: Guided transverse modes of a rectangular dielectric waveguide
   :width: 100%

   The guided transverse fields :math:`E_{pq}` (red/blue = field sign,
   core outlined), labelled by their lobe counts along :math:`x` and
   :math:`y` — the fundamental :math:`E_{11}`, the dipole pair
   :math:`E_{21}/E_{12}`, and the higher-order modes.


Cantilever beam vibration (structural)
--------------------------------------

``examples/solid/beam_vibration_analysis/vibration_cylinder.py`` runs
the classic free-vibration analysis of a clamped-free steel cylinder
(:math:`L = 1` m, :math:`R = 25` mm) in 3D linear elasticity:

.. math::

   K\,\phi = \omega^2 M\,\phi,

with :math:`K` from
:class:`~tensormesh.LinearElasticityElementAssembler` and the
consistent vector mass built from the scalar mass matrix
(:math:`M_{\text{vec}} = \rho\, M_{\text{scalar}} \otimes I_3`). The
clamped base is removed by the usual
:class:`~tensormesh.Condenser` mask. On top of the modes, a **modal
FRF** — the tip response to a transverse tip force with 1% modal
damping — shows the resonance/antiresonance structure.

The first bending frequency comes out at **37.2 Hz** vs Euler-Bernoulli
theory's 36.2 Hz (+3% with P1 tets, converging with refinement);
because the section is circular, every bending mode arrives as a
near-degenerate pair (two orthogonal bending planes), followed by the
first torsion and axial modes.

.. figure:: /_static/solid_mechanics/vibration_cylinder_modes.png
   :alt: Cantilever cylinder mode shapes
   :width: 100%

   Deformed mode shapes (displacement magnitude, exaggerated): the
   bending pairs, first torsion, and first axial mode.

.. figure:: /_static/solid_mechanics/vibration_cylinder_frf.png
   :alt: Tip displacement FRF with modal damping
   :width: 80%
   :align: center

   Dynamic amplification :math:`|u_x|/u_{\text{static}}` at the tip —
   flat at 1 below the first resonance, peaking on each bending mode,
   dropping into antiresonances between them.


Running them
------------

.. code-block:: bash

   cd examples/wave/wave_guide_mode && python waveguide_modes.py
   cd examples/solid/beam_vibration_analysis && python vibration_cylinder.py

Both expose ``run_demo(...)`` plus geometry / mode-count / mesh-size
flags (``--help`` lists them). The eigensolves run through SciPy's
shift-invert Lanczos on the assembled sparse matrices — post-processing
only, the assembly core stays torch.


What's next
-----------

* :doc:`phononic_crystal` — eigenproblems with Bloch-Floquet phase
  boundary conditions (band structures).
* :doc:`open_domain_wave` — the driven counterparts: ports and PML.
* :doc:`solid/index` — the static and nonlinear solid-mechanics ladder.

"""Ready-made assemblers for the most common FEM forms.

This package is the "battery" tier of :mod:`tensormesh.assemble`: each class
below is a worked subclass of one of the three base assemblers
(:class:`ElementAssembler`, :class:`NodeAssembler`, :class:`FacetAssembler`)
covering a textbook form — Laplace, mass, linear elasticity, Neo-Hookean
hyperelasticity, J2 plasticity, Drucker-Prager geomechanics, contact, and
constant/function loads.  Subclass these (or use the two factory functions in
:mod:`~tensormesh.assemble.builtin.common`) to skip re-deriving the weak form
for standard physics.

The built-ins are organized by physics category:

* :mod:`~tensormesh.assemble.builtin.fluid` — diffusion and mass forms;
* :mod:`~tensormesh.assemble.builtin.solid` — elasticity, plasticity,
  geomechanics, and contact;
* :mod:`~tensormesh.assemble.builtin.electromagnetic` — reserved namespace.

All names remain importable from :mod:`tensormesh.assemble.builtin` (and from
:mod:`tensormesh.assemble`) for backward compatibility.
"""

from .common import const_node_assembler, func_node_assembler
from .fluid import LaplaceElementAssembler, MassElementAssembler
from .solid import (
    ContactAssembler,
    DruckerPragerPlasticity,
    J2Plasticity,
    LinearElasticityElementAssembler,
    NeoHookeanModel,
)

__all__ = [
    "LaplaceElementAssembler",
    "MassElementAssembler",
    "LinearElasticityElementAssembler",
    "NeoHookeanModel",
    "J2Plasticity",
    "DruckerPragerPlasticity",
    "ContactAssembler",
    "const_node_assembler",
    "func_node_assembler",
]

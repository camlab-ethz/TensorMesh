"""FEM assembly: bilinear forms, linear forms, and boundary integrals.

Three base classes cover the common cases — :class:`ElementAssembler` for
volume bilinear forms (matrices), :class:`NodeAssembler` for volume linear
forms (vectors), and :class:`FacetAssembler` for boundary integrals. The
:mod:`tensormesh.assemble.builtin` module ships ready-made subclasses for
the most common forms (Laplace, mass, linear elasticity, hyperelasticity,
J2 plasticity, contact, constant/function loads).
"""

from .element_assembler import ElementAssembler
from .node_assembler import NodeAssembler
from .facet_assembler import FacetAssembler
from .builtin import (
    LaplaceElementAssembler,
    MassElementAssembler,
    LinearElasticityElementAssembler,
    NeoHookeanModel,
    J2Plasticity,
    DruckerPragerPlasticity,
    ContactAssembler,
    const_node_assembler,
    func_node_assembler,
)

__all__ = [
    "ElementAssembler",
    "NodeAssembler",
    "FacetAssembler",
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

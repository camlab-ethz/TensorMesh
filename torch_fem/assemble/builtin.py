from .element_assembler import ElementAssembler
from .node_assembler import NodeAssembler

from ..functional import dot, mul, sym, ddot, eye, trace


class LaplaceElementAssembler(ElementAssembler):
    """The element laplace assembler

    .. math::
    
        K = \\int_{\\Omega}\\nabla u \\cdot \\nabla v \\mathrm{d}x
    
    """
    def forward(self, gradu, gradv):
        K = dot(gradu, gradv)
        return K
    
class MassElementAssembler(ElementAssembler):
    """The element mass assembler
    
    .. math::
        
        K = \\int_{\\Omega} u v \\mathrm{d}x
        
    """
    def forward(self, u, v):
        K = mul(u, v)
        return K
    
__all__ = ["LaplaceElementAssembler", "MassElementAssembler"]
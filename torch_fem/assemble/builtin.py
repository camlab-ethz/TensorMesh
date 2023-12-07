from .element_assembler import ElementAssembler
from .node_assembler import NodeAssembler

from ..functional import dot, mul, sym, ddot, eye, trace


class LaplaceElementAssembler(ElementAssembler):
    r"""The element laplace assembler

    .. math::
    
        K = \int_{\Omega}\nabla u \cdot \nabla v \mathrm{d}v
    
    """
    def forward(self, gradu, gradv):
        K = dot(gradu, gradv)
        return K
    
class MassElementAssembler(ElementAssembler):
    r"""The element mass assembler
    
    .. math::
        
        K = \int_{\Omega} u v \mathrm{d}v
        
    """
    def forward(self, u, v):
        K = mul(u, v)
        return K
    
def const_node_assembler(c=1):
    class ConstNodeAssembler(NodeAssembler):
        r"""The const node assembler
        
        .. math::
    
            f = \int_{\Omega} c\cdot v \mathrm{d}v
        
        """
        def __post_init__(self, c=c):
            self.c = c 
        def forward(self, v):
            f = self.c * v
            return f
    return ConstNodeAssembler

def func_node_assembler(f=lambda x: x):
    class FuncNodeAssembler(NodeAssembler):
        r"""The func node assembler
        
        .. math::
    
            f = \int_{\Omega} f(x) v \mathrm{d}v
        
        """
        def __post_init__(self, f=f):
            self.f = f
        def forward(self, x, v):
            f = self.f(x) * v
            return f
    return FuncNodeAssembler

# class ConstNodeAssembler(NodeAssembler):
#     r"""The const node assembler
    
#     .. math::

#         f = \int_{\Omega} c\cdot v \mathrm{d}v
    
#     """
#     def __post_init__(self, c=1):
#         self.c = c
#     def forward(self, v):
#         f = self.c * v
#         return f
    
__all__ = ["LaplaceElementAssembler", "MassElementAssembler", "const_node_assembler", "func_node_assembler"]
"""Category-independent built-in load assemblers.

Factory helpers that build :class:`~tensormesh.assemble.node_assembler.NodeAssembler`
subclasses for constant and spatially-varying volume loads. These are shared by
every physics category, so they live at the top of the :mod:`builtin` package
rather than under a single physics directory.
"""

from tensormesh.assemble.node_assembler import NodeAssembler


def const_node_assembler(c = 1):
    r"""Factory: build a :class:`NodeAssembler` for a constant body load.

    **Weak form:**

    .. math::

        f_i = \int_{\Omega} c \, N_i \, \mathrm{d}\Omega

    This represents a uniform body force or source term.

    Parameters
    ----------
    c : float, optional
        Constant load value. Default: ``1``.

    Returns
    -------
    type[NodeAssembler]
        A new :class:`NodeAssembler` subclass with ``c`` baked in.

    Examples
    --------
    .. code-block:: python

        ConstLoad = const_node_assembler(c=9.81)        # gravity
        f = ConstLoad.from_mesh(mesh)(mesh.points)
    """
    class ConstNodeAssembler(NodeAssembler):
        r"""Constant load node assembler.

        .. math::

            f = \int_{\Omega} c\cdot v \mathrm{d}\Omega

        """
        def __post_init__(self, c=c):
            self.c = c
        def forward(self, v):
            f = self.c * v
            return f
    return ConstNodeAssembler

def func_node_assembler(f=lambda x: x):
    r"""Factory: build a :class:`NodeAssembler` for a spatially-varying load.

    **Weak form:**

    .. math::

        f_i = \int_{\Omega} f(\mathbf{x}) \, N_i \, \mathrm{d}\Omega

    Parameters
    ----------
    f : Callable
        Function returning the load value at a coordinate. Signature
        ``f(x) -> load``, where ``x`` has shape ``[..., dim]``.

    Returns
    -------
    type[NodeAssembler]
        A new :class:`NodeAssembler` subclass with ``f`` baked in.

    Examples
    --------
    .. code-block:: python

        source = func_node_assembler(lambda x: torch.sin(np.pi * x[..., 0]))
        rhs = source.from_mesh(mesh)(mesh.points)
    """
    class FuncNodeAssembler(NodeAssembler):
        r"""Function-based load node assembler.

        .. math::

            f = \int_{\Omega} f(\mathbf{x}) \, v \, \mathrm{d}\Omega

        """
        def __post_init__(self, f=f):
            self.f = f
        def forward(self, x, v):
            f = self.f(x) * v
            return f
    return FuncNodeAssembler

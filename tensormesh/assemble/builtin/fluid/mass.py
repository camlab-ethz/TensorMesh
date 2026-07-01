"""Mass-matrix built-in assembler."""

from tensormesh.assemble.element_assembler import ElementAssembler


class MassElementAssembler(ElementAssembler):
    r"""Mass Element Assembler.

    Assembles the mass matrix for finite element discretization. The mass matrix
    represents the :math:`L^2` inner product and is essential for time-dependent
    problems, eigenvalue problems, and :math:`L^2` projection.

    **Weak Form:**

    The mass matrix arises from terms like :math:`\int_\Omega u \, v \, \mathrm{d}\Omega`
    in the weak formulation.

    **Element Mass Matrix:**

    For each element :math:`K`, the local mass matrix entry is:

    .. math::

        M_{ij}^e = \int_{\Omega^e} N_i \, N_j \, \mathrm{d}\Omega

    where :math:`N_i, N_j` are the shape functions.

    **Applications:**

    - Time-dependent PDEs (heat equation, wave equation)
    - :math:`L^2` error computation: :math:`\|u - u_h\|_{L^2}^2 = (u-u_h)^T M (u-u_h)`
    - Eigenvalue problems: :math:`K u = \lambda M u`

    Examples
    --------
    .. code-block:: python

        mesh = Mesh.gen_rectangle(chara_length=0.1)
        M = MassElementAssembler.from_mesh(mesh)(mesh.points)
    """
    def forward(self, u, v):
        return u * v

"""Diffusion / Laplace built-in assembler."""

from tensormesh.assemble.element_assembler import ElementAssembler


class LaplaceElementAssembler(ElementAssembler):
    r"""Laplace/Diffusion Element Assembler.

    Assembles the stiffness matrix for the Laplace operator (diffusion term).
    This is the fundamental building block for solving elliptic PDEs such as
    the Poisson equation, heat equation (steady-state), and other diffusion problems.

    **Weak Form:**

    Given the Laplace equation :math:`-\nabla \cdot (\kappa \nabla u) = f`, the weak form is:

    .. math::

        \int_{\Omega} \kappa \nabla u \cdot \nabla v \, \mathrm{d}\Omega = \int_{\Omega} f v \, \mathrm{d}\Omega

    **Element Stiffness Matrix:**

    For each element :math:`K`, the local stiffness matrix entry is:

    .. math::

        K_{ij}^e = \int_{\Omega^e} \nabla N_i \cdot \nabla N_j \, \mathrm{d}\Omega

    where :math:`N_i, N_j` are the shape functions.

    **Implementation:**

    The ``forward`` method computes the integrand :math:`\nabla N_i \cdot \nabla N_j`
    at each quadrature point, which is then integrated by the base class.

    Examples
    --------
    .. code-block:: python

        mesh = Mesh.gen_rectangle(chara_length=0.1)
        K = LaplaceElementAssembler.from_mesh(mesh)(mesh.points)
    """
    def forward(self, gradu, gradv):
        return gradu @ gradv

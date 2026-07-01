"""Linear-elastic and Neo-Hookean built-in solid assemblers."""

import torch

from tensormesh.assemble.element_assembler import ElementAssembler
from tensormesh.functional.elasticity import voigt_shape_grad, voigt_stiffness


class LinearElasticityElementAssembler(ElementAssembler):
    r"""Linear Elasticity Element Assembler.

    Assembles the stiffness matrix for linear elastic materials based on
    Hooke's law. Suitable for small deformation analysis of isotropic materials.

    **Constitutive Law (Hooke's Law):**

    The stress-strain relationship for isotropic linear elasticity:

    .. math::

        \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}

    where the elasticity tensor :math:`\mathbf{C}` is defined by:

    .. math::

        C_{ijkl} = \lambda \delta_{ij} \delta_{kl} + \mu (\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})

    **Lamé Parameters:**

    .. math::

        \lambda = \frac{E \nu}{(1+\nu)(1-2\nu)}, \quad \mu = \frac{E}{2(1+\nu)}

    where :math:`E` is Young's modulus and :math:`\nu` is Poisson's ratio.

    **Strain Tensor:**

    The infinitesimal strain tensor:

    .. math::

        \boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)

    **Weak Form:**

    .. math::

        \int_{\Omega} \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \, \mathrm{d}\Omega
        = \int_{\Omega} \mathbf{f} \cdot \mathbf{v} \, \mathrm{d}\Omega
        + \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, \mathrm{d}S

    **Strain Energy Density:**

    .. math::

        \Psi = \frac{1}{2} \boldsymbol{\varepsilon} : \mathbf{C} : \boldsymbol{\varepsilon}
        = \frac{\lambda}{2} (\mathrm{tr}\, \boldsymbol{\varepsilon})^2 + \mu \, \boldsymbol{\varepsilon} : \boldsymbol{\varepsilon}

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default: ``1.0``.
    nu : float, optional
        Poisson's ratio (must satisfy :math:`-1 < \nu < 0.5`). Default: ``0.3``.

    Examples
    --------
    .. code-block:: python

        mesh = Mesh.gen_cube(chara_length=0.1)
        assembler = LinearElasticityElementAssembler.from_mesh(mesh, E=210e9, nu=0.3)
        K = assembler(mesh.points)
    """
    def __post_init__(self, E=1.0, nu=0.3):
        self.E = E
        self.nu = nu

    def forward(self, gradu, gradv):
        dim = gradu.shape[0]
        Ba = voigt_shape_grad(gradu)
        Bb = voigt_shape_grad(gradv)
        C = voigt_stiffness(self.E, self.nu, dim)
        C = C.to(dtype=gradu.dtype, device=gradu.device)
        return Ba.T @ C @ Bb

    def element_energy(self, graddisplacement):
        r"""Compute strain energy density at a quadrature point.

        .. math::

            \Psi = \frac{\lambda}{2} (\mathrm{tr}\, \boldsymbol{\varepsilon})^2
            + \mu \, \boldsymbol{\varepsilon} : \boldsymbol{\varepsilon}

        Parameters
        ----------
        graddisplacement : torch.Tensor
            Displacement gradient :math:`\nabla \mathbf{u}` of shape ``[dim, dim]``.

        Returns
        -------
        torch.Tensor
            Scalar strain energy density.
        """
        grad_u = graddisplacement
        dim = grad_u.shape[-1]

        # Strain epsilon = 0.5 (grad_u + grad_u.T)
        eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))

        # Lame parameters
        mu = self.E / (2 * (1 + self.nu))
        lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

        # Trace squared
        tr_eps = eps.diagonal(dim1=-2, dim2=-1).sum(-1)
        vol_term = 0.5 * lam * (tr_eps ** 2)

        # Double dot product eps : eps
        eps_sq = (eps * eps).sum(dim=(-2, -1))
        dev_term = mu * eps_sq

        energy = vol_term + dev_term
        return energy

class NeoHookeanModel(ElementAssembler):
    r"""Neo-Hookean Hyperelastic Material Model.

    A nonlinear hyperelastic constitutive model for large deformation analysis.
    The Neo-Hookean model is the simplest hyperelastic model and extends linear
    elasticity to the finite strain regime.

    **Deformation Gradient:**

    .. math::

        \mathbf{F} = \mathbf{I} + \nabla \mathbf{u}

    **Kinematic Quantities:**

    - Right Cauchy-Green tensor: :math:`\mathbf{C} = \mathbf{F}^T \mathbf{F}`
    - First invariant: :math:`I_1 = \mathrm{tr}(\mathbf{C}) = \|\mathbf{F}\|_F^2`
    - Jacobian (volume ratio): :math:`J = \det(\mathbf{F})`

    **Strain Energy Density:**

    .. math::

        \Psi = \frac{\mu}{2}(I_1 - d) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2

    where :math:`d` is the spatial dimension (2 or 3).

    **First Piola-Kirchhoff Stress:**

    .. math::

        \mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}
        = \mu \mathbf{F} + (\lambda \ln J - \mu) \mathbf{F}^{-T}

    **Material Parameters:**

    .. math::

        \mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default: ``1.0``.
    nu : float, optional
        Poisson's ratio. Default: ``0.3``.

    Notes
    -----
    Requires :math:`J > 0` (no element inversion). For nearly incompressible
    materials (:math:`\nu \to 0.5`), consider a mixed formulation to avoid
    volumetric locking.

    Examples
    --------
    .. code-block:: python

        mesh = Mesh.gen_cube(chara_length=0.1)
        model = NeoHookeanModel.from_mesh(mesh, E=1e6, nu=0.45)
        E_tot = model.energy(displacement)
    """
    def __post_init__(self, E=1.0, nu=0.3):
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    def element_energy(self, graddisplacement):
        r"""Compute Neo-Hookean strain energy density at a quadrature point.

        .. math::

            \Psi = \frac{\mu}{2}(I_1 - d) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2

        Parameters
        ----------
        graddisplacement : torch.Tensor
            Displacement gradient :math:`\nabla \mathbf{u}` of shape ``[dim, dim]``.

        Returns
        -------
        torch.Tensor
            Scalar strain energy density.
        """
        grad_u = graddisplacement
        dim = grad_u.shape[-1]

        # F = I + grad_u
        F = torch.eye(dim, device=grad_u.device, dtype=grad_u.dtype) + grad_u

        # Invariants
        J = torch.linalg.det(F)
        I1 = (F * F).sum() # tr(F^T F)

        log_J = torch.log(J)
        psi = (self.mu / 2) * (I1 - dim) - self.mu * log_J + (self.lam / 2) * (log_J ** 2)

        return psi

    def energy(self, u):
        r"""Compute total strain energy.

        .. math::

            \Pi = \int_{\Omega} \Psi(\mathbf{F}) \, \mathrm{d}\Omega

        Parameters
        ----------
        u : torch.Tensor
            Displacement field of shape ``[n_nodes, dim]``.

        Returns
        -------
        torch.Tensor
            Scalar total strain energy.
        """
        return super().energy(point_data={"displacement": u})

"""Ready-made assemblers for the most common FEM forms.

This module is the "battery" tier of :mod:`tensormesh.assemble`: each class
below is a worked subclass of one of the three base assemblers
(:class:`ElementAssembler`, :class:`NodeAssembler`, :class:`FacetAssembler`)
covering a textbook form — Laplace, mass, linear elasticity, Neo-Hookean
hyperelasticity, J2 plasticity, contact, and constant/function loads.
Subclass these (or use the two factory functions at the bottom) to skip
re-deriving the weak form for standard physics.
"""

import inspect
from typing import Optional, Dict, Union, Callable

import numpy as np
import torch

from .element_assembler import ElementAssembler, InputBroadcast
from .facet_assembler import FacetAssembler
from .node_assembler import NodeAssembler
from ..functional.elasticity import voigt_shape_grad, voigt_stiffness
from ..functional.plastic import drucker_prager_return_mapping, small_strain_3d
from ..vmap import vmap

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

class J2Plasticity(ElementAssembler):
    r"""J2 (von Mises) Plasticity Model with Isotropic Hardening.
    
    Implements rate-independent J2 plasticity with linear isotropic hardening
    using a return-mapping algorithm. This model is suitable for metals and
    other ductile materials under monotonic or cyclic loading.
    
    **Yield Function (von Mises):**
    
    .. math::
    
        f(\boldsymbol{\sigma}, \alpha) = \|\mathbf{s}\| - \sqrt{\frac{2}{3}}(\sigma_0 + H\alpha) \leq 0
    
    where:
    
    - :math:`\mathbf{s} = \boldsymbol{\sigma} - \frac{1}{3}\mathrm{tr}(\boldsymbol{\sigma})\mathbf{I}` is the deviatoric stress
    - :math:`\sigma_0` is the initial yield stress
    - :math:`H` is the hardening modulus
    - :math:`\alpha` is the equivalent plastic strain (internal variable)
    
    **Additive Strain Decomposition:**
    
    .. math::
    
        \boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^e + \boldsymbol{\varepsilon}^p
    
    **Flow Rule (Associated):**
    
    .. math::
    
        \dot{\boldsymbol{\varepsilon}}^p = \dot{\gamma} \frac{\partial f}{\partial \boldsymbol{\sigma}}
        = \dot{\gamma} \frac{\mathbf{s}}{\|\mathbf{s}\|}
    
    **Hardening Law:**
    
    .. math::
    
        \dot{\alpha} = \sqrt{\frac{2}{3}} \dot{\gamma}
    
    **Return Mapping Algorithm:**
    
    1. Compute trial elastic stress: :math:`\boldsymbol{\sigma}^{tr} = \mathbf{C}:(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n)`
    2. Check yield: :math:`f^{tr} = \|\mathbf{s}^{tr}\| - \sqrt{2/3}(\sigma_0 + H\alpha_n)`
    3. If :math:`f^{tr} \leq 0`: elastic step, accept trial state
    4. If :math:`f^{tr} > 0`: plastic correction
    
       .. math::
       
           \Delta\gamma = \frac{f^{tr}}{2\mu + \frac{2}{3}H}
    
    **Algorithmic Incremental Potential:**
    
    .. math::
    
        \Psi^{alg} = \frac{K}{2}(\mathrm{tr}\,\boldsymbol{\varepsilon}^e)^2 
        + \mu \|\mathbf{e}^{tr}\|^2 - \frac{1}{2}(2\mu + \frac{2}{3}H)(\Delta\gamma)^2
    
    where :math:`K = \lambda + \frac{2}{3}\mu` is the bulk modulus and
    :math:`\mathbf{e}^{tr}` is the deviatoric trial strain.

    Parameters
    ----------
    material : optional
        Material object with properties ``E``, ``nu``, ``sigma_y``, ``H``;
        when supplied, overrides the individual scalar arguments.
    E : float, optional
        Young's modulus. Default: ``200e9`` (steel).
    nu : float, optional
        Poisson's ratio. Default: ``0.3``.
    sig0 : float, optional
        Initial yield stress. Default: ``250e6``.
    H : float, optional
        Hardening modulus. Default: ``1e9``.

    Attributes
    ----------
    history : dict
        Internal state variables — plastic strain (:math:`\boldsymbol{\varepsilon}^p`)
        and equivalent plastic strain (:math:`\alpha`) — keyed by element type.

    Examples
    --------
    .. code-block:: python

        mesh = Mesh.gen_cube(chara_length=0.05)
        plasticity = J2Plasticity.from_mesh(mesh, E=200e9, nu=0.3, sig0=250e6, H=1e9)
        # In time-stepping loop:
        energy = plasticity.energy(point_data={"displacement": u})
        energy.backward()
        # After convergence:
        plasticity.update_state(u)
    """
    def __post_init__(self, material=None, E=200e9, nu=0.3, sig0=250e6, H=1e9):
        if material is not None:
            self.E = material.E
            self.nu = material.nu
            self.sig0 = material.sigma_y
            self.H = material.H
        else:
            self.E = E
            self.nu = nu
            self.sig0 = sig0
            self.H = H
        
        # Lamé parameters
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.bulk = self.lam + 2/3 * self.mu
        
        # Initialize History Variables
        self.history = {}
        for etype, trans in self.transformation.items():
            n_elem = trans.n_elements
            n_quad = trans.n_quadrature
            
            # Plastic strain tensor (trace is 0 for J2)
            eps_p = torch.zeros((n_elem, n_quad, 3, 3), device=self.device, dtype=self.dtype)
            
            # Equivalent plastic strain
            alpha = torch.zeros((n_elem, n_quad), device=self.device, dtype=self.dtype)
            
            self.history[etype] = {'eps_p': eps_p, 'alpha': alpha}

    def element_energy(self, graddisplacement, eps_p_n, alpha_n):
        r"""Compute algorithmic incremental potential energy density.

        This implements the return-mapping algorithm at the quadrature point level.

        Parameters
        ----------
        graddisplacement : torch.Tensor
            Displacement gradient :math:`\nabla \mathbf{u}`.
        eps_p_n : torch.Tensor
            Plastic strain from the previous step :math:`\boldsymbol{\varepsilon}^p_n`.
        alpha_n : torch.Tensor
            Equivalent plastic strain from the previous step :math:`\alpha_n`.

        Returns
        -------
        torch.Tensor
            Scalar incremental potential energy density.
        """
        grad_u = graddisplacement
        dim = grad_u.shape[0]
        
        # Construct 3D strain tensor
        if dim == 2:
            eps_2d = 0.5 * (grad_u + grad_u.T)
            eps = torch.nn.functional.pad(eps_2d, (0, 1, 0, 1))
        else:
            eps = 0.5 * (grad_u + grad_u.T)
        
        # Trial Elastic Step
        eps_tr = eps - eps_p_n
        tr_eps_tr = eps_tr.diagonal(dim1=-2, dim2=-1).sum(-1)
        dev_eps_tr = eps_tr - (tr_eps_tr / 3.0) * torch.eye(3, device=grad_u.device, dtype=grad_u.dtype)
        norm_dev_eps_tr = torch.norm(dev_eps_tr)
        
        # Trial yield criterion
        norm_s_tr = 2 * self.mu * norm_dev_eps_tr
        radius = np.sqrt(2/3) * (self.sig0 + self.H * alpha_n)
        f_tr = norm_s_tr - radius
        
        # Volumetric energy
        vol_energy = 0.5 * self.bulk * (tr_eps_tr**2)
        
        # Plastic multiplier
        denom = 2 * self.mu + (2/3) * self.H
        d_gamma = torch.clamp(f_tr, min=0.0) / denom
        
        # Deviatoric energy with plastic correction
        dev_energy = self.mu * (norm_dev_eps_tr**2) - 0.5 * denom * (d_gamma**2)
        
        psi = vol_energy + dev_energy
        return psi

    def update_state(self, u_vec):
        r"""Update internal state variables after load-step convergence.

        Call after the Newton-Raphson iteration converges to update the
        plastic strain and the equivalent plastic strain.

        Parameters
        ----------
        u_vec : torch.Tensor
            Converged displacement field.
        """
        with torch.no_grad():
            for etype, trans in self.transformation.items():
                cells = trans.elements
                u_elem = u_vec[cells]
                grad_u = torch.einsum('bqkx,bku->bqux', trans.shape_grad, u_elem)
                
                dim = grad_u.shape[-1]
                eps = torch.zeros(grad_u.shape[:2] + (3, 3), device=u_vec.device, dtype=u_vec.dtype)
                
                if dim == 2:
                    grad_u_2d = grad_u
                    eps[..., :2, :2] = 0.5 * (grad_u_2d + grad_u_2d.transpose(-1, -2))
                else:
                    eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))
                
                hist = self.history[etype]
                eps_p_n = hist['eps_p']
                alpha_n = hist['alpha']
                
                # Trial
                eps_tr = eps - eps_p_n
                tr_eps_tr = eps_tr.diagonal(dim1=-2, dim2=-1).sum(-1)
                dev_eps_tr = eps_tr - (tr_eps_tr.unsqueeze(-1).unsqueeze(-1) / 3.0) * torch.eye(3, device=u_vec.device, dtype=u_vec.dtype)
                
                norm_dev_eps_tr = torch.norm(dev_eps_tr, dim=(-2, -1))
                norm_s_tr = 2 * self.mu * norm_dev_eps_tr
                radius = np.sqrt(2/3) * (self.sig0 + self.H * alpha_n)
                f_tr = norm_s_tr - radius
                
                d_gamma = torch.clamp(f_tr, min=0.0) / (2 * self.mu + (2/3) * self.H)
                
                # Update direction
                norm_safe = torch.where(norm_dev_eps_tr < 1e-12, torch.ones_like(norm_dev_eps_tr), norm_dev_eps_tr)
                n_tensor = dev_eps_tr / norm_safe.unsqueeze(-1).unsqueeze(-1)
                
                yield_mask = f_tr > 0
                d_gamma_masked = d_gamma * yield_mask.float()
                
                hist['eps_p'] += d_gamma_masked.unsqueeze(-1).unsqueeze(-1) * n_tensor
                hist['alpha'] += np.sqrt(2/3) * d_gamma_masked


class DruckerPragerPlasticity(ElementAssembler):
    r"""Drucker-Prager Plasticity Model with Linear Isotropic Hardening.

    Pressure-dependent, small-strain Drucker-Prager plasticity for soils and
    weak rock, built on the pure constitutive primitive
    :func:`~tensormesh.functional.plastic.drucker_prager_return_mapping`. It mirrors the
    :class:`J2Plasticity` lifecycle — per-quadrature history, previous-step
    state passed through ``element_data``, and :meth:`update_state` after a
    converged load step — so it slots into the same energy-minimisation solvers.

    **Yield Function (tension-positive convention):**

    .. math::

        f(\boldsymbol{\sigma}, \alpha) = q + \eta\, I_1 - (k + H\alpha) \leq 0

    where :math:`I_1 = \mathrm{tr}(\boldsymbol{\sigma})`,
    :math:`q = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}`, and the cone
    coefficients :math:`\eta, k` are fitted to the Mohr-Coulomb
    triaxial-compression meridian. Because compression gives a negative
    :math:`I_1`, confinement raises the yield capacity.

    Parameters
    ----------
    material : optional
        Object with attributes ``E``, ``nu``, ``cohesion``, ``friction_angle``,
        ``dilatancy_angle`` and ``H`` (e.g.
        :class:`tensormesh.material.FrictionalMaterial`); when supplied it
        overrides the scalar arguments.
    E : float, optional
        Young's modulus. Default ``50e6``.
    nu : float, optional
        Poisson's ratio. Default ``0.30``.
    cohesion : float, optional
        Cohesion in Pa. Default ``20e3``.
    friction_angle : float, optional
        Friction angle in degrees. Default ``30.0``.
    dilatancy_angle : float or None, optional
        Dilatancy angle in degrees; ``None`` (default) means associated flow.
    H : float, optional
        Linear isotropic hardening modulus in Pa. Default ``1e6``.

    Attributes
    ----------
    history : dict
        Internal state variables — plastic strain (``eps_p``) and equivalent
        plastic strain (``alpha``) — keyed by element type.

    Examples
    --------
    .. code-block:: python

        from tensormesh.assemble import DruckerPragerPlasticity
        from tensormesh.material import FrictionalMaterial

        soil = FrictionalMaterial.from_preset("DenseSand")
        model = DruckerPragerPlasticity.from_mesh(mesh, material=soil)
        # In a load-stepping loop:
        energy = model.energy(point_data={"displacement": u},
                              element_data=model.element_data_from_history())
        energy.backward()
        # After convergence:
        model.update_state(u)
    """
    def __post_init__(self, material=None, E=50e6, nu=0.30, cohesion=20e3,
                      friction_angle=30.0, dilatancy_angle=None, H=1e6):
        if material is not None:
            self.E = material.E
            self.nu = material.nu
            self.cohesion = material.cohesion
            self.friction_angle = material.friction_angle
            self.dilatancy_angle = material.dilatancy_angle
            self.H = material.H
        else:
            self.E = E
            self.nu = nu
            self.cohesion = cohesion
            self.friction_angle = friction_angle
            self.dilatancy_angle = dilatancy_angle
            self.H = H

        # Cached elastic moduli for stress reporting.
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.bulk = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        # Initialize history variables.
        self.history = {}
        for etype, trans in self.transformation.items():
            n_elem = trans.n_elements
            n_quad = trans.n_quadrature
            eps_p = torch.zeros((n_elem, n_quad, 3, 3), device=self.device, dtype=self.dtype)
            alpha = torch.zeros((n_elem, n_quad), device=self.device, dtype=self.dtype)
            self.history[etype] = {'eps_p': eps_p, 'alpha': alpha}

    def _material_kwargs(self):
        return dict(E=self.E, nu=self.nu, cohesion=self.cohesion,
                    friction_angle=self.friction_angle,
                    dilatancy_angle=self.dilatancy_angle, H=self.H)

    def element_energy(self, graddisplacement, eps_p_n, alpha_n):
        r"""Algorithmic incremental potential density at one quadrature point.

        Delegates to :func:`~tensormesh.functional.plastic.drucker_prager_return_mapping`
        and returns its ``energy`` field.
        """
        return drucker_prager_return_mapping(
            graddisplacement, eps_p_n, alpha_n, **self._material_kwargs()
        ).energy

    def update_state(self, u_vec):
        r"""Commit per-quadrature ``eps_p`` and ``alpha`` after convergence."""
        with torch.no_grad():
            for etype, trans in self.transformation.items():
                u_elem = u_vec[trans.elements]
                grad_u = torch.einsum('bqkx,bku->bqux', trans.shape_grad, u_elem)
                hist = self.history[etype]
                result = drucker_prager_return_mapping(
                    grad_u, hist['eps_p'], hist['alpha'], **self._material_kwargs()
                )
                hist['eps_p'] = result.eps_p
                hist['alpha'] = result.alpha

    def element_data_from_history(self):
        r"""Return committed history as the ``element_data`` mapping for ``energy``."""
        return {
            "eps_p_n": {etype: h['eps_p'] for etype, h in self.history.items()},
            "alpha_n": {etype: h['alpha'] for etype, h in self.history.items()},
        }

    def max_alpha(self):
        r"""Maximum committed equivalent plastic strain over all quadrature points."""
        return torch.cat([h['alpha'].reshape(-1) for h in self.history.values()]).max()

    def mean_alpha(self):
        r"""Mean committed equivalent plastic strain over all quadrature points."""
        return torch.cat([h['alpha'].reshape(-1) for h in self.history.values()]).mean()

    def mean_stress(self, u_vec):
        r"""Average committed Cauchy stress over elements and quadrature points."""
        eye = torch.eye(3, device=self.device, dtype=self.dtype)
        stresses = []
        with torch.no_grad():
            for etype, trans in self.transformation.items():
                u_elem = u_vec[trans.elements]
                grad_u = torch.einsum('bqkx,bku->bqux', trans.shape_grad, u_elem)
                eps = small_strain_3d(grad_u)
                eps_e = eps - self.history[etype]['eps_p']
                tr_eps = eps_e.diagonal(dim1=-2, dim2=-1).sum(-1)
                dev_eps = eps_e - (tr_eps[..., None, None] / 3.0) * eye
                sigma = 2.0 * self.mu * dev_eps + self.bulk * tr_eps[..., None, None] * eye
                stresses.append(sigma.reshape(-1, 3, 3))
        return torch.cat(stresses, dim=0).mean(dim=0)


class ContactAssembler(FacetAssembler):
    r"""Contact/Boundary Facet Assembler.
    
    Assembler for integrating energy contributions over boundary facets (surfaces in 3D,
    edges in 2D). This is useful for implementing:
    
    - **Contact mechanics**: Penalty or barrier methods for non-penetration constraints
    - **Surface tension**: Capillary effects in fluid-structure interaction
    - **Pressure loads**: Follower forces that remain normal to deformed surface
    - **Robin boundary conditions**: Mixed Dirichlet-Neumann conditions
    
    **Penalty Contact Formulation:**
    
    For a penalty-based contact between surface :math:`\Gamma` and an obstacle:
    
    .. math::
    
        \Pi_{contact} = \int_{\Gamma} \frac{\kappa}{2} \langle g_n \rangle_-^2 \, \mathrm{d}S
    
    where:
    
    - :math:`g_n` is the normal gap function (negative when penetrating)
    - :math:`\langle \cdot \rangle_- = \min(\cdot, 0)` is the negative part (Macaulay bracket)
    - :math:`\kappa` is the penalty stiffness
    
    **Usage Pattern:**
    
    Subclass ``ContactAssembler`` and implement ``element_energy`` to define
    the specific contact/boundary energy density.

    Examples
    --------
    .. code-block:: python

        class PenaltyContact(ContactAssembler):
            def __post_init__(self, kappa=1e6, obstacle_y=0.0):
                self.kappa = kappa
                self.obstacle_y = obstacle_y

            def element_energy(self, x):
                gap = x[..., 1] - self.obstacle_y         # y-coordinate gap
                penetration = torch.clamp(-gap, min=0.0)
                return 0.5 * self.kappa * penetration ** 2
    """
    def energy(self, points:Optional[torch.Tensor] = None, 
                       func:Optional[Callable] = None,
                       point_data:Optional[Dict[str, torch.Tensor]] = None, 
                       element_data:Optional[Union[Dict[str, Dict[str,torch.Tensor]], Dict[str,torch.Tensor]]] = None, 
                       scalar_data:Optional[Dict[str, torch.Tensor]] = None,
                       batch_size:int = -1):
        r"""Compute total boundary / contact energy.

        Integrates ``element_energy`` over all selected boundary facets:

        .. math::

            \Pi = \int_{\Gamma} \psi(\mathbf{x}, \mathbf{u}, \ldots) \, \mathrm{d}S

        Parameters
        ----------
        points : torch.Tensor, optional
            Updated nodal coordinates; if ``None``, the cached points are used.
        func : Callable, optional
            Custom energy density to use *in place of* :meth:`element_energy`.
        point_data : dict[str, torch.Tensor], optional
            Nodal fields to interpolate at quadrature points.
        element_data : dict, optional
            Element-wise data (constant or per-quadrature).
        scalar_data : dict, optional
            Global scalar parameters.
        batch_size : int, optional
            Quadrature-point batch size; ``-1`` (default) means no batching.

        Returns
        -------
        torch.Tensor
            Scalar total boundary energy.
        """
        if point_data is None: point_data = {}
        if element_data is None: element_data = {element_type:{} for element_type in self.element_types}
        if scalar_data is None: scalar_data = {}
        
        for key, value in point_data.items():
            assert value.shape[0] == self.n_points

        if points is not None: 
            self = self.type(points.dtype).to(points.device)
            for element_type in self.element_types:
                self.transformation[element_type].update_points(points)
        else:
            points = next(iter(self.transformation.values())).points
            
        point_data["x"] = points
        
        fn = self.element_energy if func is None else func
        signature = inspect.signature(fn)
        
        broadcast_fns = [
            (lambda x: x in element_data.keys(), InputBroadcast(True, False, False, False)),
            (lambda x: x in scalar_data.keys(), InputBroadcast(False, False, False, False)),
            (lambda x: x in point_data.keys(), InputBroadcast(True, True, False, False)),
        ]
        
        element_dims = []
        quadrature_dims = []
        
        for key in signature.parameters:
            is_match = False
            for condition, broadcast in broadcast_fns:
                if condition(key):
                    element_dims.append(broadcast.element)
                    quadrature_dims.append(broadcast.quadrature)
                    is_match = True
                    break
            if not is_match:
                 raise ValueError(f"{key} is not supported for contact energy calculation.")

        element_dims = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)
        
        parallel_fn = vmap(
            vmap(fn, in_dims=quadrature_dims),
            in_dims=element_dims
        )
        
        total_energy = 0.0
        
        for element_type in self.element_types:
            trans = self.transformation[element_type]
            
            if trans.element.is_mix_facet:
                 raise NotImplementedError("Mixed facet elements not fully supported in simple energy loop yet.")
            else:
                m = self.facet_mask[element_type].item()
                elem_indices, facet_indices = torch.where(m)
                
                if len(elem_indices) == 0:
                    continue

                shape_val_sel = trans.facet_shape_val[facet_indices]
                FxW = trans.FxW[m]
                
                args = []
                for key in signature.parameters:
                    if key in point_data:
                        u_global = point_data[key]
                        nodes = trans.elements[elem_indices]
                        u_nodes = u_global[nodes]
                        val_interp = torch.einsum("sbd,sqb->sqd", u_nodes, shape_val_sel)
                        args.append(val_interp)
                    elif key in scalar_data:
                        args.append(scalar_data[key])
                    elif key in element_data:
                        args.append(element_data[key][element_type][elem_indices])
                    else:
                        raise ValueError(f"Unknown arg {key}")
                
                energy_density = parallel_fn(*args)
                energy_val = (energy_density * FxW).sum()
                total_energy += energy_val

        return total_energy

    def element_energy(self, **kwargs):
        """Override this method to define the boundary energy density."""
        raise NotImplementedError

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

__all__ = ["LaplaceElementAssembler", "MassElementAssembler", "LinearElasticityElementAssembler", "NeoHookeanModel", "J2Plasticity", "DruckerPragerPlasticity", "ContactAssembler", "const_node_assembler", "func_node_assembler"]

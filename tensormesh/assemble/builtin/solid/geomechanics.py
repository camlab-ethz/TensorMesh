"""Drucker-Prager geomechanics built-in assembler."""

import torch

from tensormesh.assemble.element_assembler import ElementAssembler
from tensormesh.functional.plastic import drucker_prager_return_mapping, small_strain_3d


class DruckerPragerPlasticity(ElementAssembler):
    r"""Drucker-Prager Plasticity Model with Linear Isotropic Hardening.

    Pressure-dependent, small-strain Drucker-Prager plasticity for soils and
    weak rock, built on the pure constitutive primitive
    :func:`~tensormesh.functional.plastic.drucker_prager_return_mapping`. It mirrors the
    :class:`~tensormesh.assemble.J2Plasticity` lifecycle — per-quadrature history,
    previous-step state passed through ``element_data``, and :meth:`update_state`
    after a converged load step — so it slots into the same energy-minimisation
    solvers.

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

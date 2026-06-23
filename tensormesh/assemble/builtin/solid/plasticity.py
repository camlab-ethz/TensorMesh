"""J2 (von Mises) plasticity built-in assembler."""

import numpy as np
import torch

from tensormesh.assemble.element_assembler import ElementAssembler


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

import math
import torch
from typing import Callable, NamedTuple, Optional, Union
from .elasticity import strain, isotropic_stress, deviatoric_stress, deviatoric_stress_norm
from .ops import divide

def update_plastic_stress(gradu:torch.Tensor, 
                          strain:torch.Tensor, 
                          stress:torch.Tensor,
                          E:Union[float,torch.Tensor] = 70.0,
                          yield_stress:Union[float,torch.Tensor] = 250.0,
                          strain_fn:Callable[[torch.Tensor],torch.Tensor] = strain,
                          stress_fn:Callable[[torch.Tensor,Union[torch.Tensor,float]],torch.Tensor] = isotropic_stress,
                         )->torch.Tensor:

    r"""
    Update stress tensor using plastic constitutive model.

    The plastic model follows von Mises yield criterion with perfect plasticity:

    .. math::

        \sigma_{\text{trial}} = \sigma + \mathbb{C}:\Delta\varepsilon

        f(\sigma_{\text{trial}}) = \|\text{dev}(\sigma_{\text{trial}})\| - \sigma_y

        \Delta\gamma = \frac{\langle f(\sigma_{\text{trial}}) \rangle}{\|\text{dev}(\sigma_{\text{trial}})\|}

        \sigma = \sigma_{\text{trial}} - \Delta\gamma\, \text{dev}(\sigma_{\text{trial}})

    where:
    
    * :math:`\sigma` is the stress tensor in :math:`\mathbb{R}^{D \times D}`
    * :math:`\mathbb{C}` is the elasticity tensor in :math:`\mathbb{R}^{D \times D \times D \times D}`
    * :math:`\varepsilon` is the strain tensor in :math:`\mathbb{R}^{D \times D}`
    * :math:`\sigma_y` is the yield stress scalar in :math:`\mathbb{R}`
    * :math:`\text{dev}` denotes the deviatoric part operator :math:`\mathbb{R}^{D \times D} \rightarrow \mathbb{R}^{D \times D}`
    * :math:`\|\cdot\|` is the von Mises norm operator :math:`\mathbb{R}^{D \times D} \rightarrow \mathbb{R}`
    * :math:`\langle \cdot \rangle` denotes the positive part operator :math:`\mathbb{R} \rightarrow \mathbb{R}`

    The model uses a trial elastic predictor followed by plastic correction if yielding occurs.
    If the trial stress exceeds the yield surface, it is projected back onto the yield surface.
    
    Parameters
    ----------
    gradu : torch.Tensor
        1D Tensor of shape [d], where d is the spatial dimension.
        Gradient of displacement field with respect to spatial coordinates.
    strain : torch.Tensor 
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Current strain tensor at the start of the timestep.
    stress : torch.Tensor
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Current stress tensor at the start of the timestep.
    E : Union[float, torch.Tensor], default=70.0
        Young's modulus. If tensor, must be 0D scalar tensor.
        Controls the elastic stiffness of the material.
    yield_stress : Union[float, torch.Tensor], default=250.0
        Yield stress threshold. If tensor, must be 0D scalar tensor.
        Material yields plastically when von Mises stress exceeds this value.
    strain_fn : Callable[[torch.Tensor], torch.Tensor], default=strain
        Function to compute strain tensor from displacement gradient.
        Default uses small strain assumption:
        
        .. math::
        
            \varepsilon_{ij} = \frac{1}{2}(\nabla u_{ij} + \nabla u_{ji}), \quad \varepsilon,\nabla u \in \mathbb{R}^{d \times d}
            
    stress_fn : Callable[[torch.Tensor, Union[float,torch.Tensor]], torch.Tensor], default=isotropic_stress
        Function to compute stress tensor from strain tensor and Young's modulus.
        Default uses isotropic linear elasticity:
        
        .. math::
        
            \sigma_{ij} = \lambda \text{tr}(\varepsilon)\delta_{ij} + 2\mu\varepsilon_{ij}, \quad \sigma,\varepsilon \in \mathbb{R}^{d \times d}
            
        where :math:`\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}`, :math:`\mu = \frac{E}{2(1+\nu)}`, :math:`E,\nu \in \mathbb{R}`, and :math:`\delta_{ij}` is the Kronecker delta

    Returns
    -------
    torch.Tensor
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Updated stress tensor after plastic correction.
    """
    # assertion
    if isinstance(E, torch.Tensor):
        assert E.numel() == 1
    if isinstance(yield_stress, torch.Tensor):
        assert yield_stress.numel() == 1
    assert gradu.dim() == 1, f"gradu should be a 1D tensor of shape [dim], but got shape {gradu.shape}"
    assert strain.dim() == 2, f"strain should be a 2D tensor of shape [dim, dim], but got shape {strain.shape}"
    assert stress.dim() == 2, f"stress should be a 2D tensor of shape [dim, dim], but got shape {stress.shape}"
    assert strain.shape == stress.shape, f"strain and stress should have same shape, but got {strain.shape} and {stress.shape}"
    assert strain.shape[0] == strain.shape[1], f"strain should be square matrix, but got shape {strain.shape}"
    assert gradu.shape[0] == strain.shape[0], f"gradu dimension should match strain dimension, but got {gradu.shape[0]} and {strain.shape[0]}"
    # get stress trial
    delta_strain = strain_fn(gradu) - strain # [dim, dim]
    assert delta_strain.shape == strain.shape, f"delta_strain should have same shape as strain, but got {delta_strain.shape} and {strain.shape}"
    stress_trial = stress_fn(delta_strain, E) + stress # [dim, dim]
    assert stress_trial.shape == stress.shape, f"stress_trial should have same shape as stress, but got {stress_trial.shape} and {stress.shape}"
    # yield function
    stress_devia = deviatoric_stress(stress_trial) # [dim, dim]
    stress_devia_norm = deviatoric_stress_norm(stress_trial) # []
    f_yield = stress_devia_norm - yield_stress
    f_yield = torch.clamp_min(f_yield, 0.)

    # update stress
    stress = stress_trial - f_yield * divide(stress_devia , stress_devia_norm)

    return stress


# ---------------------------------------------------------------------------
# Drucker-Prager constitutive primitive.
#
# Pure-math return mapping for small-strain, associated (or non-associated)
# Drucker-Prager plasticity with linear isotropic hardening.  It is independent
# of Mesh / ElementAssembler / Condenser / SparseMatrix: it consumes tensors and
# scalars and returns tensors only, so it can back a custom assembler forward().
#
# Internal stress convention is tension-positive, matching the rest of
# TensorMesh.  The yield function is
#
#     f = q + eta * I1 - (k + H * alpha) <= 0,
#
# with I1 = tr(sigma), q = sqrt(3/2 s:s), and the Drucker-Prager cone fitted to
# the Mohr-Coulomb triaxial-compression meridian.  Because compression gives a
# negative I1, confinement raises the yield capacity.
# ---------------------------------------------------------------------------
class DruckerPragerCoefficients(NamedTuple):
    """Drucker-Prager cone coefficients derived from material parameters."""

    mu: torch.Tensor          # shear modulus
    bulk: torch.Tensor        # bulk modulus
    eta: torch.Tensor         # friction slope of the yield surface
    eta_dilatancy: torch.Tensor  # dilatancy slope of the plastic flow direction
    k: torch.Tensor           # cohesion intercept
    denom: torch.Tensor       # return-mapping denominator
    H: torch.Tensor           # hardening modulus


class DruckerPragerReturn(NamedTuple):
    """Structured result of a Drucker-Prager return-mapping step.

    All fields are tensors.  ``eps_p`` and ``alpha`` are the committed
    (end-of-step) values; ``energy`` is the algorithmic incremental potential
    density used by an energy-based assembler.
    """

    energy: torch.Tensor
    stress: torch.Tensor
    eps_p: torch.Tensor
    alpha: torch.Tensor
    d_gamma: torch.Tensor
    f_trial: torch.Tensor
    yielded: torch.Tensor


def small_strain_3d(graddisplacement: torch.Tensor) -> torch.Tensor:
    r"""Return a 3D small-strain tensor from a 2D or 3D displacement gradient.

    For 2D input the in-plane symmetric strain is embedded into the upper-left
    ``2x2`` block of a ``3x3`` tensor using :func:`torch.nn.functional.pad`,
    which is safe both for directly batched inputs and under :func:`torch.vmap`
    (an in-place embed into a freshly allocated tensor is not vmap-safe).

    Parameters
    ----------
    graddisplacement : torch.Tensor
        Displacement gradient :math:`\nabla \mathbf{u}` of shape ``[..., d, d]``
        with ``d`` equal to 2 or 3.

    Returns
    -------
    torch.Tensor
        Small-strain tensor of shape ``[..., 3, 3]``.
    """
    dim = graddisplacement.shape[-1]
    sym = 0.5 * (graddisplacement + graddisplacement.transpose(-1, -2))
    if dim == 2:
        return torch.nn.functional.pad(sym, (0, 1, 0, 1))
    return sym


def drucker_prager_coefficients(
    E: Union[float, torch.Tensor],
    nu: Union[float, torch.Tensor],
    cohesion: Union[float, torch.Tensor],
    friction_angle: Union[float, torch.Tensor],
    dilatancy_angle: Optional[Union[float, torch.Tensor]] = None,
    H: Union[float, torch.Tensor] = 0.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> DruckerPragerCoefficients:
    r"""Compute Drucker-Prager cone coefficients from material parameters.

    The cone is fitted to the Mohr-Coulomb triaxial-compression meridian.  With
    ``friction_angle`` :math:`\phi` and ``dilatancy_angle`` :math:`\psi` in
    degrees,

    .. math::

        M = \frac{6 \sin\phi}{3 - \sin\phi}, \quad
        \eta = \frac{M}{3}, \quad
        k = \frac{6\, c \cos\phi}{3 - \sin\phi},

    and the plastic-flow slope :math:`\eta_\psi` uses :math:`\psi` in place of
    :math:`\phi`.  ``dilatancy_angle=None`` selects associated flow
    (:math:`\psi = \phi`).

    Scalar parameters are normalised with :func:`torch.as_tensor` so the result
    follows the requested ``dtype``/``device``.
    """
    E = torch.as_tensor(E, dtype=dtype, device=device)
    nu = torch.as_tensor(nu, dtype=dtype, device=device)
    cohesion = torch.as_tensor(cohesion, dtype=dtype, device=device)
    friction_angle = torch.as_tensor(friction_angle, dtype=dtype, device=device)
    if dilatancy_angle is None:
        dilatancy_angle = friction_angle
    dilatancy_angle = torch.as_tensor(dilatancy_angle, dtype=dtype, device=device)
    H = torch.as_tensor(H, dtype=dtype, device=device)

    mu = E / (2.0 * (1.0 + nu))
    bulk = E / (3.0 * (1.0 - 2.0 * nu))

    deg2rad = math.pi / 180.0
    sin_phi = torch.sin(friction_angle * deg2rad)
    cos_phi = torch.cos(friction_angle * deg2rad)
    sin_psi = torch.sin(dilatancy_angle * deg2rad)

    eta = (6.0 * sin_phi / (3.0 - sin_phi)) / 3.0
    eta_dilatancy = (6.0 * sin_psi / (3.0 - sin_psi)) / 3.0
    k = 6.0 * cohesion * cos_phi / (3.0 - sin_phi)

    denom = 3.0 * mu + 9.0 * bulk * eta * eta_dilatancy + H
    return DruckerPragerCoefficients(mu, bulk, eta, eta_dilatancy, k, denom, H)


def drucker_prager_return_mapping(
    graddisplacement: torch.Tensor,
    eps_p_n: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    E: Union[float, torch.Tensor],
    nu: Union[float, torch.Tensor],
    cohesion: Union[float, torch.Tensor],
    friction_angle: Union[float, torch.Tensor],
    dilatancy_angle: Optional[Union[float, torch.Tensor]] = None,
    H: Union[float, torch.Tensor] = 0.0,
) -> DruckerPragerReturn:
    r"""Drucker-Prager return mapping for one strain state.

    Implements the trial-elastic / plastic-correction return mapping for
    small-strain, linear-isotropic-hardening Drucker-Prager plasticity in the
    tension-positive convention.  The implementation is written with batched and
    :func:`torch.vmap` use in mind: every tensor reduction acts on the trailing
    ``(-2, -1)`` tensor axes, so the same function works for a single quadrature
    point (``graddisplacement`` shaped ``[d, d]``) and for a batched field
    (``[..., d, d]``).

    Parameters
    ----------
    graddisplacement : torch.Tensor
        Displacement gradient of shape ``[..., d, d]`` (``d`` = 2 or 3).
    eps_p_n, alpha_n : torch.Tensor
        Previous-step plastic strain ``[..., 3, 3]`` and equivalent plastic
        strain ``[...]``.
    E, nu, cohesion, friction_angle : float or torch.Tensor
        Material parameters; angles are in degrees.
    dilatancy_angle : float, torch.Tensor or None, optional
        Dilatancy angle in degrees.  ``None`` (default) means associated flow,
        i.e. the friction angle is used.
    H : float or torch.Tensor, optional
        Linear isotropic hardening modulus.  Default ``0.0``.

    Returns
    -------
    DruckerPragerReturn
        Structured result with the algorithmic incremental potential
        (``energy``), the committed Cauchy ``stress``, the committed ``eps_p``
        and ``alpha``, the plastic multiplier ``d_gamma``, the trial yield value
        ``f_trial`` and a boolean ``yielded`` mask.
    """
    dtype = graddisplacement.dtype
    device = graddisplacement.device
    mu, bulk, eta, eta_dilatancy, k, denom, H = drucker_prager_coefficients(
        E, nu, cohesion, friction_angle, dilatancy_angle, H, dtype=dtype, device=device
    )

    eye = torch.eye(3, dtype=dtype, device=device)

    eps = small_strain_3d(graddisplacement)
    eps_e_trial = eps - eps_p_n
    tr_eps_e = eps_e_trial.diagonal(dim1=-2, dim2=-1).sum(-1)
    dev_eps_e = eps_e_trial - (tr_eps_e[..., None, None] / 3.0) * eye
    sigma_trial = 2.0 * mu * dev_eps_e + bulk * tr_eps_e[..., None, None] * eye

    I1 = sigma_trial.diagonal(dim1=-2, dim2=-1).sum(-1)
    s = sigma_trial - (I1[..., None, None] / 3.0) * eye
    q = torch.sqrt(torch.clamp(1.5 * (s * s).sum(dim=(-2, -1)), min=1.0e-30))

    f_trial = q + eta * I1 - (k + H * alpha_n)
    d_gamma = torch.clamp(f_trial, min=0.0) / denom

    q_safe = torch.clamp(q, min=1.0e-30)
    n_dev = 1.5 * s / q_safe[..., None, None]
    flow_dir = n_dev + eta_dilatancy * eye

    eps_p = eps_p_n + d_gamma[..., None, None] * flow_dir
    alpha = alpha_n + d_gamma

    elastic_energy = 0.5 * bulk * tr_eps_e**2 + mu * (dev_eps_e * dev_eps_e).sum(dim=(-2, -1))
    energy = elastic_energy - 0.5 * denom * d_gamma**2

    eps_e = eps - eps_p
    tr_eps = eps_e.diagonal(dim1=-2, dim2=-1).sum(-1)
    dev_eps = eps_e - (tr_eps[..., None, None] / 3.0) * eye
    stress = 2.0 * mu * dev_eps + bulk * tr_eps[..., None, None] * eye

    yielded = f_trial > 0.0
    return DruckerPragerReturn(energy, stress, eps_p, alpha, d_gamma, f_trial, yielded)


def drucker_prager_yield_value(
    graddisplacement: torch.Tensor,
    eps_p_n: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    E: Union[float, torch.Tensor],
    nu: Union[float, torch.Tensor],
    cohesion: Union[float, torch.Tensor],
    friction_angle: Union[float, torch.Tensor],
    H: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Return the Drucker-Prager trial yield value ``f`` for a strain state.

    Positive values indicate the trial stress is outside the yield surface.
    This is a thin wrapper over :func:`drucker_prager_return_mapping` and does
    not depend on the dilatancy angle.
    """
    return drucker_prager_return_mapping(
        graddisplacement, eps_p_n, alpha_n,
        E=E, nu=nu, cohesion=cohesion, friction_angle=friction_angle, H=H,
    ).f_trial


__all__ = [
    "update_plastic_stress",
    "small_strain_3d",
    "drucker_prager_coefficients",
    "drucker_prager_return_mapping",
    "drucker_prager_yield_value",
    "DruckerPragerCoefficients",
    "DruckerPragerReturn",
]
    

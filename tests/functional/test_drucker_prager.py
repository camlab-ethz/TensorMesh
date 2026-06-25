"""Tests for the pure Drucker-Prager constitutive primitive."""

import torch

from tensormesh.functional import (
    DruckerPragerReturn,
    drucker_prager_coefficients,
    drucker_prager_return_mapping,
    drucker_prager_yield_value,
    small_strain_3d,
)

MATERIAL = dict(E=50.0e6, nu=0.30, cohesion=20.0e3, friction_angle=30.0, H=1.0e6)


def _zero_history():
    return (
        torch.zeros((3, 3), dtype=torch.float64),
        torch.zeros((), dtype=torch.float64),
    )


def test_elastic_step_below_yield_leaves_state_unchanged():
    eps_p_n, alpha_n = _zero_history()
    grad = torch.eye(3, dtype=torch.float64) * 1.0e-6  # tiny strain, below yield
    result = drucker_prager_return_mapping(grad, eps_p_n, alpha_n, **MATERIAL)

    assert isinstance(result, DruckerPragerReturn)
    assert float(result.d_gamma) == 0.0
    assert bool(result.yielded) is False
    assert torch.allclose(result.alpha, alpha_n)
    assert torch.allclose(result.eps_p, eps_p_n)


def test_plastic_step_above_yield_grows_alpha():
    eps_p_n, alpha_n = _zero_history()
    grad = torch.zeros((3, 3), dtype=torch.float64)
    grad[0, 1] = 0.02  # simple shear, deviatoric -> yields
    result = drucker_prager_return_mapping(grad, eps_p_n, alpha_n, **MATERIAL)

    assert float(result.d_gamma) > 0.0
    assert bool(result.yielded) is True
    assert float(result.alpha) > float(alpha_n)


def test_higher_confinement_delays_yield():
    eps_p_n, alpha_n = _zero_history()
    grad = torch.zeros((3, 3), dtype=torch.float64)
    grad[0, 1] = 0.02
    params = dict(E=50.0e6, nu=0.30, cohesion=20.0e3, friction_angle=30.0)

    f_unconfined = drucker_prager_yield_value(grad, eps_p_n, alpha_n, **params)
    grad_confined = grad + torch.eye(3, dtype=torch.float64) * (-2.0e-3)  # add compression
    f_confined = drucker_prager_yield_value(grad_confined, eps_p_n, alpha_n, **params)

    # Compression (negative I1) lowers the yield value -> confinement delays yield.
    assert float(f_confined) < float(f_unconfined)


def test_small_strain_3d_batched_2d():
    grad = torch.randn((2, 3, 2, 2), dtype=torch.float64)
    eps = small_strain_3d(grad)
    assert eps.shape == (2, 3, 3, 3)
    assert torch.allclose(eps[..., :2, :2], 0.5 * (grad + grad.transpose(-1, -2)))
    assert torch.count_nonzero(eps[..., 2, :]) == 0
    assert torch.count_nonzero(eps[..., :, 2]) == 0


def test_small_strain_3d_under_vmap():
    grad = torch.randn((5, 2, 2), dtype=torch.float64)
    out = torch.vmap(small_strain_3d)(grad)
    assert out.shape == (5, 3, 3)


def test_return_mapping_under_vmap_per_quadrature():
    grad = torch.randn((6, 2, 2), dtype=torch.float64) * 1.0e-3
    eps_p_n = torch.zeros((6, 3, 3), dtype=torch.float64)
    alpha_n = torch.zeros((6,), dtype=torch.float64)

    def energy(g, e, a):
        return drucker_prager_return_mapping(g, e, a, **MATERIAL).energy

    energies = torch.vmap(energy)(grad, eps_p_n, alpha_n)
    assert energies.shape == (6,)
    assert torch.isfinite(energies).all()


def test_dilatancy_angle_passthrough_preserves_dtype_device():
    eps_p_n, alpha_n = _zero_history()
    grad = torch.zeros((3, 3), dtype=torch.float64)
    grad[0, 1] = 0.02

    associated = drucker_prager_return_mapping(grad, eps_p_n, alpha_n, dilatancy_angle=None, **MATERIAL)
    non_associated = drucker_prager_return_mapping(grad, eps_p_n, alpha_n, dilatancy_angle=10.0, **MATERIAL)

    for result in (associated, non_associated):
        assert result.eps_p.dtype == torch.float64
        assert result.eps_p.device == grad.device

    # The dilatancy angle controls the volumetric part of the plastic flow, so a
    # non-associated angle gives a different committed plastic strain.
    vol_assoc = float(associated.eps_p.diagonal().sum())
    vol_nonassoc = float(non_associated.eps_p.diagonal().sum())
    assert vol_assoc != vol_nonassoc


def test_coefficients_associated_default():
    coef = drucker_prager_coefficients(50.0e6, 0.30, 20.0e3, 30.0, dtype=torch.float64)
    # Associated flow: dilatancy slope equals friction slope.
    assert torch.allclose(coef.eta, coef.eta_dilatancy)
    assert float(coef.mu) > 0.0
    assert float(coef.bulk) > 0.0

"""Smoke tests for the Drucker-Prager triaxial driver (public API)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[2]
    example_path = (
        repo_root
        / "examples"
        / "solid"
        / "geomechanics"
        / "drucker_prager_triaxial"
        / "drucker_prager_triaxial.py"
    )
    spec = importlib.util.spec_from_file_location("drucker_prager_triaxial", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_drucker_prager_triaxial_sanity_checks():
    module = _load_example_module()
    output = module.run_demo(n_steps=12, make_plot=False)
    sanity = output["sanity"]

    assert sanity["higher_confinement_delays_yield"]
    assert sanity["plastic_strain_monotonic"]


def test_small_strain_3d_is_batch_and_vmap_safe():
    """The public 2D plane-strain embedding must work batched and under vmap."""
    from tensormesh.functional import small_strain_3d as embed

    # Batched 2D gradients embed into 3x3 strain tensors without shape errors.
    grad_u = torch.randn((2, 3, 2, 2), dtype=torch.float64)
    eps = embed(grad_u)
    assert eps.shape == (2, 3, 3, 3)
    eps_2d = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    assert torch.allclose(eps[..., :2, :2], eps_2d)
    assert torch.count_nonzero(eps[..., 2, :]) == 0
    assert torch.count_nonzero(eps[..., :, 2]) == 0

    # The same embedding is applied per quadrature point under vmap by energy(),
    # so it must also be vmap-safe (an in-place embed would raise here).
    single = torch.randn((4, 2, 2), dtype=torch.float64)
    vmapped = torch.vmap(embed)(single)
    assert vmapped.shape == (4, 3, 3)

"""Tests for the phase-field fracture example (single-edge-notched tension)."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


torch_sla = pytest.importorskip("torch_sla")
if not hasattr(torch_sla, "vi_solve"):
    pytest.skip("the example needs torch_sla.vi_solve (torch-sla >= 0.4.0)", allow_module_level=True)


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "solid" / "phase_field_fracture" / "phase_field_fracture.py"
    spec = importlib.util.spec_from_file_location("phase_field_fracture_example", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example_module()
MODEL = dict(E=210.0, nu=0.3, Gc=2.7e-3, ell=0.04)


def _strip(model, n=(4, 20), width=0.2, height=1.0, relaxation=1.0):
    """Homogeneous tension: rollers on the bottom (u_y = 0) and the left edge
    (u_x = 0), top pulled by u_y; the right edge is free, so the stress is
    uniaxial and uniform."""
    x = np.linspace(0.0, width, n[0] + 1)
    y = np.linspace(0.0, height, n[1] + 1)
    mesh = example.structured_triangle_mesh(x, y)
    pts = mesh.points
    fixed = torch.zeros(mesh.n_points, 2, dtype=torch.bool)
    fixed[pts[:, 1] < 1e-12, 1] = True
    fixed[pts[:, 0] < 1e-12, 0] = True
    top = torch.zeros_like(fixed)
    top[pts[:, 1] > height - 1e-12, 1] = True
    fixed = (fixed | top).reshape(-1)
    return mesh, example.StaggeredPhaseField(mesh, model, fixed, relaxation=relaxation), top.reshape(-1)


@pytest.mark.parametrize("model_name", ["at1", "at2"])
def test_hessians_are_derivatives_of_the_residuals(model_name):
    """The assembled Hessians against central differences of the autograd
    residuals (exact here: the energy is quadratic in alpha, and piecewise
    quadratic in u with no kink crossed by the perturbation)."""
    model = example.PhaseFieldModel(**MODEL, model=model_name)
    mesh, solver, _ = _strip(model, n=(3, 6))
    g = torch.Generator().manual_seed(0)
    rand = lambda n: torch.rand(n, generator=g, dtype=torch.float64)  # noqa: E731
    u, du = 1e-3 * rand(mesh.n_points * 2), rand(mesh.n_points * 2) - 0.5
    alpha, dalpha = 0.8 * rand(mesh.n_points), rand(mesh.n_points) - 0.5
    eps = 1e-6
    fd_u = (solver.residual_u(u + eps * du, alpha) - solver.residual_u(u - eps * du, alpha)) / (2 * eps)
    Kdu = solver.tangent_u(u, alpha) @ du
    assert torch.allclose(Kdu, fd_u, rtol=1e-6, atol=1e-8 * float(Kdu.abs().max()))
    fd_a = (solver.residual_alpha(alpha + eps * dalpha, u) - solver.residual_alpha(alpha - eps * dalpha, u)) / (2 * eps)
    Jda = solver.hessian_alpha(u) @ dalpha
    assert torch.allclose(Jda, fd_a, rtol=1e-6, atol=1e-8 * float(Jda.abs().max()))


@pytest.mark.parametrize("relaxation", [1.0, 1.6])
@pytest.mark.parametrize("ratio", [0.97, 1.03, 1.2])
def test_at1_homogeneous_damage_matches_analytical_solution(ratio, relaxation):
    """AT1 has an elastic phase: no damage below the critical strain
    eps_c = sqrt(3 Gc (1 - nu^2) / (8 E ell)) (plane strain, uniaxial stress),
    and the homogeneous damage alpha = 1 - (eps_c / eps)^2 above it."""
    model = example.PhaseFieldModel(**MODEL, model="at1")
    mesh, solver, top = _strip(model, relaxation=relaxation)
    eps_c = math.sqrt(3 * MODEL["Gc"] * (1 - MODEL["nu"] ** 2) / (8 * MODEL["E"] * MODEL["ell"]))
    eps = ratio * eps_c
    solver.solve_step(torch.where(top, torch.full_like(solver.u, eps * 1.0), torch.zeros_like(solver.u)))
    expected = max(0.0, 1 - (eps_c / eps) ** 2)
    assert float((solver.alpha - expected).abs().max()) < 1e-6
    if expected == 0.0:
        assert torch.equal(solver.alpha, torch.zeros_like(solver.alpha))  # held on the bound alpha >= 0
    # uniaxial stress in plane strain: F = g(alpha) E / (1 - nu^2) eps * width
    F = solver.reaction(top)
    g = (1 - expected) ** 2 + model.g0
    assert F == pytest.approx(g * MODEL["E"] / (1 - MODEL["nu"] ** 2) * eps * 0.2, rel=1e-6)


def test_damage_never_decreases_and_respects_bounds():
    model = example.PhaseFieldModel(**MODEL, model="at2")
    mesh, solver, top = _strip(model, n=(3, 12))
    previous = solver.alpha.clone()
    for eps in (0.002, 0.004, 0.003, 0.006):  # with an unloading step
        solver.solve_step(torch.where(top, torch.full_like(solver.u, eps), torch.zeros_like(solver.u)))
        assert torch.all(solver.alpha >= previous - 1e-14)
        assert torch.all(solver.alpha <= 1.0)
        previous = solver.alpha.clone()


def test_prescribed_damage_is_held():
    """alpha_fixed pins damage through lower == upper: a row of broken nodes
    across the strip keeps alpha = 1 exactly and weakens the strip."""
    model = example.PhaseFieldModel(**MODEL, model="at1")
    mesh, intact, top = _strip(model)
    row = (mesh.points[:, 1] - 0.5).abs() < 1e-12
    cracked = example.StaggeredPhaseField(mesh, model, intact.u_fixed, alpha_fixed=row,
                                          alpha_value=row.to(torch.float64))
    u_bc = torch.where(top, torch.full_like(intact.u, 1e-3), torch.zeros_like(intact.u))
    intact.solve_step(u_bc)
    cracked.solve_step(u_bc)
    assert torch.equal(cracked.alpha[row], torch.ones(int(row.sum()), dtype=torch.float64))
    assert torch.all(cracked.alpha[~row] < 1.0)
    assert cracked.reaction(top) < 0.9 * intact.reaction(top)


def test_sent_crack_runs_straight_through_the_ligament():
    """Coarse single-edge-notched tension: the crack leaves the slit tip along
    y = 0.5 and runs through the ligament, the specimen loses its stiffness, and
    the dissipated energy is Gc times the crack length (0.5 mm), times the
    effective-toughness factor of a crack band one element wide (h = ell / 2)."""
    problem = replace(example.SENTProblem(), ell=0.04, h_fine=0.02, band=0.16, h_coarse=0.08,
                      increment=4e-4, max_displacement=1e-2)
    result = example.run_demo(problem, make_plot=False, verbose=False)
    F = result["reaction"]
    assert F.argmax() > 5 and F.max() > 0.3
    assert F[-1] < problem.stop_ratio * F.max()  # broken
    cracked = result["points"][result["alpha"] > 0.95]
    assert np.abs(cracked[:, 1] - 0.5).max() <= problem.h_fine + 1e-12
    assert cracked[:, 0].min() > 0.5 - problem.h_fine  # starts at the slit tip
    assert cracked[:, 0].max() > 1.0 - 1e-12  # through to the right edge
    history = np.stack([result["snapshots"][k] for k in sorted(result["snapshots"])])
    assert np.all(np.diff(history, axis=0) >= 0.0)  # damage never heals
    assert 0.5 < result["fracture_energy"][-1] / problem.Gc < 0.5 * 1.5

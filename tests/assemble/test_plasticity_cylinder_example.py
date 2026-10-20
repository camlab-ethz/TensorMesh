"""Tests for the J2 plasticity example (thick-walled cylinder under internal pressure)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "solid" / "plasticity_cylinder" / "plasticity_cylinder.py"
    spec = importlib.util.spec_from_file_location("plasticity_cylinder_example", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example_module()
COARSE = dict(n_radial=8, n_angular=8)


def _random_states(n, scale, generator):
    """Plane-strain strains, deviatoric plastic strains and equivalent plastic strains."""
    def sym(m):
        return 0.5 * (m + m.transpose(-1, -2))
    eps = torch.zeros(n, 3, 3, dtype=torch.float64)
    eps[:, :2, :2] = sym(scale * (2 * torch.rand(n, 2, 2, generator=generator, dtype=torch.float64) - 1))
    eps_p = sym(0.3 * scale * (2 * torch.rand(n, 3, 3, generator=generator, dtype=torch.float64) - 1))
    eps_p = eps_p - eps_p.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] / 3 * torch.eye(3, dtype=torch.float64)
    p = 0.5 * scale * torch.rand(n, generator=generator, dtype=torch.float64)
    return eps, eps_p, p


@pytest.mark.parametrize("H", [0.0, 2e3])
def test_return_map_and_its_consistent_tangent(H):
    """The stress lands on the yield surface at plastic points, elastic points
    keep the elastic stress, and C is the derivative of the stress update (a
    forward-mode autodiff directional derivative, for symmetric directions)."""
    mat = example.J2Material(H=H)
    g = torch.Generator().manual_seed(0)
    eps, eps_p, p = _random_states(400, 3e-3, g)
    sigma, eps_p_new, p_new, C = example.radial_return(eps, eps_p, p, mat)
    plastic = p_new > p
    assert 20 < int(plastic.sum()) < 380  # both kinds of points are tested
    q = example.von_mises(sigma)
    assert torch.allclose(q[plastic], (mat.sigma_y + H * p_new)[plastic], rtol=1e-12)
    assert torch.all(q[~plastic] <= mat.sigma_y + H * p[~plastic])
    assert torch.equal(eps_p_new[~plastic], eps_p[~plastic])
    # plastic flow is deviatoric
    assert float(eps_p_new.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-15

    d = torch.rand(400, 3, 3, generator=g, dtype=torch.float64) - 0.5
    d = 0.5 * (d + d.transpose(-1, -2))
    _, jvp = torch.func.jvp(lambda e: example.radial_return(e, eps_p, p, mat)[0], (eps,), (d,))
    assert torch.allclose(torch.einsum("nijkl,nkl->nij", C, d), jvp, rtol=1e-10, atol=1e-10 * float(jvp.abs().max()))


def test_assembled_tangent_matches_finite_differences():
    """K du against central differences of the residual, at a state beyond the
    last converged one, where the plastic points load further (away from the
    kink of the return map)."""
    problem = example.CylinderProblem(material=example.J2Material(H=4e3), control="pressure", steps=3,
                                      max_load=0.75, n_radial=3, n_angular=4)
    result = example.run_demo(problem, make_plot=False, verbose=False)
    solver = result["solver"]
    assert 0 < result["plastic_fraction"][-1] < 1
    u = 1.05 * solver.u
    g = torch.Generator().manual_seed(1)
    du = torch.where(solver.fixed, 0.0, torch.rand(u.shape, generator=g, dtype=torch.float64) - 0.5)
    du = du * 1e-3 * float(u.abs().max())
    f_ext = torch.zeros_like(u)
    _, (_, _, _, C) = solver.residual(u, f_ext)
    K = solver.tangent(quadrature_data={"C": C[..., :2, :2, :2, :2]})
    h = 1e-4
    fd = (solver.residual(u + h * du, f_ext)[0] - solver.residual(u - h * du, f_ext)[0]) / (2 * h)
    Kdu = K @ du
    assert torch.allclose(Kdu, fd, rtol=1e-6, atol=1e-7 * float(Kdu.abs().max()))


def test_quarter_ring_mesh():
    """Area of the quarter ring, nodes of the inner and outer circles, and the
    symmetry edges on the axes."""
    a, b = 100.0, 200.0
    mesh, masks = example.quarter_ring_mesh(a, b, 4, 6)
    r = torch.linalg.vector_norm(mesh.points, dim=1)
    assert torch.allclose(r[masks["inner"]], torch.full_like(r[masks["inner"]], a), rtol=1e-14)
    assert torch.allclose(r[masks["outer"]], torch.full_like(r[masks["outer"]], b), rtol=1e-14)
    assert torch.all(mesh.points[masks["x_axis"], 1] == 0) and torch.all(mesh.points[masks["y_axis"], 0] == 0)
    solver = example.J2NewtonSolver(mesh, example.J2Material(), torch.zeros(mesh.n_points * 2, dtype=torch.bool))
    assert float(solver.JxW.sum()) == pytest.approx(math.pi / 4 * (b**2 - a**2), rel=1e-5)
    assert torch.all(solver.JxW > 0)


def test_elastic_response_matches_lame():
    """Below first yield the inner-wall displacement is the Lame solution
    (plane strain) and nothing yields, under pressure and displacement control."""
    ref = example.analytical(example.CylinderProblem())
    for control, load in (("pressure", 0.5 * ref["p_e"] / ref["p_lim"]), ("displacement", 0.5)):
        problem = example.CylinderProblem(control=control, steps=1, max_load=load, **COARSE)
        result = example.run_demo(problem, make_plot=False, verbose=False)
        ref = result["analytical"]
        # 8 x 8 cells: the discretization error is about 3e-4
        assert result["pressure"][1] / result["displacement"][1] == pytest.approx(ref["stiffness"], rel=1e-3)
        assert result["plastic_fraction"][-1] == 0
        assert len(result["newton"][0]) == 1  # linear: one Newton update


def test_first_yield_at_the_inner_wall():
    """No plastic point just below the first-yield displacement; above it, the
    plastic points are in the innermost layer of elements."""
    below = example.run_demo(example.CylinderProblem(steps=1, max_load=0.99, **COARSE), make_plot=False,
                             verbose=False)
    assert below["plastic_fraction"][-1] == 0
    above = example.run_demo(example.CylinderProblem(steps=1, max_load=1.1, **COARSE), make_plot=False,
                             verbose=False)
    solver = above["solver"]
    plastic = solver.p > 0
    assert bool(plastic.any())
    layer = above["mesh"].points[solver.elements].norm(dim=-1).min(dim=1).values  # inner radius of each element
    assert float(layer[plastic.any(dim=1)].max()) == pytest.approx(100.0)


def test_limit_pressure_of_perfect_plasticity():
    """Pushing the inner wall out, the pressure saturates at the limit pressure
    (2 / sqrt 3) sigma_y ln(b / a) once the whole wall flows, and Newton with
    the consistent tangent converges in a few iterations in every step."""
    problem = example.CylinderProblem(control="displacement", steps=15, max_load=6.0, **COARSE)
    result = example.run_demo(problem, make_plot=False, verbose=False)
    p_lim = result["analytical"]["p_lim"]
    P = result["pressure"]
    assert result["plastic_fraction"][-1] == pytest.approx(1.0)
    assert P[-1] == pytest.approx(p_lim, rel=1e-2)
    assert np.all(P <= p_lim * (1 + 1e-2))
    assert np.all(np.diff(P) > -1e-6 * p_lim)  # perfect plasticity: approaches the limit from below
    assert P[-1] - P[-3] < 2e-3 * p_lim  # a plateau
    assert max(len(h) for h in result["newton"]) <= 6
    assert max(h[-1] for h in result["newton"]) <= result["solver"].tol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    problem = example.CylinderProblem(steps=6, max_load=3.0, **COARSE)
    cpu = example.run_demo(problem, make_plot=False, verbose=False)
    gpu = example.run_demo(problem, device="cuda", make_plot=False, verbose=False)
    assert np.allclose(gpu["pressure"], cpu["pressure"], rtol=1e-8)
    assert np.allclose(gpu["p"], cpu["p"], rtol=1e-6, atol=1e-12)

"""Analytical and end-to-end tests for the two Helmholtz DtN demos."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import h1vp, hankel1, jv, jvp
import torch

from tensormesh import BlochReducer
from tensormesh.sparse import SparseMatrix


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "wave" / "helmholtz_dtn"
sys.path.insert(0, str(EXAMPLE_DIR))

import circular_dtn as circle  # noqa: E402
import periodic_dtn as periodic  # noqa: E402


@pytest.mark.parametrize("mode", [-4, -1, 0, 3])
def test_hankel_symbol_matches_direct_derivative(mode: int) -> None:
    """The stable recurrence must agree with the analytical derivative."""

    wavenumber, radius = 3.7, 1.1
    order = abs(mode)
    direct = wavenumber * h1vp(order, wavenumber * radius, 1)
    direct /= hankel1(order, wavenumber * radius)
    assert circle.hankel_symbol(mode, wavenumber, radius) == pytest.approx(
        direct, rel=1.0e-12, abs=1.0e-12
    )


def test_outgoing_rayleigh_branch() -> None:
    """Propagating and evanescent modes use the outgoing square-root branch."""

    beta = periodic.outgoing_beta(
        3.0, torch.tensor([0.2, 4.0, -5.0], dtype=torch.float64)
    )
    assert beta[0].real > 0.0
    assert abs(float(beta[0].imag)) < 1.0e-14
    assert torch.all(beta[1:].imag > 0.0)


def test_rayleigh_trace_and_low_rank_block() -> None:
    """TensorMesh facet moments reproduce one Fourier trace and the DtN block."""

    mesh = periodic.build_mesh(1.0, (0.0, 0.55, 1.25), 0.10)
    boundary = abs(mesh.points[:, 1] - 1.25) < 0.005
    matrix, nodes, psi, alpha_n, beta_n = periodic.rayleigh_dtn(
        mesh, boundary, 3.0, 0.35, 1.0, 3
    )
    zero = 3
    exact_trace = (1.0 - torch.exp(-1j * alpha_n[zero])) / (1j * alpha_n[zero])
    assert abs(psi[:, zero].sum() - exact_trace) < 5.0e-6

    symbols = 1j * beta_n
    low_rank = psi.conj() @ torch.diag(symbols) @ psi.T
    explicit = sum(
        symbols[j] * psi[:, j].conj()[:, None] * psi[:, j][None, :]
        for j in range(psi.shape[1])
    )
    dense = matrix.to_torch_sparse().to_dense()
    assert torch.allclose(low_rank, explicit, atol=1.0e-11, rtol=1.0e-11)
    assert torch.allclose(dense[nodes[:, None], nodes], low_rank, atol=1.0e-12)
    assert not torch.allclose(low_rank, low_rank.conj().T)


def test_circular_trace_and_low_rank_block() -> None:
    """The circular DtN block equals its explicit modal sum."""

    mesh = circle.build_mesh("dirichlet", 0.4, 1.2, 0.16)
    boundary = abs(torch.linalg.norm(mesh.points, dim=1) - 1.2) < 0.008
    matrix, nodes, psi, _, tau = circle.circular_dtn(mesh, boundary, 4.0, 1.2, 4)
    symbols = tau / (2.0 * np.pi * 1.2)
    low_rank = psi.conj() @ torch.diag(symbols) @ psi.T
    explicit = sum(
        symbols[j] * psi[:, j].conj()[:, None] * psi[:, j][None, :]
        for j in range(psi.shape[1])
    )
    dense = matrix.to_torch_sparse().to_dense()
    assert torch.allclose(low_rank, explicit, atol=1.0e-11, rtol=1.0e-11)
    assert torch.allclose(dense[nodes[:, None], nodes], low_rank, atol=1.0e-12)


@pytest.mark.parametrize("order", [-2, 0, 3])
def test_circular_exact_conditions(order: int) -> None:
    """The three modal series satisfy their obstacle or interface conditions."""

    radius, k0, k1, a0, a1 = 0.4, 4.0, 4.0 * np.sqrt(3.0), 1.0, 0.75
    z0, z1 = k0 * radius, k1 * radius
    outgoing, _ = circle.exact_coefficients(
        "dirichlet", order, k0, k1, a0, a1, radius
    )
    assert abs(jv(order, z0) + outgoing * hankel1(order, z0)) < 1.0e-12

    outgoing, _ = circle.exact_coefficients(
        "neumann", order, k0, k1, a0, a1, radius
    )
    derivative = k0 * (jvp(order, z0, 1) + outgoing * h1vp(order, z0, 1))
    assert abs(derivative) < 1.0e-12

    outgoing, inside = circle.exact_coefficients(
        "transmission", order, k0, k1, a0, a1, radius
    )
    outside_value = jv(order, z0) + outgoing * hankel1(order, z0)
    outside_flux = a0 * k0 * (jvp(order, z0, 1) + outgoing * h1vp(order, z0, 1))
    assert abs(outside_value - inside * jv(order, z1)) < 1.0e-12
    assert abs(outside_flux - a1 * inside * k1 * jvp(order, z1, 1)) < 1.0e-12


def test_bloch_matrix_and_rhs_match_explicit_transform() -> None:
    """Bloch reduction must form both ``T^H A T`` and ``T^H b``."""

    points = np.asarray([[x, y] for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)])
    reducer = BlochReducer(points, [[1.0, 0.0]], sign=+1)
    wavevector = torch.tensor([0.4, 0.0], dtype=torch.float64)
    phase = torch.exp(1j * (reducer.node_R @ wavevector).to(periodic.COMPLEX))
    transform = torch.zeros(reducer.n_dof, reducer.n_reduced_dof, dtype=periodic.COMPLEX)
    transform[torch.arange(reducer.n_dof), reducer.master_dof] = phase

    rhs = torch.arange(1, 10, dtype=torch.float64).to(periodic.COMPLEX)
    assert torch.allclose(
        periodic.reduce_rhs(reducer, rhs, wavevector),
        transform.conj().T @ rhs,
        atol=1.0e-12,
    )
    dense = torch.arange(81, dtype=torch.float64).reshape(9, 9)
    dense = (dense + dense.T).to(periodic.COMPLEX)
    row, column = torch.meshgrid(torch.arange(9), torch.arange(9), indexing="ij")
    matrix = SparseMatrix(dense.ravel(), row.ravel(), column.ravel(), (9, 9))
    reduced = reducer.reduce(matrix, wavevector).to_torch_sparse().to_dense()
    assert torch.allclose(reduced, transform.conj().T @ dense @ transform, atol=1.0e-11)


@pytest.mark.parametrize(
    ("case", "limit"),
    [("dirichlet", 0.12), ("neumann", 0.10), ("transmission", 0.30)],
)
def test_circular_cases(case: str, limit: float) -> None:
    """Every circular case solves in complex128 with a small residual."""

    mesh, solution, _, metrics = circle.solve_circle(case, mesh_size=0.24, num_modes=4)
    assert mesh.points.dtype == torch.float64
    assert solution.dtype == torch.complex128
    assert metrics["matrix_dtype"] == torch.complex128
    assert metrics["residual"] < 1.0e-10
    assert metrics["relative_l2"] < limit


@pytest.mark.parametrize(
    ("case", "limit"),
    [("dirichlet", 0.06), ("neumann", 0.06), ("transmission", 0.08)],
)
def test_circular_convergence(case: str, limit: float) -> None:
    """Three P1 meshes show the expected nodal convergence trend."""

    sizes = np.array([0.24, 0.16, 0.10])
    errors = np.array(
        [circle.solve_circle(case, mesh_size=float(size))[3]["relative_l2"] for size in sizes]
    )
    order = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    assert np.all(np.diff(errors) < 0.0)
    assert order >= 1.5
    assert errors[-1] < limit


def test_circle_radius_is_a_live_parameter() -> None:
    """Changing the physical radius changes both mesh and exact coefficient."""

    small = circle.solve_circle("dirichlet", circle_radius=0.32, mesh_size=0.18)
    large = circle.solve_circle("dirichlet", circle_radius=0.48, mesh_size=0.18)
    assert small[0].n_points != large[0].n_points
    assert abs(small[3]["outgoing"] - large[3]["outgoing"]) > 1.0e-2


@pytest.mark.parametrize("order", [-2, 3])
def test_nonzero_circular_incident_orders(order: int) -> None:
    """Non-axisymmetric incident modes work in the complete FEM solve."""

    metrics = circle.solve_circle(
        "dirichlet", incident_order=order, mesh_size=0.18
    )[3]
    assert metrics["residual"] < 1.0e-10
    assert metrics["relative_l2"] < 7.0e-2


@pytest.mark.parametrize("case", ["slab", "dirichlet", "neumann"])
def test_periodic_cases(case: str) -> None:
    """Port amplitudes, powers, Bloch trace, and modal leakage match the oracle."""

    mesh, solution, _, metrics = periodic.solve_periodic(case)
    assert mesh.points.dtype == torch.float64
    assert solution.dtype == torch.complex128
    assert metrics["matrix_dtype"] == torch.complex128
    assert metrics["residual"] < 1.0e-10
    assert metrics["bloch_residual"] < 1.0e-10
    reflection_error = abs(metrics["reflection"] - metrics["exact_reflection"])
    reflection_error /= abs(metrics["exact_reflection"])
    assert reflection_error < 2.0e-2
    assert metrics["energy_residual"] < 1.0e-2
    assert metrics["leakage"] < 1.0e-3
    if case == "slab":
        transmission_error = abs(
            metrics["transmission"] - metrics["exact_transmission"]
        )
        transmission_error /= abs(metrics["exact_transmission"])
        assert transmission_error < 2.0e-2
        assert abs(metrics["reflectance"] + metrics["transmittance"] - 1.0) < 1.0e-2
    else:
        assert metrics["transmission"] is None
        assert metrics["transmittance"] is None
        assert abs(metrics["reflectance"] - 1.0) < 1.0e-2


def test_height_is_a_live_parameter() -> None:
    """Two layer heights produce different wall reflection amplitudes."""

    first = periodic.solve_periodic("dirichlet", height=0.45, mesh_size=0.18, num_modes=2)
    second = periodic.solve_periodic("dirichlet", height=0.65, mesh_size=0.18, num_modes=2)
    assert abs(first[3]["exact_reflection"] - second[3]["exact_reflection"]) > 1.0e-2


def test_single_rayleigh_mode_is_supported() -> None:
    """A zero cutoff keeps the incident order and has zero modal leakage."""

    metrics = periodic.solve_periodic("slab", num_modes=0, mesh_size=0.15)[3]
    assert metrics["residual"] < 1.0e-10
    assert metrics["leakage"] == 0.0


def test_invalid_user_parameters_fail_before_meshing() -> None:
    """Editable gallery constants produce concise domain errors."""

    with pytest.raises(ValueError, match="radii"):
        circle.solve_circle("dirichlet", circle_radius=1.3, dtn_radius=1.2)
    with pytest.raises(ValueError, match="propagate"):
        periodic.solve_periodic("slab", alpha=3.1)


def test_examples_are_two_self_contained_scripts() -> None:
    """The demos must not recover the removed private helper or project package."""

    assert not (EXAMPLE_DIR / "_dtn_tools.py").exists()
    for path in (EXAMPLE_DIR / "circular_dtn.py", EXAMPLE_DIR / "periodic_dtn.py"):
        text = path.read_text(encoding="utf-8")
        assert "_dtn_tools" not in text
        assert "tensormesh" + "_em_dtn" not in text
        assert "import torch_sla" not in text
        assert ".edata" not in text
        assert '.solve(rhs, backend="scipy", method="lu")' in text

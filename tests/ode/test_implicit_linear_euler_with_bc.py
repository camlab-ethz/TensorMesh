"""End-to-end tests: ImplicitLinearRungeKutta + Condenser hooks.

The hooks ``pre_solve_lhs`` / ``pre_solve_rhs`` / ``recover_stage`` are
designed to compose with :class:`tensormesh.operator.Condenser` so that
static condensation can be done *inside* a time-stepper. These tests
pin that behavior by integrating the heat equation in two ways
(verbatim handwritten loop vs. the ode module wired through the hooks)
on a small mesh and asserting machine-precision agreement.
"""
import sys
sys.path.append('../..')

import pytest
import torch

from tensormesh import Mesh, Condenser, ElementAssembler
from tensormesh.ode import ImplicitLinearEuler, MidPointLinearEuler


class _AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class _MAssembler(ElementAssembler):
    def forward(self, u, v):
        return u * v


def _build_heat_problem(device="cpu"):
    """Small 2D heat problem with homogeneous Dirichlet BC.

    Returns the mass and stiffness matrices, a random initial state
    consistent with the boundary, and a fresh Condenser.
    """
    torch.manual_seed(0)
    mesh = Mesh.gen_rectangle(chara_length=0.15, order=1, element_type="tri").to(device=device)
    M = _MAssembler.from_mesh(mesh, quadrature_order=2)()
    A = _AAssembler.from_mesh(mesh, quadrature_order=2)()
    # zero on boundary, random in the interior — keeps the test agnostic
    # to the specific mesh layout.
    u0 = torch.randn(mesh.n_points, dtype=M.edata.dtype, device=device)
    u0[mesh.boundary_mask] = 0.0
    condenser = Condenser(mesh.boundary_mask).to(device=device)
    return mesh, M, A, u0, condenser


def _manual_implicit_euler(M, A, u0, dt, n_steps, condenser, D=1.0):
    """Reference: the loop currently shipped in examples/diffusion/heat/heat.py."""
    K  = M + dt * D * D * A
    K_ = condenser(K)[0]
    U  = u0
    for _ in range(n_steps):
        F  = M @ U
        F_ = condenser.condense_rhs(F)
        U_ = K_.solve(F_)
        U  = condenser.recover(U_)
    return U


class _HeatStepper(ImplicitLinearEuler):
    """Implicit Euler for M du/dt = -D^2 A u with homogeneous Dirichlet."""
    def __init__(self, M, A, D, condenser):
        super().__init__()
        self._M  = M
        self._A  = -D * D * A
        self._cd = condenser

    def forward_M(self, t):
        return self._M

    def forward_A(self, t):
        return self._A

    def forward_B(self, t):
        return 0.0

    def pre_solve_lhs(self, K):
        return self._cd(K)[0]

    def pre_solve_rhs(self, f):
        return self._cd.restrict(f)

    def recover_stage(self, k_i):
        return self._cd.prolong(k_i)


def _run_ode_stepper(stepper_cls, M, A, u0, dt, n_steps, condenser, D=1.0):
    stepper = stepper_cls(M, A, D, condenser)
    U = u0
    t = 0.0
    for _ in range(n_steps):
        U = stepper.step(t, U, dt)
        t += dt
    return U


def test_implicit_euler_matches_manual_heat_loop():
    """Hooks + Condenser must reproduce the manual heat loop bit-for-bit."""
    _, M, A, u0, cd_manual = _build_heat_problem()
    cd_ode = Condenser(cd_manual.dirichlet_mask)

    dt, n = 1e-3, 20
    U_manual = _manual_implicit_euler(M, A, u0, dt, n, cd_manual)
    U_ode    = _run_ode_stepper(_HeatStepper, M, A, u0, dt, n, cd_ode)

    err = (U_manual - U_ode).abs().max().item()
    norm = U_manual.abs().max().item()
    assert err < 1e-12 * max(norm, 1.0), \
        f"manual vs ode disagree: abs={err:.3e}, ||U||={norm:.3e}"


def test_implicit_euler_preserves_dirichlet_zero():
    """Boundary entries must stay zero under hooks-based integration."""
    mesh, M, A, u0, cd_ode = _build_heat_problem()
    U = _run_ode_stepper(_HeatStepper, M, A, u0, 1e-3, 20, cd_ode)
    assert torch.allclose(
        U[mesh.boundary_mask],
        torch.zeros(int(mesh.boundary_mask.sum()), dtype=U.dtype, device=U.device),
        atol=1e-12,
    )


def test_default_recover_stage_is_identity():
    """Scalar Implicit-Euler with no hooks must match the closed-form solution.

    Regression check: the refactor must not change behavior in the
    no-condensation case. Mirrors ``test_implicit_linear_euler.py`` but
    is repeated here to keep this file independently informative.
    """
    u0 = torch.rand(4).double()
    dt = 0.1
    ut_gt = (1 / (1 - dt)) * u0
    ut    = ImplicitLinearEuler().step(0, u0, dt)
    assert torch.allclose(ut_gt, ut)


def test_midpoint_with_bc_runs():
    """A multi-stage scheme (s=1 but exercises stage-recovery path) also works.

    MidPointLinearEuler has s=1 too, so this is mostly a smoke test for
    the per-stage prolongation code path under a non-trivial tableau.
    """
    _, M, A, u0, cd_ode = _build_heat_problem()

    class HeatMidpoint(MidPointLinearEuler):
        def __init__(self, M, A, D, condenser):
            super().__init__()
            self._M, self._A, self._cd = M, -D * D * A, condenser
        def forward_M(self, t): return self._M
        def forward_A(self, t): return self._A
        def forward_B(self, t): return 0.0
        def pre_solve_lhs(self, K): return self._cd(K)[0]
        def pre_solve_rhs(self, f): return self._cd.restrict(f)
        def recover_stage(self, k_i): return self._cd.prolong(k_i)

    U = _run_ode_stepper(HeatMidpoint, M, A, u0, 1e-3, 20, cd_ode)
    # Energy should decay (M-norm), boundary should stay at 0.
    assert torch.isfinite(U).all()
    mesh_mask = cd_ode.dirichlet_mask
    assert U[mesh_mask].abs().max().item() < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_implicit_euler_with_bc_cuda():
    _, M, A, u0, cd_manual = _build_heat_problem(device="cuda")
    cd_ode = Condenser(cd_manual.dirichlet_mask).to(device="cuda")

    dt, n = 1e-3, 10
    U_manual = _manual_implicit_euler(M, A, u0, dt, n, cd_manual)
    U_ode    = _run_ode_stepper(_HeatStepper, M, A, u0, dt, n, cd_ode)

    assert U_ode.device.type == "cuda"
    assert torch.allclose(U_manual, U_ode, atol=1e-12, rtol=1e-12)

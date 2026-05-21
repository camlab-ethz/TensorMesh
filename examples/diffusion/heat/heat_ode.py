"""Same heat problem as ``heat.py``, but driven through ``tensormesh.ode``.
"""
import sys
sys.path.append("../../..")

import torch
from tqdm import tqdm
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.dataset import HeatMultiFrequency
from tensormesh.ode import ImplicitLinearEuler


class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """gradu, gradv: [n_dim] → scalar"""
        return gradu @ gradv


class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """u, v: scalar → scalar"""
        return u * v


class HeatStepper(ImplicitLinearEuler):
    """Backward Euler for M du/dt = -D^2 A u with homogeneous Dirichlet BC.

    Static condensation is wired through three hooks:

    * ``pre_solve_lhs`` condenses the assembled stage LHS to the inner
      DOFs.
    * ``pre_solve_rhs`` restricts the stage RHS to the inner DOFs.
      Note this uses ``Condenser.restrict`` rather than
      ``condense_rhs`` — for a stage *slope* the Dirichlet correction
      term :math:`-K_{io}\\,u_o` would be wrong (a Dirichlet DOF has
      zero time-derivative by definition, regardless of its value).
    * ``recover_stage`` prolongs the inner stage slope back to full
      DOF with zeros in the boundary slots, before the integrator
      combines it with ``u0``.
    """
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


if __name__ == '__main__':
    torch.random.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = Mesh.gen_rectangle(chara_length=0.02, order=2, element_type="tri").to(device=device)
    dataset = HeatMultiFrequency(d=16)

    u0 = dataset.initial_condition(mesh.points)

    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    M = M_asm()
    A = A_asm()

    condenser = Condenser(mesh.boundary_mask)
    dt = 0.00005
    D  = 1
    n  = 100

    stepper = HeatStepper(M, A, D, condenser)

    U = u0
    t = 0.0
    Us = [U]
    for _ in tqdm(range(n - 1), desc="Time stepping"):
        U = stepper.step(t, U, dt)
        t += dt
        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt * i) for i in tqdm(range(n), desc="Ground truth")]

    mesh.plot(
        {"FEM solution": Us, "Analytical solution": Us_gt},
        save_path="heat_ode.mp4",
        dt=dt,
        show_mesh=False,
        fix_clim=False)

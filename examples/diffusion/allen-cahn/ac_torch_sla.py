"""Allen-Cahn phase-field evolution via torch-sla's ``nonlinear_solve``.
    python ac_torch_sla.py
    python ac_torch_sla.py --device cuda --steps 200
"""
import argparse
import os
import sys
import warnings

sys.path.append("../../..")

import torch
from tqdm import tqdm

from tensormesh import ElementAssembler, NodeAssembler, Mesh
from tensormesh.dataset import PoissonMultiFrequency
from torch_sla import nonlinear_solve

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="float64 recommended")

torch.set_default_dtype(torch.float64)

DT = 1e-6
EPSILON = 220


class KAssembler(ElementAssembler):
    """Negative consistent tangent K = -dR/dc (identical to ``ac.py``)."""

    def __post_init__(self):
        self.dt = DT
        self.dcdotdc = 1.0 / DT
        self.D = lambda x: 1.0e0
        self.dD = lambda x: 0.0e0
        self.f = lambda x: -EPSILON ** 2 * x * (x ** 2 - 1)
        self.df = lambda x: -EPSILON ** 2 * (3 * x ** 2 - 1)

    def forward(self, u, v, gradu, gradv, c, gradc, cold):
        return -1.0 * (self.dcdotdc * (u * v) +
                       self.dD(c) * u * (gradv @ gradc) +
                       self.D(c) * (gradu @ gradv) -
                       self.df(c) * (u * v))


class RAssembler(NodeAssembler):
    """Backward-Euler residual R(c) (identical to ``ac.py``)."""

    def __post_init__(self):
        self.dt = DT
        self.dcdotdc = 1.0 / DT
        self.D = lambda x: 1.0e0
        self.dD = lambda x: 0.0e0
        self.f = lambda x: -EPSILON ** 2 * x * (x ** 2 - 1)
        self.df = lambda x: -EPSILON ** 2 * (3 * x ** 2 - 1)

    def forward(self, v, gradv, c, gradc, cold):
        cdot = (c - cold) / self.dt
        return cdot * v + self.D(c) * (gradv @ gradc) - self.f(c) * v


def simulate(device="cpu", chara_length=0.02, steps=200, max_iter=50,
             seed=24, linear_solver="auto", linear_method="auto"):
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="tri").to(device=device)
    dataset = PoissonMultiFrequency(K=seed, r=1)
    cold = dataset.source_term(mesh.points)
    print(f"mesh: {mesh.n_points} nodes on {device}")

    K_asm = KAssembler.from_mesh(mesh)
    R_asm = RAssembler.from_mesh(mesh)

    # F(c; cold) = 0 is the implicit backward-Euler step.
    def residual_fn(c, cold):
        return R_asm(mesh.points, point_data={"c": c, "cold": cold})

    # KAssembler gives K = -J, so return -K.values to hand nonlinear_solve J.
    def jacobian_fn(c, cold):
        K = K_asm(mesh.points, point_data={"c": c, "cold": cold})
        return (-K.values, K.row, K.col, K.shape)

    cs = [cold]
    for _ in tqdm(range(steps), desc="Time stepping"):
        c = nonlinear_solve(
            residual_fn, cold, cold,
            jacobian_fn=jacobian_fn,
            method="newton",
            atol=1e-10, tol=1e-8, max_iter=max_iter,
            linear_solver=linear_solver, linear_method=linear_method,
        )
        cold = c
        cs.append(cold)

    return mesh, cs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--chara-length", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=50,
                        help="max Newton iterations per time step")
    parser.add_argument("--linear-solver", default="auto",
                        choices=["auto", "scipy", "pytorch", "cudss", "cupy", "eigen"])
    parser.add_argument("--linear-method", default="auto")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    mesh, cs = simulate(device=args.device, chara_length=args.chara_length,
                        steps=args.steps, max_iter=args.max_iter,
                        linear_solver=args.linear_solver,
                        linear_method=args.linear_method)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "Allen-Cahn-torch-sla.mp4")
    mesh.plot(values={"phi": cs}, show_mesh=False, dt=DT, save_path=out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

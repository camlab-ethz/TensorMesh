"""Allen-Cahn via Eyre's convex-concave (convex splitting) scheme.

    python ac_convex_concave.py
    python ac_convex_concave.py --steps 200 --dt 1e-6

Same PDE as ``ac.py`` / ``ac_torch_sla.py``, but the
nonlinear reaction is *split* by the convexity of the double-well energy, so
the resulting time-stepping is unconditionally stable; see Eyre's paper
https://link.springer.com/article/10.1557/PROC-529-39.

The Allen-Cahn equation is

    c_t = Delta c + eps^2 c(1-c^2)
        = Delta c - eps^2 c^3 + eps^2 c

and it is the L2 gradient flow of the energy

    E(c) = int 1/2 |grad c|^2 + eps^2 W(c),   W(c) = 1/4 (c^2 - 1)^2.

Convex-concave splitting for the Allen Cahn equation amounts to

    (c - c_old)/dt  = Delta c - eps^2 c^3 + eps^2 c_old.

It has the advantages:
    1. Unconditionally stable: Reduces energy independent of the time-step size.
    2. The Newton step for the time-stepping problem is uniquely solvable, again
        independent of the time-step size.

The residual for Newton, assuming pure Neumann boundary conditions is

    R(c)v = int_Omega 1/dt (c - c_old) v + gradc gradv + eps^2 c^3 v - eps^2 c_old v dx.

The Jacobian of the Residual is

    R'(c)[u]v = int_Omega 1/dt u v + gradu gradv + 3 eps^2 c^2 uv dx.
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

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="float64 recommended")

torch.set_default_dtype(torch.float64)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--chara-length", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=50,
                        help="max Newton iterations per time step")
    parser.add_argument("--dt", type=float, default=5e-5,
                        help="time step (convex split is unconditionally gradient-stable)")
    parser.add_argument("--epsilon", type=float, default=220,
                        help="interface-width parameter eps in the double-well energy")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the random initial condition (reproducible)")
    parser.add_argument("--n-freq", type=int, default=24,
                        help="number of Fourier modes per axis in the initial condition")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    return parser.parse_args()


class RAssembler(NodeAssembler):
    """Residual R with the convex-concave split."""

    def __post_init__(self, dt, epsilon):
        self.dt = dt
        self.epsilon = epsilon
        self.fi = lambda x: self.epsilon ** 2 * x ** 3   # implicit
        self.fe = lambda x: self.epsilon ** 2 * x          # explicit

    def forward(self, v, gradv, c, gradc, cold):
        c_t = (c - cold) / self.dt
        return c_t * v + (gradc @ gradv) + self.fi(c) * v - self.fe(cold) * v


class KAssembler(ElementAssembler):
    """Derivative of R, implements R'(c)(u,v)."""

    def __post_init__(self, dt, epsilon):
        self.dt = dt
        self.epsilon = epsilon
        self.dfi = lambda x: -3.0 * self.epsilon ** 2 * x ** 2

    def forward(self, c, u, v, gradu, gradv):
        return 1.0 * (1.0 / self.dt * (u * v) + (gradu @ gradv) - self.dfi(c) * (u * v))


def simulate(device="cpu", chara_length=0.02, steps=200, max_iter=50, dt=1e-6, epsilon=220,
             seed=0, n_freq=24):
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="tri").to(device=device)

    generator = torch.Generator(device=mesh.points.device).manual_seed(seed)
    a = torch.empty(n_freq, n_freq, device=mesh.points.device).uniform_(-1, 1, generator=generator)
    dataset = PoissonMultiFrequency(a=a, r=1)
    cold = dataset.source_term(mesh.points)
    print(f"mesh: {mesh.n_points} nodes on {device}")

    params = dict(dt=dt, epsilon=epsilon)
    R_asm = RAssembler.from_mesh(mesh, **params)
    K_asm = KAssembler.from_mesh(mesh, **params)

    cs = [cold]

    with tqdm(range(steps), desc="Time stepping") as pbar:
        for _ in pbar:
            c = cold
            rnorm = torch.tensor(float("inf"))
            for newton_step in range(max_iter):
                point_data = {"c": c, "cold": cold}
                K = K_asm(mesh.points, point_data=point_data)
                R = R_asm(mesh.points, point_data=point_data)

                c = c + K.solve(-R)

                rnorm = torch.linalg.norm(R)
                if rnorm < 1e-10:
                    break

            cold = c
            cs.append(cold)
            pbar.set_postfix({"rnorm": float(rnorm), "newton_iter": newton_step + 1})

    return mesh, cs


def main():
    args = parse_arguments()

    mesh, cs = simulate(device=args.device, chara_length=args.chara_length,
                        steps=args.steps, max_iter=args.max_iter, dt=args.dt,
                        epsilon=args.epsilon, seed=args.seed, n_freq=args.n_freq)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"Allen-Cahn-convex-concave-dt{args.dt:g}.mp4")
    mesh.plot(values={"phi": cs}, show_mesh=False, dt=args.dt, save_path=out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

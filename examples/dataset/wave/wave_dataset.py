import sys
sys.path.append("../../..")

import os
import numpy as np
import torch
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.dataset import WaveMultiFrequency

import time
from tqdm import tqdm


class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class MAssembler(ElementAssembler):
    def forward(self, u, v):
        return u * v


def run_wave_stepping(M, cA, M_, condenser, U1, U2, n, device, desc=""):
    """Run n-2 wave time steps using central difference scheme.

    The first two snapshots (U1=u0, U2=u1) are given.
    Each subsequent step: M @ U^{n+1} = 2*M @ U^n - M @ U^{n-1} - dt^2*c^2*A @ U^n

    Parameters
    ----------
    M : SparseMatrix
        Mass matrix (full, uncondensed)
    cA : SparseMatrix
        dt^2 * c^2 * A (stiffness scaled by dt^2*c^2, full)
    M_ : SparseMatrix
        Condensed mass matrix for solve
    condenser : Condenser
    U1, U2 : torch.Tensor
        First two snapshots, shape [n_dofs] or [n_dofs, batch]
    n : int
        Total number of time steps
    """
    Us = [U1, U2]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in tqdm(range(n - 2), desc=desc):
        U_prev, U_curr = Us[-2], Us[-1]
        F = 2.0 * (M @ U_curr) - (M @ U_prev) - (cA @ U_curr)
        F_ = condenser.condense_rhs(F)
        U_ = M_.solve(F_, verbose=(i == 0))
        U  = condenser.recover(U_)
        Us.append(U)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return Us, elapsed


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(123456)

    dt = 0.001
    c  = 2.0
    n  = 100
    K  = 16
    batch_size = 1000

    #mesh = Mesh.gen_rectangle(chara_length=0.01).to(device)
    mesh = Mesh.gen_circle(chara_length=0.008, cx=0.5, cy=0.5, r=0.5).to(device)

    # Batched dataset: a has shape [batch_size, K, K]
    a = torch.zeros((batch_size, K, K)).uniform_(-1, 1)
    dataset = WaveMultiFrequency(a=a, c=c)

    u0s = dataset.initial_condition(mesh.points)  # [batch_size, n_dofs]

    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)

    M = M_asm()
    A = A_asm()
    condenser = Condenser(mesh.boundary_mask)

    # Precompute scaled stiffness: dt^2 * c^2 * A
    cA = (dt * dt * c * c) * A

    # Condense M first (sets layout for this condenser)
    M_ = condenser(M)[0]

    # First step: 2M @ U1 = 2*M @ U0 - dt^2*c^2*A @ U0  (v0 = 0)
    U0 = u0s.T  # [n_dofs, batch_size]
    F_first = 2.0 * (M @ U0) - (cA @ U0)
    F_first_ = condenser.condense_rhs(F_first)
    # K_first = 2*M, so K_first_ = 2*M_ (same layout, just scaled values)
    U1_ = (2.0 * M_).solve(F_first_)
    U1  = condenser.recover(U1_)

    print(f"DOFs: {mesh.n_points}, batch_size: {batch_size}, n_steps: {n}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- GPU solve ----
    Us_gpu, t_gpu = run_wave_stepping(
        M, cA, M_, condenser, U0.clone(), U1.clone(), n, device, desc="GPU solve")
    print(f"GPU solve:         {t_gpu:.2f}s")

    # ---- CPU solve ----
    M_cpu = M.cpu()
    cA_cpu = cA.cpu()
    condenser_cpu = Condenser(mesh.boundary_mask.cpu())
    M_cpu_ = condenser_cpu(M_cpu)[0]
    Us_cpu, t_cpu = run_wave_stepping(
        M_cpu, cA_cpu, M_cpu_, condenser_cpu,
        U0.clone().cpu(), U1.clone().cpu(), n, torch.device("cpu"), desc="CPU solve")
    print(f"CPU solve (scipy): {t_cpu:.2f}s")

    # ---- Verify correctness ----
    err = torch.max(torch.abs(Us_gpu[-1].cpu() - Us_cpu[-1])).item()
    print(f"Max error (GPU vs CPU): {err:.2e}")
    print(f"Speedup (CPU/GPU): {t_cpu/t_gpu:.2f}x")
    print("=" * 60)

    # ---- Save results ----
    save_dir = os.path.dirname(os.path.abspath(__file__))
    snapshots = torch.stack(Us_gpu).cpu().numpy()  # (n, n_dofs, batch_size)
    save_path = os.path.join(save_dir, "wave_dataset.npz")
    np.savez(
        save_path,
        snapshots=snapshots,
        points=mesh.points.cpu().numpy(),
        a=a.numpy(),
        dt=dt, c=c, n=n,
    )
    print(f"Results saved to {save_path}")
    print(f"  snapshots shape: {snapshots.shape}")

    # ---- Visualize (first sample from batch) ----
    mesh_cpu = mesh.cpu()
    mesh_cpu.plot(
        {"sample 0": [u[:, 0].cpu() for u in Us_gpu],
         "sample 1": [u[:, 1].cpu() for u in Us_gpu],
         "sample 2": [u[:, 2].cpu() for u in Us_gpu],
         "sample 3": [u[:, 3].cpu() for u in Us_gpu],
         "sample 4": [u[:, 4].cpu() for u in Us_gpu]},
        save_path="wave_dataset.mp4",
        dt=dt,
        show_mesh=False,
        linewidth=0.1,
        linecolor='black')

import sys 
sys.path.append("../../..")

import os
import numpy as np
import torch
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.dataset import HeatMultiFrequency

import time
from tqdm import tqdm

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_dim]
                gradv: torch.Tensor[n_dim]
            Returns:
            --------
                scalar
        """
        return gradu @ gradv

class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor (scalar)
                v: torch.Tensor (scalar)
            Returns:
            --------
                scalar
        """
        return u * v

def run_time_stepping(M, K_, condenser, U, n, device, desc=""):
    """Run n-1 time steps and return list of U snapshots + elapsed time."""
    Us = [U]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in tqdm(range(n-1), desc=desc):
        F = M @ U
        F_ = condenser.condense_rhs(F)
        U_ = K_.solve(F_,verbose=(i == 0))
        U  = condenser.recover(U_)
        Us.append(U)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return Us, elapsed


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(3)
    mesh = Mesh.gen_L(chara_length=0.008, element_type="tri").to(device)

    batch_size = 1000
    d = 16
    mus = torch.rand((batch_size, d))
    dataset = HeatMultiFrequency(mu=mus)
    u0s = dataset.initial_condition(mesh.points)

    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)

    M = M_asm()
    A = A_asm()

    condenser = Condenser(mesh.boundary_mask)

    dt = 0.00005
    D  = 1
    n  = 100
    K  = M + dt * D * D * A
    K_ = condenser(K)[0]

    print(f"DOFs: {mesh.n_points}, batch_size: {batch_size}, n_steps: {n}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- GPU solve (cupy or scipy fallback) ----
    U_gpu = u0s.T.clone()
    Us_gpu, t_gpu = run_time_stepping(M, K_, condenser, U_gpu, n, device, desc="GPU solve")
    print(f"GPU solve:         {t_gpu:.2f}s")

    # ---- CPU solve (scipy, data moved to CPU) ----
    M_cpu = M.cpu()
    K_cpu_mat = K.cpu()
    condenser_cpu = Condenser(mesh.boundary_mask.cpu())
    K_cpu = condenser_cpu(K_cpu_mat)[0]
    U_cpu = u0s.T.clone().cpu()
    Us_cpu, t_cpu = run_time_stepping(M_cpu, K_cpu, condenser_cpu, U_cpu, n, torch.device("cpu"), desc="CPU solve")
    print(f"CPU solve (scipy): {t_cpu:.2f}s")

    # ---- Verify correctness ----
    err = torch.max(torch.abs(Us_gpu[-1].cpu() - Us_cpu[-1])).item()
    print(f"Max error (GPU vs CPU): {err:.2e}")
    print(f"Speedup (CPU/GPU): {t_cpu/t_gpu:.2f}x")
    print("=" * 60)

    # ---- Save results to numpy file ----
    save_dir = os.path.dirname(os.path.abspath(__file__))
    snapshots = torch.stack(Us_gpu).cpu().numpy()   # (n, n_dofs, batch_size)
    save_path = os.path.join(save_dir, "heat_dataset.npz")
    np.savez(
        save_path,
        snapshots=snapshots,
        points=mesh.points.cpu().numpy(),
        mus=mus.numpy(),
        dt=dt, D=D, n=n,
    )

    print(f"Results saved to {save_path}")
    print(f"  snapshots shape: {snapshots.shape}")
    
    # ---- Visualize results (pick first sample from batch) ----
    mesh.plot(
        {"sample 0": [u[:, 0].cpu() for u in Us_gpu],
         "sample 1": [u[:, 1].cpu() for u in Us_gpu],
         "sample 2": [u[:, 2].cpu() for u in Us_gpu],
         "sample 3": [u[:, 3].cpu() for u in Us_gpu],
         "sample 4": [u[:, 4].cpu() for u in Us_gpu]},
        save_path="heat_dataset.mp4",
        dt=dt,
        show_mesh=False,
        fix_clim=False,
    )


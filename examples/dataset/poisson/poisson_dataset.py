import sys
sys.path.append("../../..")

import os
import time
import numpy as np
import torch
from tensormesh import LaplaceElementAssembler, MassElementAssembler, Mesh, Condenser, NodeAssembler
from tensormesh.dataset import PoissonMultiFrequency


class FAssembler(NodeAssembler):
    """Same weak form as examples/poisson/poisson.py: b_i = ∫ f v_i dx."""

    def forward(self, v, f):
        return v * f


def assemble_rhs_batch(mesh, M, f, F_asm):
    """Assemble batched RHS consistent with poisson.py (NodeAssembler).

    NodeAssembler returns a 1D vector per call; for nodal ``f`` the load
    ``∫ f v_i dx`` equals ``(M @ f)`` with the mass matrix from the same
    quadrature. We use ``M @ f.T`` for speed; verify sample 0 against F_asm.
    """
    b = M @ f.T  # [n_dofs, batch_size]
    b_ref = F_asm(mesh.points, point_data={"f": f[0]})
    if not torch.allclose(b[:, 0], b_ref, rtol=1e-10, atol=1e-10):
        raise RuntimeError("batched RHS (M @ f) disagrees with poisson.py FAssembler")
    return b


def solve_batch(K_, b_, condenser, device, verbose=False):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    u_ = K_.solve(b_, verbose=verbose)
    u = condenser.recover(u_)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return u.T, time.perf_counter() - t0  # [batch, n_dofs]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    chara_length = 0.008
    K_modes = 16
    batch_size = 1000

    mesh = Mesh.gen_circle(
        chara_length=chara_length, cx=0.5, cy=0.5, r=0.5
    ).to(device)

    a = torch.zeros((batch_size, K_modes, K_modes), device=device).uniform_(-1, 1)
    equation = PoissonMultiFrequency(a=a)
    points = mesh.points

    f = equation.source_term(points, domain="rectangle")  # [batch, n_dofs]

    # ---- Same assembly pattern as examples/poisson/poisson.py ----
    K_asm = LaplaceElementAssembler.from_mesh(mesh)
    F_asm = FAssembler.from_mesh(mesh)
    M_asm = MassElementAssembler.from_mesh(mesh)

    K = K_asm(mesh.points)
    M = M_asm(mesh.points)
    b = assemble_rhs_batch(mesh, M, f, F_asm)

    boundary_value = torch.zeros(mesh.n_points, device=device, dtype=mesh.dtype)
    condenser = Condenser(mesh.boundary_mask, boundary_value)
    K_, b_ = condenser(K, b)

    print(f"DOFs (total): {mesh.n_points}, inner: {K_.shape[0]}, batch_size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- GPU ----
    u_gpu, t_gpu = solve_batch(K_, b_, condenser, device, verbose=True)
    print(f"GPU solve:         {t_gpu:.2f}s")

    # ---- CPU reference ----
    mesh_cpu = mesh.cpu()
    f_cpu = f.cpu()
    K_asm_cpu = LaplaceElementAssembler.from_mesh(mesh_cpu)
    M_asm_cpu = MassElementAssembler.from_mesh(mesh_cpu)
    F_asm_cpu = FAssembler.from_mesh(mesh_cpu)
    K_cpu = K_asm_cpu(mesh_cpu.points)
    M_cpu = M_asm_cpu(mesh_cpu.points)
    b_cpu = assemble_rhs_batch(mesh_cpu, M_cpu, f_cpu, F_asm_cpu)

    condenser_cpu = Condenser(mesh_cpu.boundary_mask, boundary_value.cpu())
    K_cpu_, b_cpu_ = condenser_cpu(K_cpu, b_cpu)

    u_cpu, t_cpu = solve_batch(K_cpu_, b_cpu_, condenser_cpu, torch.device("cpu"), verbose=True)
    print(f"CPU solve (scipy): {t_cpu:.2f}s")

    err = torch.max(torch.abs(u_gpu.cpu() - u_cpu)).item()
    print(f"Max error (GPU vs CPU): {err:.2e}")
    print(f"Speedup (CPU/GPU): {t_cpu / t_gpu:.2f}x")
    print("=" * 60)

    # ---- Save ----
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "poisson_dataset.npz")
    np.savez(
        save_path,
        solutions=u_gpu.cpu().numpy(),
        sources=f.cpu().numpy(),
        points=points.cpu().numpy(),
        a=a.cpu().numpy(),
        chara_length=chara_length,
        K_modes=K_modes,
        batch_size=batch_size,
    )
    print(f"Saved: {save_path}")
    print(f"  solutions shape: {u_gpu.shape}")

    mesh_cpu.plot(
        { 
            "sample 0": u_gpu[0].cpu(),
            "sample 1": u_gpu[1].cpu(),
            "sample 2": u_gpu[2].cpu(),
            "sample 3": u_gpu[3].cpu(),
            "sample 4": u_gpu[4].cpu(),
        },
        save_path="poisson_dataset.png",
        show_mesh=False
    )

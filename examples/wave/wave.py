import sys 
sys.path.append("../..")

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensormesh import ElementAssembler, Mesh, Condenser
from tensormesh.sparse import SparseMatrix
from tensormesh.dataset import WaveMultiFrequency

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_dim]
                gradv: torch.Tensor[n_dim]
            Returns:
            --------
                M: torch.Tensor[]
        """
        return gradu @ gradv
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[]
                v: torch.Tensor[]
            Returns:
            --------
                M: torch.Tensor[n_basis]
        """
        return u * v

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    dt = 0.001
    c  = 2.0
    n  = 100
    torch.random.manual_seed(123456)

    mesh = Mesh.gen_rectangle(chara_length=0.01)
    #mesh = Mesh.gen_circle(chara_length=0.015, cx=0.5, cy=0.5, r=0.5)
    mesh = mesh.to(device)
    dataset = WaveMultiFrequency(K=16, c=c)

    u0 = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    M = M_asm()
    A = A_asm()
    condenser = Condenser(mesh.boundary_mask)

    def scale_matrix(mat, s):
        """Scale a SparseMatrix by scalar s, preserving SparseMatrix type."""
        return SparseMatrix(mat.edata * s, mat.row, mat.col, mat.shape)

    u0 = u0.to(device)
    Us  = [u0]
    v0 = torch.zeros_like(u0)
    A = scale_matrix(A, c * c)
    K = scale_matrix(M, 2.0)
    F = -(dt * dt) * (A @ u0) + 2.0 * (M @ u0) + (2.0 * dt) * (M @ v0)
    K_, F_ = condenser(K, F)
    U_     = K_.solve(F_)
    U      = condenser.recover(U_)
    M_     = scale_matrix(K_, 0.5)  # K_ = 2*M_, so M_ = K_/2
    Us.append(U)
    for _ in tqdm(range(n-2), desc="Time stepping"):
        U1, U2 = Us[-2:]

        F = 2.0 * (M @ U2) - (M @ U1) - (dt * dt) * (A @ U2)

        F_ = condenser.condense_rhs(F)

        U_ = M_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)

    # ---- Energy diagnostics.
    # The undamped wave equation conserves the mechanical energy
    #   E = 1/2 v^T M v  +  1/2 u^T A u      (A already holds c^2),
    def energy_quad(mat, x):
        return 0.5 * torch.dot(x, mat @ x)

    times, ke, pe = [], [], []
    for i, Ui in enumerate(Us):
        if i == 0:
            v = v0                                    # zero initial velocity
        elif i == len(Us) - 1:
            v = (Us[i] - Us[i - 1]) / dt              # backward difference at the end
        else:
            v = (Us[i + 1] - Us[i - 1]) / (2.0 * dt)  # centered difference
        times.append(dt * i)
        ke.append(energy_quad(M, v).item())
        pe.append(energy_quad(A, Ui).item())

    ke = torch.tensor(ke); pe = torch.tensor(pe); E = ke + pe
    drift = ((E[-1] - E[0]) / E[0]).item()
    print(f"Energy: E0 = {E[0]:.6e}, E_final = {E[-1]:.6e}, relative drift = {drift:+.2e}")

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(times, ke, label=r"kinetic  $\frac{1}{2}\,v^\top M v$", color="#2980b9")
    plt.plot(times, pe, label=r"potential  $\frac{1}{2}\,u^\top A u$", color="#27ae60")
    plt.plot(times, E,  label="total", color="#c0392b", linewidth=2)
    plt.xlabel("time"); plt.ylabel("energy")
    plt.title(f"Wave energy (relative drift {drift:+.1e})")
    plt.grid(True, alpha=0.3); plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig("wave_energy.png", dpi=150)
    plt.close()
    print("Saved wave_energy.png")

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in tqdm(range(n), desc="Ground truth")]

    mesh_cpu = mesh.to('cpu')
    Us_cpu = [u.cpu() for u in Us]
    Us_gt_cpu = [u.cpu() for u in Us_gt]

    mesh_cpu.plot({
        "FEM solution":Us_cpu,
        "Analytical solution":Us_gt_cpu},
        save_path="wave.mp4",
        dt=dt,
        show_mesh=False,
        linewidth=0.1,
        linecolor='black')
    

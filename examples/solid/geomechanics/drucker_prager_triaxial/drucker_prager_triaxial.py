"""Drucker-Prager triaxial compression example.

This example is intentionally local/example-only.  It demonstrates how a
pressure-dependent geomechanics material can be written with the same
per-quadrature history-variable pattern used by TensorMesh's built-in
J2Plasticity assembler.

The internal TensorMesh convention is tension-positive stress.  For reporting,
this script also prints compression-positive axial stress and mean pressure,
which is the convention many geomechanics readers expect.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

# Allow running this example directly from the source tree.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tensormesh.assemble import ElementAssembler
from tensormesh.dataset.mesh import gen_cube


@dataclass(frozen=True)
class DruckerPragerParameters:
    """Small-strain Drucker-Prager material parameters.

    Parameters
    ----------
    E:
        Young's modulus in Pa.
    nu:
        Poisson's ratio.
    cohesion:
        Cohesion in Pa.
    friction_angle_deg:
        Mohr-Coulomb friction angle in degrees.  The Drucker-Prager cone is
        fitted to the triaxial-compression meridian.
    H:
        Linear isotropic hardening modulus in Pa.  The local example keeps a
        small positive value for a smooth demonstration after first yield.
    """

    E: float = 50.0e6
    nu: float = 0.30
    cohesion: float = 20.0e3
    friction_angle_deg: float = 30.0
    H: float = 1.0e6


class DruckerPragerPlasticity(ElementAssembler):
    """Example-only associated Drucker-Prager plasticity assembler.

    This class is deliberately kept inside the example.  It is not a public
    TensorMesh API.  The implementation follows the same high-level lifecycle
    as J2Plasticity:

    1. store per-quadrature history in ``self.history[etype]``;
    2. pass previous-step state through ``element_data`` during energy calls;
    3. call ``update_state(u)`` after each converged load step.

    Notes
    -----
    TensorMesh uses tension-positive stress.  With compression-positive mean
    pressure ``p = -tr(sigma) / 3``, the yield function is written internally as

        f = q + eta * I1 - (k + H * alpha) <= 0,

    where ``I1 = tr(sigma)`` and ``q = sqrt(3/2 s:s)``.  Because compression
    gives negative ``I1``, confinement increases the yield stress.
    """

    def __post_init__(self, params: DruckerPragerParameters | None = None):
        if params is None:
            params = DruckerPragerParameters()

        self.params = params
        self.E = float(params.E)
        self.nu = float(params.nu)
        self.cohesion = float(params.cohesion)
        self.friction_angle_deg = float(params.friction_angle_deg)
        self.H = float(params.H)

        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.bulk = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        phi = math.radians(self.friction_angle_deg)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        # Triaxial-compression meridian fit to Mohr-Coulomb:
        # q = M p + k, p compression-positive.
        # With TensorMesh tension-positive stress, p = -I1 / 3, so
        # f = q + (M/3) I1 - k.
        self.M = 6.0 * sin_phi / (3.0 - sin_phi)
        self.eta = self.M / 3.0
        self.k = 6.0 * self.cohesion * cos_phi / (3.0 - sin_phi)

        # Associated linear Drucker-Prager return denominator for q-based f.
        self.return_denominator = 3.0 * self.mu + 9.0 * self.bulk * self.eta**2 + self.H

        self.history: Dict[str, Dict[str, torch.Tensor]] = {}
        for etype, trans in self.transformation.items():
            n_elem = trans.n_elements
            n_quad = trans.n_quadrature
            eps_p = torch.zeros((n_elem, n_quad, 3, 3), device=self.device, dtype=self.dtype)
            alpha = torch.zeros((n_elem, n_quad), device=self.device, dtype=self.dtype)
            self.history[etype] = {"eps_p": eps_p, "alpha": alpha}

    @staticmethod
    def _small_strain_3d(graddisplacement: torch.Tensor) -> torch.Tensor:
        """Return the 3D small-strain tensor for 2D or 3D input gradients."""
        dim = graddisplacement.shape[-1]
        if dim == 2:
            eps_2d = 0.5 * (graddisplacement + graddisplacement.transpose(-1, -2))
            eps = torch.zeros((3, 3), device=graddisplacement.device, dtype=graddisplacement.dtype)
            eps[:2, :2] = eps_2d
            return eps
        return 0.5 * (graddisplacement + graddisplacement.transpose(-1, -2))

    def _elastic_stress(self, eps_e: torch.Tensor) -> torch.Tensor:
        """Tension-positive isotropic elastic stress from elastic strain."""
        eye = torch.eye(3, device=eps_e.device, dtype=eps_e.dtype)
        tr_eps = eps_e.diagonal(dim1=-2, dim2=-1).sum(-1)
        dev_eps = eps_e - (tr_eps[..., None, None] / 3.0) * eye
        return 2.0 * self.mu * dev_eps + self.bulk * tr_eps[..., None, None] * eye

    @staticmethod
    def _invariants(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return I1, deviatoric stress, and q for a tension-positive stress."""
        eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
        I1 = sigma.diagonal(dim1=-2, dim2=-1).sum(-1)
        s = sigma - (I1[..., None, None] / 3.0) * eye
        s_contract_s = (s * s).sum(dim=(-2, -1))
        q = torch.sqrt(torch.clamp(1.5 * s_contract_s, min=1.0e-30))
        return I1, s, q

    def element_energy(
        self,
        graddisplacement: torch.Tensor,
        eps_p_n: torch.Tensor,
        alpha_n: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithmic incremental potential density at one quadrature point."""
        eps = self._small_strain_3d(graddisplacement)
        eps_e_trial = eps - eps_p_n
        sigma_trial = self._elastic_stress(eps_e_trial)
        I1_trial, _, q_trial = self._invariants(sigma_trial)

        f_trial = q_trial + self.eta * I1_trial - (self.k + self.H * alpha_n)
        dgamma = torch.clamp(f_trial, min=0.0) / self.return_denominator

        tr_eps_e = eps_e_trial.diagonal(dim1=-2, dim2=-1).sum(-1)
        eye = torch.eye(3, device=eps.device, dtype=eps.dtype)
        dev_eps_e = eps_e_trial - (tr_eps_e / 3.0) * eye
        elastic_energy = 0.5 * self.bulk * tr_eps_e**2 + self.mu * (dev_eps_e * dev_eps_e).sum()

        return elastic_energy - 0.5 * self.return_denominator * dgamma**2

    def update_state(self, u_vec: torch.Tensor) -> None:
        """Commit per-quadrature state after a converged load step."""
        with torch.no_grad():
            for etype, trans in self.transformation.items():
                cells = trans.elements
                u_elem = u_vec[cells]
                grad_u = torch.einsum("bqkx,bku->bqux", trans.shape_grad, u_elem)

                dim = grad_u.shape[-1]
                if dim == 2:
                    eps = torch.zeros(grad_u.shape[:2] + (3, 3), device=u_vec.device, dtype=u_vec.dtype)
                    eps[..., :2, :2] = 0.5 * (grad_u + grad_u.transpose(-1, -2))
                else:
                    eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))

                hist = self.history[etype]
                eps_p_n = hist["eps_p"]
                alpha_n = hist["alpha"]

                eps_e_trial = eps - eps_p_n
                sigma_trial = self._elastic_stress(eps_e_trial)
                I1_trial, s_trial, q_trial = self._invariants(sigma_trial)
                f_trial = q_trial + self.eta * I1_trial - (self.k + self.H * alpha_n)
                dgamma = torch.clamp(f_trial, min=0.0) / self.return_denominator

                q_safe = torch.clamp(q_trial, min=1.0e-30)
                n_dev = 1.5 * s_trial / q_safe[..., None, None]
                eye = torch.eye(3, device=u_vec.device, dtype=u_vec.dtype)
                flow_dir = n_dev + self.eta * eye

                active = (f_trial > 0.0).to(dtype=u_vec.dtype)
                dgamma = dgamma * active

                hist["eps_p"] += dgamma[..., None, None] * flow_dir
                hist["alpha"] += dgamma

    def element_data_from_history(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return history in the element_data structure expected by energy()."""
        return {
            "eps_p_n": {etype: h["eps_p"] for etype, h in self.history.items()},
            "alpha_n": {etype: h["alpha"] for etype, h in self.history.items()},
        }

    def mean_alpha(self) -> torch.Tensor:
        """Return the mean committed plastic multiplier across all quadrature points."""
        values = [hist["alpha"].reshape(-1) for hist in self.history.values()]
        return torch.cat(values).mean()

    def max_alpha(self) -> torch.Tensor:
        """Return the maximum committed plastic multiplier."""
        values = [hist["alpha"].reshape(-1) for hist in self.history.values()]
        return torch.cat(values).max()

    def mean_stress(self, u_vec: torch.Tensor) -> torch.Tensor:
        """Average committed Cauchy stress over elements and quadrature points."""
        stresses = []
        with torch.no_grad():
            for etype, trans in self.transformation.items():
                u_elem = u_vec[trans.elements]
                grad_u = torch.einsum("bqkx,bku->bqux", trans.shape_grad, u_elem)
                dim = grad_u.shape[-1]
                if dim == 2:
                    eps = torch.zeros(grad_u.shape[:2] + (3, 3), device=u_vec.device, dtype=u_vec.dtype)
                    eps[..., :2, :2] = 0.5 * (grad_u + grad_u.transpose(-1, -2))
                else:
                    eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))
                sigma = self._elastic_stress(eps - self.history[etype]["eps_p"])
                stresses.append(sigma.reshape(-1, 3, 3))
        return torch.cat(stresses, dim=0).mean(dim=0)


def affine_displacement(points: torch.Tensor, eps_diag: torch.Tensor) -> torch.Tensor:
    """Apply a diagonal small-strain tensor as nodal displacement."""
    return points * eps_diag


def triaxial_strain_path(
    axial_strain: torch.Tensor,
    confinement_pressure: float,
    params: DruckerPragerParameters,
) -> torch.Tensor:
    """Return diagonal strain for a simple triaxial-compression driver.

    ``confinement_pressure`` is compression-positive.  It is converted into an
    initial isotropic compressive strain.  The axial-loading increment then uses
    the elastic uniaxial-stress lateral strain relation to approximately keep
    lateral stress increments small before yield.
    """
    K = params.E / (3.0 * (1.0 - 2.0 * params.nu))
    eps_iso = -confinement_pressure / (3.0 * K)
    eps_x = eps_iso - params.nu * axial_strain
    eps_y = eps_iso - params.nu * axial_strain
    eps_z = eps_iso + axial_strain
    return torch.stack((eps_x, eps_y, eps_z))


def elastic_trial_yield_value(
    eps_diag: torch.Tensor,
    params: DruckerPragerParameters,
) -> torch.Tensor:
    """Closed-form elastic trial yield value for the diagonal strain path."""
    E = params.E
    nu = params.nu
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    phi = math.radians(params.friction_angle_deg)
    M = 6.0 * math.sin(phi) / (3.0 - math.sin(phi))
    eta = M / 3.0
    k = 6.0 * params.cohesion * math.cos(phi) / (3.0 - math.sin(phi))

    eye = torch.eye(3, device=eps_diag.device, dtype=eps_diag.dtype)
    eps = torch.diag(eps_diag)
    tr_eps = eps.trace()
    dev_eps = eps - tr_eps / 3.0 * eye
    sigma = 2.0 * mu * dev_eps + K * tr_eps * eye
    I1 = sigma.trace()
    s = sigma - I1 / 3.0 * eye
    q = torch.sqrt(torch.clamp(1.5 * (s * s).sum(), min=1.0e-30))
    return q + eta * I1 - k


def first_positive_index(values: List[float]) -> int | None:
    """Return the first index where a sequence becomes positive."""
    for i, value in enumerate(values):
        if value > 0.0:
            return i
    return None


def run_case(
    confinement_pressure: float,
    params: DruckerPragerParameters,
    n_steps: int = 32,
    axial_strain_final: float = -0.025,
    chara_length: float = 0.75,
) -> Dict[str, List[float]]:
    """Run one displacement-controlled triaxial-compression case."""
    dtype = torch.float64
    device = torch.device("cpu")

    mesh = gen_cube(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        front=0.0,
        back=1.0,
        chara_length=chara_length,
    )
    mesh.points = mesh.points.to(device=device, dtype=dtype)

    model = DruckerPragerPlasticity.from_mesh(mesh, params=params)
    points = mesh.points

    result: Dict[str, List[float]] = {
        "axial_strain": [],
        "axial_stress_compression_kpa": [],
        "mean_pressure_compression_kpa": [],
        "alpha_max": [],
        "elastic_trial_f_kpa": [],
    }

    for axial_strain_value in torch.linspace(0.0, axial_strain_final, n_steps, device=device, dtype=dtype):
        eps_diag = triaxial_strain_path(axial_strain_value, confinement_pressure, params)
        u = affine_displacement(points, eps_diag).detach().clone().requires_grad_(True)

        energy = model.energy(
            point_data={"displacement": u},
            element_data=model.element_data_from_history(),
        )
        # Calling backward confirms the potential is differentiable, even though
        # this affine driver has no unconstrained degrees of freedom to optimize.
        if energy.requires_grad:
            energy.backward()

        model.update_state(u)
        sigma = model.mean_stress(u)
        p_comp = -sigma.trace() / 3.0
        sigma_axial_comp = -sigma[2, 2]
        f_elastic = elastic_trial_yield_value(eps_diag, params)

        result["axial_strain"].append(float(-axial_strain_value))
        result["axial_stress_compression_kpa"].append(float(sigma_axial_comp / 1.0e3))
        result["mean_pressure_compression_kpa"].append(float(p_comp / 1.0e3))
        result["alpha_max"].append(float(model.max_alpha()))
        result["elastic_trial_f_kpa"].append(float(f_elastic / 1.0e3))

    return result


def run_demo(
    n_steps: int = 32,
    make_plot: bool = True,
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Run low- and high-confinement cases and perform sanity checks."""
    params = DruckerPragerParameters()
    cases = {
        "p0 = 0 kPa": run_case(0.0, params, n_steps=n_steps),
        "p0 = 100 kPa": run_case(100.0e3, params, n_steps=n_steps),
    }

    low = cases["p0 = 0 kPa"]
    high = cases["p0 = 100 kPa"]

    low_yield = first_positive_index(low["elastic_trial_f_kpa"])
    high_yield = first_positive_index(high["elastic_trial_f_kpa"])

    low_alpha = torch.tensor(low["alpha_max"])
    high_alpha = torch.tensor(high["alpha_max"])
    low_monotonic = bool(torch.all(low_alpha[1:] + 1.0e-12 >= low_alpha[:-1]))
    high_monotonic = bool(torch.all(high_alpha[1:] + 1.0e-12 >= high_alpha[:-1]))

    sanity = {
        "low_confinement_yield_index": low_yield,
        "high_confinement_yield_index": high_yield,
        "higher_confinement_delays_yield": (
            low_yield is not None and high_yield is not None and high_yield > low_yield
        ),
        "plastic_strain_monotonic": low_monotonic and high_monotonic,
    }

    if make_plot:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "drucker_prager_triaxial.png"
        plot_results(cases, plot_path)
        sanity["plot_path"] = str(plot_path)

    return {"cases": cases, "sanity": sanity, "params": params}


def plot_results(cases: Dict[str, Dict[str, List[float]]], output_file: Path) -> None:
    """Write a compact stress-strain and plastic-strain figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for label, result in cases.items():
        axes[0].plot(
            [100.0 * x for x in result["axial_strain"]],
            result["axial_stress_compression_kpa"],
            marker="o",
            markersize=3,
            label=label,
        )
        axes[1].plot(
            [100.0 * x for x in result["axial_strain"]],
            result["alpha_max"],
            marker="o",
            markersize=3,
            label=label,
        )

    axes[0].set_xlabel("Axial compression strain [%]")
    axes[0].set_ylabel("Axial stress, compression positive [kPa]")
    axes[0].set_title("Drucker-Prager triaxial driver")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Axial compression strain [%]")
    axes[1].set_ylabel("Maximum plastic multiplier")
    axes[1].set_title("Committed plastic history")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32, help="Number of load steps per case.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run only the numerical driver and sanity checks.",
    )
    args = parser.parse_args()

    output = run_demo(n_steps=args.steps, make_plot=not args.no_plot)
    sanity = output["sanity"]

    print("Drucker-Prager triaxial compression example")
    print("Internal convention: stress is tension-positive.")
    print("Reported axial stress and mean pressure are compression-positive.")
    print()
    print("Sanity checks")
    for key, value in sanity.items():
        print(f"  {key}: {value}")

    if not sanity["higher_confinement_delays_yield"]:
        raise RuntimeError("Expected higher confinement to delay yield.")
    if not sanity["plastic_strain_monotonic"]:
        raise RuntimeError("Expected committed plastic history to be monotonic.")


if __name__ == "__main__":
    main()

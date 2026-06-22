"""Drucker-Prager triaxial compression example.

This example drives a pressure-dependent geomechanics material through a simple
triaxial-compression strain path using TensorMesh's public Drucker-Prager API:
the :class:`tensormesh.assemble.DruckerPragerPlasticity` assembler, the
:class:`tensormesh.material.FrictionalMaterial` container, and the constitutive
primitive :func:`tensormesh.functional.drucker_prager_yield_value`.

The internal TensorMesh convention is tension-positive stress.  For reporting,
this script also prints compression-positive axial stress and mean pressure,
which is the convention many geomechanics readers expect.  The assembler follows
the same per-quadrature history lifecycle as the built-in J2 plasticity model:
previous-step state is passed through ``element_data`` and ``update_state(u)`` is
called after each converged load step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

# Allow running this example directly from the source tree.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tensormesh.assemble import DruckerPragerPlasticity
from tensormesh.material import FrictionalMaterial
from tensormesh.functional import drucker_prager_yield_value
from tensormesh.dataset.mesh import gen_cube


def affine_displacement(points: torch.Tensor, eps_diag: torch.Tensor) -> torch.Tensor:
    """Apply a diagonal small-strain tensor as nodal displacement."""
    return points * eps_diag


def triaxial_strain_path(
    axial_strain: torch.Tensor,
    confinement_pressure: float,
    material: FrictionalMaterial,
) -> torch.Tensor:
    """Return diagonal strain for a simple triaxial-compression driver.

    ``confinement_pressure`` is compression-positive.  It is converted into an
    initial isotropic compressive strain.  The axial-loading increment then uses
    the elastic uniaxial-stress lateral strain relation to approximately keep
    lateral stress increments small before yield.
    """
    K = material.E / (3.0 * (1.0 - 2.0 * material.nu))
    eps_iso = -confinement_pressure / (3.0 * K)
    eps_x = eps_iso - material.nu * axial_strain
    eps_y = eps_iso - material.nu * axial_strain
    eps_z = eps_iso + axial_strain
    return torch.stack((eps_x, eps_y, eps_z))


def elastic_trial_yield_value(
    eps_diag: torch.Tensor,
    material: FrictionalMaterial,
) -> torch.Tensor:
    """Elastic trial Drucker-Prager yield value for a diagonal strain path.

    Uses the public constitutive primitive
    :func:`tensormesh.functional.drucker_prager_yield_value` with zero plastic
    history, so this reports the same yield value the assembler sees on a trial
    elastic step.
    """
    zeros = torch.zeros((3, 3), device=eps_diag.device, dtype=eps_diag.dtype)
    return drucker_prager_yield_value(
        torch.diag(eps_diag),
        zeros,
        torch.zeros((), device=eps_diag.device, dtype=eps_diag.dtype),
        E=material.E,
        nu=material.nu,
        cohesion=material.cohesion,
        friction_angle=material.friction_angle,
        H=material.H,
    )


def first_positive_index(values: List[float]) -> int | None:
    """Return the first index where a sequence becomes positive."""
    for i, value in enumerate(values):
        if value > 0.0:
            return i
    return None


def run_case(
    confinement_pressure: float,
    material: FrictionalMaterial,
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

    model = DruckerPragerPlasticity.from_mesh(mesh, material=material)
    points = mesh.points

    result: Dict[str, List[float]] = {
        "axial_strain": [],
        "axial_stress_compression_kpa": [],
        "mean_pressure_compression_kpa": [],
        "alpha_max": [],
        "elastic_trial_f_kpa": [],
    }

    for axial_strain_value in torch.linspace(0.0, axial_strain_final, n_steps, device=device, dtype=dtype):
        eps_diag = triaxial_strain_path(axial_strain_value, confinement_pressure, material)
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
        f_elastic = elastic_trial_yield_value(eps_diag, material)

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
    # Associated Drucker-Prager soil (dilatancy_angle defaults to the friction
    # angle), matching the parameters this example has always used.
    material = FrictionalMaterial(
        name="ExampleSoil",
        E=50.0e6,
        nu=0.30,
        cohesion=20.0e3,
        friction_angle=30.0,
        H=1.0e6,
    )
    cases = {
        "p0 = 0 kPa": run_case(0.0, material, n_steps=n_steps),
        "p0 = 100 kPa": run_case(100.0e3, material, n_steps=n_steps),
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

    return {"cases": cases, "sanity": sanity, "material": material}


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

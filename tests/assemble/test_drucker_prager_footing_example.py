"""Tests for the Drucker-Prager footing geomechanics example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_module(name: str, *relative_parts: str):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root.joinpath("examples", "solid", "geomechanics", *relative_parts)
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so the example's dataclass annotations (evaluated under
    # ``from __future__ import annotations``) can resolve the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_footing():
    return _load_module(
        "drucker_prager_footing_example",
        "drucker_prager_footing",
        "drucker_prager_footing.py",
    )


def _is_non_decreasing(values, tol: float = 1.0e-9) -> bool:
    return all(values[i + 1] >= values[i] - tol for i in range(len(values) - 1))


def test_footing_settles_and_plasticity_grows_and_localizes():
    example = _load_footing()

    result = example.run_demo(
        make_plot=False,
        n_steps=6,
        chara_length=0.60,
        final_pressure=300.0e3,
        cohesion=10.0e3,
        dtype=torch.float64,
    )

    assert result["n_nodes"] > 0
    assert result["n_steps"] == 6

    # Settles downward and develops plasticity.
    assert result["final_footing_settlement_m"] > 0.0
    assert result["final_max_settlement_m"] > 0.0
    assert result["final_max_alpha"] > 0.0

    # Committed plastic history and settlement grow monotonically with load.
    assert _is_non_decreasing(result["max_alphas"], tol=1.0e-15)
    assert _is_non_decreasing(result["footing_settlements_m"])

    # Plasticity localizes under the centered footing, not at the far boundary.
    footing_width = 1.2
    assert abs(result["plastic_centroid_x"]) < footing_width
    assert -2.5 < result["plastic_centroid_y"] < 0.0


def test_low_load_is_near_elastic():
    footing = _load_footing()
    elastic = _load_module("elastic_footing_example", "elastic_footing", "elastic_footing.py")

    pressure = 60.0e3
    # High cohesion at a low load keeps the response below yield.
    dp = footing.run_demo(
        make_plot=False,
        n_steps=4,
        chara_length=0.60,
        final_pressure=pressure,
        cohesion=80.0e3,
        dtype=torch.float64,
    )

    assert dp["final_footing_settlement_m"] > 0.0
    assert dp["final_max_alpha"] < 1.0e-6  # essentially elastic

    # Below yield, the incremental potential reduces to linear elasticity, so the
    # settlement should match the elastic footing example closely.
    el = elastic.run_demo(
        make_plot=False,
        chara_length=0.60,
        footing_pressure=pressure,
        dtype=torch.float64,
    )
    rel_diff = abs(dp["final_footing_settlement_m"] - el["footing_settlement_m"]) / el[
        "footing_settlement_m"
    ]
    assert rel_diff < 0.10


def test_higher_cohesion_delays_plasticity():
    example = _load_footing()

    low_cohesion = example.run_demo(
        make_plot=False,
        n_steps=5,
        chara_length=0.60,
        final_pressure=300.0e3,
        cohesion=10.0e3,
        dtype=torch.float64,
    )
    high_cohesion = example.run_demo(
        make_plot=False,
        n_steps=5,
        chara_length=0.60,
        final_pressure=300.0e3,
        cohesion=40.0e3,
        dtype=torch.float64,
    )

    assert high_cohesion["final_max_alpha"] < low_cohesion["final_max_alpha"]

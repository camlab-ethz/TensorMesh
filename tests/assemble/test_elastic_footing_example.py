"""Tests for the elastic footing geomechanics example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "examples"
        / "solid"
        / "geomechanics"
        / "elastic_footing"
        / "elastic_footing.py"
    )
    spec = importlib.util.spec_from_file_location("elastic_footing_example", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so the example's dataclass annotations (evaluated under
    # ``from __future__ import annotations``) can resolve the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_elastic_footing_balances_load_and_settles_downward():
    example = _load_example_module()

    result = example.run_demo(
        make_plot=False,
        chara_length=0.55,
        footing_pressure=80.0e3,
        dtype=torch.float64,
        device="cpu",
    )

    assert result["n_nodes"] > 0
    assert result["n_loaded_nodes"] >= 2
    assert result["n_free_dofs"] > 0

    assert result["total_vertical_load_N_per_m"] < 0.0
    assert result["vertical_reaction_N_per_m"] > 0.0

    assert result["footing_settlement_m"] > 0.0
    assert result["max_settlement_m"] > 0.0
    assert result["min_vertical_displacement_m"] < 0.0

    assert result["load_balance_relative_error"] < 1.0e-4


def test_elastic_footing_response_scales_linearly_with_load():
    example = _load_example_module()

    low = example.run_demo(
        make_plot=False,
        chara_length=0.60,
        footing_pressure=60.0e3,
        dtype=torch.float64,
        device="cpu",
    )
    high = example.run_demo(
        make_plot=False,
        chara_length=0.60,
        footing_pressure=120.0e3,
        dtype=torch.float64,
        device="cpu",
    )

    ratio = high["footing_settlement_m"] / low["footing_settlement_m"]

    assert 1.95 < ratio < 2.05

"""Smoke tests for the example-only Drucker-Prager triaxial driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[2]
    example_path = (
        repo_root
        / "examples"
        / "solid"
        / "geomechanics"
        / "drucker_prager_triaxial"
        / "drucker_prager_triaxial.py"
    )
    spec = importlib.util.spec_from_file_location("drucker_prager_triaxial", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_drucker_prager_triaxial_sanity_checks():
    module = _load_example_module()
    output = module.run_demo(n_steps=12, make_plot=False)
    sanity = output["sanity"]

    assert sanity["higher_confinement_delays_yield"]
    assert sanity["plastic_strain_monotonic"]

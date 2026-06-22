"""Tests for the table-backed FrictionalMaterial container."""

import pytest

from tensormesh.material import FrictionalMaterial


def test_from_preset_returns_expected_fields():
    soil = FrictionalMaterial.from_preset("DenseSand")
    assert soil.name == "DenseSand"
    assert soil.E == 50.0e6
    assert soil.nu == 0.30
    assert soil.cohesion == 5000.0
    assert soil.friction_angle == 35.0
    assert soil.dilatancy_angle == 5.0
    assert soil.rho == 1900.0
    assert "illustrative" in soil.description.lower()


def test_preset_names_includes_expected():
    names = FrictionalMaterial.preset_names()
    assert "DenseSand" in names
    assert "LooseSand" in names
    assert "SoftClay" in names
    assert len(names) >= 4


def test_unknown_preset_raises_with_clear_message():
    with pytest.raises(KeyError) as exc:
        FrictionalMaterial.from_preset("Unobtainium")
    assert "Unobtainium" in str(exc.value)
    assert "Available presets" in str(exc.value)


def test_direct_constructor_minimal_no_name_required():
    soil = FrictionalMaterial(E=50.0e6, nu=0.30, cohesion=5.0e3, friction_angle=35.0)
    assert soil.name == "Custom"
    assert soil.dilatancy_angle is None  # associated flow by default
    assert soil.H == 0.0
    assert soil.rho is None


def test_direct_constructor_full():
    soil = FrictionalMaterial(
        E=50.0e6, nu=0.30, cohesion=5.0e3, friction_angle=35.0,
        dilatancy_angle=5.0, H=25.0e3, rho=1900.0, name="MySoil",
    )
    assert soil.name == "MySoil"
    assert soil.dilatancy_angle == 5.0
    assert soil.H == 25.0e3
    mu, lam = soil.lame_params
    assert mu > 0.0 and lam > 0.0

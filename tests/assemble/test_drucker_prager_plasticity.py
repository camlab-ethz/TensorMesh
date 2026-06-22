"""Tests for the built-in DruckerPragerPlasticity assembler (public API)."""

import torch

from tensormesh.assemble import DruckerPragerPlasticity
from tensormesh.material import FrictionalMaterial
from tensormesh.dataset.mesh import gen_rectangle


def _build_model(dtype=torch.float64):
    mesh = gen_rectangle(chara_length=0.6, left=-2.0, right=2.0, bottom=-2.0, top=0.0)
    mesh.points = mesh.points.to(dtype=dtype)
    soil = FrictionalMaterial(E=50.0e6, nu=0.30, cohesion=20.0e3, friction_angle=30.0, H=1.0e6)
    return DruckerPragerPlasticity.from_mesh(mesh, material=soil), mesh


def _simple_shear(points, gamma):
    u = torch.zeros_like(points)
    u[:, 0] = gamma * points[:, 1]
    return u.detach().clone().requires_grad_(True)


def test_from_mesh_with_material_initializes_history():
    model, _ = _build_model()
    assert len(model.history) > 0
    for hist in model.history.values():
        assert "eps_p" in hist and "alpha" in hist
        assert hist["eps_p"].shape[-2:] == (3, 3)
        assert hist["alpha"].shape[0] == hist["eps_p"].shape[0]


def test_history_tensors_dtype_and_device():
    model, mesh = _build_model()
    for hist in model.history.values():
        assert hist["eps_p"].dtype == torch.float64
        assert hist["alpha"].dtype == torch.float64
        assert hist["eps_p"].device == mesh.points.device


def test_scalar_construction_matches_material():
    mesh = gen_rectangle(chara_length=0.8, left=-1.0, right=1.0, bottom=-1.0, top=0.0)
    mesh.points = mesh.points.to(dtype=torch.float64)
    model = DruckerPragerPlasticity.from_mesh(
        mesh, E=50.0e6, nu=0.30, cohesion=5.0e3, friction_angle=35.0,
        dilatancy_angle=35.0, H=25.0e3,
    )
    assert model.friction_angle == 35.0
    assert model.cohesion == 5.0e3
    assert len(model.history) > 0


def test_energy_below_yield_is_finite_and_differentiable():
    model, mesh = _build_model()
    u = (mesh.points * torch.tensor([0.0, -1.0e-4], dtype=torch.float64)).detach().clone().requires_grad_(True)
    energy = model.energy(point_data={"displacement": u}, element_data=model.element_data_from_history())
    assert torch.isfinite(energy)
    energy.backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()


def test_update_state_does_not_decrease_alpha_and_yields():
    model, mesh = _build_model()
    points = mesh.points
    prev = float(model.max_alpha())
    for step in range(1, 5):
        u = _simple_shear(points, 0.01 * step)
        model.energy(point_data={"displacement": u}, element_data=model.element_data_from_history())
        model.update_state(u)
        current = float(model.max_alpha())
        assert current >= prev - 1.0e-15
        prev = current
    assert prev > 0.0  # plasticity developed under monotonic shear


def test_confinement_reduces_plasticity():
    points_model, mesh = _build_model()
    points = mesh.points

    # Pure shear (deviatoric, no confinement).
    shear = _build_model()[0]
    u_shear = _simple_shear(points, 0.03)
    shear.energy(point_data={"displacement": u_shear}, element_data=shear.element_data_from_history())
    shear.update_state(u_shear)
    alpha_shear = float(shear.max_alpha())

    # Same shear plus isotropic compression (confinement).
    confined = _build_model()[0]
    u_conf = u_shear.detach() + points * torch.tensor([-0.01, -0.01], dtype=torch.float64)
    u_conf = u_conf.detach().clone().requires_grad_(True)
    confined.energy(point_data={"displacement": u_conf}, element_data=confined.element_data_from_history())
    confined.update_state(u_conf)
    alpha_conf = float(confined.max_alpha())

    assert alpha_shear > 0.0
    assert alpha_conf < alpha_shear  # confinement raises capacity, reduces plasticity

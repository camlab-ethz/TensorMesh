"""The constant pressure mode lies in the kernel of the free momentum rows.

For a Taylor-Hood form :math:`(B^\\top 1)_i = -\\int \\nabla\\cdot v_i =
-\\oint v_i\\cdot n`, which vanishes exactly for every velocity basis
function that is zero on the boundary — the identity that makes the
pressure determined only up to a constant. A solve does **not** catch a
violation (the system is still consistent and the fields look right);
only probing :math:`K\\,(0, 1)` directly does. Two shipped defects broke
it silently: a boundary mask that missed curved-boundary nodes (their
rows are not free) and an order-2 mesh whose edge nodes were not
reordered (wrong geometry map).
"""
import os
import tempfile

import pytest
import torch

from tensormesh import Field, Mesh, MixedElementAssembler


class Stokes(MixedElementAssembler):
    fields = [Field(trial="u", test="v", order=2, components=2),
              Field(trial="p", test="q", order=1)]

    def forward(self, gradu, p, gradv, q):
        return (gradu * gradv).sum() - p * gradv.diagonal().sum() - q * gradu.diagonal().sum()


def _free_momentum_kernel_norm(mesh, topological=False):
    asm = Stokes.from_mesh(mesh.double())
    lay = asm.layout
    K = asm()
    Kc = K @ lay.cat(u=0.0, p=1.0)
    if topological:
        fixed_u = lay.boundary_mask("u")                         # dofmap-carried field
    else:
        fixed_u = lay.dof_mask("u", mesh.boundary_mask)          # node-carried field
    free = ~fixed_u & ~lay.dof_mask("p")
    return float(Kc[free].norm())


@pytest.mark.parametrize("make", [
    pytest.param(lambda: Mesh.gen_rectangle(chara_length=0.1, element_type="tri", order=2), id="rect-tri6"),
    pytest.param(lambda: Mesh.gen_rectangle(chara_length=0.1, element_type="quad", order=2), id="rect-quad9"),
    pytest.param(lambda: Mesh.gen_circle(chara_length=0.06, cx=.5, cy=.5, r=.5, order=2), id="circle-curved"),
    pytest.param(lambda: Mesh.gen_hollow_circle(chara_length=0.15, element_type="tri", order=2), id="annulus-curved"),
    pytest.param(lambda: Mesh.gen_hollow_rectangle(chara_length=0.08, element_type="tri", order=2), id="rect-with-hole"),
])
def test_constant_pressure_mode_is_in_the_kernel(make):
    assert _free_momentum_kernel_norm(make()) < 1e-10


def test_constant_pressure_mode_topological_p2_on_p1_mesh():
    mesh = Mesh.gen_circle(chara_length=0.06, cx=.5, cy=.5, r=.5)      # linear, curved boundary
    assert _free_momentum_kernel_norm(mesh, topological=True) < 1e-10


def test_kernel_survives_vtu_roundtrip_with_reorder():
    mesh = Mesh.gen_circle(chara_length=0.06, cx=.5, cy=.5, r=.5, order=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "circle2.vtu")
        mesh.save(path)
        again = Mesh.read(path, reorder=True)
    assert _free_momentum_kernel_norm(again) < 1e-10

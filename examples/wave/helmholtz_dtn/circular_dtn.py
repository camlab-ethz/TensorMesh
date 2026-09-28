# %%
"""Circular Helmholtz scattering with a Fourier--Hankel DtN boundary.


Edit the constants below and run this file. The three cases share the same
TensorMesh variational assembly but impose a Dirichlet obstacle, a Neumann
obstacle, or a penetrable circular interface.

The file is intentionally self-contained and follows the TensorMesh gallery
order: parameters, variational form, mesh, DtN map, solve, exact field, and
plotting. TensorMesh performs finite-element quadrature, global scatter, and
strong boundary condensation; SciPy and Gmsh only provide exact data and mesh
geometry.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import h1vp, hankel1, hankel1e, jv, jvp
import torch

from tensormesh import Condenser, ElementAssembler, FacetAssembler, Mesh
from tensormesh.sparse import SparseMatrix


# -----------------------------------------------------------------------------
# User parameters
# Material j has k_j = omega * sqrt(b_j / a_j). Keeping a_j explicit matters:
# it also weights the interface flux and the exterior DtN contribution.
OMEGA = 4.0
A0, B0 = 1.0, 1.0
A1, B1 = 0.75, 2.25
R_C = 0.4
R_GAMMA = 1.2
INCIDENT_ORDER = 0
NUM_MODES = 50
MESH_SIZE = 0.10
CASES = ("dirichlet", "neumann", "transmission")
OUTPUT_DIR = Path(__file__).parents[3] / "docs/source/_static/wave/helmholtz_dtn"

REAL = torch.float64
COMPLEX = torch.complex128


# -----------------------------------------------------------------------------
# TensorMesh variational and boundary-trace assemblers
class HelmholtzAssembler(ElementAssembler):
    """Evaluate the weighted Helmholtz element integrand.

    TensorMesh supplies basis values and gradients, quadrature, geometry
    transformations, and the scatter from element entries to global COO.
    """

    def forward(self, gradu, gradv, u, v, a, b, omega_sq):
        return a * (gradu @ gradv) - omega_sq * b * u * v


class FourierTrace(FacetAssembler):
    """Integrate traces against ``exp(-i m theta)`` using real channels.

    Facet geometry remains real. Cosine and negative-sine channels are assembled
    separately and combined into complex Fourier moments after facet assembly.
    """

    def __post_init__(self, modes):
        self.modes = modes.to(dtype=REAL, device=self.device)

    def forward(self, v, x):
        phase = self.modes * torch.atan2(x[1], x[0])
        channels = torch.stack((torch.cos(phase), -torch.sin(phase)), dim=-1)
        return v[:, None, None] * channels[None, :, :]


# -----------------------------------------------------------------------------
# Volume and nonlocal boundary assembly
def assemble_volume(mesh, omega, a, b):
    """Use TensorMesh for complete element quadrature and scatter."""

    # Geometry stays float64; only the assembled algebra is promoted to complex128.
    assembler = HelmholtzAssembler.from_mesh(mesh, quadrature_order=4)
    assembler.type(COMPLEX)
    omega_sq = torch.full_like(a, omega * omega, dtype=COMPLEX)
    return assembler(
        points=mesh.points,
        element_data={
            "a": a.to(COMPLEX),
            "b": b.to(COMPLEX),
            "omega_sq": omega_sq,
        },
    )


def modal_trace(mesh, boundary_mask, modes):
    """Return boundary nodes and the TensorMesh modal moment matrix Psi."""

    # Retain global node order for scattering the dense DtN boundary block.
    nodes = torch.nonzero(boundary_mask, as_tuple=False).flatten()
    elements = mesh.elements()
    if isinstance(elements, torch.Tensor):
        elements = {mesh.default_element_type: elements}
    # TensorMesh extracts boundary facets and supplies trace bases/quadrature.
    assembler = FourierTrace.from_elements(
        mesh.points,
        elements,
        boundary_mask,
        quadrature_order=7,
        dtype=REAL,
        project="reduce",
        modes=modes,
    )
    raw = assembler(mesh.points).reshape(mesh.n_points, modes.numel(), 2)
    # Reconstruct exp(-i*m*theta) only after the real facet transformation.
    psi = torch.complex(raw[..., 0], raw[..., 1]).to(COMPLEX)
    return nodes, psi[nodes]


def combine_sparse(*terms):
    """Merge volume and DtN entries even when their COO layouts differ."""

    rows = torch.cat([matrix.row_indices for _, matrix in terms]).to(torch.long)
    columns = torch.cat([matrix.col_indices for _, matrix in terms]).to(torch.long)
    values = torch.cat(
        [coefficient * matrix.values.to(COMPLEX) for coefficient, matrix in terms]
    )
    coo = torch.sparse_coo_tensor(
        torch.stack((rows, columns)), values, terms[0][1].shape, dtype=COMPLEX
    )
    return SparseMatrix.from_sparse_coo(coo)


# -----------------------------------------------------------------------------
# Circular Fourier--Hankel DtN operator
def hankel_symbol(mode, wavenumber, radius):
    """Stable value of ``k H_n'(kR) / H_n(kR)``."""

    order = abs(int(mode))
    argument = wavenumber * radius
    ratio = hankel1e(order + 1, argument) / hankel1e(order, argument)
    return order / radius - wavenumber * ratio


def circular_dtn(mesh, boundary_mask, wavenumber, radius, num_modes):
    """Assemble the nonlocal DtN block in four visible steps."""

    # 1. Fourier modes and Hankel symbols.
    modes = torch.arange(-num_modes, num_modes + 1, dtype=REAL)
    tau = torch.tensor(
        [hankel_symbol(mode, wavenumber, radius) for mode in modes],
        dtype=COMPLEX,
    )
    # 2. TensorMesh facet integration.
    nodes, psi = modal_trace(mesh, boundary_mask, modes)
    # 3. Low-rank modal product. psi.T is deliberately not psi.conj().T.
    block = psi.conj() @ torch.diag(tau / (2.0 * math.pi * radius)) @ psi.T
    # 4. Scatter the complete boundary block into global COO coordinates.
    count = nodes.numel()
    matrix = SparseMatrix(
        block.reshape(-1),
        nodes.repeat_interleave(count),
        nodes.repeat(count),
        (mesh.n_points, mesh.n_points),
    )
    return matrix, nodes, psi, modes.to(torch.long), tau


def solve_general(matrix, rhs):
    """Solve the complex indefinite system and report its relative residual."""

    # Use TensorMesh's public sparse API and select LU explicitly because the
    # Helmholtz-plus-DtN matrix is complex, indefinite, and generally non-Hermitian.
    solution = matrix.solve(rhs, backend="scipy", method="lu")
    residual = matrix @ solution - rhs
    return solution, float(torch.linalg.norm(residual) / torch.linalg.norm(rhs))


# -----------------------------------------------------------------------------
# Geometry and analytical reference field
def build_mesh(case, circle_radius, dtn_radius, mesh_size):
    """Build an annulus or a conforming two-material disk, then load TensorMesh."""

    # A single Gmsh branch handles both obstacle and interface topology;
    # Mesh.from_file then hands all finite-element work to TensorMesh.
    import gmsh

    with tempfile.TemporaryDirectory(prefix="tensormesh_circle_") as directory:
        filename = Path(directory) / "circle.msh"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("circular_dtn")
            outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, dtn_radius, dtn_radius)
            inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, circle_radius, circle_radius)
            if case == "transmission":
                surfaces, _ = gmsh.model.occ.fragment([(2, outer)], [(2, inner)])
            else:
                surfaces, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(
                2, [tag for dimension, tag in surfaces if dimension == 2]
            )
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(2)
            gmsh.write(str(filename))
            mesh = Mesh.from_file(str(filename), reorder=False)
        finally:
            gmsh.finalize()
    mesh.points = mesh.points.to(REAL)
    return mesh


def exact_coefficients(case, order, k0, k1, a0, a1, radius):
    """Return the outgoing coefficient and optional interior coefficient."""

    z0, z1 = k0 * radius, k1 * radius
    # Obstacle coefficients follow directly from the scalar wall condition.
    if case == "dirichlet":
        return complex(-jv(order, z0) / hankel1(order, z0)), None
    if case == "neumann":
        return complex(-jvp(order, z0, 1) / h1vp(order, z0, 1)), None
    # The 2x2 system enforces continuity of u and a*partial_r(u) at R_C.
    matrix = np.array(
        [
            [hankel1(order, z0), -jv(order, z1)],
            [a0 * k0 * h1vp(order, z0, 1), -a1 * k1 * jvp(order, z1, 1)],
        ],
        dtype=np.complex128,
    )
    rhs = -np.array([jv(order, z0), a0 * k0 * jvp(order, z0, 1)])
    outgoing, interior = np.linalg.solve(matrix, rhs)
    return complex(outgoing), complex(interior)


def exact_field(points, case, order, k0, k1, a0, a1, radius):
    """Evaluate the exact total field at mesh nodes."""

    outgoing, interior = exact_coefficients(
        case, order, k0, k1, a0, a1, radius
    )
    radial = np.linalg.norm(points, axis=1)
    angular = np.exp(1j * order * np.arctan2(points[:, 1], points[:, 0]))
    safe_radial = np.where(radial < radius, radius, radial)
    outside = (
        jv(order, k0 * safe_radial)
        + outgoing * hankel1(order, k0 * safe_radial)
    ) * angular
    if case != "transmission":
        return outside
    inside = interior * jv(order, k1 * radial) * angular
    return np.where(radial <= radius + 1.0e-12, inside, outside)


# -----------------------------------------------------------------------------
# End-to-end circular solves
def solve_circle(
    case,
    *,
    omega=OMEGA,
    a0=A0,
    b0=B0,
    a1=A1,
    b1=B1,
    circle_radius=R_C,
    dtn_radius=R_GAMMA,
    incident_order=INCIDENT_ORDER,
    num_modes=NUM_MODES,
    mesh_size=MESH_SIZE,
):
    """Solve one case and return ``mesh, numerical, exact, metrics``."""

    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    if min(omega, a0, b0, a1, b1, mesh_size) <= 0.0:
        raise ValueError("omega, material coefficients, and mesh_size must be positive")
    if not 0.0 < circle_radius < dtn_radius:
        raise ValueError("radii must satisfy 0 < circle_radius < dtn_radius")
    if num_modes < abs(incident_order):
        raise ValueError("num_modes must be at least abs(incident_order)")
    k0 = omega * math.sqrt(b0 / a0)
    k1 = omega * math.sqrt(b1 / a1)
    mesh = build_mesh(case, circle_radius, dtn_radius, mesh_size)
    # P1 coefficients are attached per triangle; no local stiffness loop is used.
    triangles = mesh.elements("triangle")
    centroids = mesh.points[triangles].mean(dim=1)
    interior = torch.linalg.norm(centroids, dim=1) < circle_radius
    if case != "transmission":
        interior[:] = False
    a = torch.where(interior, a1, a0).to(REAL)
    b = torch.where(interior, b1, b0).to(REAL)

    # Identify the physical and artificial circles from mesh coordinates.
    tolerance = 0.05 * mesh_size
    radial = torch.linalg.norm(mesh.points, dim=1)
    outer = torch.abs(radial - dtn_radius) < tolerance
    inner = torch.abs(radial - circle_radius) < tolerance
    # Weak operator: volume contribution minus a0 times the exterior DtN map.
    volume = assemble_volume(mesh, omega, a, b)
    dtn, nodes, psi, modes, tau = circular_dtn(
        mesh, outer, k0, dtn_radius, num_modes
    )
    matrix = combine_sparse((1.0, volume), (-a0, dtn))

    # g=(partial_r-T)u_inc is nonzero only in the selected incident order.
    incident = int(torch.nonzero(modes == incident_order).item())
    argument = k0 * dtn_radius
    datum = k0 * jvp(incident_order, argument, 1)
    datum -= tau[incident].item() * jv(incident_order, argument)
    coefficients = torch.zeros(modes.numel(), dtype=COMPLEX)
    coefficients[incident] = datum
    rhs = torch.zeros(mesh.n_points, dtype=COMPLEX)
    rhs[nodes] = a0 * (psi.conj() @ coefficients)

    # Dirichlet is imposed strongly; Neumann is the natural weak boundary.
    if case == "dirichlet":
        condenser = Condenser(
            inner,
            dirichlet_value=torch.zeros(int(inner.sum()), dtype=COMPLEX),
        )
        reduced_matrix, reduced_rhs = condenser(matrix, rhs)
        reduced_solution, residual = solve_general(reduced_matrix, reduced_rhs)
        solution = condenser.recover(reduced_solution)
    else:
        solution, residual = solve_general(matrix, rhs)

    # The exact field is used only for verification, never for FE assembly.
    points = mesh.points.cpu().numpy()
    exact = exact_field(
        points, case, incident_order, k0, k1, a0, a1, circle_radius
    )
    exact_tensor = torch.as_tensor(exact, dtype=COMPLEX)
    relative_l2 = float(
        torch.linalg.norm(solution - exact_tensor) / torch.linalg.norm(exact_tensor)
    )
    outgoing, inside_coefficient = exact_coefficients(
        case, incident_order, k0, k1, a0, a1, circle_radius
    )
    metrics = {
        "residual": residual,
        "relative_l2": relative_l2,
        "k0": k0,
        "k1": k1,
        "outgoing": outgoing,
        "interior": inside_coefficient,
        "matrix_dtype": matrix.values.dtype,
    }
    return mesh, solution, exact, metrics


# -----------------------------------------------------------------------------
# Visualization and script entry point
def plot_case(mesh, solution, exact, case):
    """Plot numerical, exact, and error fields with TensorMesh."""

    numerical = solution.detach().cpu()
    reference = torch.as_tensor(exact, dtype=COMPLEX)
    fields = {
        "Re(u_h)": numerical.real,
        "Re(u_exact)": reference.real,
        "error": abs(numerical - reference),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh.plot(
        fields,
        save_path=str(OUTPUT_DIR / f"circular_{case}.png"),
        show_mesh=False,
    )

    # TensorMesh intentionally leaves figures open for interactive use.
    import matplotlib.pyplot as plt

    plt.close("all")


def main():
    for case in CASES:
        mesh, solution, exact, metrics = solve_circle(case)
        print(f"\n{case}: k0={metrics['k0']:.6g}, k1={metrics['k1']:.6g}")
        print(f"  residual={metrics['residual']:.3e}")
        print(f"  relative nodal L2 error={metrics['relative_l2']:.3e}")
        print(f"  outgoing coefficient={metrics['outgoing']}")
        if metrics["interior"] is not None:
            print(f"  interior coefficient={metrics['interior']}")
        plot_case(mesh, solution, exact, case)


if __name__ == "__main__":
    main()

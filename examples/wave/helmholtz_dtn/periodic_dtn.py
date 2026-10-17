"""Periodic Helmholtz scattering with Rayleigh DtN boundaries.

Edit the constants below and run this file. The three cases are a two-port
finite slab, a Dirichlet-backed layer, and a Neumann-backed layer.

The file is intentionally self-contained and follows the TensorMesh gallery
order: parameters, variational form, periodic mesh, Rayleigh DtN map, Bloch
reduction, exact amplitudes, and plotting. TensorMesh performs finite-element
quadrature, global scatter, Bloch reduction, and strong boundary condensation.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from tensormesh import BlochReducer, Condenser, ElementAssembler, FacetAssembler, Mesh
from tensormesh.sparse import SparseMatrix


# -----------------------------------------------------------------------------
# User parameters
# Material j has k_j = omega * sqrt(b_j / a_j). Keeping a_j explicit matters:
# it also weights the interface flux and the exterior DtN contribution.
OMEGA = 3.0
A0, B0 = 1.0, 1.0
A1, B1 = 0.8, 2.25
PERIOD = 1.0
HEIGHT = 0.55
ALPHA = 0.35
AMPLITUDE = 1.0 + 0.0j
NUM_MODES = 3
MESH_SIZE = 0.06
PADDING = 0.70
CASES = ("slab", "dirichlet", "neumann")
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


class RayleighTrace(FacetAssembler):
    """Integrate traces against ``exp(-i alpha_n x)`` using real channels.

    Facet geometry remains real. Cosine and negative-sine channels are assembled
    separately and combined into complex Rayleigh moments after facet assembly.
    """

    def __post_init__(self, alpha_n):
        self.alpha_n = alpha_n.to(dtype=REAL, device=self.device)

    def forward(self, v, x):
        phase = self.alpha_n * x[0]
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


def modal_trace(mesh, boundary_mask, alpha_n):
    """Return boundary nodes and the TensorMesh modal moment matrix Psi."""

    # Retain global node order for scattering the dense DtN boundary block.
    nodes = torch.nonzero(boundary_mask, as_tuple=False).flatten()
    elements = mesh.elements()
    if isinstance(elements, torch.Tensor):
        elements = {mesh.default_element_type: elements}
    # TensorMesh extracts boundary facets and supplies trace bases/quadrature.
    assembler = RayleighTrace.from_elements(
        mesh.points,
        elements,
        boundary_mask,
        quadrature_order=7,
        dtype=REAL,
        project="reduce",
        alpha_n=alpha_n,
    )
    raw = assembler(mesh.points).reshape(mesh.n_points, alpha_n.numel(), 2)
    # Reconstruct exp(-i*alpha_n*x) only after the real facet transformation.
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
# Rayleigh DtN operator
def outgoing_beta(wavenumber, alpha_n):
    """Square root with positive real or positive imaginary outgoing branch."""

    # Propagating modes use beta>0; evanescent modes use Im(beta)>0.
    beta = torch.sqrt(
        torch.tensor(wavenumber * wavenumber, dtype=COMPLEX)
        - alpha_n.to(COMPLEX) ** 2
    )
    beta = torch.where(beta.imag < 0.0, -beta, beta)
    return torch.where((abs(beta.imag) < 1.0e-14) & (beta.real < 0.0), -beta, beta)


def rayleigh_dtn(mesh, boundary_mask, wavenumber, alpha, period, num_modes):
    """Assemble one Rayleigh DtN block in four visible steps."""

    # 1. Horizontal and vertical modal wavenumbers.
    orders = torch.arange(-num_modes, num_modes + 1, dtype=REAL)
    alpha_n = alpha + 2.0 * math.pi * orders / period
    beta_n = outgoing_beta(wavenumber, alpha_n)
    # 2. TensorMesh facet integration.
    nodes, psi = modal_trace(mesh, boundary_mask, alpha_n)
    # 3. Low-rank modal product. psi.T is deliberately not psi.conj().T.
    block = psi.conj() @ torch.diag(1j * beta_n / period) @ psi.T
    # 4. Scatter the complete boundary block into global COO coordinates.
    count = nodes.numel()
    matrix = SparseMatrix(
        block.reshape(-1),
        nodes.repeat_interleave(count),
        nodes.repeat(count),
        (mesh.n_points, mesh.n_points),
    )
    return matrix, nodes, psi, alpha_n, beta_n


def solve_general(matrix, rhs):
    """Solve the complex indefinite system and report its relative residual."""

    # Use TensorMesh's public sparse API and select LU explicitly because the
    # Helmholtz-plus-DtN matrix is complex, indefinite, and generally non-Hermitian.
    solution = matrix.solve(rhs, backend="scipy", method="lu")
    residual = matrix @ solution - rhs
    return solution, float(torch.linalg.norm(residual) / torch.linalg.norm(rhs))


# -----------------------------------------------------------------------------
# Periodic layered geometry and analytical reference field
def build_mesh(period, levels, mesh_size):
    """Build a conforming layered cell with matching left/right nodes."""

    import gmsh

    with tempfile.TemporaryDirectory(prefix="tensormesh_periodic_") as directory:
        filename = Path(directory) / "mesh.msh"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("periodic_dtn")
            # Fragmentation creates conforming interfaces at all y-levels.
            rectangles = [
                (
                    2,
                    gmsh.model.occ.addRectangle(
                        0.0, low, 0.0, period, high - low
                    ),
                )
                for low, high in zip(levels[:-1], levels[1:])
            ]
            surfaces, _ = gmsh.model.occ.fragment(
                [rectangles[0]], rectangles[1:]
            )
            gmsh.model.occ.synchronize()
            tags = [tag for dimension, tag in surfaces if dimension == 2]
            gmsh.model.addPhysicalGroup(2, tags)
            curves = {
                tag
                for dimension, tag in gmsh.model.getBoundary(
                    [(2, tag) for tag in tags], combined=False, oriented=False
                )
                if dimension == 1
            }
            # setPeriodic needs matching side segments in identical y-order.
            sides = {"left": [], "right": []}
            tolerance = 1.0e-8 * max(1.0, period)
            for tag in curves:
                xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(1, tag)
                if abs(xmin) < tolerance and abs(xmax) < tolerance:
                    sides["left"].append(tag)
                elif abs(xmin - period) < tolerance and abs(xmax - period) < tolerance:
                    sides["right"].append(tag)
            for side in sides.values():
                side.sort(key=lambda tag: gmsh.model.occ.getCenterOfMass(1, tag)[1])
            # Affine map from a left-side point to its right-side partner.
            translation = [
                1.0, 0.0, 0.0, period,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
            gmsh.model.mesh.setPeriodic(
                1, sides["right"], sides["left"], translation
            )
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(2)
            gmsh.write(str(filename))
            mesh = Mesh.from_file(str(filename), reorder=False)
        finally:
            gmsh.finalize()
    mesh.points = mesh.points.to(REAL)
    return mesh


def exact_solution(points, case, omega, alpha, height, bottom, top, amplitude,
                   a0, b0, a1, b1):
    """Evaluate the exact layered field and return its port amplitudes."""

    k0 = omega * math.sqrt(b0 / a0)
    k1 = omega * math.sqrt(b1 / a1)
    # beta and gamma are the exterior and layer vertical wavenumbers.
    beta = complex(np.sqrt(k0 * k0 - alpha * alpha + 0.0j))
    gamma = complex(np.sqrt(k1 * k1 - alpha * alpha + 0.0j))
    if beta.imag < 0.0:
        beta = -beta
    if gamma.imag < 0.0:
        gamma = -gamma
    incident_h = amplitude * np.exp(-1j * beta * (height - top))
    reflected_h = np.exp(1j * beta * (height - top))

    if case == "slab":
        # The four unknown amplitudes enforce field and weighted-flux continuity.
        down_h, up_h = np.exp(-1j * gamma * height), np.exp(1j * gamma * height)
        transmitted_0 = np.exp(-1j * beta * (0.0 - bottom))
        system = np.array(
            [
                [reflected_h, -down_h, -up_h, 0.0],
                [1j * a0 * beta * reflected_h,
                 1j * a1 * gamma * down_h,
                 -1j * a1 * gamma * up_h, 0.0],
                [0.0, 1.0, 1.0, -transmitted_0],
                [0.0, -1j * a1 * gamma, 1j * a1 * gamma,
                 1j * a0 * beta * transmitted_0],
            ],
            dtype=np.complex128,
        )
        rhs = np.array(
            [-incident_h, 1j * a0 * beta * incident_h, 0.0, 0.0]
        )
        reflection, down, up, transmission = np.linalg.solve(system, rhs)
    else:
        # A sine/cosine layer basis enforces the D/N wall condition exactly.
        if case == "dirichlet":
            shape = np.sin(gamma * height)
            derivative = gamma * np.cos(gamma * height)
        else:
            shape = np.cos(gamma * height)
            derivative = -gamma * np.sin(gamma * height)
        system = np.array(
            [
                [reflected_h, -shape],
                [1j * a0 * beta * reflected_h, -a1 * derivative],
            ],
            dtype=np.complex128,
        )
        rhs = np.array([-incident_h, 1j * a0 * beta * incident_h])
        reflection, standing = np.linalg.solve(system, rhs)
        transmission = None

    x, y = points[:, 0], points[:, 1]
    bloch = np.exp(1j * alpha * x)
    upper = bloch * (
        amplitude * np.exp(-1j * beta * (y - top))
        + reflection * np.exp(1j * beta * (y - top))
    )
    if case == "slab":
        layer = bloch * (down * np.exp(-1j * gamma * y) + up * np.exp(1j * gamma * y))
        lower = bloch * transmission * np.exp(-1j * beta * (y - bottom))
        field = np.where(y >= height, upper, np.where(y <= 0.0, lower, layer))
    else:
        shape_y = np.sin(gamma * y) if case == "dirichlet" else np.cos(gamma * y)
        field = np.where(y >= height, upper, bloch * standing * shape_y)
    return np.asarray(field), complex(reflection), (
        None if transmission is None else complex(transmission)
    )


def reduce_rhs(reducer, rhs, wavevector):
    """Form ``T^H rhs`` with the conjugated Bloch phase."""

    # master_dof with the conjugate phase implements the test-side T^H.
    phase = torch.exp(1j * (reducer.node_R @ wavevector).to(COMPLEX))
    reduced = torch.zeros(reducer.n_reduced_dof, dtype=COMPLEX)
    reduced.index_add_(0, reducer.master_dof, phase.conj() * rhs)
    return reduced


def scattering_coefficients(
    top_data, bottom_data, solution, period, incident, amplitude, exterior_a
):
    """Return modal amplitudes ``r, t`` and power fractions ``R, T``."""

    top_modes = top_data[2].T @ solution[top_data[1]] / period
    reflected = top_modes.clone()
    reflected[incident] -= amplitude
    reflection = complex(reflected[incident].item())

    incident_weight = torch.real(exterior_a * top_data[4][incident])

    def power(amplitudes, beta):
        weights = torch.real(exterior_a * beta)
        propagating = torch.where(
            weights > 0.0,
            weights / incident_weight * abs(amplitudes / amplitude) ** 2,
            torch.zeros_like(weights),
        )
        return float(propagating.sum())

    reflectance = power(reflected, top_data[4])
    transmission = transmittance = None
    modal_vectors = [reflected]
    if bottom_data is not None:
        transmitted = bottom_data[2].T @ solution[bottom_data[1]] / period
        transmission = complex(transmitted[incident].item())
        transmittance = power(transmitted, bottom_data[4])
        modal_vectors.append(transmitted)

    # N=0 is a valid one-mode truncation, with no non-incident mode to measure.
    leakage = 0.0
    if reflected.numel() > 1:
        other = torch.arange(reflected.numel()) != incident
        leakage = max(
            float(torch.max(abs(vector[other] / amplitude)))
            for vector in modal_vectors
        )
    return reflection, transmission, reflectance, transmittance, leakage


def bloch_trace_residual(mesh, solution, period, alpha, tolerance):
    """Measure ``u(L,y)-exp(i alpha L)u(0,y)`` on paired side nodes."""

    left = torch.nonzero(abs(mesh.points[:, 0]) < tolerance).flatten()
    right = torch.nonzero(abs(mesh.points[:, 0] - period) < tolerance).flatten()
    left = left[torch.argsort(mesh.points[left, 1])]
    right = right[torch.argsort(mesh.points[right, 1])]
    phase = torch.exp(1j * torch.tensor(alpha * period, dtype=REAL)).to(COMPLEX)
    return float(torch.max(abs(solution[right] - phase * solution[left])))


# -----------------------------------------------------------------------------
# End-to-end periodic solves
def solve_periodic(
    case,
    *,
    omega=OMEGA,
    a0=A0,
    b0=B0,
    a1=A1,
    b1=B1,
    period=PERIOD,
    height=HEIGHT,
    alpha=ALPHA,
    amplitude=AMPLITUDE,
    num_modes=NUM_MODES,
    mesh_size=MESH_SIZE,
    padding=PADDING,
):
    """Solve one case and return ``mesh, numerical, exact, metrics``."""

    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    if min(omega, a0, b0, a1, b1, period, height, mesh_size, padding) <= 0.0:
        raise ValueError("frequency, materials, geometry, and mesh_size must be positive")
    if num_modes < 0:
        raise ValueError("num_modes must be non-negative")
    if abs(amplitude) == 0.0:
        raise ValueError("amplitude must be nonzero")
    k0 = omega * math.sqrt(b0 / a0)
    k1 = omega * math.sqrt(b1 / a1)
    if abs(alpha) >= k0:
        raise ValueError("the incident mode must propagate: abs(alpha) < k0")
    retained_alpha = (
        alpha + 2.0 * math.pi * order / period
        for order in range(-num_modes, num_modes + 1)
    )
    if any(
        math.isclose(abs(alpha_n), k0, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for alpha_n in retained_alpha
    ):
        raise ValueError("retained Rayleigh modes must avoid a Wood anomaly")
    if math.isclose(abs(alpha), k1, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("the layer mode must avoid grazing incidence")
    bottom = -padding if case == "slab" else 0.0
    top = height + padding
    levels = (bottom, 0.0, height, top) if case == "slab" else (0.0, height, top)
    mesh = build_mesh(period, levels, mesh_size)
    # Attach piecewise-constant material data to each conforming triangle.
    triangles = mesh.elements("triangle")
    y_centroid = mesh.points[triangles, 1].mean(dim=1)
    in_layer = (y_centroid > 0.0) & (y_centroid < height)
    a = torch.where(in_layer, a1, a0).to(REAL)
    b = torch.where(in_layer, b1, b0).to(REAL)
    tolerance = 0.05 * mesh_size
    top_mask = abs(mesh.points[:, 1] - top) < tolerance
    bottom_mask = abs(mesh.points[:, 1] - bottom) < tolerance
    # The slab has two radiation ports; wall cases have only the top port.
    volume = assemble_volume(mesh, omega, a, b)
    top_data = rayleigh_dtn(mesh, top_mask, k0, alpha, period, num_modes)
    terms = [(1.0, volume), (-a0, top_data[0])]
    bottom_data = None
    if case == "slab":
        bottom_data = rayleigh_dtn(
            mesh, bottom_mask, k0, alpha, period, num_modes
        )
        terms.append((-a0, bottom_data[0]))
    matrix = combine_sparse(*terms)

    # The incident load g=-2*i*beta_0*A occupies the zeroth mode only.
    incident = num_modes
    coefficients = torch.zeros(2 * num_modes + 1, dtype=COMPLEX)
    coefficients[incident] = -2j * top_data[4][incident] * amplitude
    rhs = torch.zeros(mesh.n_points, dtype=COMPLEX)
    rhs[top_data[1]] = a0 * (top_data[2].conj() @ coefficients)

    # Constraint order matters: assemble volume + DtN first, apply Bloch
    # test/trial phases second, then condense wall master DOFs when required.
    reducer = BlochReducer(mesh.points.cpu().numpy(), [[period, 0.0]], sign=+1)
    wavevector = torch.tensor([alpha, 0.0], dtype=REAL)
    reduced_matrix = reducer.reduce(matrix, wavevector)
    reduced_rhs = reduce_rhs(reducer, rhs, wavevector)
    if case == "dirichlet":
        # Map the full bottom trace, including both corners, to unique masters.
        master_mask = torch.zeros(reducer.n_reduced_dof, dtype=torch.bool)
        master_mask[torch.unique(reducer.master_dof[bottom_mask])] = True
        condenser = Condenser(
            master_mask,
            dirichlet_value=torch.zeros(int(master_mask.sum()), dtype=COMPLEX),
        )
        inner_matrix, inner_rhs = condenser(reduced_matrix, reduced_rhs)
        inner_solution, residual = solve_general(inner_matrix, inner_rhs)
        reduced_solution = condenser.recover(inner_solution)
    else:
        reduced_solution, residual = solve_general(reduced_matrix, reduced_rhs)
    # Recover in reverse order: Condenser above, then BlochReducer here.
    solution = reducer.recover(reduced_solution, wavevector)

    # Lowercase values are amplitudes; uppercase values are power fractions.
    reflection, transmission, reflectance, transmittance, leakage = (
        scattering_coefficients(
            top_data,
            bottom_data,
            solution,
            period,
            incident,
            amplitude,
            a0,
        )
    )

    # The flat-layer field checks FEM nodal values and port amplitudes.
    points = mesh.points.cpu().numpy()
    exact, exact_r, exact_t = exact_solution(
        points, case, omega, alpha, height, bottom, top, amplitude,
        a0, b0, a1, b1,
    )
    exact_tensor = torch.as_tensor(exact, dtype=COMPLEX)
    relative_l2 = float(
        torch.linalg.norm(solution - exact_tensor) / torch.linalg.norm(exact_tensor)
    )
    # Check the quasi-periodic trace independently of reducer internals.
    bloch_residual = bloch_trace_residual(
        mesh, solution, period, alpha, tolerance
    )
    metrics = {
        "residual": residual,
        "relative_l2": relative_l2,
        "k0": k0,
        "k1": k1,
        "reflection": reflection,
        "transmission": transmission,
        "reflectance": reflectance,
        "transmittance": transmittance,
        "exact_reflection": exact_r,
        "exact_transmission": exact_t,
        "energy_residual": abs(reflectance + (transmittance or 0.0) - 1.0),
        "bloch_residual": bloch_residual,
        "leakage": leakage,
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
        save_path=str(OUTPUT_DIR / f"periodic_{case}.png"),
        show_mesh=False,
    )

    # TensorMesh intentionally leaves figures open for interactive use.
    import matplotlib.pyplot as plt

    plt.close("all")


def main():
    for case in CASES:
        mesh, solution, exact, metrics = solve_periodic(case)
        print(f"\n{case}: k0={metrics['k0']:.6g}, k1={metrics['k1']:.6g}")
        print(f"  residual={metrics['residual']:.3e}")
        print(f"  relative nodal L2 error={metrics['relative_l2']:.3e}")
        print(f"  r={metrics['reflection']}, R={metrics['reflectance']:.8f}")
        if metrics["transmission"] is not None:
            print(f"  t={metrics['transmission']}, T={metrics['transmittance']:.8f}")
        print(f"  energy residual={metrics['energy_residual']:.3e}")
        print(f"  Bloch residual={metrics['bloch_residual']:.3e}")
        plot_case(mesh, solution, exact, case)


if __name__ == "__main__":
    main()

"""Harmonic extension solver for boundary condition data.

For each BC dimension and each of {alpha, beta, g}, solves a Laplace equation
-Δφ = 0 with Dirichlet BCs determined by the BC type of each boundary segment:

  - Dirichlet segment: alpha=1, beta=0, g=g_value
  - Neumann segment:   alpha=0, beta=1, g=g_value
  - Robin segment:     alpha=alpha_value, beta=1, g=g_value

Ported from fenicsx-main/generate_dataset.py:176-234.
"""

import torch
import numpy as np
from typing import List, Dict

from tensormesh import LaplaceElementAssembler, Mesh, Condenser
from boundary import SegmentBCs


def solve_harmonic_extensions(mesh: Mesh,
                               segments: List[SegmentBCs],
                               ndims: int = 1,
                               ) -> List[Dict[str, np.ndarray]]:
    """Solve harmonic extension problems for BC data.

    For each dimension d in range(ndims) and each key in {alpha, beta, g}:
      Solve -Δφ = 0 with all-Dirichlet BCs on the boundary.
      The Dirichlet values depend on the BC type of each segment.

    Args:
        mesh: TensorMesh Mesh object
        segments: list of SegmentBCs with indices and values filled
        ndims: number of BC dimensions (1 for Poisson, 2 for elasticity)

    Returns:
        extensions: list of dicts, one per dimension.
            extensions[d]['alpha']: [n_points] numpy array
            extensions[d]['beta']:  [n_points] numpy array
            extensions[d]['g']:     [n_points] numpy array
    """
    n_points = mesh.points.shape[0]
    dtype = mesh.points.dtype
    device = mesh.points.device

    # Pre-assemble stiffness matrix (shared across all extension solves)
    assembler = LaplaceElementAssembler.from_mesh(mesh)
    K = assembler(mesh.points)

    # Zero RHS (Laplace equation)
    f_zero = torch.zeros(n_points, dtype=dtype, device=device)

    extensions = []

    for d in range(ndims):
        ext_d = {}
        for key in ['alpha', 'beta', 'g']:
            # Build Dirichlet BCs for this extension problem
            dirichlet_mask = mesh.boundary_mask.clone()
            dirichlet_values = torch.zeros(n_points, dtype=dtype, device=device)

            for seg in segments:
                if seg.indices is None or len(seg.indices) == 0:
                    continue
                idx = torch.tensor(seg.indices, dtype=torch.long, device=device)
                dim_bc = seg.dims[d]

                if dim_bc.type == 'dirichlet':
                    bc_map = {
                        'g': dim_bc.values['g'] if dim_bc.values else np.zeros(len(seg.indices)),
                        'alpha': np.ones(len(seg.indices)),
                        'beta': np.zeros(len(seg.indices)),
                    }
                elif dim_bc.type == 'neumann':
                    bc_map = {
                        'g': dim_bc.values['g'] if dim_bc.values else np.zeros(len(seg.indices)),
                        'alpha': np.zeros(len(seg.indices)),
                        'beta': np.ones(len(seg.indices)),
                    }
                elif dim_bc.type == 'robin':
                    bc_map = {
                        'g': dim_bc.values['g'] if dim_bc.values else np.zeros(len(seg.indices)),
                        'alpha': dim_bc.values.get('alpha', np.zeros(len(seg.indices))) if dim_bc.values else np.zeros(len(seg.indices)),
                        'beta': np.ones(len(seg.indices)),
                    }
                else:
                    raise ValueError(f"Unknown BC type: {dim_bc.type}")

                dirichlet_values[idx] = torch.tensor(
                    bc_map[key], dtype=dtype, device=device)

            # Solve: -Δφ = 0 with these Dirichlet BCs
            condenser = Condenser(dirichlet_mask, dirichlet_values)
            K_inner, f_inner = condenser(K, f_zero)
            u_inner = K_inner.solve(f_inner)
            u = condenser.recover(u_inner)

            ext_d[key] = u.detach().cpu().numpy()

        extensions.append(ext_d)

    return extensions

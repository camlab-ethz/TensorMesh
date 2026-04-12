"""Main dataset generation script.

Generates Poisson and elasticity datasets matching the format from
arXiv 2602.04923 (Mousavi, Mishra, De Lorenzis).

Usage:
    python generate_dataset.py --problem poisson --shape circle --id bc1 --size 256
    python generate_dataset.py --problem poisson --shape boomerang --id bc4 --size 10
    python generate_dataset.py --problem elasticity --shape circlehollow --id m1 --size 10
"""

import argparse
import logging
import os
import time

import numpy as np
import torch

from mesh_gen import create_mesh
from boundary import (
    BCGenerator, Dirichlet, Neumann, Robin, RandomBCTypes,
    NeumannHomogenous, DirichletHomogenous,
    get_centered_radial_cosine,
    ConstantFunctionGenerator, RandomRadialSines,
)
from bc_segments import draw_valid_bcs, build_bc_masks_and_values
from poisson_solver import solve_poisson
from elasticity_solver import solve_hyperelasticity
from harmonic_extension import solve_harmonic_extensions
from sdf import compute_sdf_at_nodes, compute_sdf_on_grid
from geometry import (get_boomerang_boundary, circle_boundary, polygon_boundary,
                      get_circlehollow_holes, get_squarehollow_holes,
                      get_boomcircletri_holes)
from output import initialize_file, store_sample


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

def _random_bc_types(modes_d=8, modes_n=6, modes_r=(6, 3), C=(0., 0.), R=1.0):
    return RandomBCTypes(
        ranges={'dirichlet': (1.0, 4.0), 'neumann': (2.0, 10.0),
                'robin': ((2.0, 10.0), (0.2, 0.6))},
        modes={'dirichlet': modes_d, 'neumann': modes_n, 'robin': modes_r},
        C=C, R=R,
    )


def get_poisson_config(shape: str, identifier: str):
    """Get Poisson dataset configuration."""
    if shape in ['circle', 'square']:
        C, R, modes_bc1, modes_d = (0., 0.), 1.0, 12, 8
    elif shape == 'boomerang':
        C, R, modes_bc1, modes_d = (0., -0.375), 0.625, 6, 6
    else:
        raise ValueError(f"Shape '{shape}' not supported for Poisson")

    base = {'ndims': 1, 'variable_bcs': True, 'center': C, 'variables': {'source': 1}}

    if identifier == 'bc1':
        return {**base,
                'bc_generator_exterior': BCGenerator(ndims=1, dists=[
                    Dirichlet(rng=(2.0, 10.0), modes=modes_bc1, C=C, R=R)]),
                'source_generator': ConstantFunctionGenerator(
                    func=get_centered_radial_cosine(ord=2, scale=20.0)),
                'minimum_non_dirichlet_length': 0.0,
                'maximum_non_dirichlet_length': 1.0}
    elif identifier in ['bc4', 'bc5']:
        config = {**base,
                  'bc_generator_exterior': BCGenerator(ndims=1, dists=[
                      Dirichlet(rng=(1.0, 4.0), modes=modes_d, C=C, R=R),
                      _random_bc_types(modes_d, 6 if shape != 'boomerang' else 4,
                                       (6, 3) if shape != 'boomerang' else (4, 3), C, R),
                      _random_bc_types(modes_d, 6 if shape != 'boomerang' else 4,
                                       (6, 3) if shape != 'boomerang' else (4, 3), C, R),
                      _random_bc_types(modes_d, 6 if shape != 'boomerang' else 4,
                                       (6, 3) if shape != 'boomerang' else (4, 3), C, R),
                  ]),
                  'minimum_non_dirichlet_length': 0.2,
                  'maximum_non_dirichlet_length': 0.51}
        if identifier == 'bc4':
            config['source_generator'] = ConstantFunctionGenerator(
                func=get_centered_radial_cosine(ord=2, scale=20.0))
        else:
            config['source_generator'] = RandomRadialSines(modes=2, ord=2, scale=20.0)
        return config
    else:
        raise ValueError(f"Unknown Poisson identifier: {identifier}")


def get_elasticity_config(shape: str, identifier: str):
    """Get elasticity dataset configuration."""
    if shape not in ['circlehollow', 'squarehollow', 'boomcircletri']:
        raise ValueError(f"Shape '{shape}' not supported for elasticity")

    # Hole names per shape
    hole_names = {
        'circlehollow': ['purple', 'brown'],
        'squarehollow': ['purple', 'brown'],
        'boomcircletri': ['purple', 'pink', 'brown'],
    }
    holes = hole_names[shape]

    if identifier == 'm1':
        E, nu = 200.0e3, 0.3
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 - 2 * nu) * (1 + nu))
        char_stress, disp_rng, trac_rng, load_steps = 100, (0.1e-3, 0.5e-3), (4, 40), 2
    elif identifier == 'm2':
        E, nu = 10.0e3, 0.3
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 - 2 * nu) * (1 + nu))
        char_stress, disp_rng, trac_rng, load_steps = 10, (0.4e-3, 2e-3), (0.4, 4), 4
    elif identifier == 'm3':
        mu, lam = 10.0, 1000.0
        char_stress, disp_rng, trac_rng, load_steps = 50, (0.1, 0.4), (0.5, 3), 25
    else:
        raise ValueError(f"Unknown elasticity identifier: {identifier}")

    # Non-dimensionalize
    disp_rng_nd = [d / 1 for d in disp_rng]  # char_length=1
    trac_rng_nd = [t / char_stress for t in trac_rng]

    bc_gen_exterior = BCGenerator(ndims=2, dists=[
        NeumannHomogenous(),
        Dirichlet(rng=disp_rng_nd, modes=2),
        Neumann(rng=trac_rng_nd, modes=2),
        Neumann(rng=trac_rng_nd, modes=2),
    ])

    bc_gen_holes = {
        name: BCGenerator(ndims=2, dists=[
            DirichletHomogenous() if name == 'brown' else NeumannHomogenous()])
        for name in holes
    }

    return {
        'ndims': 2,
        'variable_bcs': True,
        'center': (0., 0.),
        'variables': {'strain': 4, 'cauchystress': 4},
        'bc_generator_exterior': bc_gen_exterior,
        'bc_generator_holes': bc_gen_holes,
        'minimum_non_dirichlet_length': 0.0,
        'maximum_non_dirichlet_length': 1.0,
        'lame_mu': mu,
        'lame_lambda': lam,
        'characteristic_stress': char_stress,
        'load_steps': load_steps,
    }


def get_config(problem: str, shape: str, identifier: str):
    if problem == 'poisson':
        return get_poisson_config(shape, identifier)
    elif problem == 'elasticity':
        return get_elasticity_config(shape, identifier)
    else:
        raise ValueError(f"Unknown problem: {problem}")


# ---------------------------------------------------------------------------
# Boundary curve points for SDF (cached)
# ---------------------------------------------------------------------------

def get_boundary_curve_points(shape: str, n_points: int = 1000) -> np.ndarray:
    """Get dense boundary curve points for SDF computation."""
    if shape == 'circle':
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])
    elif shape == 'square':
        pts = []
        n_side = n_points // 4
        for x in np.linspace(-1, 1, n_side, endpoint=False):
            pts.append([x, -1])
        for y in np.linspace(-1, 1, n_side, endpoint=False):
            pts.append([1, y])
        for x in np.linspace(1, -1, n_side, endpoint=False):
            pts.append([x, 1])
        for y in np.linspace(1, -1, n_side, endpoint=False):
            pts.append([-1, y])
        return np.array(pts)
    elif shape == 'boomerang':
        return get_boomerang_boundary(n_points)
    elif shape == 'circlehollow':
        return circle_boundary(0., 0., 1.0, n_points)
    elif shape == 'squarehollow':
        return polygon_boundary([(-1, -1), (1, -1), (1, 1), (-1, 1)], n_points // 4)
    elif shape == 'boomcircletri':
        return get_boomerang_boundary(n_points)
    else:
        raise NotImplementedError(f"Boundary curve for '{shape}' not implemented")


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_poisson_sample(mesh, config, boundary_curve_pts):
    """Generate one Poisson sample with full output data."""
    ndims = config['ndims']
    center = config.get('center', (0., 0.))
    points_np = mesh.points.detach().cpu().numpy()

    # Draw valid BCs
    shape_bcs = draw_valid_bcs(
        bc_generator=config['bc_generator_exterior'],
        min_non_dirichlet_length=config.get('minimum_non_dirichlet_length', 0.0),
        max_non_dirichlet_length=config.get('maximum_non_dirichlet_length', 1.0),
        ndims=ndims,
    )

    # Build masks and values
    bc_data = build_bc_masks_and_values(
        mesh_points=mesh.points, boundary_mask=mesh.boundary_mask,
        segments=shape_bcs, ndims=ndims, center=center,
    )

    # Draw source function
    source_func = config['source_generator'].draw()

    # Solve Poisson
    u = solve_poisson(
        mesh=mesh, source_func=source_func,
        boundary_masks={k: bc_data[k] for k in ['dirichlet_mask', 'neumann_mask', 'robin_mask']},
        boundary_values={k: bc_data[k] for k in ['dirichlet_values', 'neumann_values',
                                                   'robin_g_values', 'robin_alpha_values']},
    )

    # Source at nodes
    source_values = source_func(points_np)

    # Harmonic extensions
    segments = bc_data['segments']
    extensions = solve_harmonic_extensions(mesh, segments, ndims=ndims)

    # Collect BC info
    bc_info = _collect_bc_info(segments, ndims=ndims)

    return {
        'solution': u.detach().cpu().numpy(),
        'variables': {'source': source_values},
        'boundary': bc_info,
        'extensions': extensions,
        'coordinates': points_np,
    }


def generate_elasticity_sample(mesh, config):
    """Generate one elasticity (hyperelasticity) sample."""
    ndims = config['ndims']
    center = config.get('center', (0., 0.))
    points_np = mesh.points.detach().cpu().numpy()

    # Draw valid BCs for exterior
    shape_bcs = draw_valid_bcs(
        bc_generator=config['bc_generator_exterior'],
        min_non_dirichlet_length=config.get('minimum_non_dirichlet_length', 0.0),
        max_non_dirichlet_length=config.get('maximum_non_dirichlet_length', 1.0),
        ndims=ndims,
    )

    # Assign exterior nodes to segments
    bc_data = build_bc_masks_and_values(
        mesh_points=mesh.points, boundary_mask=mesh.boundary_mask,
        segments=shape_bcs, ndims=ndims, center=center,
    )
    segments = bc_data['segments']

    # Draw BCs for holes (if any)
    hole_segments = {}
    bc_gen_holes = config.get('bc_generator_holes', None)
    if bc_gen_holes:
        for hole_name, hole_gen in bc_gen_holes.items():
            hole_bcs = hole_gen.draw()
            # For holes, we don't do angle-based assignment — each hole has one segment
            # covering all its boundary nodes. We need to identify hole boundary nodes.
            # For now, assign based on angle from (0,0) — works for simple cases
            # TODO: improve for complex multi-hole geometries
            for seg in hole_bcs:
                seg.indices = np.array([], dtype=np.int64)  # placeholder
                for d in range(ndims):
                    seg.dims[d].values = {}
            hole_segments[hole_name] = hole_bcs

    # Solve hyperelasticity
    mu_nd = config['lame_mu'] / config['characteristic_stress']
    lam_nd = config['lame_lambda'] / config['characteristic_stress']

    u, strain, cauchy_stress = solve_hyperelasticity(
        mesh=mesh,
        mu_nd=mu_nd,
        lam_nd=lam_nd,
        segments_exterior=segments,
        segments_holes=hole_segments if hole_segments else None,
        load_steps=config['load_steps'],
        ndims=ndims,
        verbose=False,
    )

    # Evaluate BC values at assigned nodes for storage
    from bc_segments import evaluate_bc_values
    evaluate_bc_values(segments, points_np, ndims)

    # Harmonic extensions (2D: 6 Poisson solves)
    all_segs = list(segments)
    for hole_segs in hole_segments.values():
        all_segs.extend(hole_segs)
    extensions = solve_harmonic_extensions(mesh, all_segs, ndims=ndims)

    # Collect BC info per dimension
    bc_info = _collect_bc_info_nd(all_segs, ndims=ndims)

    return {
        'solution': u,
        'variables': {'strain': strain, 'cauchystress': cauchy_stress},
        'boundary': bc_info,
        'extensions': extensions,
        'coordinates': points_np,
    }


def _collect_bc_info_nd(segments, ndims=2):
    """Collect BC info for multi-dimensional problems (per dimension 0)."""
    # For the HDF5 format, boundaries are stored per-dimension
    # For now, collect dimension 0 (simplification — full version stores per dim)
    dir_idx, dir_g = [], []
    neu_idx, neu_g = [], []
    rob_idx, rob_g, rob_a = [], [], []
    for seg in segments:
        if seg.indices is None or len(seg.indices) == 0:
            continue
        for d in range(ndims):
            bc = seg.dims[d]
            if bc.type == 'dirichlet' and bc.values and 'g' in bc.values:
                dir_idx.append(seg.indices)
                dir_g.append(bc.values['g'])
            elif bc.type == 'neumann' and bc.values and 'g' in bc.values:
                neu_idx.append(seg.indices)
                neu_g.append(bc.values['g'])
    cat = lambda arrs, dt=np.float64: np.concatenate(arrs) if arrs else np.array([], dtype=dt)
    return {
        'dirichlet': {'indices': cat(dir_idx, np.int32), 'g': cat(dir_g)},
        'neumann': {'indices': cat(neu_idx, np.int32), 'g': cat(neu_g)},
        'robin': {'indices': np.array([], dtype=np.int32), 'g': np.array([]), 'alpha': np.array([])},
    }


def _collect_bc_info(segments, ndims=1):
    dir_idx, dir_g = [], []
    neu_idx, neu_g = [], []
    rob_idx, rob_g, rob_a = [], [], []
    for seg in segments:
        if seg.indices is None or len(seg.indices) == 0:
            continue
        bc = seg.dims[0]
        if bc.type == 'dirichlet' and bc.values:
            dir_idx.append(seg.indices)
            dir_g.append(bc.values['g'])
        elif bc.type == 'neumann' and bc.values:
            neu_idx.append(seg.indices)
            neu_g.append(bc.values['g'])
        elif bc.type == 'robin' and bc.values:
            rob_idx.append(seg.indices)
            rob_g.append(bc.values['g'])
            rob_a.append(bc.values['alpha'])
    cat = lambda arrs, dt=np.float64: np.concatenate(arrs) if arrs else np.array([], dtype=dt)
    return {
        'dirichlet': {'indices': cat(dir_idx, np.int32), 'g': cat(dir_g)},
        'neumann': {'indices': cat(neu_idx, np.int32), 'g': cat(neu_g)},
        'robin': {'indices': cat(rob_idx, np.int32), 'g': cat(rob_g), 'alpha': cat(rob_a)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate FEM dataset')
    parser.add_argument('--problem', type=str, default='poisson')
    parser.add_argument('--shape', type=str, default='circle')
    parser.add_argument('--id', type=str, default='bc1')
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./data')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'])
    parser.add_argument('--save-hdf5', action='store_true', help='Save full HDF5 output')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = f'{args.problem}-{args.shape}-{args.id}'
    logger.info(f'Generating dataset: {dataset_name}, size={args.size}')

    config = get_config(args.problem, args.shape, args.id)
    ndims = config['ndims']
    variables = config['variables']

    # Create mesh
    logger.info('Creating mesh...')
    order = 1 if args.problem == 'poisson' else 2
    mesh = create_mesh(args.shape, order=order)
    if args.dtype == 'float64':
        mesh = mesh.double()
    n_points = mesh.points.shape[0]
    n_boundary = mesh.boundary_mask.sum().item()
    logger.info(f'Mesh: {n_points} points, {n_boundary} boundary')
    points_np = mesh.points.detach().cpu().numpy()

    # Pre-compute SDF (shared across all samples for fixed geometry)
    logger.info('Computing SDF...')
    boundary_curve_pts = get_boundary_curve_points(args.shape, n_points=1000)
    boundary_node_indices = torch.where(mesh.boundary_mask)[0].cpu().numpy()

    # Get hole boundary points for hollow geometries
    hole_curve_pts = None
    if args.shape == 'circlehollow':
        hole_curve_pts = get_circlehollow_holes(n_points=500)
    elif args.shape == 'squarehollow':
        hole_curve_pts = get_squarehollow_holes(n_points=500)
    elif args.shape == 'boomcircletri':
        hole_curve_pts = get_boomcircletri_holes(n_points=500)

    sdf_nodes = compute_sdf_at_nodes(points_np, boundary_curve_pts, boundary_node_indices,
                                      hole_boundary_points=hole_curve_pts)
    bbox_grid_arrays, sdf_bbox_grid, sdf_gradient = compute_sdf_on_grid(
        points_np, boundary_curve_pts, hole_boundary_points=hole_curve_pts)
    logger.info(f'SDF: range=[{sdf_nodes.min():.4f}, {sdf_nodes.max():.4f}]')

    # Initialize HDF5 output
    hdf5_file = None
    if args.save_hdf5:
        os.makedirs(args.output_dir, exist_ok=True)
        hdf5_path = os.path.join(args.output_dir, f'{dataset_name}.nc')
        hdf5_file = initialize_file(hdf5_path, args.size, ndims, variables)
        logger.info(f'Output: {hdf5_path}')

    # Generate samples
    n_success = 0
    n_errors = 0
    for i in range(args.size):
        t0 = time.time()
        try:
            if args.problem == 'poisson':
                sample = generate_poisson_sample(mesh, config, boundary_curve_pts)
            elif args.problem == 'elasticity':
                sample = generate_elasticity_sample(mesh, config)
            else:
                raise NotImplementedError(f"Problem '{args.problem}' not supported")

            # Store to HDF5
            if hdf5_file is not None:
                store_sample(
                    file=hdf5_file,
                    i_sample=n_success,
                    coordinates=sample['coordinates'],
                    solution=sample['solution'],
                    source_or_variables=sample['variables'],
                    sdf_nodes=sdf_nodes,
                    sdf_gradient=sdf_gradient,
                    bbox_grid_arrays=bbox_grid_arrays,
                    sdf_bbox_grid=sdf_bbox_grid,
                    extensions=sample['extensions'],
                    boundary_info=sample['boundary'],
                    ndims=ndims,
                )

            n_success += 1
            dt = time.time() - t0
            if n_success % max(1, args.size // 10) == 0:
                logger.info(f'  Sample {n_success}/{args.size} done ({dt:.2f}s)')

        except Exception as e:
            n_errors += 1
            logger.warning(f'  Sample {i+1} failed: {e}')
            import traceback
            traceback.print_exc()

    if hdf5_file is not None:
        hdf5_file.close()

    logger.info(f'Generated {n_success}/{args.size} samples ({n_errors} errors)')


if __name__ == '__main__':
    main()

"""Generate 1 sample per dataset and save visualization.

Produces a PNG for each of the 18 datasets showing the solution field.

Usage:
    python visualize_all.py
    python visualize_all.py --output-dir ./figures
"""

import argparse
import os
import logging
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mesh_gen import create_mesh
from generate_dataset import (
    get_config, generate_poisson_sample, generate_elasticity_sample,
    get_boundary_curve_points,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

ALL_DATASETS = [
    ('poisson', 'circle', 'bc1'),
    ('poisson', 'circle', 'bc4'),
    ('poisson', 'circle', 'bc5'),
    ('poisson', 'square', 'bc1'),
    ('poisson', 'square', 'bc4'),
    ('poisson', 'square', 'bc5'),
    ('poisson', 'boomerang', 'bc1'),
    ('poisson', 'boomerang', 'bc4'),
    ('poisson', 'boomerang', 'bc5'),
    ('elasticity', 'circlehollow', 'm1'),
    ('elasticity', 'circlehollow', 'm2'),
    ('elasticity', 'circlehollow', 'm3'),
    ('elasticity', 'squarehollow', 'm1'),
    ('elasticity', 'squarehollow', 'm2'),
    ('elasticity', 'squarehollow', 'm3'),
    ('elasticity', 'boomcircletri', 'm1'),
    ('elasticity', 'boomcircletri', 'm2'),
    ('elasticity', 'boomcircletri', 'm3'),
]


def plot_scalar_field(points, cells, values, title, save_path, cmap='RdBu_r'):
    """Plot a scalar field on a triangular mesh."""
    x, y = points[:, 0], points[:, 1]

    # Get triangle connectivity (first 3 nodes for P2)
    tri = cells[:, :3]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal')
    triang = mtri.Triangulation(x, y, tri)

    vmax = max(abs(values.min()), abs(values.max()))
    if vmax < 1e-15:
        vmax = 1.0
    im = ax.tripcolor(triang, values, shading='gouraud', cmap=cmap,
                       vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
    cb.formatter.set_powerlimits((0, 0))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_vector_field(points, cells, u, strain, stress, title, save_path):
    """Plot displacement, strain, and stress fields for elasticity."""
    x, y = points[:, 0], points[:, 1]
    tri = cells[:, :3]
    triang = mtri.Triangulation(x, y, tri)

    fields = {
        '$u_x$': u[:, 0],
        '$u_y$': u[:, 1],
        '$\\varepsilon_{11}$': strain[:, 0],
        '$\\varepsilon_{12}$': strain[:, 1],
        '$\\sigma_{11}$': stress[:, 0],
        '$\\sigma_{12}$': stress[:, 1],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(title, fontsize=13)

    for ax, (name, vals) in zip(axes.flat, fields.items()):
        ax.set_aspect('equal')
        vmax = max(abs(vals.min()), abs(vals.max()))
        if vmax < 1e-15:
            vmax = 1.0
        im = ax.tripcolor(triang, vals, shading='gouraud', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=25)
        cb.formatter.set_powerlimits((0, 0))
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_triangle_cells(mesh):
    """Get triangle connectivity from mesh (handles triangle and triangle6)."""
    cell_keys = list(mesh.cells.keys())
    for k in cell_keys:
        if 'triangle' in k:
            return mesh.cells[k].cpu().numpy()
    raise RuntimeError(f"No triangle cells found in {cell_keys}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mesh_cache = {}

    for problem, shape, ident in ALL_DATASETS:
        dataset_name = f'{problem}-{shape}-{ident}'
        logger.info(f'Generating {dataset_name}...')
        t0 = time.time()

        # Reset random state per dataset for reproducibility
        np.random.seed(args.seed)

        config = get_config(problem, shape, ident)
        order = 1 if problem == 'poisson' else 2

        # Cache meshes
        mesh_key = (shape, order)
        if mesh_key not in mesh_cache:
            mesh_cache[mesh_key] = create_mesh(shape, order=order).double()
        mesh = mesh_cache[mesh_key]

        points_np = mesh.points.detach().cpu().numpy()
        cells_np = get_triangle_cells(mesh)

        try:
            if problem == 'poisson':
                boundary_curve_pts = get_boundary_curve_points(shape, n_points=1000)
                sample = generate_poisson_sample(mesh, config, boundary_curve_pts)

                save_path = os.path.join(args.output_dir, f'{dataset_name}.png')
                plot_scalar_field(
                    points_np, cells_np, sample['solution'],
                    title=dataset_name, save_path=save_path,
                )
            else:
                sample = generate_elasticity_sample(mesh, config)

                save_path = os.path.join(args.output_dir, f'{dataset_name}.png')
                plot_vector_field(
                    points_np, cells_np,
                    u=sample['solution'],
                    strain=sample['variables']['strain'],
                    stress=sample['variables']['cauchystress'],
                    title=dataset_name, save_path=save_path,
                )

            dt = time.time() - t0
            logger.info(f'  Saved {save_path} ({dt:.1f}s)')

        except Exception as e:
            logger.error(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()

    logger.info(f'Done. Figures in {args.output_dir}/')


if __name__ == '__main__':
    main()

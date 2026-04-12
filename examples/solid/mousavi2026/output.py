"""HDF5 output routines matching original dataset format.

Ported from fenicsx-main/generate_dataset.py:62-99 (initialize) and 242-282 (store).
"""

import h5py
import numpy as np
from typing import Dict, List, Optional


def initialize_file(filepath: str,
                    n_samples: int,
                    ndims: int = 1,
                    variables: Dict[str, int] = None) -> h5py.File:
    """Create and initialize an HDF5 file matching the original format.

    Args:
        filepath: output file path
        n_samples: number of samples to allocate
        ndims: 1 for Poisson, 2 for elasticity
        variables: dict of variable_name -> n_components
            e.g. {'source': 1} or {'strain': 4, 'cauchystress': 4}

    Returns:
        open h5py.File (caller must close)
    """
    if variables is None:
        variables = {'source': 1}

    vlen_int = h5py.vlen_dtype(np.dtype('int32'))
    vlen_flt = h5py.vlen_dtype(np.dtype('float64'))

    f = h5py.File(filepath, 'w')
    f.attrs['count'] = 0

    # Bounding box group
    g_bbox = f.create_group('bbox')
    g_bbox.create_dataset('grid', shape=(n_samples, 1, 2, 256), dtype='float64')
    g_bbox.create_dataset('sdf', shape=(n_samples, 1, 1, 256, 256), dtype='float64')

    # Coordinates (variable-length)
    f.create_dataset('coordinates', shape=(n_samples, 1, 2), dtype=vlen_flt)

    # Interior group
    g_int = f.create_group('interior')
    g_int.create_dataset('sdf', shape=(n_samples, 1, 1), dtype=vlen_flt)
    g_int.create_dataset('sdf_grad', shape=(n_samples, 1, 2), dtype=vlen_flt)
    g_int.create_dataset('solution', shape=(n_samples, 1, ndims), dtype=vlen_flt)

    for vname, vdims in variables.items():
        g_int.create_dataset(vname, shape=(n_samples, 1, vdims), dtype=vlen_flt)

    # Extensions group
    g_ext = g_int.create_group('extensions')
    for dim in range(ndims):
        g_ext_dim = g_ext.create_group(str(dim))
        for vname in ['alpha', 'beta', 'g']:
            g_ext_dim.create_dataset(vname, shape=(n_samples, 1, 1), dtype=vlen_flt)

    # Boundaries group
    g_bnd = f.create_group('boundaries')
    for dim in range(ndims):
        g_bnd_dim = g_bnd.create_group(str(dim))
        for bc_type in ['dirichlet', 'neumann', 'robin']:
            g_bc = g_bnd_dim.create_group(bc_type)
            g_bc.create_dataset('indices', shape=(n_samples, 1, 1), dtype=vlen_int)
            g_bc.create_dataset('g', shape=(n_samples, 1, 1), dtype=vlen_flt)
            if bc_type == 'robin':
                g_bc.create_dataset('alpha', shape=(n_samples, 1, 1), dtype=vlen_flt)

    return f


def store_sample(file: h5py.File,
                 i_sample: int,
                 coordinates: np.ndarray,
                 solution: np.ndarray,
                 source_or_variables: dict,
                 sdf_nodes: np.ndarray,
                 sdf_gradient: np.ndarray,
                 bbox_grid_arrays: list,
                 sdf_bbox_grid: np.ndarray,
                 extensions: List[Dict[str, np.ndarray]],
                 boundary_info: dict,
                 ndims: int = 1):
    """Store one sample to the HDF5 file.

    Args:
        file: open h5py.File
        i_sample: sample index
        coordinates: [n_points, 2]
        solution: [n_points] for Poisson or [n_points, 2] for elasticity
        source_or_variables: dict of variable arrays, e.g.
            {'source': [n_points]} or {'strain': [n_points, 4], 'cauchystress': [n_points, 4]}
        sdf_nodes: [n_points] SDF at mesh nodes
        sdf_gradient: [n_points, 2] SDF gradient
        bbox_grid_arrays: [x_array, y_array] for grid
        sdf_bbox_grid: [256, 256] SDF on grid
        extensions: list of dicts per dim, each with 'alpha', 'beta', 'g' arrays
        boundary_info: dict with 'dirichlet'/'neumann'/'robin' keys, each with 'indices', 'g', optionally 'alpha'
        ndims: 1 for Poisson, 2 for elasticity
    """
    n_points = coordinates.shape[0]

    # Coordinates: store as [2, n_points]
    file['coordinates'][i_sample, 0, :] = coordinates.T

    # Bounding box
    file['bbox']['grid'][i_sample, 0] = np.stack(bbox_grid_arrays)
    file['bbox']['sdf'][i_sample, 0, 0] = sdf_bbox_grid

    # Interior SDF
    file['interior']['sdf'][i_sample, 0, 0] = sdf_nodes
    file['interior']['sdf_grad'][i_sample, 0, :] = sdf_gradient.T

    # Solution
    if solution.ndim == 1:
        file['interior']['solution'][i_sample, 0, 0] = solution
    else:
        for d in range(ndims):
            file['interior']['solution'][i_sample, 0, d] = solution[:, d]

    # Variables (source, strain, stress)
    for vname, vdata in source_or_variables.items():
        if vdata.ndim == 1:
            file['interior'][vname][i_sample, 0, 0] = vdata
        else:
            for d in range(vdata.shape[1]):
                file['interior'][vname][i_sample, 0, d] = vdata[:, d]

    # Extensions
    for dim in range(ndims):
        for key in ['alpha', 'beta', 'g']:
            file['interior']['extensions'][str(dim)][key][i_sample, 0, 0] = extensions[dim][key]

    # Boundaries
    for dim in range(ndims):
        for bc_type in ['dirichlet', 'neumann', 'robin']:
            bc_data = boundary_info.get(bc_type, {})
            indices = bc_data.get('indices', np.array([], dtype=np.int32))
            g_vals = bc_data.get('g', np.array([]))

            file['boundaries'][str(dim)][bc_type]['indices'][i_sample, 0, 0] = indices.astype(np.int32)
            file['boundaries'][str(dim)][bc_type]['g'][i_sample, 0, 0] = g_vals.astype(np.float64)

            if bc_type == 'robin':
                alpha_vals = bc_data.get('alpha', np.array([]))
                file['boundaries'][str(dim)][bc_type]['alpha'][i_sample, 0, 0] = alpha_vals.astype(np.float64)

    file.attrs['count'] = i_sample + 1

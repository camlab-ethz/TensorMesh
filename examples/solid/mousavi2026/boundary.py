"""Random boundary condition function generators.

Ports the angle-based Fourier BC functions from the original FEniCSx code
(fenicsx-main/src/functions/boundary.py and domain.py).
"""

import numpy as np
import torch
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta
from typing import Dict, Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Random scalar/vectorial BC functions (angle-based Fourier modes)
# ---------------------------------------------------------------------------

def get_scalar_function(value: float):
    """Returns a constant scalar function f(x) = value."""
    def f(x):
        # x: [n_points, 2] numpy array
        return np.zeros(x.shape[0]) + value
    return f


def get_random_scalar_function(rng: Tuple[float, float], modes: int = 2,
                                C: Tuple[float, float] = (0., 0.), R: float = 1.0):
    """Returns a random angle-based Fourier scalar function.

    f(x) = scale * sin(theta + shift_mul) * sum_k(coeff_k * sin((k+1)*theta + shift_k))
    where theta = arctan2(x[:,1] - C[1], x[:,0] - C[0]) / R

    Args:
        rng: (low, high) amplitude range for uniform scale sampling
        modes: number of Fourier modes
        C: center point for angle computation
        R: radial scaling factor for angle
    """
    center = np.array(C)
    scale = np.random.uniform(*rng)
    shifts = np.random.uniform(low=0., high=2 * np.pi, size=(modes,))
    # Random partition of [0, 1] into `modes` parts → coefficients
    coeffs = sorted([0.] + list(np.random.uniform(size=(modes - 1,))) + [1.])
    coeffs = np.subtract(coeffs[1:], coeffs[:-1])
    shift_mul = np.random.uniform(low=0., high=2 * np.pi)

    def f(x):
        # x: [n_points, 2] numpy array
        dx = x - center[np.newaxis, :]
        theta = np.arctan2(dx[:, 1], dx[:, 0]) / R
        terms = [
            coeffs[k] * np.sin((k + 1) * theta + shifts[k])
            for k in range(modes)
        ]
        return scale * np.sin(theta + shift_mul) * np.sum(np.stack(terms), axis=0)

    return f


def get_random_vectorial_function(rng, modes, C=(0., 0.), R=1.0,
                                   ndim: int = 1, ax: int = None):
    """Returns a random vectorial function composed of scalar functions.

    Args:
        ndim: number of vector components
        ax: if given, only component `ax` is random (others zero)
    """
    if ax is None:
        fs = [get_random_scalar_function(rng, modes, C, R) for _ in range(ndim)]
    else:
        fs = [
            (get_random_scalar_function(rng, modes, C, R) if i == ax
             else get_scalar_function(0.0))
            for i in range(ndim)
        ]

    def f(x):
        return np.stack([fi(x) for fi in fs], axis=-1)

    return f


# ---------------------------------------------------------------------------
# Source function generators
# ---------------------------------------------------------------------------

def get_centered_radial_cosine(ord: int = 2, scale: float = 20.0, freq: float = 2.0):
    """Source function f(x) = scale * cos(freq * 2*pi * ||x||_ord).

    Returns a callable that takes [n_points, 2] numpy array.
    """
    def f(x):
        r = np.linalg.norm(x, axis=-1, ord=ord)
        return scale * np.cos(freq * (2 * np.pi) * r)
    return f


class FunctionDistribution(metaclass=ABCMeta):
    """Base class for random function distributions."""
    @abstractmethod
    def draw(self) -> Callable:
        pass


class ConstantFunctionGenerator(FunctionDistribution):
    """Always returns the same function."""
    def __init__(self, func):
        self.func = func

    def draw(self):
        return self.func


class RandomRadialSines(FunctionDistribution):
    """Random radial sine source function.

    f(x) = scale * sum_k(coeff_k * sin(2*(k+1)*pi*r + shift_k))
    where r = ||x - random_center||_ord
    """
    def __init__(self, modes: int = 2, ord: int = 2, scale: float = 20.):
        self.modes = modes
        self.ord = ord
        self.scale = scale

    def draw(self):
        center = np.random.uniform(-1, 1, size=(1, 2))
        shifts = np.random.uniform(low=0., high=2 * np.pi, size=(self.modes,))
        coeffs = sorted([0.] + list(np.random.uniform(size=(self.modes - 1,))) + [1.])
        coeffs = np.subtract(coeffs[1:], coeffs[:-1])

        def f(x):
            r = np.linalg.norm(x - center, axis=-1, ord=self.ord)
            terms = [
                coeffs[k] * np.sin(2 * (k + 1) * np.pi * r + shifts[k])
                for k in range(self.modes)
            ]
            return self.scale * np.sum(np.stack(terms), axis=0)

        return f


# ---------------------------------------------------------------------------
# BC distribution classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionBC:
    """Boundary condition specification for one spatial dimension."""
    type: str  # 'dirichlet', 'neumann', or 'robin'
    functions: Dict[str, Callable]  # {'g': func} or {'g': func, 'alpha': func}
    values: Optional[Dict[str, np.ndarray]] = None  # filled later with nodal values
    dist: str = 'unknown'


class BCDistribution(metaclass=ABCMeta):
    @abstractmethod
    def draw(self) -> DimensionBC:
        pass


class Dirichlet(BCDistribution):
    def __init__(self, rng, modes, C=(0., 0.), R=1.0):
        self.rng = rng
        self.modes = modes
        self.C = C
        self.R = R

    def draw(self):
        return DimensionBC(
            type='dirichlet',
            functions={'g': get_random_scalar_function(rng=self.rng, modes=self.modes, C=self.C, R=self.R)},
            dist='Dirichlet',
        )


class Neumann(BCDistribution):
    def __init__(self, rng, modes, C=(0., 0.), R=1.0):
        self.rng = rng
        self.modes = modes
        self.C = C
        self.R = R

    def draw(self):
        return DimensionBC(
            type='neumann',
            functions={'g': get_random_scalar_function(rng=self.rng, modes=self.modes, C=self.C, R=self.R)},
            dist='Neumann',
        )


class Robin(BCDistribution):
    def __init__(self, rng, modes, C=(0., 0.), R=1.0):
        """rng = ((g_low, g_high), (alpha_low, alpha_high))
           modes = (g_modes, alpha_modes)"""
        self.rng = rng
        self.modes = modes
        self.C = C
        self.R = R

    def draw(self):
        return DimensionBC(
            type='robin',
            functions={
                'g': get_random_scalar_function(rng=self.rng[0], modes=self.modes[0], C=self.C, R=self.R),
                'alpha': get_random_scalar_function(rng=self.rng[1], modes=self.modes[1], C=self.C, R=self.R),
            },
            dist='Robin',
        )


class RandomBCTypes(BCDistribution):
    """Randomly selects Dirichlet, Neumann, or Robin (33% each)."""
    def __init__(self, ranges, modes, C=(0., 0.), R=1.0):
        self.ranges = ranges
        self.modes = modes
        self.C = C
        self.R = R

    def draw(self):
        a = np.random.rand()
        if a < .33:
            return DimensionBC(
                type='dirichlet',
                functions={'g': get_random_scalar_function(
                    rng=self.ranges['dirichlet'], modes=self.modes['dirichlet'], C=self.C, R=self.R)},
                dist='RandomBCTypes',
            )
        elif a < .66:
            return DimensionBC(
                type='neumann',
                functions={'g': get_random_scalar_function(
                    rng=self.ranges['neumann'], modes=self.modes['neumann'], C=self.C, R=self.R)},
                dist='RandomBCTypes',
            )
        else:
            return DimensionBC(
                type='robin',
                functions={
                    'g': get_random_scalar_function(
                        rng=self.ranges['robin'][0], modes=self.modes['robin'][0], C=self.C, R=self.R),
                    'alpha': get_random_scalar_function(
                        rng=self.ranges['robin'][1], modes=self.modes['robin'][1], C=self.C, R=self.R),
                },
                dist='RandomBCTypes',
            )


class NeumannHomogenous(BCDistribution):
    def __init__(self, value: float = 0.0):
        self.value = value

    def draw(self):
        return DimensionBC(
            type='neumann',
            functions={'g': get_scalar_function(value=self.value)},
            dist='NeumannHomogenous',
        )


class DirichletHomogenous(BCDistribution):
    def __init__(self, value: float = 0.0):
        self.value = value

    def draw(self):
        return DimensionBC(
            type='dirichlet',
            functions={'g': get_scalar_function(value=self.value)},
            dist='DirichletHomogenous',
        )


# ---------------------------------------------------------------------------
# Segment and BCGenerator
# ---------------------------------------------------------------------------

SEGMENT_COLORS = ['orange', 'cyan', 'red', 'blue', 'green', 'yellow', 'purple', 'brown', 'pink']


@dataclass
class SegmentBCs:
    """Boundary conditions on one boundary segment."""
    name: str
    center: float  # parametric center in [0, 1]
    radius: float  # parametric half-width
    dims: List[DimensionBC]  # one per spatial dimension
    indices: Optional[np.ndarray] = None  # boundary node indices (filled later)

    def __getitem__(self, item):
        return self.dims[item]

    def contains(self, a: np.ndarray) -> np.ndarray:
        """Check if parametric values `a` (in [0,1]) lie in this segment.
        Handles periodic wrapping.
        """
        lo = self.center - self.radius
        hi = self.center + self.radius
        if lo < 0:
            return (a >= lo + 1) | (a < hi)
        elif hi > 1:
            return (a >= lo) | (a < hi - 1)
        else:
            return (a >= lo) & (a < hi)


class BCGenerator:
    """Generates random boundary segments with random BC types.

    Args:
        ndims: number of spatial dimensions for BC (1 for Poisson, 2 for elasticity)
        dists: list of BCDistribution objects, one per segment
        joints: optional fixed joint positions (if None, drawn randomly)
    """
    def __init__(self, ndims: int = 1, dists: List[BCDistribution] = [],
                 joints: List[float] = None):
        self.dists = dists
        self.ndims = ndims
        self.n = len(dists)
        assert self.n > 0
        if joints is not None:
            assert len(joints) == self.n
            self.joints = joints + [joints[0] + 1]
        else:
            self.joints = joints

    def draw(self) -> List[SegmentBCs]:
        # Draw random joints or use fixed
        if self.joints is None:
            joints = sorted(list(np.random.uniform(size=(self.n,))))
            joints = joints + [joints[0] + 1]
            dists_permutations = [np.random.permutation(self.n) for _ in range(self.ndims)]
        else:
            joints = self.joints
            dists_permutations = [np.arange(self.n) for _ in range(self.ndims)]

        bcs = []
        for idx in range(self.n):
            center = 0.5 * (joints[idx + 1] + joints[idx])
            if center > 1:
                center -= 1
            radius = 0.5 * (joints[idx + 1] - joints[idx])
            bcs.append(SegmentBCs(
                name=SEGMENT_COLORS[idx],
                center=center,
                radius=radius,
                dims=[self.dists[dists_permutations[d][idx]].draw() for d in range(self.ndims)],
            ))
        return bcs

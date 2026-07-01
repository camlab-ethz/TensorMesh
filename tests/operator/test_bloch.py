"""Tests for the BlochReducer operator (Bloch-Floquet periodic BC)."""

import sys
sys.path.append("../..")

import math

import numpy as np
import torch

from tensormesh import BlochReducer
from tensormesh.sparse import SparseMatrix


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def grid_points(nx, ny, ax=1.0, ay=1.0):
    """Structured node grid on [0,ax] x [0,ay]; opposite faces carry matching
    nodes (the periodicity precondition), so no gmsh is needed for the tests."""
    xs = np.linspace(0.0, ax, nx)
    ys = np.linspace(0.0, ay, ny)
    return np.array([[x, y] for y in ys for x in xs], dtype=float)


def random_sym_sparse(n, seed, complex_=False):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g, dtype=torch.float64)
    A = A + A.T
    if complex_:
        B = torch.rand(n, n, generator=g, dtype=torch.float64)
        A = A.to(torch.complex128) + 1j * (B - B.T)        # Hermitian
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return SparseMatrix(A.reshape(-1), row.reshape(-1), col.reshape(-1), (n, n))


def dense_T(bloch, k):
    """Explicit dense Floquet transform T(k) from the reducer's own pairing."""
    phase = bloch._phase(k)
    T = torch.zeros(bloch.n_dof, bloch.n_reduced_dof, dtype=torch.complex128)
    T[torch.arange(bloch.n_dof), bloch.master_dof] = phase
    return T


# --------------------------------------------------------------------------- #
class TestPairing:
    def test_master_count_square(self):
        # 5x5 grid on a unit square: masters = interior+low faces = 4x4 = 16
        pts = grid_points(5, 5)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        assert bloch.n_nodes == 25
        assert bloch.n_masters == 16            # (5-1) * (5-1)

    def test_right_maps_to_left(self):
        pts = grid_points(4, 4, ax=2.0, ay=2.0)
        bloch = BlochReducer(pts, [[2.0, 0.0], [0.0, 2.0]])
        # node on the right face (x=2) must map to the same-y node on x=0
        md = bloch.master_dof.numpy()
        for i, p in enumerate(pts):
            if np.isclose(p[0], 2.0) and p[1] < 2.0 - 1e-9:
                j = np.where(np.all(np.isclose(pts, [0.0, p[1]]), axis=1))[0][0]
                assert md[i] == md[j]
        # the lattice translation of a right-face node is +a1
        R = bloch.node_R.numpy()
        i = np.where(np.all(np.isclose(pts, [2.0, 0.0]), axis=1))[0][0]
        assert np.allclose(R[i], [2.0, 0.0])

    def test_corner_maps_to_origin(self):
        pts = grid_points(4, 4)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        md = bloch.master_dof.numpy()
        corner = np.where(np.all(np.isclose(pts, [1.0, 1.0]), axis=1))[0][0]
        origin = np.where(np.all(np.isclose(pts, [0.0, 0.0]), axis=1))[0][0]
        assert md[corner] == md[origin]

    def test_non_orthogonal_lattice(self):
        # rhombic (60 deg) cell: a1=(1,0), a2=(0.5, sqrt(3)/2). Build matching
        # points so right/top faces have images; check pairing succeeds.
        a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, math.sqrt(3) / 2])
        n = 4
        pts = np.array([i / (n - 1) * a1 + j / (n - 1) * a2
                        for j in range(n) for i in range(n)])
        bloch = BlochReducer(pts, [a1.tolist(), a2.tolist()])
        assert bloch.n_masters == (n - 1) * (n - 1)     # float-space pairing works


# --------------------------------------------------------------------------- #
class TestReduction:
    def test_matches_dense_triple_product(self):
        pts = grid_points(5, 5)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        K = random_sym_sparse(bloch.n_dof, seed=0)
        for k in ([0.3, 1.1], [math.pi, math.pi], [0.0, 0.0]):
            Kr = bloch.reduce(K, k).to_dense()
            T = dense_T(bloch, k)
            ref = T.conj().T @ K.to_dense().to(torch.complex128) @ T
            assert torch.allclose(Kr, ref, atol=1e-10), f"k={k}"

    def test_reduced_is_hermitian(self):
        pts = grid_points(6, 4)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        K = random_sym_sparse(bloch.n_dof, seed=1)
        Kr = bloch.reduce(K, [0.7, -0.4]).to_dense()
        assert torch.allclose(Kr, Kr.conj().T, atol=1e-10)

    def test_gamma_is_real_phase_one(self):
        pts = grid_points(5, 5)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        # at k=0 every phase is 1 -> reduction is a real index-merge of a real K
        K = random_sym_sparse(bloch.n_dof, seed=2)
        Kr = bloch.reduce(K, [0.0, 0.0]).to_dense()
        assert torch.allclose(Kr.imag, torch.zeros_like(Kr.imag), atol=1e-12)

    def test_shape(self):
        pts = grid_points(5, 5)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        K = random_sym_sparse(bloch.n_dof, seed=3)
        Kr = bloch.reduce(K, [1.0, 2.0])
        assert tuple(Kr.shape) == (16, 16)

    def test_dense_matches_sparse(self):
        # a dense torch.Tensor input must reduce to the same thing as the
        # equivalent SparseMatrix (for hand-assembled beam/truss systems)
        pts = grid_points(5, 4)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        K = random_sym_sparse(bloch.n_dof, seed=9)
        k = [0.6, -0.9]
        Kr_sparse = bloch.reduce(K, k).to_dense()
        Kr_dense = bloch.reduce(K.to_dense(), k)
        assert torch.allclose(Kr_sparse, Kr_dense, atol=1e-10)
        # and both equal the explicit dense triple product
        T = dense_T(bloch, k)
        ref = T.conj().T @ K.to_dense().to(torch.complex128) @ T
        assert torch.allclose(Kr_dense, ref, atol=1e-10)


# --------------------------------------------------------------------------- #
class TestPartialPeriodicity:
    def test_single_direction(self):
        # strip periodic in y only (one lattice vector); x faces stay free
        pts = grid_points(5, 4, ax=3.0, ay=1.0)
        bloch = BlochReducer(pts, [[0.0, 1.0]])          # only a2
        # masters: drop the top row (y=1) -> 5 * (4-1) = 15
        assert bloch.n_masters == 15
        K = random_sym_sparse(bloch.n_dof, seed=4)
        Kr = bloch.reduce(K, [0.0, 0.9]).to_dense()
        T = dense_T(bloch, [0.0, 0.9])
        ref = T.conj().T @ K.to_dense().to(torch.complex128) @ T
        assert torch.allclose(Kr, ref, atol=1e-10)


# --------------------------------------------------------------------------- #
class TestVectorDOFs:
    def test_two_dofs_per_node(self):
        pts = grid_points(4, 4)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]], dofs_per_node=2)
        assert bloch.n_dof == 2 * 16
        assert bloch.n_reduced_dof == 2 * 9
        K = random_sym_sparse(bloch.n_dof, seed=5)
        Kr = bloch.reduce(K, [0.5, 0.5]).to_dense()
        T = dense_T(bloch, [0.5, 0.5])
        ref = T.conj().T @ K.to_dense().to(torch.complex128) @ T
        assert torch.allclose(Kr, ref, atol=1e-10)


# --------------------------------------------------------------------------- #
class TestRecover:
    def test_recover_phase(self):
        pts = grid_points(5, 5)
        bloch = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]])
        k = [0.3, 1.7]
        u_red = torch.randn(bloch.n_reduced_dof, dtype=torch.complex128)
        u_full = bloch.recover(u_red, k)
        T = dense_T(bloch, k)
        assert torch.allclose(u_full, T @ u_red, atol=1e-12)

    def test_eig_consistency(self):
        # generalized eig of (Kr, Mr) must be invariant to the phase sign
        pts = grid_points(6, 6)
        b1 = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]], sign=-1)
        b2 = BlochReducer(pts, [[1.0, 0.0], [0.0, 1.0]], sign=+1)
        K = random_sym_sparse(b1.n_dof, seed=7)
        # SPD mass matrix M = B B^T + n I
        g = torch.Generator().manual_seed(8)
        B = torch.rand(b1.n_dof, b1.n_dof, generator=g, dtype=torch.float64)
        Md = B @ B.T + b1.n_dof * torch.eye(b1.n_dof, dtype=torch.float64)
        row, col = torch.meshgrid(torch.arange(b1.n_dof), torch.arange(b1.n_dof), indexing="ij")
        M = SparseMatrix(Md.reshape(-1), row.reshape(-1), col.reshape(-1), (b1.n_dof,) * 2)
        k = [0.8, -0.5]
        w = {}
        for tag, b in (("m", b1), ("p", b2)):
            Kr = b.reduce(K, k).to_dense(); Mr = b.reduce(M, k).to_dense()
            Kr = 0.5 * (Kr + Kr.conj().T); Mr = 0.5 * (Mr + Mr.conj().T)
            w[tag] = torch.sort(_gen_eig(Kr, Mr)).values
        assert torch.allclose(w["m"], w["p"], atol=1e-8)


def _gen_eig(Kr, Mr):
    L = torch.linalg.cholesky(Mr)
    Li = torch.linalg.solve_triangular(L, torch.eye(Mr.shape[0], dtype=Mr.dtype), upper=False)
    A = Li @ Kr @ Li.conj().T
    return torch.linalg.eigvalsh(0.5 * (A + A.conj().T))

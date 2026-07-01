"""Bloch-Floquet periodic boundary conditions for assembled FEM systems.

This module provides :class:`BlochReducer`, the periodic counterpart of
:class:`~tensormesh.operator.condense.Condenser`.  Where the :class:`Condenser`
eliminates Dirichlet DOFs, :class:`BlochReducer` ties together the DOFs on
opposite faces of a periodic unit cell with a wavevector-dependent Floquet phase

.. math::

    u(\\mathbf r + \\mathbf R) = e^{\\,i\\,\\mathbf k\\cdot\\mathbf R}\\, u(\\mathbf r),
    \\qquad \\mathbf R = \\sum_j n_j\\, \\mathbf a_j,

and reduces an assembled operator to the independent (master) DOFs:

.. math::

    A_r(\\mathbf k) \\;=\\; T(\\mathbf k)^{H}\\, A\\, T(\\mathbf k),

where :math:`T(\\mathbf k)` maps the master DOFs to all DOFs with the phase
above.  For a band structure one reduces both the stiffness ``K`` and the mass
``M`` and solves the generalized Hermitian eigenproblem
:math:`K_r u = \\lambda M_r u` at each :math:`\\mathbf k`.

The reduction is built **without forming** :math:`T`: it remaps the COO indices
of ``A`` onto the master DOFs, weights each entry by
:math:`\\overline{\\phi_i}\\,\\phi_j`, and coalesces, so :math:`A_r` is produced
in one pass over the non-zeros of ``A``.

Examples
--------
.. code-block:: python

    import torch
    from tensormesh import Mesh, BlochReducer
    from tensormesh.assemble import LaplaceElementAssembler, MassElementAssembler

    # a periodic unit-cell mesh (opposite faces must carry matching nodes)
    K = LaplaceElementAssembler.from_mesh(mesh)()
    M = MassElementAssembler.from_mesh(mesh)()

    bloch = BlochReducer(mesh.points, lattice_vectors=[[a, 0.0], [0.0, a]])
    for k in k_path:                          # k = (kx, ky)
        Kr, Mr = bloch.reduce_system(K, M, k)
        w2 = torch.linalg.eigvalsh(           # generalized eig via Cholesky outside
            ...)
"""
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from ..sparse import SparseMatrix

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence]


class BlochReducer(nn.Module):
    """Bloch-Floquet periodic reduction of an assembled FEM operator.

    Parameters
    ----------
    points : array-like, shape ``[n_nodes, dim]``
        Nodal coordinates of the unit-cell mesh.  Opposite periodic faces
        **must carry matching nodes** (e.g. a mesh built with
        ``gmsh.model.mesh.setPeriodic``); a node on the ``+a_j`` face is paired
        with its image one lattice vector back.
    lattice_vectors : array-like, shape ``[n_lat, dim]``
        The **periodic** lattice vectors ``a_j`` (1, 2 or 3 of them).  Pass only
        the directions that are actually periodic — e.g. a single vector for a
        waveguide periodic in one direction.
    dofs_per_node : int, optional
        Number of DOFs per node (1 for scalar acoustics/Helmholtz, ``dim`` for
        elasticity, 6 for a 3D frame).  Default 1.

        The components are assumed **node-major (component-interleaved)**: DOF
        ``d`` of node ``i`` lives at global index ``i * dofs_per_node + d``, so
        the global vector is ``[n0_x, n0_y, n1_x, n1_y, ...]``.  This is the
        layout TensorMesh's vector assemblers and projector produce -- the
        :class:`~tensormesh.assemble.NodeAssembler` integral is returned
        ``flatten()``-ed from shape ``[n_nodes, dofs_per_node]`` -- so a Bloch
        reduction of a vector operator assembled by
        :class:`~tensormesh.assemble.LinearElasticityElementAssembler` lines up
        with its ``K``/``M`` without any DOF re-ordering.
    tol : float, optional
        Absolute coordinate tolerance for node matching (default scales with the
        bounding box: ``1e-7 * diag``).
    sign : int, optional
        Sign convention of the Floquet phase ``exp(sign * i k·R)``; ``-1``
        (default) gives ``u(r+R) = exp(-i k·R) u(r)`` on the master→slave map.
        The eigenvalues are independent of this choice.

    Attributes
    ----------
    n_nodes, n_masters : int
        Node counts before / after reduction.
    n_dof, n_reduced_dof : int
        DOF counts before / after reduction (``n_* * dofs_per_node``).

    Notes
    -----
    Like :class:`Condenser`, this is a :class:`torch.nn.Module`; the pairing
    buffers move with ``.to(device)``.  The pairing is computed once at
    construction (geometry only); :meth:`reduce` is called per wavevector.
    """

    master_dof: torch.Tensor          # [n_dof] long: full DOF -> reduced DOF
    node_R: torch.Tensor              # [n_nodes, dim] float: lattice translation R_i
    _points: torch.Tensor

    def __init__(self,
                 points: ArrayLike,
                 lattice_vectors: ArrayLike,
                 dofs_per_node: int = 1,
                 tol: Optional[float] = None,
                 sign: int = -1):
        super().__init__()
        pts = np.asarray(_to_numpy(points), dtype=float)
        lat = np.asarray(_to_numpy(lattice_vectors), dtype=float)
        if lat.ndim == 1:
            lat = lat[None, :]
        dim = pts.shape[1]
        assert lat.shape[1] == dim, \
            f"lattice_vectors dim {lat.shape[1]} != points dim {dim}"
        self.dofs_per_node = int(dofs_per_node)
        self.sign = int(sign)
        self.n_nodes = pts.shape[0]
        self.dim = dim

        if tol is None:
            diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            tol = 1e-7 * max(diag, 1.0)
        self.tol = float(tol)

        master_node, trans = self._pair(pts, lat, self.tol)   # [n_nodes], [n_nodes, n_lat]

        # compact master labels to 0..n_masters-1
        uniq = np.unique(master_node)
        relabel = {int(m): r for r, m in enumerate(uniq)}
        master_compact = np.fromiter((relabel[int(m)] for m in master_node),
                                     dtype=np.int64, count=self.n_nodes)
        self.n_masters = len(uniq)
        self.n_dof = self.n_nodes * self.dofs_per_node
        self.n_reduced_dof = self.n_masters * self.dofs_per_node

        # full DOF -> reduced DOF: node block expands to dofs_per_node lanes
        nd = self.dofs_per_node
        master_dof = (master_compact[:, None] * nd + np.arange(nd)[None, :]).ravel()
        node_R = trans @ lat                              # [n_nodes, dim], R_i = sum n_j a_j

        self.register_buffer("master_dof", torch.as_tensor(master_dof, dtype=torch.long))
        self.register_buffer("node_R", torch.as_tensor(node_R, dtype=torch.float64))
        self.register_buffer("_points", torch.as_tensor(pts, dtype=torch.float64),
                             persistent=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pair(pts, lat, tol):
        """Greedy position matching: translate each node back along the lattice
        vectors until no further image exists; that endpoint is its master.

        Returns (master_node[n_nodes], translations[n_nodes, n_lat]).  Works for
        any number of periodic directions and for partial periodicity (a node
        whose ``-a_j`` image is outside the mesh stays its own master).

        The translated position is kept in **float** space and re-hashed each
        step (``cand = cur - a_j``), not subtracted in integer-hash space: for a
        non-orthogonal lattice with irrational components ``round(x + y)`` need
        not equal ``round(x) + round(y)``, so integer-space subtraction would
        drift and mis-pair.  Since ``cand`` equals the master position to float
        precision, re-rounding it lands exactly on the master's hash key.
        """
        n_lat = lat.shape[0]
        N = pts.shape[0]

        def key(p):
            return tuple(np.round(p / tol).astype(np.int64))

        lookup = {key(pts[i]): i for i in range(N)}
        master = np.arange(N, dtype=np.int64)
        trans = np.zeros((N, n_lat), dtype=np.int64)
        for i in range(N):
            cur = pts[i].copy()                      # float position
            n = np.zeros(n_lat, dtype=np.int64)
            moved = True
            while moved:
                moved = False
                for j in range(n_lat):
                    cand = cur - lat[j]              # float subtraction
                    idx = lookup.get(key(cand))
                    if idx is not None and idx != i:
                        cur = cand
                        n[j] += 1
                        moved = True
            master[i] = lookup[key(cur)]
            trans[i] = n
        return master, trans

    # ------------------------------------------------------------------ #
    def _phase(self, k) -> torch.Tensor:
        """Per-DOF Floquet phase exp(sign * i k·R_i), shape [n_dof] complex."""
        k = torch.as_tensor(k, dtype=torch.float64, device=self.node_R.device).reshape(-1)
        assert k.shape[0] == self.dim, f"k must have length {self.dim}, got {k.shape[0]}"
        kR = self.node_R @ k                                  # [n_nodes]
        node_phase = torch.exp(self.sign * 1j * kR.to(torch.complex128))
        return node_phase.repeat_interleave(self.dofs_per_node)   # [n_dof]

    def reduce(self, matrix, k):
        """Return the reduced operator ``T(k)^H A T(k)``.

        Parameters
        ----------
        matrix : SparseMatrix or torch.Tensor
            Assembled operator of shape ``[n_dof, n_dof]`` (real or complex).
            A :class:`SparseMatrix` is reduced sparsely (COO index remap +
            coalesce) and returns a :class:`SparseMatrix`; a dense
            ``torch.Tensor`` is reduced densely and returns a dense complex
            ``torch.Tensor`` (handy for hand-assembled systems such as a beam /
            truss lattice that does not go through the sparse assembler).
        k : array-like, shape ``[dim]``
            Wavevector.

        Returns
        -------
        SparseMatrix or torch.Tensor
            Reduced operator of shape ``[n_reduced_dof, n_reduced_dof]``
            (complex).
        """
        assert matrix.shape[0] == self.n_dof, \
            f"matrix shape {tuple(matrix.shape)} != [{self.n_dof}, {self.n_dof}]"
        if isinstance(matrix, SparseMatrix):
            return self._reduce_sparse(matrix, k)
        if isinstance(matrix, torch.Tensor):
            return self._reduce_dense(matrix, k)
        raise TypeError(f"reduce() expects a SparseMatrix or torch.Tensor, "
                        f"got {type(matrix).__name__}")

    def _reduce_sparse(self, matrix: SparseMatrix, k) -> SparseMatrix:
        phase = self._phase(k).to(self.master_dof.device)
        row = matrix.row.to(self.master_dof.device)
        col = matrix.col.to(self.master_dof.device)
        val = matrix.edata.to(torch.complex128)

        new_row = self.master_dof[row]
        new_col = self.master_dof[col]
        new_val = torch.conj(phase[row]) * phase[col] * val      # conj(phi_i) phi_j A_ij

        n = self.n_reduced_dof
        coo = torch.sparse_coo_tensor(
            torch.stack([new_row, new_col]), new_val, (n, n)).coalesce()
        idx = coo.indices()
        return SparseMatrix(coo.values(), idx[0], idx[1], (n, n))

    def _reduce_dense(self, A: torch.Tensor, k) -> torch.Tensor:
        phase = self._phase(k).to(A.device)
        n = self.n_reduced_dof
        W = (torch.conj(phase)[:, None] * phase[None, :]) * A.to(torch.complex128)
        md = self.master_dof.to(A.device)
        flat = (md[:, None] * n + md[None, :]).reshape(-1)       # scatter targets
        Ar = torch.zeros(n * n, dtype=torch.complex128, device=A.device)
        Ar.index_add_(0, flat, W.reshape(-1))
        return Ar.view(n, n)

    def reduce_system(self, K: SparseMatrix, M: SparseMatrix, k):
        """Convenience: reduce a stiffness/mass pair, ``(K_r, M_r)``."""
        return self.reduce(K, k), self.reduce(M, k)

    def recover(self, u_reduced: torch.Tensor, k) -> torch.Tensor:
        """Scatter a reduced-DOF field back to all DOFs with the Floquet phase.

        The scatter-back counterpart of :meth:`reduce`, named to mirror
        :meth:`~tensormesh.operator.condense.Condenser.recover`:

        ``u_full[i] = exp(sign i k·R_i) * u_reduced[master(i)]``.

        Parameters
        ----------
        u_reduced : torch.Tensor, shape ``[n_reduced_dof, ...]``
        k : array-like, shape ``[dim]``
        """
        assert u_reduced.shape[0] == self.n_reduced_dof, \
            f"u_reduced shape {tuple(u_reduced.shape)} != [{self.n_reduced_dof}, ...]"
        phase = self._phase(k).to(u_reduced.device)
        gathered = u_reduced[self.master_dof.to(u_reduced.device)]
        shape = [self.n_dof] + [1] * (u_reduced.dim() - 1)
        return phase.reshape(shape) * gathered


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


BlochReducer.__autodoc__ = ["reduce", "reduce_system", "recover"]

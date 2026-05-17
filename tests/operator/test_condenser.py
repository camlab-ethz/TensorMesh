"""Tests for the Condenser operator (boundary condition handling)."""

import sys
sys.path.append("../..")

import torch
import torch.nn as nn
import numpy as np
import pytest

from tensormesh import Mesh, Condenser
from tensormesh.sparse import SparseMatrix
from tensormesh.assemble import LaplaceElementAssembler


class TestCondenserBasic:
    """Basic Condenser functionality tests."""
    
    def test_condenser_creation(self):
        """Test Condenser initialization."""
        mask = torch.tensor([True, False, False, True])
        condenser = Condenser(mask)
        
        assert condenser.dirichlet_mask.shape == (4,)
        assert condenser.dirichlet_value.shape == (2,)  # 2 boundary nodes
        assert torch.all(condenser.dirichlet_value == 0)
    
    def test_condenser_with_values(self):
        """Test Condenser with non-zero Dirichlet values."""
        mask = torch.tensor([True, False, False, True])
        values = torch.tensor([1.0, 0.0, 0.0, 2.0])
        condenser = Condenser(mask, values)
        
        assert torch.allclose(condenser.dirichlet_value, torch.tensor([1.0, 2.0]))
    
    def test_condenser_with_partial_values(self):
        """Test Condenser with values only for boundary nodes."""
        mask = torch.tensor([True, False, False, True])
        values = torch.tensor([1.0, 2.0])  # Only boundary values
        condenser = Condenser(mask, values)
        
        assert torch.allclose(condenser.dirichlet_value, torch.tensor([1.0, 2.0]))


class TestCondenserOperations:
    """Tests for Condenser operations."""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple 4x4 sparse linear system."""
        # Simple tridiagonal matrix
        row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        col = torch.tensor([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
        val = torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.]).double()
        
        K = SparseMatrix(val, row, col, (4, 4))
        f = torch.tensor([1., 0., 0., 1.]).double()
        mask = torch.tensor([True, False, False, True])
        
        return K, f, mask
    
    def test_condense_matrix(self, simple_system):
        """Test matrix condensation."""
        K, f, mask = simple_system
        condenser = Condenser(mask)
        
        K_inner, f_inner = condenser(K, f)
        
        # Inner system should be 2x2 (2 free DOFs)
        assert K_inner.shape == (2, 2)
        assert f_inner.shape == (2,)
    
    def test_recover_solution(self, simple_system):
        """Test solution recovery."""
        K, f, mask = simple_system
        condenser = Condenser(mask, torch.tensor([0., 0., 0., 0.]).double())
        
        K_inner, f_inner = condenser(K, f)
        u_inner = K_inner.solve(f_inner)
        u_full = condenser.recover(u_inner)
        
        # Full solution should have 4 DOFs
        assert u_full.shape == (4,)
        # Boundary values should be 0
        assert torch.allclose(u_full[mask], torch.zeros(2).double(), atol=1e-10)
    
    def test_update_dirichlet(self, simple_system):
        """Test updating Dirichlet values."""
        K, f, mask = simple_system
        condenser = Condenser(mask)
        
        # Initial values are 0
        assert torch.all(condenser.dirichlet_value == 0)
        
        # Update values
        new_values = torch.tensor([1.0, 2.0])
        condenser.update_dirichlet(new_values)
        
        assert torch.allclose(condenser.dirichlet_value, new_values)


class TestCondenserRestrictProlong:
    """``restrict`` / ``prolong`` are pure linear projections (no Dirichlet correction)."""

    def test_restrict_drops_boundary_entries(self):
        mask  = torch.tensor([True, False, False, True])
        cd    = Condenser(mask, torch.tensor([99.0, 99.0]))  # nonzero dirichlet
        f     = torch.tensor([10., 1., 2., 20.]).double()
        # restrict must not subtract any Dirichlet correction term.
        assert torch.allclose(cd.restrict(f), torch.tensor([1., 2.]).double())

    def test_prolong_zeros_the_boundary(self):
        mask  = torch.tensor([True, False, False, True])
        cd    = Condenser(mask, torch.tensor([99.0, 99.0]))  # nonzero dirichlet
        f_in  = torch.tensor([1., 2.]).double()
        # boundary slots must be 0, not the Dirichlet value.
        assert torch.allclose(cd.prolong(f_in), torch.tensor([0., 1., 2., 0.]).double())

    def test_restrict_prolong_inverse_on_inner(self):
        mask = torch.tensor([True, False, False, True, False])
        cd   = Condenser(mask)
        f    = torch.tensor([10., 1., 2., 20., 3.]).double()
        # restrict followed by prolong zeroes the boundary entries but
        # leaves inner entries untouched.
        f_round = cd.prolong(cd.restrict(f))
        expected = f.clone(); expected[mask] = 0.
        assert torch.allclose(f_round, expected)

    def test_layout_not_required(self):
        """restrict / prolong only need dirichlet_mask, not a prior __call__."""
        mask = torch.tensor([True, False, False, True])
        cd   = Condenser(mask)
        # __call__ was never invoked: layout buffers are still None.
        assert cd.is_inner_dof is None
        f = torch.tensor([10., 1., 2., 20.]).double()
        cd.restrict(f)
        cd.prolong(torch.tensor([1., 2.]).double())


class TestCondenserWithMesh:
    """Integration tests with actual mesh."""
    
    def test_poisson_with_condenser(self):
        """Test solving Poisson equation with boundary conditions."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        
        # Assemble stiffness matrix
        K_asm = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=2)
        K = K_asm()
        
        # Create RHS
        f = torch.ones(mesh.n_points).double()
        
        # Apply boundary conditions
        condenser = Condenser(mesh.boundary_mask)
        K_inner, f_inner = condenser(K, f)
        
        # Solve
        u_inner = K_inner.solve(f_inner)
        u = condenser.recover(u_inner)
        
        # Check solution shape
        assert u.shape == (mesh.n_points,)
        
        # Boundary values should be 0
        assert torch.allclose(u[mesh.boundary_mask], torch.zeros(mesh.boundary_mask.sum()).double(), atol=1e-10)
        
        # Interior values should be positive (for this problem)
        interior_mask = ~mesh.boundary_mask
        assert torch.all(u[interior_mask] > 0)


class TestCondenserModule:
    """Verify that Condenser participates in the nn.Module ecosystem."""

    def test_is_nn_module(self):
        condenser = Condenser(torch.tensor([True, False, False, True]))
        assert isinstance(condenser, nn.Module)

    def test_no_learnable_parameters(self):
        condenser = Condenser(torch.tensor([True, False, False, True]))
        assert list(condenser.parameters()) == []

    def test_persistent_buffers_in_state_dict(self):
        condenser = Condenser(torch.tensor([True, False, False, True]))
        sd = condenser.state_dict()
        assert "dirichlet_mask"  in sd
        assert "dirichlet_value" in sd
        # Lazy index buffers are non-persistent and must not leak into state_dict.
        for name in ("inner_row", "inner_col",
                     "ou2in_row", "ou2in_col",
                     "is_inner_edge", "is_ou2in_edge",
                     "is_inner_dof", "is_outer_dof"):
            assert name not in sd, f"{name} should be a non-persistent buffer"

    def test_to_cpu_smoke(self):
        # Round-tripping through .to('cpu') is a no-op on CPU, but the call
        # exercises the buffer-iteration machinery that .to(other_device) uses.
        mask  = torch.tensor([True, False, False, True])
        value = torch.tensor([1.0, 2.0])
        condenser = Condenser(mask, value).to("cpu")
        assert condenser.dirichlet_mask.device.type == "cpu"
        assert condenser.dirichlet_value.device.type == "cpu"

    def test_lazy_buffers_move_with_module(self):
        """After a __call__ populates the lazy buffers, .to() must move them too."""
        K = SparseMatrix(
            torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.]).double(),
            torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]),
            torch.tensor([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]),
            (4, 4),
        )
        mask      = torch.tensor([True, False, False, True])
        condenser = Condenser(mask)
        condenser(K, torch.zeros(4).double())   # populates lazy buffers

        for name in ("inner_row", "inner_col", "is_inner_dof"):
            buf = getattr(condenser, name)
            assert buf is not None and buf.device.type == "cpu"

        condenser = condenser.to("cpu")  # idempotent on CPU, must not error
        assert condenser.inner_row.device.type   == "cpu"
        assert condenser.is_inner_dof.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_to_cuda_moves_all_buffers(self):
        mask      = torch.tensor([True, False, False, True])
        condenser = Condenser(mask, torch.tensor([1.0, 2.0])).cuda()

        assert condenser.dirichlet_mask.is_cuda
        assert condenser.dirichlet_value.is_cuda

        K = SparseMatrix(
            torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.]).double().cuda(),
            torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]).cuda(),
            torch.tensor([0, 1, 0, 1, 2, 1, 2, 3, 2, 3]).cuda(),
            (4, 4),
        )
        K_inner, _ = condenser(K, torch.ones(4).double().cuda())
        assert K_inner.edata.is_cuda
        for name in ("inner_row", "is_inner_dof"):
            assert getattr(condenser, name).is_cuda


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


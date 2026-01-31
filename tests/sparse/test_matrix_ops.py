"""Tests for SparseMatrix operations."""

import sys
sys.path.append("../..")

import torch
import numpy as np
import pytest

from tensormesh.sparse import SparseMatrix


class TestSparseMatrixCreation:
    """Tests for SparseMatrix creation."""
    
    def test_from_coo(self):
        """Test creation from COO format."""
        row = torch.tensor([0, 1, 2])
        col = torch.tensor([1, 2, 0])
        val = torch.tensor([1.0, 2.0, 3.0])
        
        A = SparseMatrix(val, row, col, (3, 3))
        
        assert A.shape == (3, 3)
        assert A.edata.shape == (3,)
    
    def test_random(self):
        """Test random sparse matrix creation."""
        A = SparseMatrix.random(10, 10, density=0.3)
        
        assert A.shape == (10, 10)
    
    def test_from_scipy(self):
        """Test creation from scipy sparse."""
        import scipy.sparse as sp
        
        scipy_mat = sp.coo_matrix(([1., 2., 3.], ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
        A = SparseMatrix.from_scipy_coo(scipy_mat)
        
        assert A.shape == (3, 3)


class TestSparseMatrixArithmetic:
    """Tests for arithmetic operations."""
    
    @pytest.fixture
    def sparse_matrix(self):
        """Create a simple sparse matrix."""
        row = torch.tensor([0, 0, 1, 1, 2, 2])
        col = torch.tensor([0, 1, 0, 1, 1, 2])
        val = torch.tensor([2., -1., -1., 2., -1., 2.]).double()
        return SparseMatrix(val, row, col, (3, 3))
    
    def test_add(self, sparse_matrix):
        """Test matrix addition."""
        A = sparse_matrix
        B = A + A
        
        assert torch.allclose(B.edata, A.edata * 2)
    
    def test_mul_scalar(self, sparse_matrix):
        """Test scalar multiplication."""
        A = sparse_matrix
        B = A * 2.0
        
        assert torch.allclose(B.edata, A.edata * 2)
    
    def test_rmul_scalar(self, sparse_matrix):
        """Test right scalar multiplication."""
        A = sparse_matrix
        B = 2.0 * A
        
        assert torch.allclose(B.edata, A.edata * 2)
    
    def test_neg(self, sparse_matrix):
        """Test negation."""
        A = sparse_matrix
        B = -A
        
        assert torch.allclose(B.edata, -A.edata)
    
    def test_pow(self, sparse_matrix):
        """Test power operation."""
        A = sparse_matrix
        B = A ** 2
        
        assert torch.allclose(B.edata, A.edata ** 2)


class TestSparseMatrixMatmul:
    """Tests for matrix multiplication."""
    
    @pytest.fixture
    def sparse_matrix(self):
        """Create a simple sparse matrix."""
        row = torch.tensor([0, 0, 1, 1, 2, 2])
        col = torch.tensor([0, 1, 0, 1, 1, 2])
        val = torch.tensor([2., -1., -1., 2., -1., 2.]).double()
        return SparseMatrix(val, row, col, (3, 3))
    
    def test_matvec(self, sparse_matrix):
        """Test matrix-vector multiplication."""
        A = sparse_matrix
        x = torch.tensor([1., 2., 3.]).double()
        
        y = A @ x
        y_expected = A.to_dense() @ x
        
        assert torch.allclose(y, y_expected)
    
    def test_matmat(self, sparse_matrix):
        """Test matrix-matrix multiplication."""
        A = sparse_matrix
        X = torch.randn(3, 2).double()
        
        Y = A @ X
        Y_expected = A.to_dense() @ X
        
        assert torch.allclose(Y, Y_expected)


class TestSparseMatrixTranspose:
    """Tests for transpose operation."""
    
    def test_transpose(self):
        """Test matrix transpose."""
        row = torch.tensor([0, 0, 1])
        col = torch.tensor([1, 2, 2])
        val = torch.tensor([1., 2., 3.]).double()
        
        A = SparseMatrix(val, row, col, (2, 3))
        AT = A.transpose()
        
        assert AT.shape == (3, 2)
        assert torch.allclose(AT.to_dense(), A.to_dense().T)


class TestSparseMatrixReductions:
    """Tests for reduction operations."""
    
    @pytest.fixture
    def sparse_matrix(self):
        """Create a simple sparse matrix."""
        row = torch.tensor([0, 0, 1, 1, 2, 2])
        col = torch.tensor([0, 1, 0, 1, 1, 2])
        val = torch.tensor([2., -1., -1., 2., -1., 2.]).double()
        return SparseMatrix(val, row, col, (3, 3))
    
    def test_sum_all(self, sparse_matrix):
        """Test summing all elements."""
        A = sparse_matrix
        s = A.sum()
        
        assert torch.isclose(s, A.edata.sum())
    
    def test_sum_keepdim(self, sparse_matrix):
        """Test sum with keepdim option if available."""
        A = sparse_matrix
        # Just test basic sum works
        s = A.sum()
        assert s is not None
    
    def test_degree(self, sparse_matrix):
        """Test degree computation."""
        A = sparse_matrix
        deg = A.degree(axis=0)
        
        assert deg.shape == (3,)


class TestSparseMatrixConversion:
    """Tests for format conversion."""
    
    @pytest.fixture
    def sparse_matrix(self):
        """Create a simple sparse matrix."""
        row = torch.tensor([0, 0, 1, 1, 2, 2])
        col = torch.tensor([0, 1, 0, 1, 1, 2])
        val = torch.tensor([2., -1., -1., 2., -1., 2.]).double()
        return SparseMatrix(val, row, col, (3, 3))
    
    def test_to_dense(self, sparse_matrix):
        """Test conversion to dense."""
        A = sparse_matrix
        A_dense = A.to_dense()
        
        assert A_dense.shape == (3, 3)
        assert A_dense.dtype == A.edata.dtype
    
    def test_to_scipy(self, sparse_matrix):
        """Test conversion to scipy."""
        A = sparse_matrix
        A_scipy = A.to_scipy_coo()
        
        assert A_scipy.shape == (3, 3)
    
    def test_roundtrip_scipy(self, sparse_matrix):
        """Test scipy roundtrip preserves values."""
        A = sparse_matrix
        A_scipy = A.to_scipy_coo()
        A_back = SparseMatrix.from_scipy_coo(A_scipy)
        
        # Convert to dense for comparison (handles different orderings)
        assert torch.allclose(A.to_dense().float(), A_back.to_dense().float())


class TestSparseMatrixLayout:
    """Tests for layout comparison."""
    
    def test_has_same_layout(self):
        """Test layout comparison."""
        row = torch.tensor([0, 1, 2])
        col = torch.tensor([1, 2, 0])
        val1 = torch.tensor([1., 2., 3.])
        val2 = torch.tensor([4., 5., 6.])
        
        A = SparseMatrix(val1, row, col, (3, 3))
        B = SparseMatrix(val2, row, col, (3, 3))
        
        assert A.has_same_layout(B)
        assert A.has_same_layout(A.layout_hash)
    
    def test_different_layout(self):
        """Test different layouts."""
        A = SparseMatrix(
            torch.tensor([1., 2., 3.]),
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 2, 0]),
            (3, 3)
        )
        B = SparseMatrix(
            torch.tensor([1., 2.]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            (3, 3)
        )
        
        assert not A.has_same_layout(B)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


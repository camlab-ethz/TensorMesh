"""Tests for the Mesh class."""

import sys
sys.path.append("../..")

import torch
import numpy as np
import pytest
import os
import tempfile

from tensormesh import Mesh


class TestMeshCreation:
    """Tests for Mesh creation and generation."""
    
    def test_gen_rectangle(self):
        """Test rectangle mesh generation."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        
        assert mesh.n_points > 0
        assert mesh.n_elements > 0
        assert mesh.dim == 2
        assert mesh.default_element_type == "triangle"
    
    def test_gen_rectangle_quad(self):
        """Test rectangle mesh with quad elements."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="quad")
        
        assert mesh.default_element_type == "quad"
    
    def test_gen_circle(self):
        """Test circle mesh generation."""
        mesh = Mesh.gen_circle(chara_length=0.2, cx=0.5, cy=0.5, r=0.5)
        
        assert mesh.n_points > 0
        assert mesh.dim == 2
    
    def test_gen_cube(self):
        """Test cube mesh generation."""
        mesh = Mesh.gen_cube(chara_length=0.3)
        
        assert mesh.n_points > 0
        assert mesh.dim == 3
        assert mesh.default_element_type == "tetra"


class TestMeshProperties:
    """Tests for Mesh properties."""
    
    @pytest.fixture
    def mesh_2d(self):
        return Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
    
    @pytest.fixture
    def mesh_3d(self):
        return Mesh.gen_cube(chara_length=0.4)
    
    def test_n_points(self, mesh_2d):
        """Test n_points property."""
        assert mesh_2d.n_points == mesh_2d.points.shape[0]
    
    def test_n_elements(self, mesh_2d):
        """Test n_elements property."""
        elements = mesh_2d.elements()
        assert mesh_2d.n_elements == elements.shape[0]
    
    def test_dim(self, mesh_2d, mesh_3d):
        """Test dim property."""
        assert mesh_2d.dim == 2
        assert mesh_3d.dim == 3
    
    def test_device(self, mesh_2d):
        """Test device property."""
        assert mesh_2d.device == torch.device("cpu")
    
    def test_dtype(self, mesh_2d):
        """Test dtype property."""
        assert mesh_2d.dtype in [torch.float32, torch.float64]
    
    def test_boundary_mask(self, mesh_2d):
        """Test boundary_mask property."""
        mask = mesh_2d.boundary_mask
        
        assert mask.shape == (mesh_2d.n_points,)
        assert mask.dtype == torch.bool
        assert mask.any()  # At least some boundary nodes


class TestMeshRepr:
    """Tests for Mesh string representation."""
    
    def test_repr_basic(self):
        """Test __repr__ doesn't crash."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        repr_str = repr(mesh)
        
        assert "Mesh" in repr_str
        assert "points" in repr_str
        assert "cells" in repr_str
    
    def test_repr_empty_data(self):
        """Test __repr__ with empty cell_data."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        # Clear cell_data to test empty handling
        mesh.cell_data.clear()
        
        # Should not crash
        repr_str = repr(mesh)
        assert "Mesh" in repr_str


class TestMeshIO:
    """Tests for Mesh I/O operations."""
    
    def test_save_load_vtu(self):
        """Test saving and loading VTU format."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.vtu")
            mesh.save(filepath)
            
            assert os.path.exists(filepath)
            
            # Load back
            mesh_loaded = Mesh.read(filepath)
            assert mesh_loaded.n_points == mesh.n_points
    
    def test_to_meshio(self):
        """Test conversion to meshio format."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        meshio_mesh = mesh.to_meshio()
        
        assert meshio_mesh.points.shape[0] == mesh.n_points


class TestMeshElements:
    """Tests for elements() method."""
    
    def test_elements_default(self):
        """Test elements() with default type."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        elements = mesh.elements()
        
        assert isinstance(elements, torch.Tensor)
        assert elements.dim() == 2
    
    def test_elements_by_type(self):
        """Test elements() with specific type."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        elements = mesh.elements("triangle")
        
        assert isinstance(elements, torch.Tensor)
    
    def test_elements_all(self):
        """Test that mesh.cells contains all element types."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        # mesh.cells is a BufferDict containing all element types
        assert hasattr(mesh.cells, 'keys')
        assert "triangle" in mesh.cells.keys()


class TestMeshClone:
    """Tests for mesh cloning."""
    
    def test_clone(self):
        """Test mesh cloning."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        original_point = mesh.points[0, 0].item()
        mesh_cloned = mesh.clone()
        
        assert mesh_cloned.n_points == mesh.n_points
        assert mesh_cloned.n_elements == mesh.n_elements
        
        # Clone should have same coordinates
        assert torch.allclose(mesh_cloned.points, mesh.points, atol=1e-6)


class TestMeshPointData:
    """Tests for point data registration."""
    
    def test_register_point_data(self):
        """Test registering point data."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        
        data = torch.randn(mesh.n_points)
        # Use a unique key name
        key = "test_data_unique_12345"
        mesh.register_point_data(key, data)
        
        assert key in mesh.point_data.keys()
    
    def test_register_point_data_wrong_size(self):
        """Test registering point data with wrong size."""
        mesh = Mesh.gen_rectangle(chara_length=0.3, element_type="tri")
        
        data = torch.randn(mesh.n_points + 10)  # Wrong size
        
        with pytest.raises(AssertionError):
            mesh.register_point_data("bad_data", data)


class TestMeshPlot:
    """Tests for Mesh.plot static visualization."""

    def test_plot_single_field_dict(self):
        """plot with a one-key dict must not fail (ncols=1 Axes handling)."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        x, y = mesh.points[:, 0], mesh.points[:, 1]
        u = torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = f.name

        try:
            mesh.plot({"u": u}, save_path=save_path, show_mesh=True, show=False)
            assert os.path.isfile(save_path)
            assert os.path.getsize(save_path) > 0
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


import meshio
import numpy as np
import torch
import os 
import gmsh 
import re
import warnings
if __name__ == '__main__':
    import sys 
    sys.path.append("..")
    from mesh import Mesh
else:
    from .mesh import Mesh


DIGIT = 4

def rectangle(chara_length=0.1,
             order=1,
             cell_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
             visualize=False,
             cache_path=".gmsh_cache/tmp.msh"):
    """
        Parameters:
        -----------
            chara_length: float
                the characteristic length of the mesh
            order: int
                the order of the mesh
            cell_type: str
                the type of the element, e.g., 'quad', 'tri'
            left: float
                the left boundary of the rectangle
            right: float
                the right boundary of the rectangle
            bottom: float
                the bottom boundary of the rectangle
            top: float
                the top boundary of the rectangle
            visualize: bool
                whether to visualize the mesh
            cache_path: str
                the path to store the mesh
        Returns:
        --------
            None
    """
    assert left < right, f"left must be smaller than right, but got {left} >= {right}"
    assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert cell_type in ["quad", "tri"], f"cell_type must be 'quad' or 'tri', but got {cell_type}"
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    width, height = right - left, top - bottom

    gmsh.initialize()
    gmsh.model.add("rectangle")

    rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)

    gmsh.model.occ.synchronize()

    if cell_type == "quad":
        # Set transfinite meshing
        gmsh.model.mesh.setTransfiniteSurface(rectangle, "Right")
        # Apply the recombine algorithm to generate quad elements
        gmsh.model.mesh.setRecombine(2, rectangle)

    # Set the element order to 2 to generate second-order elements
    gmsh.option.setNumber("Mesh.ElementOrder", order)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    if visualize:
        gmsh.fltk.run()

    # Save the mesh
    gmsh.write(cache_path)

    # Finalize Gmsh
    gmsh.finalize()

    mesh = Mesh.from_file(cache_path)

    is_left_boundary  = mesh.points[:, 0] == left
    is_right_boundary = mesh.points[:, 0] == right
    is_bottom_boundary= mesh.points[:, 1] == bottom
    is_top_boundary   = mesh.points[:, 1] == top
    is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_left_boundary", is_left_boundary)
    mesh.register_point_data("is_right_boundary", is_right_boundary)
    mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
    mesh.register_point_data("is_top_boundary", is_top_boundary)

    return mesh

if __name__ == '__main__':
    mesh = rectangle(cell_type="quad", chara_length=0.1, order=2, visualize=False)
    print(mesh)
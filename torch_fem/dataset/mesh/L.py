import meshio
import numpy as np
import torch
import os 
import gmsh 
import re
import warnings
if __name__ == '__main__':
    import sys 
    sys.path.append("../..")
    from mesh import Mesh
else:
    from ...mesh import Mesh

def gen_L(chara_length=0.1,
             order=1,
             element_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0, 
             top_inner=0.5,
             right_inner=0.5,
             visualize=False,
             cache_path=None):
    """
        Parameters:
        -----------
            chara_length: float
                the characteristic length of the mesh
            order: int
                the order of the mesh
            element_type: str
                the type of the element, e.g., 'quad', 'tri'
            left: float
                the left boundary of the Lshape
            right: float
                the right boundary of the Lshape
            bottom: float
                the bottom boundary of the Lshape
            top: float
                the top boundary of the Lshape
            top_inner: float
                the top inner boundary of the Lshape
            right_inner:
                the right inner boundary of the Lshape
            visualize: bool
                whether to visualize the mesh
            cache_path: str
                the path to store the mesh
        Returns:
        --------
            Mesh
    """
    assert left < right, f"left must be smaller than right, but got {left} >= {right}"
    assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

    if cache_path is None:
        cache_path = f".gmsh_cache/L_{left}_{right}_{bottom}_{top}_{top_inner}_{right_inner}_{chara_length}_{order}_{element_type}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):

        width, height = right - left, top - bottom

        gmsh.initialize()
        gmsh.model.add("rectangle")

        rectangle_outer = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
        rectangle_inner = gmsh.model.occ.addRectangle(right_inner, top_inner, 0, right-right_inner, top-top_inner)

        gmsh.model.occ.synchronize()

        _ = gmsh.model.occ.cut([(2,rectangle_outer)], [(2,rectangle_inner)])

        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            # gmsh.model.mesh.setTransfiniteSurface(rectangle_outer, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, rectangle_outer)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(2, [rectangle_outer])
        gmsh.model.setPhysicalName(2, 1, "domain")

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
    is_L_top_boundary = mesh.points[:, 1] == top_inner
    is_L_right_boundary = mesh.points[:, 0] == right_inner
    is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary | is_L_top_boundary | is_L_right_boundary
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_left_boundary", is_left_boundary)
    mesh.register_point_data("is_right_boundary", is_right_boundary)
    mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
    mesh.register_point_data("is_top_boundary", is_top_boundary)
    mesh.register_point_data("is_L_top_boundary", is_L_top_boundary)
    mesh.register_point_data("is_L_right_boundary", is_L_right_boundary)

    return mesh


if __name__ == '__main__':
    mesh = gen_L(element_type="quad", chara_length=0.1, order=2, visualize=False)
    print(mesh)
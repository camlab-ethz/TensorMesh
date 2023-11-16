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

def gen_rectangle(chara_length=0.1,
             order=1,
             element_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0,
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
            Mesh
    """
    assert left < right, f"left must be smaller than right, but got {left} >= {right}"
    assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

    if cache_path is None:
        cache_path = f".gmsh_cache/rectangle_{left}_{right}_{bottom}_{top}_{chara_length}_{order}_{element_type}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):

        width, height = right - left, top - bottom

        gmsh.initialize()
        gmsh.model.add("rectangle")

        rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)

        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            gmsh.model.mesh.setTransfiniteSurface(rectangle, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, rectangle)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(2, [rectangle])
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
    is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_left_boundary", is_left_boundary)
    mesh.register_point_data("is_right_boundary", is_right_boundary)
    mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
    mesh.register_point_data("is_top_boundary", is_top_boundary)

    return mesh

def gen_hollow_rectangle(chara_length=0.1,
             order=1,
             element_type="quad",
             outer_left=0.0, outer_right=1.0, outer_bottom=0.0, outer_top=1.0,
             inner_left = 0.25,  inner_right=0.75,
             inner_bottom =0.25, inner_top=0.75,
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
            outer_left: float
                the left boundary of the rectangle
            outer_right: float
                the right boundary of the rectangle
            outer_bottom: float
                the bottom boundary of the rectangle
            outer_top: float
                the outer_top boundary of the rectangle
            inner_left:float
                the inner left boundary of the rectangle
            inner_right:float 
                the  inner right boundary of the rectangle
            inner_bottom:float 
                the inner bottom boundary of the rectangle
            inner_top:float 
                the inner top boundary of the rectangle
            visualize: bool
                whether to visualize the mesh
            cache_path: str
                the path to store the mesh
    """
    assert inner_left < inner_right, f"inner_left must be smaller than inner_right, but got {inner_left} >= {inner_right}"
    assert inner_bottom < inner_top, f"inner_bottom must be smaller than inner_top, but got {inner_bottom} >= {inner_top}"
    assert inner_left > outer_left, f"inner_left must be greated than left, but got {inner_left} <= {outer_left}"
    assert inner_right < outer_right, f"inner_right must be smaller than right, but got {inner_right} >= {outer_right}"
    assert inner_bottom > outer_bottom, f"inner_bottom must be greater than bottom, but  got {inner_bottom} <= {outer_bottom}"
    assert inner_top < outer_top, f"inner_top must be smaller than outer_top, but got {inner_top} > {outer_top}"
    assert outer_left < outer_right, f"left must be smaller than right, but got {outer_left} >= {outer_right}"
    assert outer_bottom < outer_top, f"outer_bottom must be smaller than outer_top, but got {outer_bottom} >= {outer_top}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

    if cache_path is None:
        cache_path = f".gmsh_cache/hollow_rectangle_{outer_left}_{outer_right}_{outer_bottom}_{outer_top}_{inner_left}_{inner_right}_{inner_bottom}_{inner_top}_{chara_length}_{order}_{element_type}.msh"
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):

        width, height = outer_right - outer_left, outer_top - outer_bottom
        inner_width, inner_height = inner_right - inner_left, inner_top - inner_bottom
        gmsh.initialize()
        gmsh.model.add("rectangle")

        rectangle_outer = gmsh.model.occ.addRectangle(outer_left, outer_bottom, 0, width, height)
        rectangle_inner = gmsh.model.occ.addRectangle(inner_left, inner_bottom, 0, inner_width, inner_height)

        gmsh.model.occ.synchronize()

        _ = gmsh.model.occ.cut([(2,rectangle_outer)], [(2,rectangle_inner)])

        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            # gmsh.model.mesh.setTransfiniteSurface(rectangle, "Right")
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

    is_outer_left_boundary  = mesh.points[:, 0] == outer_left
    is_outer_right_boundary = mesh.points[:, 0] == outer_right
    is_outer_bottom_boundary= mesh.points[:, 1] == outer_bottom
    is_outer_top_boundary   = mesh.points[:, 1] == outer_top
    is_inner_left_boundary   = mesh.points[:,0] == inner_left
    is_inner_right_boundary  = mesh.points[:,0] == inner_right 
    is_inner_bottom_boundary = mesh.points[:,1] == inner_bottom
    is_inner_top_boundary    = mesh.points[:,1] == inner_top
    is_outer_boundary       = is_outer_left_boundary | is_outer_right_boundary | is_outer_bottom_boundary | is_outer_top_boundary
    is_inner_boundary       = is_inner_left_boundary | is_inner_right_boundary | is_inner_bottom_boundary | is_inner_top_boundary
    is_boundary             = is_inner_boundary | is_outer_boundary
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_inner_left_boundary", is_inner_left_boundary)
    mesh.register_point_data("is_outer_left_boundary", is_outer_left_boundary)
    mesh.register_point_data("is_inner_right_boundary", is_inner_right_boundary)
    mesh.register_point_data("is_outer_right_boundary", is_outer_right_boundary)
    mesh.register_point_data("is_inner_bottom_boundary", is_inner_bottom_boundary)
    mesh.register_point_data("is_outer_bottom_boundary", is_outer_bottom_boundary)
    mesh.register_point_data("is_inner_top_boundary", is_inner_top_boundary)
    mesh.register_point_data("is_outer_top_boundary", is_outer_top_boundary)

    return mesh


if __name__ == '__main__':
    mesh = gen_hollow_rectangle(element_type="quad", chara_length=0.1, order=2, visualize=False)
    print(mesh)
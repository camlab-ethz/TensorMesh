import meshio
import numpy as np
import torch
import os
import re
import warnings
if __name__ == '__main__':
    import sys 
    sys.path.append("../..")
    from mesh import Mesh
else:
    from ...mesh import Mesh
    from ._boundary import boundary_tolerance, near, within, register_boundary_masks

def gen_L(chara_length=0.1,
             order=1,
             element_type="quad",
             left=0.0, right=1.0, bottom=0.0, top=1.0, 
             top_inner=0.5,
             right_inner=0.5,
             visualize=False,
             cache_path=None):
    """
    Parameters
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

    Returns
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
        import gmsh

        width, height = right - left, top - bottom

        gmsh.initialize()
        gmsh.model.add("rectangle")

        rectangle_outer = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
        rectangle_inner = gmsh.model.occ.addRectangle(right_inner, top_inner, 0, right-right_inner, top-top_inner)

        gmsh.model.occ.synchronize()

        cut_result = gmsh.model.occ.cut([(2,rectangle_outer)], [(2,rectangle_inner)])
        gmsh.model.occ.synchronize()
        rectangle_outer_cut = cut_result[0][0][1]
        rectangle_inner_cut = cut_result[1][0][0][1]


        if element_type == "quad":
            # Set transfinite meshing
            # gmsh.model.mesh.setTransfiniteSurface(rectangle_outer, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, rectangle_outer_cut)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)
        boundary_lines_outer = gmsh.model.getBoundary([(2, rectangle_outer_cut)], oriented=False)
        boundary_lines_inner = gmsh.model.getBoundary([(2, rectangle_inner_cut)], oriented=False)
        boundary_lines = boundary_lines_outer + boundary_lines_inner
        line_group = gmsh.model.addPhysicalGroup(1, [line[1] for line in boundary_lines])
        gmsh.model.setPhysicalName(1, line_group, "boundary")

        gmsh.model.addPhysicalGroup(2, [rectangle_outer_cut])
        gmsh.model.setPhysicalName(2, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        if visualize:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(cache_path)

        # Finalize Gmsh
        gmsh.finalize()

    mesh = Mesh.from_file(cache_path,  reorder=True)

    tol = boundary_tolerance(chara_length)
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    register_boundary_masks(
        mesh,
        is_left_boundary=near(x, left, tol),
        is_right_boundary=near(x, right, tol),
        is_bottom_boundary=near(y, bottom, tol),
        is_top_boundary=near(y, top, tol),
        # the re-entrant sides: only the segments bounding the cut-out corner
        is_L_top_boundary=near(y, top_inner, tol) & (x >= right_inner - tol),
        is_L_right_boundary=near(x, right_inner, tol) & (y >= top_inner - tol),
    )

    return mesh


if __name__ == '__main__':
    mesh = gen_L(element_type="quad", chara_length=0.1, order=2, visualize=False)
    print(mesh)
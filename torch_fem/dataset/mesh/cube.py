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

def gen_cube(chara_length=0.1,
             order=1,
             left=0.0, right=1.0, 
             bottom=0.0, top=1.0,
             front=0.0, back=1.0,
             visualize=False,
             cache_path=None):
    """
    Parameters
    ----------
    chara_length: float, optional
        the characteristic length of the mesh,
        default: :obj:`0.1`
    order: int, optional
        the order of the basis function,
        default: :obj:`1`
    left: float, optional
        the left boundary of the cube,
        default: :obj:`0.0`
    right: float, optional
        the right boundary of the cube,
        default: :obj:`1.0`
    bottom: float, optional
        the bottom boundary of the cube,
        default: :obj:`0.0`
    top: float, optional
        the top boundary of the cube,
        default: :obj:`1.0`
    front: float, optional
        the front boundary of the cube,
        default: :obj:`0.0`
    back: float, optional
        the back boundary of the cube,
        default: :obj:`1.0`
    visualize: bool, optional
        whether to visualize the mesh,
        default: :obj:`False`
    cache_path: str, optional
        the path to save the mesh, if :obj:`None`, it will be decided by :meth:`torch_fem.dataset.mesh.gen_cube`,
        default: :obj:`None`
    Returns
    -------
    torch_fem.mesh.Mesh
        the mesh object
    """
    assert left < right, f"left must be smaller than right, but got {left} >= {right}"
    assert bottom < top, f"bottom must be smaller than top, but got {bottom} >= {top}"
    assert front < back, f"front must be smaller than back, but got {front} >= {back}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    

    cache_path = cache_path or f".gmsh_cache/cube_{left}_{right}_{bottom}_{top}_{front}_{back}_{chara_length}_{order}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):
        width, height,depth = right - left, top - bottom, back - front

        gmsh.initialize()
        gmsh.model.add("cube")

        cube = gmsh.model.occ.addBox(left, bottom, front, width, height, depth)

        gmsh.model.occ.synchronize()

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(3, [cube])
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(3)

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
    is_front_boundary = mesh.points[:, 2] == front
    is_back_boundary  = mesh.points[:, 2] == back
    is_boundary       = is_left_boundary | is_right_boundary | is_bottom_boundary | is_top_boundary | is_front_boundary | is_back_boundary
    mesh.register_point_data("is_boundary", is_boundary)
    mesh.register_point_data("is_left_boundary", is_left_boundary)
    mesh.register_point_data("is_right_boundary", is_right_boundary)
    mesh.register_point_data("is_bottom_boundary", is_bottom_boundary)
    mesh.register_point_data("is_top_boundary", is_top_boundary)
    mesh.register_point_data("is_front_boundary", is_front_boundary)
    mesh.register_point_data("is_back_boundary", is_back_boundary)
    return mesh


def gen_hollow_cube(chara_length=0.1,
             order=1,
             outer_left=0.0, outer_right=1.0, 
             outer_bottom=0.0, outer_top=1.0,
             outer_front=0.0, outer_back=1.0,
             inner_left=0.25, inner_right=0.75,
             inner_bottom=0.25, inner_top=0.75,
             inner_front=0.25, inner_back=0.75,
             visualize=False,
             cache_path=".gmsh_cache/tmp.msh"):
    """
    Parameters
    ----------
    chara_length: float, optional
        the characteristic length of the mesh,
        default: :obj:`0.1`
    order: int, optional
        the order of the basis function,
        default: :obj:`1`
    outer_left: float, optional
        the left boundary of the outer cube,
        default: :obj:`0.0`
    outer_right: float, optional
        the right boundary of the outer cube,
        default: :obj:`1.0`
    outer_bottom: float, optional
        the bottom boundary of the outer cube,
        default: :obj:`0.0`
    outer_top: float, optional
        the top boundary of the outer cube,
        default: :obj:`1.0`
    outer_front: float, optional
        the front boundary of the outer cube,
        default: :obj:`0.0`
    outer_back: float, optional
        the back boundary of the outer cube,
        default: :obj:`1.0`
    inner_left: float, optional
        the left boundary of the inner cube,
        default: :obj:`0.25`
    inner_right: float, optional
        the right boundary of the inner cube,
        default: :obj:`0.75`
    inner_bottom: float, optional
        the bottom boundary of the inner cube,
        default: :obj:`0.25`
    inner_top: float, optional
        the top boundary of the inner cube,
        default: :obj:`0.75`
    inner_front: float, optional
        the front boundary of the inner cube,
        default: :obj:`0.25`
    inner_back: float, optional
        the back boundary of the inner cube,
        default: :obj:`0.75`
    visualize: bool, optional
        whether to visualize the mesh,
        default: :obj:`False`
    cache_path: str, optional
        the path to save the mesh, if :obj:`None`, it will be decided by :meth:`torch_fem.dataset.mesh.gen_hollow_cube`,
        default: :obj:`None`

    Returns
    -------
    torch_fem.mesh.Mesh
        the mesh object 
    """
    assert outer_left < inner_left < inner_right < outer_right, f"outer_left < inner_left  < inner_right < outer_right, but got {outer_left} < {inner_left} < {inner_right} < {outer_right}"
    assert outer_bottom < inner_bottom < inner_top < outer_top, f"outer_bottom < inner_bottom < inner_top < outer_top, but got {outer_bottom} < {inner_bottom} < {inner_top} < {outer_top}"
    assert outer_front < inner_front < inner_back < outer_back, f"outer_front < inner_front < inner_back < outer_back, but got {outer_front} < {inner_front} < {inner_back} < {outer_back}"

    if cache_path is None:
        cache_path = f".gmsh_cache/cube_{outer_left}_{outer_right}_{outer_bottom}_{outer_top}_{outer_front}_{outer_back}_{inner_left}_{inner_right}_{inner_bottom}_{inner_top}_{inner_front}_{inner_back}_{chara_length}_{order}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):

        outer_width, outer_height, outer_depth = outer_right - outer_left, outer_top - outer_bottom, outer_back - outer_front
        inner_width, inner_height, inner_depth = inner_right - inner_left, inner_top - inner_bottom, inner_back - inner_front

        gmsh.initialize()
        gmsh.model.add("cube")

        cube_outer = gmsh.model.occ.addBox(outer_left, outer_bottom, outer_front, outer_width, outer_height, outer_depth)
        cube_inner = gmsh.model.occ.addBox(inner_left, inner_bottom, inner_front, inner_width, inner_height, inner_depth)
        
        gmsh.model.occ.synchronize()

        _ = gmsh.model.occ.cut([(3, cube_outer)], [(3, cube_inner)])

        gmsh.model.occ.synchronize()

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(3, [cube_outer])
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(3)

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
    is_outer_front_boundary = mesh.points[:, 2] == outer_front 
    is_outer_back_boundary  = mesh.points[:, 2] == outer_back
    is_inner_left_boundary  = mesh.points[:, 0] == inner_left
    is_inner_right_boundary = mesh.points[:, 0] == inner_right 
    is_inner_bottom_boundary= mesh.points[:, 1] == inner_bottom
    is_inner_top_boundary   = mesh.points[:, 1] == inner_top
    is_inner_front_boundary = mesh.points[:, 2] == inner_front
    is_inner_back_boundary  = mesh.points[:, 2] == inner_back
    is_outer_boundary       = is_outer_left_boundary | is_outer_right_boundary | is_outer_bottom_boundary | is_outer_top_boundary | is_outer_front_boundary | is_outer_back_boundary
    is_inner_boundary       = is_inner_left_boundary | is_inner_right_boundary | is_inner_bottom_boundary | is_inner_top_boundary | is_inner_front_boundary | is_inner_front_boundary
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
    mesh.register_point_data("is_inner_front_boundary", is_inner_front_boundary)
    mesh.register_point_data("is_outer_front_boundary", is_outer_front_boundary)
    mesh.register_point_data("is_inner_back_boundary", is_inner_back_boundary)
    mesh.register_point_data("is_outer_back_boundary", is_outer_back_boundary)

    return mesh 

if __name__ == '__main__':
    mesh = gen_hollow_cube(chara_length=0.1, order=2, visualize=False)
    print(mesh)
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

def gen_circle(chara_length=0.1,
             order=1,
             element_type="tri",
             cx = 0.0, cy = 0.0, r = 1.0,
             visualize=False,
             cache_path=None,
             verbose=False):
    """
    Parameters
    -----------
        chara_length: float
            The characteristic length of the mesh
        order: int
            The order of the elements
        element_type: str
            The type of the elements. Must be one of "quad" or "tri"
        cx: float
            The x-coordinate of the center of the circle
        cy: float
            The y-coordinate of the center of the circle
        r: float
            The radius of the circle
        visualize: bool
            Whether to visualize the mesh
        cache_path: str
            The path to save the mesh
        verbose: bool
            Whether to print detailed information

    Returns
    --------
        None
    """
    assert r > 0, f"r must be positive, but got {r} <= 0"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"

    if cache_path is None:
        cache_path = f".gmsh_cache/circle_{cx}_{cy}_{r}_{chara_length}_{order}_{element_type}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):
        import gmsh

        gmsh.initialize()
        gmsh.model.add("Circle")

        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        else:
            gmsh.option.setNumber("General.Terminal", 1)

        circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)

        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            gmsh.model.mesh.setTransfiniteSurface(circle, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, circle)
           
        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        boundary_lines = gmsh.model.getBoundary([(2, circle)], oriented=False)
        line_group = gmsh.model.addPhysicalGroup(1, [line[1] for line in boundary_lines])
        gmsh.model.setPhysicalName(1, line_group, "boundary")

        gmsh.model.addPhysicalGroup(2, [circle])
        gmsh.model.setPhysicalName(2, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        if visualize:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(cache_path)

        # Finalize Gmsh
        gmsh.finalize()

    mesh = Mesh.from_file(cache_path, reorder=True)

    # Topological detection: ``radius == r`` missed the boundary nodes that
    # gmsh places one ulp off the circle (7 of 53 at chara_length=0.06).
    register_boundary_masks(mesh)

    if verbose:
        print(f"Generated circle mesh with center ({cx}, {cy}), radius {r}, characteristic length {chara_length}, order {order}, element type {element_type}")

    return mesh

def gen_hollow_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None,
             verbose=False):
    """
    Parameters
    -----------
        chara_length: float
            The characteristic length of the mesh
        order: int
            The order of the elements
        element_type: str
            The type of the elements. Must be one of "quad" or "tri"
        cx: float
            The x-coordinate of the center of the circle
        cy: float
            The y-coordinate of the center of the circle
        r_inner: float
            The inner radius of the circle
        r_outer: float
            The outer radius of the circle
        visualize: bool
            Whether to visualize the mesh
        cache_path: str
            The path to save the mesh
        verbose: bool
            Whether to print detailed information

    Returns
    --------
        Mesh
    """

    assert r_inner > 0, f"r_inner must be positive, but got {r_inner} <= 0"
    assert r_outer > 0, f"r_outer must be positive, but got {r_outer} <= 0"
    assert r_outer > r_inner, f"r_outer must be greater than r_inner, but got {r_outer} <= {r_inner}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"
   
    if cache_path is None:
        cache_path = f".gmsh_cache/hollow_circle_{cx}_{cy}_{r_inner}_{r_outer}_{chara_length}_{order}_{element_type}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):
        import gmsh

        gmsh.initialize()
        gmsh.model.add("HollowCircle")

        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        else:
            gmsh.option.setNumber("General.Terminal", 1)

        circle_inner = gmsh.model.occ.addDisk(cx, cy, 0, r_inner, r_inner,)
        circle_outer = gmsh.model.occ.addDisk(cx, cy, 0, r_outer, r_outer)

        gmsh.model.occ.synchronize()

        # occ.cut CONSUMES both inputs; ``circle_outer`` / ``circle_inner``
        # are stale after it. Work with the cut result only.
        cut, _ = gmsh.model.occ.cut([(2, circle_outer)], [(2, circle_inner)])
        gmsh.model.occ.synchronize()
        surfaces = [tag for dim, tag in cut if dim == 2]

        if element_type == "quad":
            # Apply the recombine algorithm to generate quad elements
            for surface in surfaces:
                gmsh.model.mesh.setRecombine(2, surface)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        boundary_lines = gmsh.model.getBoundary([(2, s) for s in surfaces], oriented=False)
        line_group = gmsh.model.addPhysicalGroup(1, [tag for dim, tag in boundary_lines if dim == 1])
        gmsh.model.setPhysicalName(1, line_group, "boundary")

        surface_group = gmsh.model.addPhysicalGroup(2, surfaces)
        gmsh.model.setPhysicalName(2, surface_group, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        if visualize:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(cache_path)

        # Finalize Gmsh
        gmsh.finalize()

    mesh = Mesh.from_file(cache_path, reorder=True)

    tol = boundary_tolerance(chara_length)
    radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2)
    register_boundary_masks(
        mesh,
        is_inner_boundary=near(radius, r_inner, tol),
        is_outer_boundary=near(radius, r_outer, tol),
    )

    if verbose:
        print(f"Generated hollow circle mesh with center ({cx}, {cy}), inner radius {r_inner}, outer radius {r_outer}, characteristic length {chara_length}, order {order}, element type {element_type}")

    return mesh

if __name__ == '__main__':
    mesh = gen_hollow_circle(element_type="quad", chara_length=0.1, order=2, visualize=False, verbose=True)
    print(mesh)
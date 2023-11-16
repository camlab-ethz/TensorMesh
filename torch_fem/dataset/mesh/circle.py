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

def gen_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r = 1.0,
             visualize=False,
             cache_path=None):
    """
        Parameters:
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
        Returns:
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

        gmsh.initialize()
        gmsh.model.add("Circle")

        circle = gmsh.model.occ.addDisk(cx, cy, 0, r)

        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            gmsh.model.mesh.setTransfiniteSurface(circle, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, circle)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

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

    mesh = Mesh.from_file(cache_path)

    radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2)
    is_boundary = radius == r
    mesh.register_point_data("is_boundary", is_boundary)

    return mesh

def gen_hollow_circle(chara_length=0.1,
             order=1,
             element_type="quad",
             cx = 0.0, cy = 0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
    """
        Parameters:
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
        Returns:
        --------
            Mesh
    """

    assert r_inner > 0, f"r_inner must be positive, but got {r_inner} <= 0"
    assert r_outer > 0, f"r_outer must be positive, but got {r_outer} <= 0"
    assert r_outer > r_inner, f"r_outer must be greater than r_inner, but got {r_outer} <= {r_inner}"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"
    assert element_type in ["quad", "tri"], f"element_type must be 'quad' or 'tri', but got {element_type}"
   
    if cache_path is None:
        cache_path = f".gmsh_cache/circle_{cx}_{cy}_{r_inner}_{r_outer}_{chara_length}_{order}_{element_type}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):

        gmsh.initialize()
        gmsh.model.add("HollowCircle")

        circle_inner = gmsh.model.occ.addDisk(cx, cy, 0, r_inner, r_inner,)
        circle_outer = gmsh.model.occ.addDisk(cx, cy, 0, r_outer, r_outer)

        gmsh.model.occ.synchronize()

        hollow_entity, _ = gmsh.model.occ.cut([(2, circle_outer)], [(2, circle_inner)])
        hollow_circle = hollow_entity[0][-1]
        gmsh.model.occ.synchronize()

        if element_type == "quad":
            # Set transfinite meshing
            # gmsh.model.mesh.setTransfiniteSurface(circle_outer, "Right")
            # Apply the recombine algorithm to generate quad elements
            gmsh.model.mesh.setRecombine(2, circle_outer)

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(2, [circle_inner])
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

    radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2)
    is_inner_boundary = torch.isclose(radius, torch.ones_like(radius) * r_inner)
    is_outer_boundary = torch.isclose(radius, torch.ones_like(radius) * r_outer)
    is_boundary = is_inner_boundary | is_outer_boundary
    mesh.register_point_data("is_inner_boundary", is_inner_boundary)
    mesh.register_point_data("is_outer_boundary", is_outer_boundary)
    mesh.register_point_data("is_boundary", is_boundary)

    return mesh

if __name__ == '__main__':
    mesh = gen_hollow_circle(element_type="quad", chara_length=0.1, order=2, visualize=False)
    print(mesh)
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

def gen_sphere(chara_length=0.1,
             order=1,
              cx = 0.0, cy = 0.0, cz=0.0, r = 1.0,
             visualize=False,
             cache_path=None):
    """
    Parameters
    -----------
        chara_length: float
            the characteristic length of the mesh
        order: int
            the order of the mesh
        cx: float
            the x cooridnate of the center of the sphere
        cy: float
            the y cooridnate of the center of the sphere
        cz: float
            the z coorindate of the center of the sphere
        r:  float
            the radius  of the sphere
        visualize: bool
            whether to visualize the mesh
        cache_path: str
            the path to store the mesh

    Returns
    --------
        None
    """
    assert r > 0, f"r must be positive, but got {r} <= 0"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"

    if cache_path is None:
        cache_path = f".gmsh_cache/sphere_{cx}_{cy}_{cz}_{r}_{chara_length}_{order}.msh"

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):
        import gmsh

        gmsh.initialize()
        gmsh.model.add("cube")

        sphere = gmsh.model.occ.addSphere(cx, cy, cz, r)

        gmsh.model.occ.synchronize()

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        gmsh.model.addPhysicalGroup(3, [sphere])
        gmsh.model.setPhysicalName(3, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(3)

        if visualize:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(cache_path)

        # Finalize Gmsh
        gmsh.finalize()

    # reorder=True was missing here: at order=2 the tetra10 edge nodes
    # came out in gmsh order and every cell was silently distorted.
    mesh = Mesh.from_file(cache_path, reorder=True)

    register_boundary_masks(mesh)
    return mesh


def gen_hollow_sphere(chara_length=0.1,
             order=1,
              cx = 0.0, cy = 0.0, cz=0.0, r_inner = 1.0, r_outer = 2.0,
             visualize=False,
             cache_path=None):
    """
    Parameters
    -----------
        chara_length: float
            the characteristic length of the mesh
        order: int
            the order of the mesh
        cx: float
            the x cooridnate of the center of the sphere
        cy: float
            the y cooridnate of the center of the sphere
        cz: float
            the z coorindate of the center of the sphere
        r_inner:  float
            the inner radius of the sphere
        r_outer: float 
            the outer radius of the sphere

    Returns
    --------
        None
    """
    assert r_outer > r_inner, f"r_outer must be grearter than r_inner, but got {r_outer} <= {r_inner}"
    assert r_inner > 0, f"r_inner must be positive, but got {r_inner} <= 0"
    assert chara_length > 0, f"chara_length must be positive, but got {chara_length} <= 0"

    if cache_path is None:
        cache_path = f".gmsh_cache/hollow_sphere_{cx}_{cy}_{cz}_{r_inner}_{r_outer}_{chara_length}_{order}.msh"
   
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if not os.path.exists(cache_path):
        import gmsh

        gmsh.initialize()
        gmsh.model.add("cube")

        sphere_inner = gmsh.model.occ.addSphere(cx, cy, cz, r_inner)
        sphere_outer = gmsh.model.occ.addSphere(cx, cy, cz, r_outer)

        gmsh.model.occ.synchronize()

        # occ.cut CONSUMES both inputs; work with the cut result only.
        cut, _ = gmsh.model.occ.cut([(3, sphere_outer)], [(3, sphere_inner)])
        gmsh.model.occ.synchronize()
        volumes = [tag for dim, tag in cut if dim == 3]

        # Set the element order to 2 to generate second-order elements
        gmsh.option.setNumber("Mesh.ElementOrder", order)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)

        volume_group = gmsh.model.addPhysicalGroup(3, volumes)
        gmsh.model.setPhysicalName(3, volume_group, "domain")


        # Generate the mesh
        gmsh.model.mesh.generate(3)

        if visualize:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(cache_path)

        # Finalize Gmsh
        gmsh.finalize()

    mesh = Mesh.from_file(cache_path,  reorder=True)

    tol = boundary_tolerance(chara_length)
    radius = torch.sqrt((mesh.points[:, 0] - cx)**2 + (mesh.points[:, 1] - cy)**2 + (mesh.points[:, 2] - cz)**2)
    register_boundary_masks(
        mesh,
        is_inner_boundary=near(radius, r_inner, tol),
        is_outer_boundary=near(radius, r_outer, tol),
    )
    return mesh

if __name__ == '__main__':
    mesh = gen_hollow_sphere(chara_length=0.1, order=2, visualize=True)
    print(mesh)
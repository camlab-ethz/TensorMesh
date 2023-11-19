import gmsh 
import os
from ...mesh import Mesh
from ...shape import element_type2dimension, element_type2order


class MeshGen:
    """
    Parameters
    ----------
    element_type : str, optional
        If ``element_type`` is ``None``, then it will generate a mix mesh. 
        Otherwise, the order and element will be determined by ``element_type``.
    dimension : int, optional
        The dimension of the mesh, e.g., :math:`2` or :math:`3`.
    order : int, optional
        The order of the element, e.g., :math:`1`, :math:`2`.
    chara_length : float, optional
        The characteristic length of the mesh. The smaller the value, the more dense 
        the mesh. Default is :math:`0.1`.

    
    Examples
    --------

    1. generate rectangle mesh with triangle elements:

        .. code-block:: python

            from torch_fem import MeshGen
            generator = MeshGen(element_type="tri") # triangle mesh for 2d
            generator.addRectangle(0,0,1,1) # add a rectangle
            mesh = generator.gen().plot() # generate and visualize the mesh
        
    2. generate mixed mesh with left triangle and right rectangle 

        .. code-block:: python    
            
            from torch_fem import MeshGen
            mesh_gen = MeshGen(element_type=None, chara_length=0.1, order=2)
            mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
            mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
            mesh_gen.remove_circle(0.5,0.5,0.1)
            mesh_gen.gen().plot()


    """
    def __init__(self, element_type=None, dimension=2, order=1, chara_length=0.1, cache_path="./tmp.msh"):
        if element_type is not None:
            order     = element_type2order[element_type]
            dimension = element_type2dimension[element_type]
        
        self.dimension = dimension
        self.order     = order
        self.chara_length = chara_length
        self.element_type = element_type
        self.cache_path   = cache_path
        gmsh.initialize()
        gmsh.model.add("geometry")

        self.objects = {}
        self.default_objects = []
        self.quad_objects = []
           
    def add_rectangle(self, left, bottom, width, height, element="tri"):
        """add a rectangle to the geometry
        Parameters
        ----------
        left: float
            the left boundary of the rectangle
        bottom: float
            the bottom boundary of the rectangle
        width: float
            the width of the rectangle
        height: float
            the height of the rectangle
        element: str, optional
            the type of the element, can be "tri" or "quad"
            default: "tri"

        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        if self.element_type is not None:
            element = "tri" if self.element_type.startswith("triangle") else "quad"
        assert element in ["tri", "quad"]
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]rectangle({left},{bottom},{width},{height})"
        self.default_objects.append(name)
        self.objects[name] = (2,rectangle)
        if element == "quad":
            # gmsh.model.mesh.setTransfiniteSurface(rectangle)
            self.quad_objects.append(name)
        return self

    def remove_rectangle(self, left, bottom, width, height):
        """remove the rectangle from the geometry
        Parameters
        ----------
        left: float
            the left boundary of the rectangle
        bottom: float
            the bottom boundary of the rectangle
        width: float
            the width of the rectangle
        height: float
            the height of the rectangle
        
        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(2,rectangle)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]rectangle({left},{bottom},{width},{height})"
        self.objects[name] = (2,rectangle)
        return self

    def add_circle(self, cx, cy, r, element="tri"):
        """add a circle to the geometry
        Parameters
        ----------
        cx: float
            the x coordinate of the center
        cy: float
            the y coordinate of the center
        r: float
            the radius of the circle
        element: str, optional
            the type of the element, can be "tri" or "quad"
            default: "tri"

        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        if self.element_type is not None:
            element = "tri" if self.element_type.startswith("triangle") else "quad"
        assert element in ["tri", "quad"]
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]circle({cx},{cy},{r})"
        self.default_objects.append(name)
        self.objects[name] = (2,circle)
        if element == "quad":
            gmsh.model.mesh.setRecombine(2, circle)
        return self

    def remove_circle(self, cx, cy, r):
        """remove the cirlce from the geometry
        Parameters
        ----------
        cx: float
            the x coordinate of the center
        cy: float
            the y coordinate of the center
        r: float
            the radius of the circle

        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(2,circle)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]circle({cx},{cy},{r})"
        self.objects[name] = (2,circle)
        return self

    def add_cube(self, x, y, z, dx, dy, dz):
        """add a cube to the geometry, only works for 3d
        Parameters
        ----------
        x: float
            the x coordinate of the center
        y: float
            the y coordinate of the center
        z: float
            the z coordinate of the center
        dx: float
            the width of the cube
        dy: float
            the height of the cube
        dz: float
            the depth of the cube

        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        cube = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        name = f"[{len(self.objects)}]cube({x},{y},{z},{dx},{dy},{dz})"
        self.default_objects.append(name)
        self.objects[name] = (3,cube)
        return self

    def remove_cube(self, x, y, z, dx, dy, dz):
        """remove the cube from the geometry, only works for 3d
        Parameters
        ----------
        x: float
            the x coordinate of the center
        y: float
            the y coordinate of the center
        z: float
            the z coordinate of the center
        dx: float
            the width of the cube
        dy: float
            the height of the cube
        dz: float
            the depth of the cube

        Returns
        -------
        MeshGen
            the mesh generator itself        
        """
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        cube = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(3,cube)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]cube({x},{y},{z},{dx},{dy},{dz})"
        self.objects[name] = (3,cube)
        return self

    def add_sphere(self, x, y, z, r):
        """add a sphere to the geometry, only works for 3d
        Parameters
        ----------
        x: float
            the x coordinate of the center
        y: float
            the y coordinate of the center
        z: float
            the z coordinate of the center
        r: float
            the radius of the sphere

        Returns
        -------
        MeshGen
            the mesh generator itself
        """
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        sphere = gmsh.model.occ.addSphere(x, y, z, r)
        name = f"[{len(self.objects)}]sphere({x},{y},{z},{r})"
        self.default_objects.append(name)
        self.objects[name] = (3,sphere)
        return self

    def remove_sphere(self, x, y, z, r):
        """remove the sphere from the geometry, only works for 3d
        Parameters
        ----------
        x: float
            the x coordinate of the center
        y: float
            the y coordinate of the center
        z: float
            the z coordinate of the center
        r: float
            the radius of the sphere

        Returns
        -------
        MeshGen
            the mesh generator itself

        """
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        sphere = gmsh.model.occ.addSphere(x, y, z, r)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(3,sphere)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]sphere({x},{y},{z},{r})"
        self.objects[name] = (3,sphere)
        return self

    def gen(self, show=False):
        """generate the mesh from the geometry
        Parameters
        ----------
        show: bool, optional
            whether to show the mesh in the gmsh gui
            default: :obj:`False`

        Returns
        -------
        torch_fem.mesh.Mesh
            the generated mesh
        
        """
        if self.element_type is None:
            for obj in self.quad_objects:
                gmsh.model.mesh.setRecombine(*self.objects[obj])
        elif self.element_type.startswith("quad"):
            for obj in self.default_objects:
                gmsh.model.mesh.setRecombine(*self.objects[obj])
        
        gmsh.option.setNumber("Mesh.ElementOrder", self.order)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.chara_length)

        gmsh.model.addPhysicalGroup(self.dimension, [self.objects[i][1] for i in self.default_objects])
        gmsh.model.setPhysicalName(self.dimension, 1, "domain")

        # Generate the mesh
        gmsh.model.mesh.generate(self.dimension)

        if show:
            gmsh.fltk.run()

        # Save the mesh
        gmsh.write(self.cache_path)

        # Finalize Gmsh
        gmsh.finalize()

        mesh = Mesh.from_file(self.cache_path)

        os.remove(self.cache_path)

        return mesh
import gmsh 
import os
from ...mesh import Mesh
from ...shape import element_type2dimension, element_type2order


class MeshGen:
    def __init__(self, element_type=None, dimension=2, order=1, chara_length=0.1, cache_path="./tmp.msh"):
        """
            Parameters:
            -----------
                element_type: str
                    if element_type is None, then it will generate mix mesh 
                    else, the order, element will be find out by element_type
                dimension: int
                    the dimension of the mesh
                order: int
                    the order of the mesh
                chara_length: float
                    the characteristic length of the mesh
                    default: 0.1
        
            Usage:
            >>> generator = MeshGenerator(dimension=2) # mixed mesh for 2d
            >>> generator.addRectangle(0,0,1,1) # add a rectangle
            >>> mesh = generator.gen() # generate the mesh

            >>> generator = MeshGenerator(element_type="triangle") # triangle mesh 
            >>> generator.addRectangle(0,0,1,1) # add a rectangle
            >>> generator.removeCircle(0.5,0.5,0.2) # remove a circle
            >>> mesh = generator.gen() # generate the mesh
        """
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
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        rectangle = gmsh.model.occ.addRectangle(left, bottom, 0, width, height)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(2,rectangle)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]rectangle({left},{bottom},{width},{height})"
        self.objects[name] = (2,rectangle)
        return self

    def add_circle(self, cx, cy, r, element="tri"):
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
        assert self.dimension == 2, f"dimension must be 2, but got {self.dimension}"
        circle = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(2,circle)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]circle({cx},{cy},{r})"
        self.objects[name] = (2,circle)
        return self

    def add_cube(self, x, y, z, dx, dy, dz):
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        cube = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        name = f"[{len(self.objects)}]cube({x},{y},{z},{dx},{dy},{dz})"
        self.default_objects.append(name)
        self.objects[name] = (3,cube)
        return self

    def remove_cube(self, x, y, z, dx, dy, dz):
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        cube = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(3,cube)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]cube({x},{y},{z},{dx},{dy},{dz})"
        self.objects[name] = (3,cube)
        return self

    def add_sphere(self, x, y, z, r):
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        sphere = gmsh.model.occ.addSphere(x, y, z, r)
        name = f"[{len(self.objects)}]sphere({x},{y},{z},{r})"
        self.default_objects.append(name)
        self.objects[name] = (3,sphere)
        return self

    def remove_sphere(self, x, y, z, r):
        assert self.dimension == 3, f"dimension must be 3, but got {self.dimension}"
        sphere = gmsh.model.occ.addSphere(x, y, z, r)
        difference, _ = gmsh.model.occ.cut([self.objects[i] for i in self.default_objects], [(3,sphere)])
        gmsh.model.occ.synchronize()
        name = f"[{len(self.objects)}]sphere({x},{y},{z},{r})"
        self.objects[name] = (3,sphere)
        return self

    def gen(self, show=False):
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
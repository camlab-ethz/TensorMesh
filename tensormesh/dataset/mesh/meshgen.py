import importlib.util
import os
import sys
from warnings import warn
from ...mesh import Mesh
from ...element import element_type2dimension, element_type2order


def _lazy_import(name):
    """Defer loading a native module until first attribute access.
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        class _Missing:
            def __getattr__(self, attr):
                raise ImportError(
                    f"{name!r} is required for this functionality. "
                    f"Install it with: pip install {name}"
                )
        return _Missing()
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


gmsh = _lazy_import("gmsh")


abbr2element_type = {
    "tri"  : "triangle",
    "quad" : "quad",
    "tet"  : "tetra",
    "hex"  : "hexahedron",
    "pri"  : "pyramid", 
    "pyr"  : "wedge",
}
element_type2abbr = {v: k for k, v in abbr2element_type.items()}

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

            from tensormesh import MeshGen
            generator = MeshGen(element_type="tri") # triangle mesh for 2d
            generator.addRectangle(0,0,1,1) # add a rectangle
            mesh = generator.gen().plot() # generate and visualize the mesh
        
    2. generate mixed mesh with left triangle and right rectangle 

        .. code-block:: python    
            
            from tensormesh import MeshGen
            mesh_gen = MeshGen(element_type=None, chara_length=0.1, order=2)
            mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
            mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
            mesh_gen.remove_circle(0.5,0.5,0.1)
            mesh_gen.gen().plot()


    """
    def __init__(self, element_type=None,
                 dimension:int=2,
                 order:int=1,
                 chara_length:float=0.1,
                 cache_path:str="./tmp.msh",
                 verbose:bool = False):
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
        self.hex_objects = []

        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
        else:
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.Verbosity", 1)


           
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
            element = element_type2abbr[self.element_type]
        assert element in ["tri", "quad"], f"element should be `tri` or `quad`, got {element}"
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
            element = element_type2abbr[self.element_type]
        assert element in ["tri", "quad"], f"element should be `tri` or `quad` ,got {element}"
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

    def add_cube(self, x, y, z, dx, dy, dz, element:str="tet"):
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
        if self.element_type is not None:
            element = element_type2abbr[self.element_type]
        assert element in ["hex", "tet"], f"element should be in `hex`, `tet`, got {element}"
        cube = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        gmsh.model.occ.synchronize()
        
        name = f"[{len(self.objects)}]cube({x},{y},{z},{dx},{dy},{dz})"
        self.default_objects.append(name)
        self.objects[name] = (3, cube)
        
        if element == "hex":
            # Get boundary faces after synchronization
            self.hex_objects.append(name)
            
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

        not supported for hex element because of algorithm issue

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

        # Check if we're trying to use hex elements
        if self.element_type == "hexahedron" or any(obj in self.hex_objects for obj in self.default_objects):
            warn("Spheres cannot be properly meshed with hexahedral elements. Using tetrahedral elements instead.")

        sphere = gmsh.model.occ.addSphere(x, y, z, r)
        gmsh.model.occ.synchronize()
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
        tensormesh.Mesh
            the generated mesh

        """
        # for obj in self.quad_objects:
        #     gmsh.model.mesh.setRecombine(*self.objects[obj])
        # for obj in self.hex_objects:
        #     gmsh.option.setNumber("Mesh.RecombineAll", 1)
        #     gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Mesh.Algorithm3D = 1 for Delaunay
        #     faces = gmsh.model.occ.getEntities(2)
        #     for face in faces:
        #         gmsh.model.mesh.setRecombine(2, face[1])

        if self.quad_objects:
            # Apply recombination only to specific surfaces marked for quads
            for obj in self.quad_objects:
                dim, tag = self.objects[obj]
                gmsh.model.mesh.setRecombine(dim, tag)

        if self.hex_objects:
            # Settings for mixed 3D meshes
            for obj in self.hex_objects:
                dim, tag = self.objects[obj]
                # Set recombination for the volume
                gmsh.model.mesh.setRecombine(dim, tag)
                # Set recombination for all boundary faces of this volume
                faces = gmsh.model.getBoundary([(dim, tag)])
                for face in faces:
                    gmsh.model.mesh.setRecombine(2, abs(face[1]))
            
            # Additional settings for better mixed mesh generation
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom algorithm
            gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 1)
            # Disable pyramid generation for now to avoid non-manifold issues
            gmsh.option.setNumber("Mesh.Pyramids", 0)
        

        if self.dimension == 3:
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Use Frontal algorithm for hex meshes
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            
            # Additional settings for better hex mesh generation
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.QualityType", 2)  # SICN quality measure

            # Use only tetrahedral elements for transition
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    

            # Better handling of curved surfaces
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)
            gmsh.option.setNumber("Mesh.MinimumCurvePoints", 8)

            # Disable size extension from boundary for more uniform mesh
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        
            
            
            # Limit the number of optimization steps
            # gmsh.option.setNumber("Mesh.OptimizeMaxNbIterations", 50)
            
            if self.order > 2:
                print(f"Warning: Reducing order from {self.order} to 2 for 3d elements to ensure stable meshing")
                self.order = 2

        
        
        gmsh.option.setNumber("Mesh.ElementOrder", self.order)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.chara_length)
        
        # More controlled mesh size for hex elements
        if self.hex_objects:
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.chara_length * 0.8)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.chara_length * 1.2)
        
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

        mesh = Mesh.from_file(self.cache_path,  reorder=True)


        os.remove(self.cache_path)

        return mesh
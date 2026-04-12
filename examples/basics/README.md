# Visualization

Scripts for visualizing finite element building blocks in TensorMesh: basis points, shape functions, element node ordering, and mesh generation with field data.

## `basis.py` — Basis Point Locations

Plots the spatial distribution of interpolation nodes on the reference element for all seven element types (Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism) at orders 1--4.

### Usage

```bash
python basis.py
```

### Output

- `output/basis_points/<element>_comparison.png`: numbered node positions with element edges

## `basis_fn.py` — Basis Function Shapes

Visualizes the polynomial shape functions for each element type at orders 1--4. 1D basis functions are plotted as curves; 2D as 3D surfaces; 3D elements produce one figure per order.

### Usage

```bash
python basis_fn.py
```

### Output

- `output/linear.png`, `output/triangle.png`, `output/quadrilateral.png`: 1D/2D basis functions
- `output/<element>/<order>.png`: 3D basis functions

> **Note:** Orders 3--4 may produce warnings about basis function accuracy due to Vandermonde matrix ill-conditioning.

## `element_gallery.py` — Node Ordering (TensorMesh vs Gmsh/VTK)

Side-by-side comparison of TensorMesh internal node numbering versus Gmsh/VTK node numbering for 2D and 3D elements at orders 2--4. Useful for verifying element reordering when importing/exporting meshes.

### Usage

```bash
python element_gallery.py
```

### Output

- `<element>_p2p3p4_order_compare.png`: two-row layout (top: TensorMesh/FEniCS, bottom: Gmsh/VTK)

## `plot_mesh.py` — Mesh Generation and Field Visualization

Demonstrates `MeshGen`-based mesh generation and visualization: structured/unstructured meshes (triangle, quad, tet, hex), hybrid meshes with holes, node/element adjacency graphs, and scalar field plotting in 2D and 3D.

### Usage

```bash
python plot_mesh.py
```

### Output

- `output/rectangle_mesh.png`, `output/cube_mesh.png`, `output/circle_mesh.png`: basic meshes
- `output/hybrid_mesh2d.png`: mixed tri+quad mesh with circular hole
- `output/node_adj_2d.png`, `output/ele_adj_2d.png`: adjacency graphs (2D)
- `output/node_adj_3d.png`, `output/ele_adj_3d.png`: adjacency graphs (3D)
- `output/point_value_2d.png`, `output/point_value_3d.png`: nodal scalar field
- `output/element_value_2d.png`, `output/element_value_3d.png`: element-wise scalar field

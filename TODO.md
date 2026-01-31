# TODO List

## Core Features

### Element Support
- [x] Line (order 1-4)
- [x] Triangle (order 1-4)
- [x] Quadrilateral (order 1-4)
- [x] Tetrahedron (order 1-4)
- [x] Hexahedron (order 1-4)
- [x] Pyramid (order 1-4)
- [x] Prism (order 1-4)

### Assemblers
- [x] ElementAssembler (stiffness matrix assembly)
- [x] NodeAssembler (load vector assembly)
- [x] FacetAssembler (boundary integrals)
- [x] Built-in assemblers (Laplace, Mass, LinearElasticity)
- [x] Energy-based assembly (`ElementAssembler.energy()`)

### Sparse Solvers
- [x] SciPy backend
- [x] PETSc backend
- [x] CuPy/cuSOLVER backend
- [ ] Distributed solving (MPI)

### Mesh
- [x] Mesh generation (rectangle, circle, L-shape, cube, sphere, etc.)
- [x] Mixed element mesh support
- [x] Node/element adjacency
- [x] Graph partitioning
- [ ] Adaptive mesh refinement
- [ ] Distributed mesh

### Visualization
- [x] Matplotlib backend
- [x] PyVista backend
- [x] Animation support
- [x] Static plotting
- [ ] Interactive 3D visualization

## Examples

### Completed
- [x] Poisson equation (2D/3D)
- [x] Heat equation
- [x] Wave equation
- [x] Linear elasticity
- [x] Nonlinear Poisson
- [x] Solid mechanics (hyperelasticity, plasticity)
- [x] Topology optimization
- [x] Defect detection (inverse problem)

### Planned
- [ ] Heat equation PINN
- [ ] Wave equation PINN
- [ ] Dynamic elasticity
- [ ] Fluid-structure interaction
- [ ] Phase field fracture

## Documentation
- [x] API reference
- [x] Installation guide
- [x] Example tutorials
- [ ] Theory documentation
- [ ] Video tutorials

## Testing
- [x] Element tests
- [x] Assembler tests
- [x] Sparse solver tests
- [x] Mesh tests
- [ ] Integration tests
- [ ] Performance benchmarks

## Future
- [ ] Strong form to weak form automation
- [ ] Higher-order time integration
- [ ] Model order reduction
- [ ] Neural network integration examples

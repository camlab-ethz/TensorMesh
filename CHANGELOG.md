# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- PyVista visualization interface (`plot_value`, `mesh_to_pyvista`)
- `FacetAssembler` for boundary integrals
- Energy-based assembly with `ElementAssembler.energy()` method
- `compile()` method for `NodeAssembler` optimization
- High-order element support (up to order 4) for all element types
- Pyramid and Prism element types
- Topology optimization examples
- Solid mechanics examples (hyperelasticity, plasticity)
- Fluid mechanics examples (Navier-Stokes)

### Changed
- Improved documentation structure
- Updated visualization module exports

### Fixed
- Fixed typo in `facet_assembler.py` filename

## [0.1.0] - 2023-11-16

### Added 

- Element, Facet, Node Assembler 
- Sparse Matrix and sparse solver for different backend 
    - scipy
    - libtorch
    - petsc
    - cusolve
- Ordinary Differential Equation 
- Mesh generation / adjacency

### Changed

### Deprecated

### Fixed

### Removed

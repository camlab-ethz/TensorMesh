# Changelog

User-visible changes to TensorMesh (PyPI `tensormesh-fem`), newest first.
The layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/):
each release groups its lines under Added, Changed, Fixed and Removed.

A line that starts with **Breaking** changes numerical results, quadrature
point counts, a dependency floor or a public name. Read those lines before
upgrading.

## [Unreleased]

## [0.3.0] - 2026-11-21

### Added

- Exodus II mesh I/O. `Mesh.read` and `Mesh.save` handle `.e`, `.exo`, `.ex2`, `.exii`, `.gen` and `.g` files (or `file_format="exodus"`) through `tensormesh.mesh.exodus.read_exodus` / `write_exodus`, and keep the element-block ids (`cell_data["block_id"]`), node sets and side sets. Needs netCDF4: `pip install "tensormesh-fem[exodus]"`. Contributed by Max Nezdyur. ([#61])
- `Mesh.point_sets` and `Mesh.side_sets`, named sets of points and of `(cell, local facet)` pairs, with `register_point_set`, `register_side_set`, `point_set_mask` and `side_set_mask`. The `point_sets` of meshio meshes are kept as well. ([#61])
- `FacetAssembler.from_mesh(boundary_mask=...)` takes the name of a side set or of a point set, or a side-set mapping. A side set selects exactly its facets, interior ones included, which a node mask cannot do. ([#61])
- Quadrature rules of any order. Line rules above 7 points and triangle and tetrahedron rules above degree 7 are generated as Gauss–Jacobi conical products; quadrilateral, hexahedron, prism, pyramid and facet rules build on them. Contributed by Fabian Rohner. ([#72])
- `quadrature_data` argument of `ElementAssembler.__call__`, `ElementAssembler.energy` and `NodeAssembler.__call__`: tensors of shape `[n_element, n_quadrature, ...]` whose value at the current quadrature point reaches `forward` by name (consistent tangents, stresses, history variables). A key that repeats a `point_data`, `element_data` or `scalar_data` key, or a built-in name such as `u` or `x`, raises `ValueError`. ([#70])
- Facet integrands of `FacetAssembler` and `FacetBilinearAssembler` can take `n`, the outward unit normal at the facet quadrature point, also available as `Transformation.facet_normal`. A `point_data` key named `n` takes precedence. ([#69])
- `BlochReducer.reduce_rhs(b, k)` reduces a load vector to the master DOFs, and `BlochReducer.reduce_mask(mask)` maps a full-DOF mask onto the reduced DOFs. ([#65])
- `tensormesh.element.PyramidBasisFns`, the rational basis of the pyramid, with the `map` / `grad` interface of `Polynomials`. ([#64])
- Example and gallery page: Helmholtz scattering with exact Dirichlet-to-Neumann boundary conditions, for circular scatterers and for periodic slabs and walls (`examples/wave/helmholtz_dtn/`). Contributed by Yu Gao. ([#57], [#65])
- Example and gallery page: phase-field fracture of a single-edge-notched specimen with AT1 or AT2 damage, solved with `torch_sla.vi_solve` (`examples/solid/phase_field_fracture/`). ([#66])
- Example and gallery page: J2 plasticity of a thick-walled cylinder, solved by Newton's method with the consistent tangent (`examples/solid/plasticity_cylinder/`). ([#71])

### Changed

- **Breaking:** `quadrature_order`, and `order` in `Element.get_quadrature` / `get_facet_quadrature`, is the polynomial degree the rule integrates exactly on every shape. It used to count Gauss points per direction on lines, quadrilaterals, hexahedra and pyramids and in the height of prisms. On those shapes, and on the edge and quadrilateral facets of any cell, an order of 3 or more now selects `order // 2 + 1` points per direction instead of `order` (3 → 2, 4 → 3, 5 → 3, 6 → 4, 7 → 4), so integrals and the shape of per-quadrature-point data change. A point count `n` of the old convention is order `2 * n - 1` now. Triangle and tetrahedron rules and the default order 2 are unchanged. ([#73], [#78])
- `Mesh.read` on an Exodus file uses the new reader instead of meshio's. It always converts the connectivity to TensorMesh node ordering and ignores `reorder`. ([#61])
- `MixedElementAssembler.from_mesh` no longer rejects a default quadrature degree above 7, so fields of order 4 and higher work without an explicit `quadrature_order`. ([#72])
- An invalid quadrature order raises `ValueError` instead of `AssertionError`, and NumPy integers are accepted. ([#72])
- **Breaking:** torch-sla ≥ 0.4.0 is required (was ≥ 0.3.0), also by the `cudss`, `pyamg` and `gpu` extras and by `requirements.txt`. ([#67])
- An assembly call over all quadrature points (`batch_size=-1`, the `ElementAssembler` default) caches the shape gradients and `JxW` on the `Transformation`, and later calls reuse them instead of inverting every element Jacobian again. The cache stays in memory between calls. Chunked calls (`batch_size > 0`, the `NodeAssembler` default) cache nothing, as before. ([#68])
- Element Jacobians up to 3×3 are inverted in closed form. A degenerate element (zero Jacobian determinant) now raises `ValueError`. ([#68])

### Fixed

- **Breaking:** tetrahedron quadrature at orders 5, 6 and 7 was exact only to degrees 4, 5 and 6. These orders now use 15, 24 and 64 points (14, 15 and 24 before), so results on tetrahedra and the shape of per-quadrature-point data change at these orders. ([#72])
- **Breaking:** pyramid elements (`pyramid`, `pyramid14`) use the conforming rational space of Bergot, Cohen & Duruflé (2010) instead of a non-conforming polynomial space, and the pyramid quadrature weights now include the Jacobian of the collapsed-cube map. Without it every non-constant integrand was wrong (∫z dV came out as 1/6, exact value 1/12). Results on meshes with pyramid cells change. The number of quadrature points does not. `Pyramid.get_basis_fns` returns a `PyramidBasisFns` instead of a `Polynomials`. ([#64])
- **Breaking:** `FacetAssembler` and `FacetBilinearAssembler`, and with them `robin_operator` and `port_source`, integrate a facet only if a single cell references it and all of its nodes are in the boundary mask. An interior facet whose nodes are all flagged, such as the diagonal of a corner cell or a chord of a curved boundary, was integrated once from each side and is now skipped, so boundary integrals on such meshes change (perimeter of a 4×4 triangulated unit square: 5.414 before, 4 now). ([#69])
- **Breaking:** `Element.get_outwards_facet_normal` returns outward reference normals. It returned inward ones. ([#69])
- **Breaking:** facet quadrature is built in the dtype of the points instead of float32, which had limited float64 facet integrals to a relative accuracy of about 1e-7. Float64 boundary integrals change at that level. ([#69])
- **Breaking:** assemblers honour new `points` after the geometry has been cached. Before, `ElementAssembler` after an `energy()` call, `FacetAssembler` from its second call on, and a compiled `NodeAssembler` returned the result for the old coordinates. ([#68])
- With points that require grad, a second `backward` through an assembler no longer fails with "Trying to backward through the graph a second time", and new points are no longer ignored. ([#68])
- `FacetAssembler` runs on prism and pyramid cells. That path raised an error before. ([#69])
- `FacetAssembler.from_elements` no longer overwrites the entries of the caller's connectivity dict. ([#69])
- `ElementAssembler.energy` with `batch_size > 0` and per-quadrature-point `element_data` no longer fails with a shape error. ([#70])
- `plot_3d` evaluates basis functions only inside the reference element. Plots of tetrahedra, prisms and pyramids showed values outside it. ([#64])
- Test suite: the quadrature weight-sum tests compare float64 sums. `tests/quadrature/test_quad.py::test_sum[3]` failed on aarch64. ([#63])
- Test suite: `test_implicit_euler_with_bc_cuda` compares the two CUDA solves at the accuracy of the iterative solver instead of 1e-12. It failed on GPUs without cuDSS. ([#79])

## [0.2.1] - 2026-09-28

### Added

- `Mesh.topological_boundary_mask()` detects boundary nodes by facet incidence. `mesh.boundary_mask` falls back to it on imported meshes. ([#58])
- `is_inner_boundary` / `is_outer_boundary` masks on the hollow generators. ([#58])
- `Mesh.check_node_ordering`, run by the `Mesh` constructor. ([#58])
- `BufferDict` supports `in` and iteration over keys. ([#58])
- Fifteen Colab notebooks that cover the example gallery, linked from the README and the docs. ([#52], [#53], [#54], [#55])

### Changed

- **Breaking:** loading an order ≥ 2 Gmsh/VTK mesh without `reorder=True` raises `ValueError` instead of assembling a wrong geometry map without notice. A `.vtu` written by `mesh.save` is VTK-ordered, so read it back with `reorder=True`. ([#58])
- **Breaking:** the per-side generator masks (`is_left_boundary`, …) are tolerance predicates restricted to the boundary. An interior node that happened to sit on such a coordinate is no longer flagged. ([#58])

### Fixed

- Nodes that gmsh places one ulp off a curved boundary were missing from `is_boundary` (`gen_circle`: 7 of 53 at order 1, 16 of 106 at order 2), and their Dirichlet condition was dropped without notice. ([#58])
- `gen_hollow_rectangle` and `gen_hollow_circle` always raised `Unknown model face`, `gen_hollow_cube` returned a stale cached mesh, and `gen_sphere(order=2)` was never reordered. ([#58])
- `layout_signature` stays stable across vector reassembly, so one `Condenser` can be reused across time steps and Picard iterations. ([#51])

## [0.2.0] - 2026-07-22

### Added

- `MixedElementAssembler`: multi-field block systems written as one scalar integrand over fields declared with `Field(...)`. `assembler.layout` (`BlockLayout`) provides boundary masks, `split` / `cat` and `prolong` / `restrict`, and `assemble_vector` builds load vectors. Field order is independent of mesh order, so Taylor–Hood P2–P1 runs on linear Gmsh meshes. ([#38])
- `BlochReducer`: Bloch–Floquet periodic boundary conditions, with three phononic-crystal examples validated against COMSOL. Contributed by Nicolas Cayuela. ([#36])
- `tensormesh.distributed`: `DistributedMesh` partitioning (RCB, spectral, METIS), the `@distributed` assembler decorator, `DSparseMatrix` and `Condenser` dispatch, for multi-GPU runs under `torchrun`. Contributed by walkerchi.
- Open-domain wave operators: `FacetBilinearAssembler` for boundary bilinear forms, `cartesian_pml` with `AnisotropicLaplaceElementAssembler` / `ScaledMassElementAssembler`, and `robin_operator` / `port_source`, with four examples. Contributed by Nicolas Cayuela. ([#39])
- `DruckerPragerPlasticity` and the `FrictionalMaterial` presets are public API, with geomechanics examples (kilickursat). Allen–Cahn convex–concave splitting example (MariusZeinhofer).
- Complex dtypes are supported from assembly through condensation to the solve, with a Helmholtz example.
- The `[pyamg]` extra. `[gpu]` now installs cuDSS and PyAMG.
- User-guide chapters on mixed assembly and distributed FEM, and gallery chapters on phononic crystals, open-domain waves, modal analysis and Taylor–Hood Stokes, in English and Chinese.

### Changed

- **Breaking:** torch-sla ≥ 0.3.0 is required.
- The fluid examples use Taylor–Hood elements instead of stabilized P1–P1.

### Removed

- **Breaking:** the `[cupy]` extra, because torch-sla dropped its CuPy backend. Use the default CPU stack, `[cudss]`, `[pyamg]` or `[gpu]`.

### Fixed

- `MeshGen.remove_sphere` no longer hangs on spheres smaller than the mesh size. ([#47])

## [0.1.1] - 2026-05-26

### Fixed

- `import tensormesh` no longer leaves a broken gmsh `LazyLoader` stub in `sys.modules`. The stub caused `'_LazyModule' object has no attribute 'initialize'` on Python 3.12 with gmsh 4.15 (for example on Colab), which broke mesh generation and `python -m tensormesh.verify_install`.

## [0.1.0] - 2026-05-23

First public release, as `tensormesh-fem` on PyPI (import name `tensormesh`).

### Added

- Tensorized assembly ([TensorGalerkin](https://arxiv.org/abs/2602.05052)) of weak forms written in Python, with eager execution and autograd through assembly and solve, on CPU and GPU.
- Triangle, quadrilateral, tetrahedron, hexahedron, pyramid and prism elements, mesh generators, and Gmsh and VTKHDF5 I/O.
- Linear, nonlinear and eigenvalue solves through torch-sla (SciPy, native PyTorch, CuPy and cuDSS backends). The `[gpu]` extra installs CuPy and cuDSS.
- Boundary conditions and time integration.
- `python -m tensormesh.verify_install` to check an installation.

[Unreleased]: https://github.com/camlab-ethz/TensorMesh/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/camlab-ethz/TensorMesh/releases/tag/v0.3.0
[0.2.1]: https://github.com/camlab-ethz/TensorMesh/releases/tag/v0.2.1
[0.2.0]: https://github.com/camlab-ethz/TensorMesh/releases/tag/v0.2.0
[0.1.1]: https://github.com/camlab-ethz/TensorMesh/releases/tag/v0.1.1
[0.1.0]: https://github.com/camlab-ethz/TensorMesh/releases/tag/v0.1.0

[#36]: https://github.com/camlab-ethz/TensorMesh/pull/36
[#38]: https://github.com/camlab-ethz/TensorMesh/pull/38
[#39]: https://github.com/camlab-ethz/TensorMesh/pull/39
[#47]: https://github.com/camlab-ethz/TensorMesh/pull/47
[#51]: https://github.com/camlab-ethz/TensorMesh/pull/51
[#52]: https://github.com/camlab-ethz/TensorMesh/pull/52
[#53]: https://github.com/camlab-ethz/TensorMesh/pull/53
[#54]: https://github.com/camlab-ethz/TensorMesh/pull/54
[#55]: https://github.com/camlab-ethz/TensorMesh/pull/55
[#57]: https://github.com/camlab-ethz/TensorMesh/pull/57
[#58]: https://github.com/camlab-ethz/TensorMesh/pull/58
[#61]: https://github.com/camlab-ethz/TensorMesh/pull/61
[#63]: https://github.com/camlab-ethz/TensorMesh/pull/63
[#64]: https://github.com/camlab-ethz/TensorMesh/pull/64
[#65]: https://github.com/camlab-ethz/TensorMesh/pull/65
[#66]: https://github.com/camlab-ethz/TensorMesh/pull/66
[#67]: https://github.com/camlab-ethz/TensorMesh/pull/67
[#68]: https://github.com/camlab-ethz/TensorMesh/pull/68
[#69]: https://github.com/camlab-ethz/TensorMesh/pull/69
[#70]: https://github.com/camlab-ethz/TensorMesh/pull/70
[#71]: https://github.com/camlab-ethz/TensorMesh/pull/71
[#72]: https://github.com/camlab-ethz/TensorMesh/pull/72
[#73]: https://github.com/camlab-ethz/TensorMesh/discussions/73
[#78]: https://github.com/camlab-ethz/TensorMesh/pull/78
[#79]: https://github.com/camlab-ethz/TensorMesh/pull/79

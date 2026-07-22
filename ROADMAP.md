# Roadmap

Planned directions for core source code of TensorMesh, roughly ordered by priority. Solver-side items (backends, complex adjoint, multi-GPU) live in the [torch-sla ROADMAP](https://github.com/sparsexlab/torch-sla).

_Last updated: 2026-07._

## 1. Mixed (multi-field / block) assembly for Lagrange spaces — ✅ shipped

Shipped as `MixedElementAssembler` + `Field` (`tensormesh/assemble/mixed_assembler.py`): the fields are declared once (e.g. `Field(trial="u", test="v", order=2, components=2)` + `Field(trial="p", test="q", order=1)` for Taylor–Hood P2–P1) and the weak form is written as a single scalar integrand with tensor-valued trial/test arguments; the assembler extracts every coupling block by one-hot evaluation and scatters into one block `SparseMatrix`, while `assembler.layout` provides the block-DOF helpers (offsets, `dof_mask`/`dof_index`, `split`/`cat`, `restrict`/`prolong`) that the hand-rolled fluids assemblers used to reimplement — see the rewritten `examples/fluid/cavity/cavity.py` (true Taylor–Hood, stabilization-free). Field orders beyond {1, mesh order} are handled by item 2. The block machinery is the foundation that the complex (item 3) and mixed-element (item 5) work both reuse.

## 2. Generalized order pairs via topological Lagrange DOF maps — ✅ shipped (2D any order; 3D vertex+edge)

Decouple the **field order from the mesh order**: the mesh order (fixed at import, e.g. by gmsh) is a *geometry* property, while each field's DOF carriers are generated from mesh **topology** by `lagrange_dofmap` (`tensormesh/assemble/topology.py`) — one DOF per vertex, `order−1` per unique **oriented** edge (element-local slots flip when an element traverses the edge backwards, which is what keeps `order ≥ 3` spaces C⁰-conforming), plus per-cell interiors. Slots are classified by coordinate-matching each family's reference nodes, so no internal-ordering assumptions; the geometry map stays isoparametric at mesh order, making sub- *and* super-parametric fields correct on curved elements. This unlocks: Taylor–Hood P2–P1 **directly on linear gmsh imports** (no re-meshing at order 2), generalized pairs like P3–P2, and 3D P2-on-P1 tetrahedra — all verified to machine precision against exactly-representable manufactured solutions (`tests/assemble/test_generalized_pairs.py`). Supporting API: `layout.boundary_mask(f)` (topological facet-incidence boundary detection — no `is_boundary` point data or geometric tolerance needed), `layout.dof_mask(f, where=...)` coordinate predicates, generalized `points`/`restrict`/`prolong`, and the `field_data={"w": ("u", w)}` channel that interpolates Picard iterates with the owning field's own basis. Remaining: **3D face-DOF orientation** (P3+ tetra, P2+ hex) — deliberately deferred because it shares the facet-orientation layer with item 5; quadrature tables cap at degree 7, so pairs are practical up to P3 until the tables grow. Load vectors are assembled space-aware by `MixedElementAssembler.assemble_vector` (the linear-form counterpart of `__call__`: same argument dispatch and block layout, test arguments only, `field_data` supported) — see the rewritten transient `examples/fluid/cylinder_flow` for the full pattern.

## 3. Complex-valued FEM → Helmholtz, PML, metamaterial topology optimization

**Status**: scalar complex Helmholtz is **unblocked end-to-end** ([`examples/wave/helmholtz/`](examples/wave/helmholtz/)), and the **PML + open-boundary layer shipped** via PR #39: `cartesian_pml` (stretched-coordinate coefficients), `AnisotropicLaplaceElementAssembler` / `ScaledMassElementAssembler` (complex tensor/scalar coefficients), `FacetBilinearAssembler` (boundary bilinear forms), and `robin_operator` / `port_source` (first-order absorbing / plane-wave ports) — exercised by the port-driven Helmholtz resonator and the PML-framed optical ring resonator ([`examples/wave/`](examples/wave/)). The metamaterial TopOpt example is the remaining work.

Unblock the assembly stack for complex-valued systems so a complex element matrix can flow end-to-end into a complex-symmetric LDLᵀ / Hermitian LDLᴴ solve — enabling time-harmonic Helmholtz with PML and, on top of it, topology optimization of acoustic and (2D / scalar) electromagnetic metamaterials.

Why it fits the current architecture: PML is a **volume** modification (complex, anisotropic coefficients `A(x)`, `c(x)` in the coordinate-stretched layer), not a boundary condition, so it maps directly onto the existing `ElementAssembler` — a tensor-valued complex coefficient is expressible via `point_data` (the assembly `einsum` ellipsis already carries tensor fields). No new facet/boundary assembler is needed.

Assembly-side unblock landed (see commit history): `ElementAssembler.type()` now accepts complex dtypes; `Polynomial` / `Transformation` keep geometry / basis buffers real even when the assembler is cast to complex (complex content enters through `point_data`, not the mesh); `ReduceProjector`'s fp64 upcast picks `complex128` instead of stripping the imaginary part; `SparseProjector` honours its `dtype` parameter; and `ElementAssembler.__call__` einsum sites promote real `shape_val` / `shape_grad` / `jxw` to the coefficient's complex dtype on demand (`torch.einsum` doesn't auto-promote across complex / real the way `*` does).

Solver-side dependency (the complex solve **and the correct complex adjoint** — essential for TopOpt, where a wrong adjoint yields silently wrong design sensitivities) lives in torch-sla and has shipped — see torch-sla `linear_solve.py` / `nvmath_backend.py` for the complex LDLᵀ / LDLᴴ path and `tests/test_complex_support.py` for the gradcheck.

Remaining work for item 3:
- ~~PML example proper~~ — ✅ shipped in PR #39 (`cartesian_pml` + anisotropic/scaled assemblers; optical ring resonator example).
- Metamaterial TopOpt example wiring SIMP + adjoint through the complex Helmholtz path; the OC kernel may need swapping for MMA on wave objectives. Prerequisite worth adding first: an autograd check through `robin_operator` / the PML coefficient fields (the design sensitivities flow through both).

Topology-optimization scaffolding mostly exists: the density → SIMP → filter → OC pipeline is already proven on real problems (`tensormesh/optimizer/oc.py`, `examples/inverse_design/`). The wave objective is real (e.g. `|u|²` at a target point), so autograd's real-loss convention holds — but classic OC assumes monotone, compliance-like sensitivities, so a wave objective may want MMA instead.

Scope note: scalar complex Helmholtz (complex Lagrange) covers acoustics and 2D / scalar (TE/TM) electromagnetics. **Full-vector 3D electromagnetics** needs H(curl) Nédélec elements — gated on item 5.

## 4. Distributed mixed assembly (`@distributed` × `MixedElementAssembler`)

The two 2026-07 features don't compose yet: `distributed(MixedAssembler)` raises `TypeError` at the decorator's MRO gate (it accepts `ElementAssembler` / `NodeAssembler` subclasses only), and lifting the gate alone would not be enough — two structural gaps sit underneath:

- **Global block-DOF numbering.** The distributed merge remaps per-submesh COO triples to global coordinates through `orig_nid` (local node → global node), which assumes DOFs *are* mesh nodes. The mixed layout is `dof = offset_f + node_local·c_f + comp` with **global-N-dependent per-field offsets** (each submesh's `from_mesh` would compute subdomain-sized offsets), and generalized-order fields carry DOFs on **topological edges that have no cross-partition identity** — two ranks assign unrelated ids to the same partition-boundary edge.
- **Row-owner semantics.** `DSparseTensor.partition` assigns row ownership from node coordinates (RCB); block DOFs need carrier-based ownership instead (vertex DOFs → node owner; edge DOFs → e.g. min-rank of the edge's endpoints).

Implementation route (reuses the existing infrastructure): build the mixed `BlockLayout` **once on the global mesh** (pure topology — cheap and rank-deterministic), map each submesh's field DOFs into the global block numbering by identifying edges through their *global vertex pairs*, then feed the existing merge / all-to-all row-routing path; `assemble_vector` follows the same map. Land the nodes-kind-only case first (P2–P1 on order-2 meshes, where every field DOF is a mesh node and `orig_nid` + per-field `g2l` suffice), then the dofmap-kind fields as an increment. **Deliberately sequenced after** the single-field distributed examples are validated on a multi-GPU cluster, so failures attribute cleanly.

## 5. P0, then Raviart–Thomas (H(div)) and Nédélec (H(curl)) elements

TensorMesh today ships **continuous Lagrange nodal elements only** (`Line`, `Triangle`, `Quadrilateral`, `Tetrahedron`, `Hexahedron`, `Pyramid`, `Prism`, plus higher-order nodal variants), and the element abstraction is **nodal-Lagrange to the core**: scalar basis with the nodal interpolation property (`φᵢ(xⱼ) = δᵢⱼ`, see `element.py`), the standard covariant (gradient) map `∇ₓφ = J⁻ᵀ∇_ξφ`, and node-based global DOFs.

To enable structure-preserving discretization of problems that are naturally posed on H(div) and H(curl) (e.g. Maxwell's equations, Darcy's flow) the following types of elements will be implemented: 

- **Nédélec element**: H(curl)-conforming; tangential continuity across elements; DoFs defined on edges, faces and elements;
- **Raviart-Thomas**: H(div)-conforming; normal continuity across elements; DoFs defined on faces and elements;
- **Discontinous**: $L^2$-conforming; no continuity across elements; DoFs on elements.

This entails extension in the following three aspects:
- **Vector-valued** — `shape_val` gains a component dimension `[n_q, n_basis, dim]` and are defined by facet-flux moments `∫_facet v·n = δ`.
- **Piola transform** — The pullbacks of these vector-valued elements are different from the one for the nodal element.
- **Facet DOFs + orientation/sign** — global DOFs extends to facets of all dimensions, not just nodes. This requires a unique-facet enumeration as DOF carriers and a per-element ±1 sign convention so shared-facet normals agree. The geometric facet machinery (`get_facet`, `get_edge`, facet quadrature, `Transformation.facets`) already exists, and item 2 shipped the *unsigned Lagrange half* of the DOF layer (unique-edge enumeration + per-element orientation flip in `lagrange_dofmap`, facet-incidence boundary detection); what remains here is face enumeration/orientation and the ±1 sign convention (the projector currently scatters to node indices, and `reorder` handles node permutations only).

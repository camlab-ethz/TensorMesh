# Helmholtz (complex) — manufactured plane-wave solution

End-to-end validation of the complex-coefficient FEM assembly path
(ROADMAP item 2). Solves the interior Helmholtz problem

```
-Δu(x,y) - k² u(x,y) = 0       in Ω = (0,1)²
            u(x,y)   = g(x,y)  on ∂Ω
```

with the analytic plane wave `u_exact(x,y) = exp(i k x)` as the
Dirichlet boundary data — the body force is zero because the plane
wave satisfies the Helmholtz operator pointwise.

## Why this example exists

- Exercises a complex `point_data` coefficient (`k²`) flowing through
  `ElementAssembler.__call__` into the assembled SparseMatrix.
- Tests complex Dirichlet condensation in `Condenser`.
- Tests the complex linear solve via `SparseMatrix.solve` (delegates
  to torch-sla's complex factorizations).
- Reports L2 error against the analytic solution and shows the
  expected ~`O(h²)` convergence (modulo Helmholtz pollution at
  moderate k).

## Run it

```bash
python helmholtz.py                                  # default k=2π
python helmholtz.py --k 12.566 --chara-length 0.05   # k=4π
python helmholtz.py --no-plot                        # just the convergence table
```

Sample output (k = 2π, complex128):

```
h=0.200  n_dofs=  44  L2 err = 1.529e-01
h=0.100  n_dofs= 143  L2 err = 5.274e-02
h=0.050  n_dofs= 509  L2 err = 1.506e-02
h=0.025  n_dofs=1934  L2 err = 3.935e-03
```

## Cross-validation against scikit-fem

`tests/assemble/test_helmholtz_example.py::test_helmholtz_cross_validates_against_scikit_fem`
hands the same `(points, cells)` to scikit-fem's `MeshTri` and assembles
the same form with its built-in `laplace` / `mass` integrators, solves
via `scipy.sparse.linalg.spsolve`, and compares node-by-node.
On `h = 0.1` (143 nodes), `k = 2π`:

```
max |u_tensormesh - u_skfem|              = 2.3e-15
max |u_tensormesh - u_skfem| / max |u|    = 2.3e-15      # machine ε
both vs analytic exp(ikx)                 = 5.274e-02   # identical to 4 sig figs
```

i.e. the two pipelines agree to floating-point precision and inherit
the same discretisation error, which is the strongest correctness
signal available for the complex assembly path.

## What's next (PML follow-up)

The current example uses a constant scalar `k²` coefficient. The
infrastructure already supports anisotropic complex *tensor*
coefficients via `point_data` — the natural extension is to wrap the
domain in a PML absorbing layer with coordinate-stretched
`A(x), c(x)` and run a scattering example. See ROADMAP item 2.

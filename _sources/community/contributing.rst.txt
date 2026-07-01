Contributing
============

TensorMesh is small enough that one well-aimed patch can move the
needle. This page covers the practical loop: get a dev environment,
run the tests, build the docs, and open a PR.

If you have an idea but aren't sure where it fits, post on
:doc:`github_discussions` (*Ideas & RFCs* category) or drop a line on
:doc:`discord`. For confirmed bugs and specific feature requests, the
issue tracker is the right entry point — see :doc:`github_issues`.


Getting set up
--------------

Requirements:

* **Python ≥ 3.10**
* **PyTorch ≥ 2.0** (with CUDA if you want to run the GPU paths)

Everything else installs from PyPI in the step below — including
``torch-sla`` (``>= 0.2.1``), the hard dependency that provides every
sparse-solver backend. No C++ toolchain or manual build step is
required.

Clone and install in editable mode:

.. code-block:: bash

   git clone https://github.com/camlab-ethz/TensorMesh.git
   cd TensorMesh
   pip install -e ".[test]"

Optional extras: ``cupy`` / ``cudss`` (GPU sparse-direct solver
backends), ``gpu`` (both at once), and ``example`` (Plotly for some
figure scripts).

Smoke-test the install by running the bundled checker, which solves a
tiny Poisson problem on CPU (and on GPU if available) and reports
which sparse-solver backends are wired up:

.. code-block:: bash

   python -m tensormesh.verify_install

If both the CPU and the CUDA solve print sensible errors against the
analytical reference, you're good.


Running the tests
-----------------

Tests live under ``tests/``, mirroring the package layout
(``tests/element/``, ``tests/sparse/``, ``tests/ode/``, …). They use
plain pytest:

.. code-block:: bash

   pytest tests/                                    # full suite
   pytest tests/sparse/test_spsolve.py -v           # one file
   pytest tests/element/test_basis.py::test_triangle_basis -v   # one test
   pytest tests/ --cov=tensormesh                   # coverage report

A few conventions worth knowing before writing new tests:

* **Default to ``float64`` and ``cpu``** in new tests; add a GPU
  variant guarded by ``@pytest.mark.skipif(not torch.cuda.is_available())``
  when the code path you're exercising has device/dtype handling.
* **Avoid NumPy inside assembler ``forward()`` methods.** Assemblers
  run under ``torch.vmap``, which is silently incompatible with NumPy
  arrays — stay in torch.
* **Name tests after what they exercise**, not the file. Duplicate
  function names across test files run fine but make ``pytest -k``
  filtering noisy.


Building the docs
-----------------

The docs use Sphinx with the ``pydata_sphinx_theme``. Build locally:

.. code-block:: bash

   cd docs
   make html

Rendered output goes to ``docs/_build/html/``; open
``docs/_build/html/index.html`` to view. A clean build currently emits
a handful of pre-existing warnings — a stale ``ExplicitRungeKutta``
cross-reference in ``example_gallery/wave.rst`` plus a few NumPy-style
docstring-formatting warnings — so the working rule is simply: don't
add new ones, and fixing an existing one is always welcome.

The site is **bilingual EN/ZH** via Sphinx's gettext workflow. The
canonical English source is the ``.rst`` files; Chinese strings live
in ``.po`` files under ``docs/source/locale/zh_CN/``. To regenerate the
``.po`` after editing English text:

.. code-block:: bash

   make gettext                    # extract strings
   sphinx-intl update -p _build/gettext -l zh_CN   # merge into .po

To build the Chinese site or both sites together:

.. code-block:: bash

   make zh         # Chinese build only
   make html-all   # both EN and ZH

If your PR only touches English text, you can leave the ``.po`` files
alone — a maintainer will batch the i18n update.


Code style
----------

There is no automated formatter / linter enforced in CI today, so the
working rule is **match the file you're editing**. In practice that
means:

* 4-space indent, PEP 8 in spirit, line length around 88-100 characters.
* **NumPy-style docstrings** with Summary → Parameters → Returns →
  Notes → Examples (in that order). Type hints are encouraged but not
  required; if you add them, prefer ``torch.Tensor`` over the generic
  ``Any``.
* Use the existing **``BufferDict`` / ``BufferList``** containers
  (``tensormesh.nn``) for ``nn.Module``-compatible tensor collections.
  Don't roll a plain ``dict`` of tensors when crossing
  ``nn.Module`` boundaries.
* Keep ``forward()`` methods pure tensor code — no Python control flow
  on tensor values, no NumPy, no ``.item()`` calls unless the result
  is for a print/log.


Submitting a pull request
-------------------------

The mechanical flow:

1. **Fork & branch from ``main``.** Branch naming is informal; ``feat/<short>``
   / ``fix/<short>`` / ``docs/<short>`` matches the commit-message style.
2. **Write tests.** A PR that touches numerical code without an
   associated test is unlikely to be merged. CPU + ``float64`` is the
   minimum; add a GPU variant if the touched code branches on device.
3. **Run the relevant tests and a docs build** locally; both should be
   clean.
4. **Open the PR**, link the issue with ``Fixes #NN`` (or
   ``Refs #NN`` for partial fixes), and fill in the body explaining the
   *why* — not just the *what*, since the diff already covers that.
5. **Commit message style** follows the rest of the project:
   ``<type>(<scope>): <imperative summary>``, e.g.
   ``fix(ode): propagate u0.dtype/device through step()``. Types in
   use: ``feat``, ``fix``, ``docs``, ``test``, ``refactor``, ``perf``,
   ``ci``.

What reviewers will look for:

* Tests that exercise the path you changed, including the failure
  mode before your fix if applicable.
* No new Sphinx warnings if you touched docstrings or ``.rst``.
* No accidental backwards-incompatibility — if a public symbol is
  renamed or removed, the PR description should call it out
  explicitly.
* Numerical changes (new element, new quadrature, new solver) should
  include at least one analytical-reference check.

This is a small maintainer team; response times are best-effort. If a
PR has been silent for more than a week or two, a polite ping on
:doc:`discord` ``#dev`` is welcome.


Sign-off and the Developer Certificate of Origin
------------------------------------------------

Every commit that lands on ``main`` must be signed off under the
`Developer Certificate of Origin <https://developercertificate.org/>`_
(DCO) — the same lightweight contribution policy used by the Linux
kernel, PyTorch, vLLM, and most Apache-licensed projects. The full
text lives in the ``DCO`` file at the repository root; the gist is
that you certify the code is yours to contribute (or that it came
from a properly-licensed source) and that you are submitting it under
the project's Apache 2.0 license.

The mechanics are trivial — pass ``-s`` to ``git commit`` and Git
appends the line for you:

.. code-block:: bash

   git commit -s -m "fix(ode): propagate dtype through step()"

producing a trailer of the form::

   Signed-off-by: Your Name <your@email>

Use your real legal name and an email you control. A pull request
whose commits lack a valid ``Signed-off-by`` trailer will not be
merged.

If you forget the sign-off on a commit you have already pushed:

.. code-block:: bash

   # for the latest commit only
   git commit --amend -s --no-edit
   git push --force-with-lease

   # for an older commit, or the whole branch
   git rebase --signoff main
   git push --force-with-lease

A maintainer cannot sign on your behalf — sign-off is the contributor's
own certification.


Project layout
--------------

A rough map of where things live, for new contributors:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Contents
   * - ``tensormesh/mesh/``
     - ``Mesh``, generators (``gen_rectangle``, ``gen_cube``, …), I/O.
   * - ``tensormesh/element/``
     - Element types (``Line``, ``Triangle``, …), basis functions,
       quadrature, Gmsh/VTK reordering.
   * - ``tensormesh/assemble/``
     - ``ElementAssembler`` / ``NodeAssembler`` / ``FacetAssembler``
       base classes, built-in Laplace / Mass / Linear-elasticity
       assemblers.
   * - ``tensormesh/sparse/``
     - ``SparseMatrix`` (extends ``torch_sla.SparseTensor``),
       ``spsolve`` dispatch, ``nonlinear_solve``.
   * - ``tensormesh/operator/``
     - ``Condenser`` (Dirichlet BC via static condensation).
   * - ``tensormesh/ode/``
     - Time-integration schemes — explicit/implicit linear Euler,
       midpoint, RK base classes.
   * - ``tensormesh/functional/``
     - Tensor utilities for FEM (Voigt notation, gradients).
   * - ``tensormesh/dataset/``
     - Batched mesh generation + ``…MultiFrequency`` source samplers
       for ML training data.
   * - ``tensormesh/nn/``
     - ``BufferDict`` / ``BufferList``.
   * - ``tests/``
     - Test suite, mirroring the package layout.
   * - ``examples/``
     - End-to-end worked examples.
   * - ``docs/``
     - Sphinx sources, build configuration, and translation files.

When in doubt, grep for a nearby symbol — the package is small enough
that ``rg <SymbolName> tensormesh/`` will usually point you at the
right neighbourhood.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 
import datetime 
sys.path.insert(0, os.path.abspath('../..'))
import tensormesh    

project = 'TensorMesh'
author = 'Shizheng Wen, Mingyuan Chi'
copyright = f'{datetime.datetime.now().year}, TensorMesh Contributors'
version = tensormesh.__version__
release = version

# -- Internationalization ----------------------------------------------------
# Source language; build other languages with `-D language=zh_CN` etc.
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False  # one .po per source .rst file
gettext_uuid = True      # stable message IDs across pot regenerations

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
     'sphinx.ext.autodoc',
     'sphinx.ext.napoleon',
     'sphinx.ext.viewcode',
     'sphinx.ext.mathjax',
     'sphinx.ext.githubpages',
     'sphinx.ext.intersphinx',
     'sphinx.ext.autosummary',
     'sphinx_design',
     'nbsphinx'
]


html_theme = 'furo'

# Override the default `f"{project} {release} documentation"` so the sidebar
# brand renders as a clean "TensorMesh" — required for the two-color wordmark
# CSS in _static/custom.css to work.
html_title = 'TensorMesh'

html_theme_options = {
    'light_logo': 'logo.png',
    'dark_logo':  'logo.png',
    'sidebar_hide_name': False,

    # Brand palette — overrides Furo's defaults so the whole site picks up the
    # tmblue / tmteal pair from the logo (see logo.tex).
    'light_css_variables': {
        'color-brand-primary': '#5B6EE8',  # tmblue: sidebar active links, TOC highlight
        'color-brand-content': '#149B8E',  # tmteal: in-content anchor links
    },
    'dark_css_variables': {
        'color-brand-primary': '#8B9AFF',
        'color-brand-content': '#2DC9B8',
    },

    # Furo doesn't ship Font Awesome — inline SVG marks for footer links.
    'footer_icons': [
        {
            'name': 'GitHub',
            'url':  'https://github.com/camlab-ethz/TensorMesh',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49'
                '-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2'
                '-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 '
                '0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 '
                '2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07'
                '-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
                '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>'
                '</svg>'
            ),
            'class': '',
        },
        {
            'name': 'Discord',
            'url':  'https://discord.gg/EC9kbHSnrx',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path d="M13.545 2.907a13.2 13.2 0 0 0-3.257-1.011.05.05 0 0 0-.052.025'
                'c-.141.25-.297.576-.406.833a12.2 12.2 0 0 0-3.658 0 8 8 0 0 0-.412-.833'
                '.05.05 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.05.05 0 0 0-.021.019'
                'C.356 6.024-.213 9.047.066 12.032c.002.015.01.029.02.037a13.3 13.3 0 0 0'
                ' 3.995 2.01.05.05 0 0 0 .056-.019c.308-.42.582-.86.818-1.329a.05.05 0 0 0'
                '-.027-.06 8.7 8.7 0 0 1-1.248-.595.05.05 0 0 1-.005-.085c.079-.053.16-.104'
                '.248-.152a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.05.05 0 0 1'
                ' .053.007c.08.048.16.1.248.152a.05.05 0 0 1-.004.085 8.2 8.2 0 0 1-1.249.594'
                '.05.05 0 0 0-.027.06c.24.465.514.906.817 1.329a.05.05 0 0 0 .056.02 13.2 13.2'
                ' 0 0 0 4.001-2.02.05.05 0 0 0 .021-.036c.334-3.451-.559-6.449-2.366-9.106a.04'
                '.04 0 0 0-.021-.02zM5.68 10.77c-.789 0-1.438-.724-1.438-1.61 0-.886.637-1.61'
                ' 1.438-1.61.807 0 1.45.73 1.438 1.61 0 .886-.637 1.61-1.438 1.61zm4.842 0c-.788 0'
                '-1.438-.724-1.438-1.61 0-.886.637-1.61 1.438-1.61.807 0 1.451.73 1.438 1.61 0'
                ' .886-.63 1.61-1.438 1.61z"></path>'
                '</svg>'
            ),
            'class': '',
        },
    ],
}

# Sidebar layout: insert language switcher between brand and search.
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'language-switcher.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}

html_favicon = '_static/logo.png'
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

# Preserve source-level expressions for default argument values so
# function-object defaults render as e.g. `strain_fn=strain` rather than
# `strain_fn=<function strain>` (which is just repr() at doc-build time).
autodoc_preserve_defaults = True

# Render parameter type hints with their unqualified ("short") name —
# e.g. `Tensor` instead of `~torch.Tensor` — and hyperlink them via
# intersphinx. Without this, Sphinx 9 leaks the `~` short-name marker
# into the rendered signature for parameters (return types are unaffected).
python_use_unqualified_type_names = True

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python':     ('https://docs.python.org/3', None),
    'numpy':      ('https://numpy.org/doc/stable/', None),
    'pandas':     ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'torch':      ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy':      ('https://docs.scipy.org/doc/scipy/', None),
}

# nitpicky mode is always on: any broken cross-reference becomes a build
# warning so that stale docstrings (renamed classes, dropped modules,
# wrong roles) are caught at PR time. The ignore lists below cover
# targets that are *expected* to fail resolution and are not worth fixing
# at source:
nitpicky = True

#
#   - ``optional``: napoleon parses ``x : float, optional`` as two types,
#     emitting a cross-ref for ``optional`` that never resolves.
#   - quoted forward references like ``'Element'``: PEP 484 string forms of
#     a class name (`-> 'Element'`) that Sphinx fails to dereference.
#   - ``np.ndarray``: ndarray is autodoc'd under the alias `np.` which is
#     not in the numpy inventory; refactoring each docstring to
#     ``numpy.ndarray`` would be noisy churn.
#   - ``meshio*``: meshio has no public Sphinx documentation site, so its
#     types cannot be cross-referenced.
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'np.ndarray'),
    ('py:class', 'meshio._mesh.Mesh'),
    ('py:class', 'meshio.Mesh'),
    ('py:mod',   'meshio'),
    # torch.nn.Parameter is registered as torch.nn.parameter.Parameter in
    # the upstream inventory; torch.float32 / torch.float64 are not
    # cross-referenced by the torch docs at all.
    ('py:class', 'torch.nn.Parameter'),
    ('py:class', 'torch.float32'),
    ('py:class', 'torch.float64'),
    ('py:obj',   'torch.float32'),
    ('py:obj',   'torch.float64'),
    # SparseMatrix.solve is inherited from torch_sla.SparseTensor; not
    # registered in our inventory and not part of the public torch_sla docs.
    ('py:meth', 'tensormesh.sparse.SparseMatrix.solve'),
    ('py:class', 'tensormesh.sparse.SparseMatrix.solve'),
    # torch_sla has no public Sphinx site.
    ('py:class', 'torch_sla.sparse_tensor.SparseTensor'),
    ('py:class', 'torch_sla.distributed.DSparseTensor'),
    # TypeVar on the @distributed decorator's signature — autodoc renders
    # ``Type[T]`` but the TypeVar itself has no documentation target.
    ('py:class', 'tensormesh.distributed.assembler.T'),
    # Informal napoleon type phrases in cartesian_pml's docstring
    # ("sequence of (lo, hi)", "float or sequence of float").
    ('py:class', 'sequence'),
    ('py:class', 'lo'),
    ('py:class', 'hi'),
    # Bare class names from autodoc-rendered type hints; we keep
    # python_use_unqualified_type_names = True for readability, which
    # prevents these from auto-prefixing with their (mostly matplotlib)
    # module path and so they don't resolve in nitpicky mode.
    ('py:class', 'Tensor'),
    ('py:class', 'plt.Axes'),
    ('py:class', 'PolyCollection'),
    ('py:class', 'PathCollection'),
    ('py:class', 'Path3DCollection'),
    ('py:class', 'AxesImage'),
    # PolynomialTensor is informal docstring shorthand for ``Polynomials``;
    # not a class that actually exists.
    ('py:class', 'PolynomialTensor'),
    # torch.Float (note the capital F) is a NumPy-ism — torch uses
    # torch.float / torch.float32 / torch.float64; the docstring refers to
    # the older name. Not in any inventory.
    ('py:class', 'torch.Float'),
    ('py:class', 'torch.sparse_csr_matrix'),
    ('py:class', 'torch._VariableFunctionsClass.sparse_csr_tensor'),
    # Dunder and private attribute references — Sphinx cannot resolve them
    # without explicit class scoping; we use them in narrative docstrings.
    ('py:obj',   '__setitem__'),
    ('py:attr',  '_data'),
    ('py:attr',  '_buffers'),
    ('py:attr',  '_parameters'),
    ('py:meth',  '_apply'),
    ('py:meth',  '__getitem__'),
    # default_element_type is a property on Mesh; cross-ref from class
    # docstring fails because it's not in scope by name alone.
    ('py:attr',  'default_element_type'),
    # ModuleDict is torch.nn.ModuleDict — the bare name doesn't resolve
    # via intersphinx even with python_use_unqualified_type_names = True.
    ('py:class', 'ModuleDict'),
    ('py:class', 'Axes'),
    ('py:class', 'Axes3D'),
    # pyvista has no Sphinx site we cross-reference.
    ('py:class', 'pyvista.DataSet'),
    ('py:class', 'ScipySparseMatrix'),
    ('py:class', 'DSparseTensor'),
    # Scientific-notation numeric literals in :obj: roles, missed by the
    # earlier `0.1`-style cleanup script (only regex'd decimal form).
    ('py:obj',   '1e-3'),
    ('py:obj',   '1e-4'),
    # torch.nn.Module.type's inherited signature uses dst_type — see
    # nitpick_ignore_regex below for the py:class case; this is py:attr.
    ('py:attr',  'dst_type'),
]

nitpick_ignore_regex = [
    # Quoted forward references (PEP 484 string class names).
    ('py:class', r"'[A-Za-z_][A-Za-z0-9_]*'"),
    # Autodoc emits the source-file path tensormesh.sparse.matrix.SparseMatrix.*
    # for SparseMatrix's self-returning methods (.float(), .double(), .half(),
    # .cpu()) when they use 'SparseMatrix' as a string forward reference.
    # The rendered HTML link is correct; only the title attribute is wrong.
    ('py:class', r"tensormesh\.sparse\.matrix\.SparseMatrix\..*"),
    # napoleon misparses prose written in the parameter type slot of NumPy
    # docstrings as if it were a type annotation; rather than rewriting
    # every occurrence we accept these targeted false positives.
    # `default=foo`, `default is "bar"` — defaults inlined into the type slot.
    ('py:class', r"default[ =].*"),
    # `If is_mix_facet is True` — sentence fragments in the type slot.
    ('py:class', r"If .*"),
    # `Polynomial n_vars=...` etc. — pseudo type expressions with shape spec.
    ('py:class', r"Polynomial(s)?( n_vars=.*)?"),
    # `dim+1`, `int int`, `**n_vars`, `n_point` — assorted shape/type-column noise.
    ('py:class', r"(dim\+1|int int|\*\*n_vars|n_point)"),
    # NumPy-style choice annotation `param : {'a', 'b'}, optional` is split
    # by napoleon on the comma — the resulting tokens `{'a'` and `'b'}` are
    # not types we want to cross-reference.
    ('py:class', r"\{'[^']+'"),
    ('py:class', r"'[^']+'\}"),
    # torch.nn.Module.type's inherited signature uses dst_type as its
    # parameter name; autodoc carries the parent's type annotation onto
    # subclasses that don't override the docstring.
    ('py:class', r"dst_type"),
]

# Custom inline roles available in every .rst file. Used in index.rst to color
# "Tensor" / "Mesh" in the H1 with the brand palette (see _static/custom.css).
rst_prolog = """
.. role:: tensor-blue
.. role:: mesh-teal
"""

napoleon_google_docstring = False

autosummary_generate = True

autodoc_member_order = 'groupwise'

# -- Options for HTML output -------------------------------------------------

def skip(app, what, name, obj, skip, options):
     # print(f"what: {what}, name: {name}, obj: {obj}, skip: {skip}, options: {options}\n")
     if hasattr(obj, '__autodoc__'):
          return not name in obj.__autodoc__
     
     return skip

def setup(app):
     app.connect('autodoc-skip-member', skip)
     app.add_css_file('custom.css')

# These settings can also help with documentation structure
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Enable better section numbering
numfig = True
numfig_secnum_depth = 2
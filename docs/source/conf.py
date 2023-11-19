# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 
import datetime 
sys.path.insert(0,"../..")
import torch_fem    
import torch_fem_sphinx_theme


project = 'torch_fem'
author  = 'walkerchi'
copyright = f'{datetime.datetime.now().year}, {author}'
version = torch_fem.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
     'sphinx.ext.autodoc',
     'sphinx.ext.napoleon',
     'sphinx.ext.viewcode',
     'sphinx.ext.mathjax',
     'sphinx.ext.githubpages',
     'sphinx.ext.intersphinx',
     'nbsphinx'
]


html_theme = 'torch_fem_sphinx_theme'
html_logo  = ('https://raw.githubusercontent.com/walkerchi/torch_fem_sphinx_theme/'
             'master/torch_fem_sphinx_theme/static/img/torch_fem_logo.webp')
html_favicon =('https://raw.githubusercontent.com/walkerchi/torch_fem_sphinx_theme/'
             'master/torch_fem_sphinx_theme/static/img/torch_fem_logo.webp')
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

exclude_patterns = []

napoleon_google_docstring = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output




def skip(app, what, name, obj, skip, options):
     print(f"what: {what}, name: {name}, obj: {obj}, skip: {skip}, options: {options}\n")
     if hasattr(obj, '__autodoc__'):
          return not name in obj.__autodoc__
     
     return skip

def setup(app):
    app.connect('autodoc-skip-member', skip)

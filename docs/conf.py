# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Add project root to sys.path
                                          # This allows Sphinx to find GAT_r_adaptivity

# You can remove the following line if src is indeed not needed for GAT_r_adaptivity docs
sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DNN_mesh_refienement_for_FEM'
copyright = '2025, Erling Tennøy Nordtvedt' # Consider updating the year if it's a typo
author = 'Erling Tennøy Nordtvedt'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary", # Often used with autodoc for summary tables
    "sphinx.ext.napoleon",    # For Google/NumPy style docstrings
    "sphinx_autodoc_typehints", # To handle type hints better
    "sphinx_rtd_theme",       # Add the theme here if it's also an extension (often not needed just for theme)
]
napoleon_use_param = True
# napoleon_google_docstring = True # Uncomment if you use Google style docstrings
# napoleon_numpy_docstring = True  # Uncomment if you use NumPy style docstrings

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# The html_header is one way to add custom CSS/JS.
# Consider adding custom CSS to a file in _static and linking it via html_css_files if it's more than a line or two.
# html_css_files = [
#     'css/custom.css', # Example: if you have _static/css/custom.css
# ]

html_header = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""" # This is fine if it works for you.

# -- Options for autodoc -----------------------------------------------------
# autodoc_member_order = 'bysource' # Optional: 'alphabetical', 'groupwise', or 'bysource'
#utosummary_generate = True # Important for sphinx.ext.autosummary to generate stub files

# (CRITICAL FOR PREVIOUS ERRORS) -- Mocking imports
# If you still have ModuleNotFoundErrors for libraries like torch, jax, dolfin, etc.,
# and they are not essential for generating the docstrings (e.g., you can't/don't want to
# install them in your documentation build environment), mock them:
autodoc_mock_imports = [
    "torch",
    "torch_geometric",  # <-- Add this line
    "jax",
    "dolfin",
    "dolfin_dg",
    "fenics",
    "mpi4py", # Common FEniCS dependency
    "petsc4py", # Common FEniCS dependency
    # Add any other external libraries that GAT_r_adaptivity imports
    # but are not available or needed when building documentation.
]

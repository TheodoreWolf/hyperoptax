# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the source directory to Python path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Hyperoptax"
copyright = "2025, Theo Wolf"
author = "Theo Wolf"
release = "0.1.6"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for NumPy/Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other project docs
    # "myst_parser",  # Markdown support (temporarily disabled)
    # "nbsphinx",  # Jupyter notebook support (may cause issues)
]

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary configuration
autosummary_generate = True

# Napoleon configuration (for better docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping to link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/TheodoreWolf/hyperoptax",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "repository_branch": "main",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "logo": {
        "text": "Hyperoptax",
        "alt_text": "Hyperoptax - Parallel hyperparameter tuning with JAX",
    },
}

# Additional HTML configuration
html_title = "Hyperoptax Documentation"
html_short_title = "Hyperoptax"

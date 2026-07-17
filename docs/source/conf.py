# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as package_version

# Add the source directory to Python path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Hyperoptax"
copyright = "2025-2026, Theo Wolf"
author = "Theo Wolf"
release = package_version("hyperoptax")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build"]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

# Include type hints in the description rather than the signature so that
# Sphinx Napoleon + autodoc produce cleaner function/class signatures.
autodoc_typehints = "description"

# Napoleon configuration (for better docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping to link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/TheodoreWolf/hyperoptax",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "repository_branch": "main",
    "home_page_in_toc": True,
    # Make the left sidebar navigation static
    # Show only top-level items by default
    "show_navbar_depth": 1,
    # Disable expansion to keep a fixed navigation layout
    "collapse_navbar": True,
    "logo": {
        "image_light": "_static/manifold.png",
        "image_dark": "_static/manifold.png",
        "text": "Hyperoptax",
        "alt_text": "Hyperoptax - Parallel hyperparameter tuning with JAX",
    },
}

# Additional HTML configuration
html_title = "Hyperoptax Documentation"
html_short_title = "Hyperoptax"

# Disable the built-in page-level Table of Contents sidebar so the navigation
# bar doesn’t jump around when heading structures differ between pages.
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}

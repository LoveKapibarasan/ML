"""Sphinx configuration for the EV Smart Charging (ML) documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "EV Smart Charging"
copyright = "2026, AI-Charge"
author = "AI-Charge"

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# reStructuredText is the source format for all docs pages.
source_suffix = ".rst"

# -- Napoleon (Google-style docstrings) ---------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autosummary ---------------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

# -- Autodoc --------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Third-party packages imported at module scope (torch/CUDA, DB drivers, live
# API clients, etc.) are not installed in the docs build environment and are
# not needed to render docstrings — mock them out instead of requiring a full
# training/serving environment just to build docs.
autodoc_mock_imports = [
    "stable_baselines3",
    "gymnasium",
    "psycopg2",
    "openmeteo_requests",
    "requests_cache",
    "retry_requests",
    "entsoe",
    "fastapi",
    "dotenv",
    "holidays",
]

# -- Intersphinx ----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output ------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

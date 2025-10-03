# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Cross-Facility Federated Learning"
copyright = "2025, Gianluca Mittone, Alberto Mulone, Giulio Malenza, Robert Birke, Marco Aldinucci"
author = (
    "Gianluca Mittone, Alberto Mulone, Giulio Malenza, Robert Birke, Marco Aldinucci"
)
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    # "sphinx.ext.doctest",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "autoapi.extension",
]

# autosummary_generate = True
autoapi_dirs = ["../../xffl"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # alabaster
html_static_path = ["_static"]

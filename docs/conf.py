# Configuration file for the Sphinx documentation builder.

# [https://www.sphinx-doc.org/en/master/usage/configuration.html](https://www.sphinx-doc.org/en/master/usage/configuration.html)

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add your 'codes' folder to sys.path so Sphinx can import your modules

sys.path.insert(0, os.path.abspath('../../codes'))

# -- Project information -----------------------------------------------------

project = 'Solar Wind Prediction'
authors = 'Rohan R. Poojary, Dattaraj M. Dhuri'
copyright = '2025, Rohan R. Poojary'
release = 'v0.1'

# -- General configuration ---------------------------------------------------

extensions = [
'sphinx.ext.autodoc',          # Include docstrings from modules
'sphinx.ext.napoleon',         # Support for Google and NumPy docstrings
'sphinx.ext.autosummary',      # Generate summary tables automatically
'sphinx_autodoc_typehints',    # Show Python type hints in docs
'myst_parser',                 # Optional: for Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Generate autosummary pages automatically

autosummary_generate = True

# Mock imports that may not be available

autodoc_mock_imports = ['IPS_OMNI_make_data']  # Avoid import errors if excluded

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # ReadTheDocs theme
html_static_path = ['_static']

# -- Custom setup ------------------------------------------------------------

def setup(app):
pass  # You can add custom Sphinx setup hooks here if needed

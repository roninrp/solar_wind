# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('../code'))

def run_apidoc(_):
    src = os.path.join(os.path.dirname(__file__), '../code')
    dst = os.path.join(os.path.dirname(__file__))
    subprocess.call(['sphinx-apidoc', '-f', '-e', '-M', '-o', dst, src])

def setup(app):
    app.connect('builder-inited', run_apidoc)

project = 'Solar Wind Prediction'
copyright = '2025, Rohan R. Poojary'
author = 'Rohan R. Poojary'
release = 'v0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # For Google/NumPy style docstrings
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

suppress_warnings = ["toc.not_included"]
autodoc_mock_imports = ["IPS_OMNI_make_data"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = 'sphinx_rtd_theme'
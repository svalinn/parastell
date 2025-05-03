# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ParaStell'
copyright = '2023-2025 UW-Madison Computational Nuclear Engineering Research Group'
author = 'Connor Moreno, Edgar Pflug, Paul Wilson, Eitan Weinstein, Enrique Miralles-Dolz, Joshua Smandych, Paul Romano, Bowen Quan, Jonathan Shimwell'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.apidoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = ['_templates']
exclude_patterns = []


apidoc_module_dir = "../../parastell"
apidoc_module_first = True
apidoc_separate_modules = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

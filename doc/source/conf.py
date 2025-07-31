# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ParaStell'
copyright = '2023-2025 UW-Madison Computational Nuclear Engineering Research Group'
author = 'Connor Moreno, Edgar Pflug, Paul Wilson, Eitan Weinstein, Enrique Miralles-Dolz, Joshua Smandych, Paul Romano, Bowen Quan, Jonathan Shimwell'

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

version = importlib.metadata.version("parastell")
release = version
html_logo = "parastell_logo.svg"
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

autodoc_typehints = "both"
autodoc_mock_imports = ["cubit"]
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_typehints_description_target = "all"
autodoc_member_order = "groupwise"
# Display the version
display_version = True
autodoc_default_options = {
    "autosummary": True,
    "show-inheritance": True,
    "inherited-members": True,
}

apidoc_module_dir = "../../parastell"
apidoc_module_first = True
apidoc_separate_modules = True
apidoc_excluded_paths = [
    "tests"
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

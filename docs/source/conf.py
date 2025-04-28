"""Configuration file for sphinx documentation."""

import os
import shutil
from datetime import datetime
from pathlib import Path

from sphinx.ext import apidoc

import qdesignoptimizer

current_year = datetime.now().year
project = "qdesignoptimizer"
copyright = f"{current_year}, 202Q-lab"
author = "202Q-lab"
release = qdesignoptimizer.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_last_updated_by_git",
    "sphinx_mdinclude",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_favicon = "favicon.ico"

html_title = "qdesignoptimizer · " + release

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "flyout_display": "hidden",
    "version_selector": True,
    "language_selector": True,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_show_sourcelink = False

# -- Intersphinx  -------------------------------------------------------------
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
autodoc_member_order = "bysource"

# nbsphinx configuration
nbsphinx_execute = "never"


github_doc_root = "https://github.com/rtfd/recommonmark/tree/master/doc/"


def setup(app):
    print("Copying example notebooks into docs/source/_projects")
    run_apidoc(None)
    source = Path(__file__).parent
    project_root = source.parents[1]

    def all_but_ipynb(dir, contents):
        result = []
        for c in contents:
            if os.path.isfile(os.path.join(dir, c)) and (
                not (c.endswith(".ipynb") or c.endswith(".png"))
            ):
                result += [c]
        return result

    shutil.rmtree(
        os.path.join(project_root, "docs/source/_projects"), ignore_errors=True
    )
    shutil.copytree(
        os.path.join(project_root, "projects"),
        os.path.join(project_root, "docs/source/_projects"),
        ignore=all_but_ipynb,
    )
    app.add_css_file("my_theme.css")

    def clean_examples_dir(_app, _exception):
        shutil.rmtree(
            os.path.join(project_root, "docs/source/_projects"), ignore_errors=True
        )

    app.connect("build-finished", clean_examples_dir)


def run_apidoc(_):
    """Extract autodoc directives from package structure."""
    source = Path(__file__).parent
    docs_dest = os.path.join(source, "api-reference")
    if os.path.isdir(docs_dest):
        shutil.rmtree(docs_dest, ignore_errors=False, onerror=None)
    package = os.path.join(source.parents[1], "src", "qdesignoptimizer")

    apidoc.main(["--module-first", "-o", str(docs_dest), str(package), "--separate"])

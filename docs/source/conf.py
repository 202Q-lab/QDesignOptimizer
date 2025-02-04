"""Configuration file for sphinx documentation."""

import shutil
from datetime import datetime
from pathlib import Path

from recommonmark.transform import AutoStructify
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
    #  "recommonmark",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_last_updated_by_git",
    "sphinx_mdinclude",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_favicon = "favicon.ico"

html_title = "qdesignoptimizer · " + release

html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/202Q-lab/qdesignoptimizer/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    # "light_logo": "202Q-lab_logo.png",
    # "dark_logo": "202Q-lab_logo_dark.png",
    "light_css_variables": {
        "color-brand-primary": "#6400FF",
        "color-brand-secondary": "#6400FF",
        "color-brand-content": "#6400FF",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/202Q-lab/qdesignoptimizer",
            "html": """
                    <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                        <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                    </svg>
                """,
            "class": "",
        },
    ],
}
# html_static_path = ["_static"]
html_show_sourcelink = False

# -- Intersphinx  -------------------------------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
autodoc_member_order = "bysource"


github_doc_root = "https://github.com/rtfd/recommonmark/tree/master/doc/"


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: github_doc_root + url,
            "auto_toc_tree_section": "Contents",
        },
        True,
    )
    app.add_transform(AutoStructify)


def run_apidoc(_):
    """Extract autodoc directives from package structure."""
    source = Path(__file__).parent
    docs_dest = source / "api-reference"
    if docs_dest.is_dir():
        shutil.rmtree(docs_dest, ignore_errors=False, onerror=None)
    package = source.parents[1] / "src" / "qdesignoptimizer"
    apidoc.main(["--module-first", "-o", str(docs_dest), str(package), "--separate"])

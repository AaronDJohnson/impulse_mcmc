# Configuration file for the Sphinx documentation builder.
#
# Build locally with:
#   python -m sphinx -b html docs docs/_build/html
#
# The build is fully offline: no intersphinx, no remote assets.

import json
import os
import shutil
import sys

import impulse

# -- Path setup --------------------------------------------------------------

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, os.pardir))
sys.path.insert(0, REPO_ROOT)

# -- Project information -----------------------------------------------------

project = "impulse-mcmc"
author = "Aaron D. Johnson"
copyright = "2026, Aaron D. Johnson"
version = impulse.__version__
release = impulse.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

# Deliberately NO intersphinx: docs must build offline (CI and local).

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Autodoc / autosummary / napoleon ----------------------------------------

# Docstrings are numpydoc-style; napoleon parses them (no numpydoc package).
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Summary tables on the API pages link to autodoc entries on the same pages;
# no stub generation needed.
autosummary_generate = False

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# -- nbsphinx ----------------------------------------------------------------

# The example notebooks run 20k+ iteration samplers. NEVER execute them in a
# docs build; the pages below render the outputs stored in the repository.
nbsphinx_execute = "never"

nbsphinx_prolog = """
.. note::

   This page is a static rendering of a Jupyter notebook from the
   repository's ``examples/`` directory. The stored outputs come from a full
   run (tens of thousands of iterations); the notebook is **not** executed
   during documentation builds.
"""

# -- Notebook staging ---------------------------------------------------------
# The rendered notebooks live in examples/ at the repo root. nbsphinx needs
# them inside the Sphinx source directory, so copy them into docs/examples/
# at build time (docs/examples/*.ipynb is gitignored). Notebooks whose first
# cell is not a top-level markdown heading get a title cell prepended so
# Sphinx can title the page.

_NOTEBOOKS = {
    # filename -> title to inject when the notebook lacks a leading "# ..."
    "sinusoidal_model.ipynb": "Sinusoid fitting with parallel tempering",
    "rjmcmc_sinusoids.ipynb": "RJMCMC model selection: counting sinusoids",
    "high_dimensional_test.ipynb": None,  # already starts with a title cell
}


def _stage_notebooks() -> None:
    dst_dir = os.path.join(DOCS_DIR, "examples")
    os.makedirs(dst_dir, exist_ok=True)
    for name, title in _NOTEBOOKS.items():
        src = os.path.join(REPO_ROOT, "examples", name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            continue
        with open(src) as fp:
            nb = json.load(fp)
        cells = nb.get("cells", [])
        has_title = (
            cells
            and cells[0].get("cell_type") == "markdown"
            and "".join(cells[0].get("source", [])).lstrip().startswith("# ")
        )
        if title is not None and not has_title:
            cells.insert(
                0,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}"],
                },
            )
        # Normalize to nbformat 4.5 with cell ids so nbformat stops warning
        # about missing-id cells during the build.
        if nb.get("nbformat", 4) == 4 and nb.get("nbformat_minor", 0) < 5:
            nb["nbformat_minor"] = 5
        for i, cell in enumerate(cells):
            cell.setdefault("id", f"cell-{i}")
        nb["cells"] = cells
        with open(dst, "w") as fp:
            json.dump(nb, fp)


_stage_notebooks()


# -- Docstring fix-ups ---------------------------------------------------------
# A few upstream docstrings contain constructs that docutils misparses
# (|...| looks like a substitution reference; one numpydoc parameter entry
# wraps its comma-separated name list over several lines, which napoleon
# cannot parse). Repair them here, BEFORE napoleon runs (priority < 500),
# so the build stays warning-clean without patching the source tree.


def _escape_pipes(app, what, name, obj, options, lines):
    targets = ("|z - median(z)|", "|quantile - 0.5|")
    for i, line in enumerate(lines):
        for target in targets:
            if target in line:
                lines[i] = line.replace(target, "\\" + target[:-1] + "\\|")


def _join_wrapped_param_names(app, what, name, obj, options, lines):
    """Merge numpydoc parameter-name lines that wrap across multiple lines.

    ``RJPTSampler``'s docstring lists ~26 parameter names in one entry,
    wrapped over four lines; napoleon needs them on a single line.
    """
    if name != "impulse.RJPTSampler":
        return
    i = 0
    while i < len(lines):
        if lines[i].startswith("buffer_size, groups"):
            j = i
            while j < len(lines) and lines[j].rstrip().endswith(","):
                j += 1
            if j < len(lines) and j > i:
                merged = " ".join(line.strip() for line in lines[i : j + 1])
                lines[i : j + 1] = [merged]
        i += 1


def setup(app):
    app.connect("autodoc-process-docstring", _escape_pipes, priority=400)
    app.connect("autodoc-process-docstring", _join_wrapped_param_names, priority=400)

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = f"impulse-mcmc {release}"
html_static_path: list[str] = []

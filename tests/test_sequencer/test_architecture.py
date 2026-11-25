"""Layering, held structurally (the setup-analyzer pattern).

* the engine imports no backend, no GUI, no QDO, no qiskit-metal;
* only ``app`` imports Dash;
* backends import QDO / qiskit-metal only inside functions;
* no module in the package touches ``sys.path``.

A breach would let the package stop being liftable into QDO, or make a
measurement setup pull in Ansys tooling, without any test failing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "src" / "qdesignoptimizer" / "sequencer"


def _modules(sub: str = ""):
    root = _PKG / sub if sub else _PKG
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(_PKG).parts
        # examples/ holds design projects, not library code: a design
        # folder puts itself on the import path (its modules import each
        # other by bare name) and imports QDO at the top, as every real
        # design does.
        if "tests" in parts or "examples" in parts:
            continue
        yield path


def _imports(path: Path, top_level_only: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else ast.walk(tree)
    names = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            names.add(prefix + (node.module or ""))
    return names


def test_engine_imports_only_itself_and_the_standard_library():
    allowed_third_party = {"yaml"}
    for path in _modules("engine"):
        for name in _imports(path, top_level_only=True):
            if name.startswith("."):
                assert not name.lstrip(".").startswith(
                    ("app", "backends")
                ), f"{path.name} imports {name}"
                continue
            top = name.split(".")[0]
            assert (
                top in allowed_third_party or top in _STDLIB
            ), f"{path.name} imports {name} at module scope"


def test_only_the_app_imports_dash():
    for path in _modules():
        if "app" in path.relative_to(_PKG).parts:
            continue
        names = _imports(path, top_level_only=False)
        assert not any(
            n.split(".")[0] == "dash" for n in names
        ), f"{path.name} imports dash"


def test_backends_import_qdo_and_metal_lazily():
    heavy = {"qdesignoptimizer", "qiskit_metal", "pyEPR", "pyaedt"}
    for path in _modules("backends"):
        names = _imports(path, top_level_only=True)
        assert not any(
            n.split(".")[0] in heavy for n in names
        ), f"{path.name} imports {names & heavy} at module scope"


def test_no_module_touches_sys_path():
    for path in _modules():
        assert "sys.path" not in path.read_text(
            encoding="utf-8"
        ), f"{path.name} touches sys.path"


_STDLIB = set(sys.stdlib_module_names)

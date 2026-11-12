"""Find and build a project's backend.

A reference is ``<target>:<factory>``. The target is either a ``.py``
file (a project's ``sequencer.py``) or a dotted module
(``qdesignoptimizer.sequencer.backends.fake``); the factory is called with no
arguments and returns the backend.

A file target is imported under a name derived from its path, leaving
the import path alone; whatever the project file itself does to find
its own flat modules is the project's business. It is re-imported when
the file changes, so "Reload catalog" in the GUI picks up an edit.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .backend import Backend

_LOADED: dict[tuple[str, str], Any] = {}


class BackendError(RuntimeError):
    """The backend could not be found or built."""


def load_backend(ref: str, base_dir: Path) -> Backend:
    """Import and build a backend.

    Args:
        ref: ``<file.py or module>:<factory>``.
        base_dir: Folder a file target is relative to.

    Raises:
        BackendError: If the target or factory cannot be found, or the
            object built does not look like a backend.
    """
    if not ref or ":" not in ref:
        raise BackendError(f"backend {ref!r} must read <file.py or module>:<factory>")
    target, factory_name = ref.rsplit(":", 1)
    module = _import_target(target, Path(base_dir))
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise BackendError(f"{target} has no {factory_name!r}")
    try:
        backend = factory()
    except Exception as exc:  # noqa: BLE001 -- reported
        raise BackendError(f"{ref} failed: {exc}") from exc
    missing = [
        m
        for m in (
            "name",
            "words",
            "out_dir",
            "catalog",
            "generate",
            "run",
            "commit",
        )
        if not hasattr(backend, m)
    ]
    if missing:
        raise BackendError(f"{ref} built an object without {missing}")
    return backend


def _import_target(target: str, base_dir: Path):
    if not target.endswith(".py"):
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            raise BackendError(f"cannot import {target}: {exc}") from exc
    path = (base_dir / target).resolve()
    if not path.is_file():
        raise BackendError(f"backend file {path} does not exist")
    key = (str(path), str(path.stat().st_mtime_ns))
    if key in _LOADED:
        return _LOADED[key]
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    name = f"_qdo_sequencer_backend_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BackendError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise BackendError(f"importing {path} failed: {exc}") from exc
    _LOADED[key] = module
    return module

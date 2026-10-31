"""Pick the implementation a lesson's tests run against.

Lesson tests never import the code under test directly. They ask for an
``impl`` fixture, which calls :func:`load_impl`:

* If ``PARSRAG_LEARNER_PATH`` is set (the grader does this for
  ``python -m grader run NN --path DIR``), ``DIR/<module>.py`` is loaded.
* Otherwise the reference module from the ``parsrag`` package is imported.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

LEARNER_PATH_ENV = "PARSRAG_LEARNER_PATH"


def load_impl(module: str, reference: str) -> ModuleType:
    """Load the learner's ``<module>.py``, or else ``reference`` (e.g. ``parsrag.normalize``)."""
    learner_dir = os.environ.get(LEARNER_PATH_ENV)
    if not learner_dir:
        return importlib.import_module(reference)

    directory = Path(learner_dir).resolve()
    path = directory / f"{module}.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected {path} — the grader looks for '{module}.py' in --path. "
            f"/ فایل «{module}.py» در پوشهٔ داده‌شده پیدا نشد."
        )
    # Let the learner's file import sibling helpers from its own folder.
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    name = f"learner_{module}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

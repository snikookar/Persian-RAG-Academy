from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def ld() -> ModuleType:
    """The loaders module under test (learner's starter or parsrag.loaders)."""
    return load_impl("loaders", "parsrag.loaders")

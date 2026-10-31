from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def ck() -> ModuleType:
    """The chunking module under test (learner's starter or parsrag.chunking)."""
    return load_impl("chunking", "parsrag.chunking")

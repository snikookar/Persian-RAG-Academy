from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def rg() -> ModuleType:
    """The facade module under test (learner's rag.py or parsrag.rag)."""
    return load_impl("rag", "parsrag.rag")

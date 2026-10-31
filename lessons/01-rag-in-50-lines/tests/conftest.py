from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def rag() -> ModuleType:
    """The naive_rag module under test (learner's starter or parsrag.naive_rag)."""
    return load_impl("naive_rag", "parsrag.naive_rag")

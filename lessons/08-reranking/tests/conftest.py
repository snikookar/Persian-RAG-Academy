from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def rr() -> ModuleType:
    """The rerank module under test (learner's rerank.py or parsrag.rerank)."""
    return load_impl("rerank", "parsrag.rerank")

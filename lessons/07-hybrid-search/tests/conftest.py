from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def rt() -> ModuleType:
    """The retrieval module under test (learner's retrieval.py or parsrag.retrieval)."""
    return load_impl("retrieval", "parsrag.retrieval")

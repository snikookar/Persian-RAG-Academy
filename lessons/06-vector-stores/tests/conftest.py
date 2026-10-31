from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def vs() -> ModuleType:
    """The stores module under test (learner's stores.py or parsrag.stores)."""
    return load_impl("stores", "parsrag.stores")

from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def sec() -> ModuleType:
    """The security module under test (learner's security.py or parsrag.security)."""
    return load_impl("security", "parsrag.security")

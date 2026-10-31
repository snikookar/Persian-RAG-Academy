from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def doctor() -> ModuleType:
    """The doctor module under test (learner's starter or parsrag.doctor)."""
    return load_impl("doctor", "parsrag.doctor")

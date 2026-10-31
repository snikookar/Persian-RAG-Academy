from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def nz() -> ModuleType:
    """The normalize module under test (learner's starter or parsrag.normalize)."""
    return load_impl("normalize", "parsrag.normalize")

from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    """The generation module under test (learner's generate.py or parsrag.generate)."""
    return load_impl("generate", "parsrag.generate")

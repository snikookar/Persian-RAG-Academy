from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def ag() -> ModuleType:
    """The agent module under test (learner's agent.py or parsrag.agent)."""
    return load_impl("agent", "parsrag.agent")

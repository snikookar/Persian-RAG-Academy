from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def gm() -> ModuleType:
    """Generation metrics under test (learner's generation.py or parsrag.eval.generation)."""
    return load_impl("generation", "parsrag.eval.generation")


@pytest.fixture(scope="module")
def st() -> ModuleType:
    """Bootstrap statistics under test (learner's stats.py or parsrag.eval.stats)."""
    return load_impl("stats", "parsrag.eval.stats")


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    """The regression gate under test (learner's gate.py or parsrag.eval.gate)."""
    return load_impl("gate", "parsrag.eval.gate")

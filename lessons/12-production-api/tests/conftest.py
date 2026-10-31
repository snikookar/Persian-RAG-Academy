from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def cm() -> ModuleType:
    """The cache module under test (learner's cache.py or parsrag.api.cache)."""
    return load_impl("cache", "parsrag.api.cache")


@pytest.fixture(scope="module")
def rl() -> ModuleType:
    """The rate-limit module under test (learner's ratelimit.py or parsrag.api.ratelimit)."""
    return load_impl("ratelimit", "parsrag.api.ratelimit")

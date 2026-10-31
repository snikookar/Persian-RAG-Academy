from types import ModuleType

import pytest

from grader import load_impl


@pytest.fixture(scope="module")
def metrics() -> ModuleType:
    """Retrieval metrics under test (learner's metrics.py or parsrag.eval.retrieval)."""
    return load_impl("metrics", "parsrag.eval.retrieval")


@pytest.fixture(scope="module")
def tfidf() -> ModuleType:
    """TF-IDF embedding under test (learner's tfidf.py or parsrag.embeddings)."""
    return load_impl("tfidf", "parsrag.embeddings")

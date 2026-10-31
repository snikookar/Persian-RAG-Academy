"""Evaluation helpers: QA datasets and retrieval metrics. Lesson 10 adds generation metrics."""

from parsrag.eval.dataset import QAItem, load_qa
from parsrag.eval.retrieval import (
    RetrievalReport,
    contains_evidence,
    evaluate_retrieval,
    hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "QAItem",
    "RetrievalReport",
    "contains_evidence",
    "evaluate_retrieval",
    "hit_rate",
    "load_qa",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]

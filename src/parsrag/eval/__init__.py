"""Evaluation: QA datasets, retrieval metrics (lessons 04–05) and generation metrics (lesson 10).

The end-to-end runner lives in :mod:`parsrag.eval.pipeline` and the
command line in ``python -m parsrag.eval`` (``run`` / ``gate``).
"""

from parsrag.eval.dataset import QAItem, load_qa
from parsrag.eval.gate import compare_metrics
from parsrag.eval.generation import LLMJudge, citation_precision, faithfulness, token_f1
from parsrag.eval.retrieval import (
    RetrievalReport,
    contains_evidence,
    evaluate_retrieval,
    hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from parsrag.eval.stats import bootstrap_ci, paired_bootstrap

__all__ = [
    "LLMJudge",
    "QAItem",
    "RetrievalReport",
    "bootstrap_ci",
    "citation_precision",
    "compare_metrics",
    "contains_evidence",
    "evaluate_retrieval",
    "faithfulness",
    "hit_rate",
    "load_qa",
    "ndcg_at_k",
    "paired_bootstrap",
    "recall_at_k",
    "reciprocal_rank",
    "token_f1",
]

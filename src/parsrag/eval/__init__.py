"""Evaluation helpers. Lesson 10 grows this into a full evaluation toolkit."""

from parsrag.eval.dataset import QAItem, load_qa
from parsrag.eval.retrieval import hit_rate

__all__ = ["QAItem", "hit_rate", "load_qa"]

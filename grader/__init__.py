"""Automatic grader for Persian RAG Academy lessons.

Usage::

    python -m grader list
    python -m grader run 02 --path lessons/02-normalization/starter
    python -m grader check-solutions
"""

from grader.loader import LEARNER_PATH_ENV, load_impl
from grader.paths import FIXTURES, LESSONS, REPO_ROOT

__all__ = ["FIXTURES", "LEARNER_PATH_ENV", "LESSONS", "REPO_ROOT", "load_impl"]

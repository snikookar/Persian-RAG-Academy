"""Well-known repository paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS = REPO_ROOT / "lessons"
FIXTURES = REPO_ROOT / "data" / "fixtures"
CORPUS = FIXTURES / "corpus"
QA_FILE = FIXTURES / "qa.jsonl"
QA_PARAPHRASE_FILE = FIXTURES / "qa_paraphrase.jsonl"

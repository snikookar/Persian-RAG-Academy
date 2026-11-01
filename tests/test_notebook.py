"""The Colab notebook is committed clean: valid nbformat 4, outputs cleared, cells tagged."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_notebook import NOTEBOOKS, check_structure

NOTEBOOK = NOTEBOOKS / "benchmark_real_embeddings.ipynb"


def test_committed_notebook_passes_the_structure_check() -> None:
    assert check_structure(NOTEBOOK) == []


def test_offline_cells_cover_the_benchmark_table_and_chart() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    offline = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "offline" in c["metadata"].get("tags", [])
    ]
    joined = "\n".join(offline)
    for needle in ("qa_paraphrase.jsonl", "HashingEmbedding", "markdown_table", "savefig"):
        assert needle in joined


def test_check_catches_outputs_untagged_cells_and_keys(tmp_path: Path) -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = next(c for c in nb["cells"] if c["cell_type"] == "code")
    code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["hi\n"]}]
    code["execution_count"] = 3
    code["metadata"]["tags"] = []
    code["source"] = ["key = 'sk-" + "a" * 24 + "'\n"]
    bad = NOTEBOOKS / "_tmp_bad.ipynb"
    bad.write_text(json.dumps(nb), encoding="utf-8")
    try:
        errors = "\n".join(check_structure(bad))
    finally:
        bad.unlink()
    expected = ("outputs are not cleared", "execution_count", "exactly one of", "API key", "Colab")
    for needle in expected:
        assert needle in errors

"""Check the Colab notebooks offline (stdlib only, plus what the offline cells import).

    python scripts/check_notebook.py                 # structure only
    python scripts/check_notebook.py --run           # also execute the offline cells

Structure checks, for every ``notebooks/*.ipynb``:

- valid JSON, nbformat 4, every cell has the expected keys;
- outputs cleared (no outputs, ``execution_count`` null, no widget state);
- every code cell is tagged ``offline`` or ``network`` (and offline cells are plain Python:
  no ``!`` shell or ``%`` magic lines);
- no API keys or tokens in the source;
- an "Open in Colab" badge that points at this very file on ``main``.

``--run`` executes the ``offline`` cells of ``benchmark_real_embeddings.ipynb`` in order, in
one namespace, in a temporary directory, with every socket connection refused. It then
checks that the offline baseline rows the notebook prints are exactly the rows committed in
lesson 05's real-model results table, so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
LESSON05 = ROOT / "lessons" / "05-embeddings" / "README.fa.md"
COLAB = "https://colab.research.google.com/github/snikookar/Persian-RAG-Academy/blob/main/"
SECRET = re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")
TAGS = {"offline", "network"}


def source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def check_structure(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{rel}: invalid JSON ({exc})"]
    errors: list[str] = []
    if nb.get("nbformat") != 4:
        errors.append(f"{rel}: nbformat must be 4")
    if "widgets" in nb.get("metadata", {}):
        errors.append(f"{rel}: notebook metadata contains widget state")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        return [*errors, f"{rel}: no cells"]
    if COLAB + rel not in source(cells[0]):
        errors.append(f"{rel}: the first cell needs an 'Open in Colab' badge linking {COLAB}{rel}")
    for i, cell in enumerate(cells):
        where = f"{rel}: cell {i}"
        kind = cell.get("cell_type")
        if kind not in {"code", "markdown", "raw"} or "source" not in cell:
            errors.append(f"{where}: malformed cell")
            continue
        text = source(cell)
        if SECRET.search(text):
            errors.append(f"{where}: looks like it contains an API key or token")
        if kind != "code":
            continue
        if cell.get("outputs"):
            errors.append(f"{where}: outputs are not cleared")
        if cell.get("execution_count") is not None:
            errors.append(f"{where}: execution_count is not null")
        tags = set(cell.get("metadata", {}).get("tags", [])) & TAGS
        if len(tags) != 1:
            errors.append(f"{where}: tag the code cell with exactly one of {sorted(TAGS)}")
        if "offline" in tags and any(
            line.lstrip().startswith(("!", "%")) for line in text.splitlines()
        ):
            errors.append(f"{where}: offline cells must be plain Python (no ! or % lines)")
    return errors


def block_network() -> None:
    """Refuse every outgoing connection and name resolution for the rest of the process."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network access is disabled while checking the notebook")

    socket.socket.connect = refuse  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign,assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def run_offline(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    offline = [
        source(c)
        for c in nb["cells"]
        if c["cell_type"] == "code" and "offline" in c.get("metadata", {}).get("tags", [])
    ]
    block_network()
    os.environ["PARSRAG_REPO"] = str(ROOT)
    namespace: dict[str, object] = {"__name__": "__main__"}
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for i, code in enumerate(offline):
                print(f"--- offline cell {i + 1}/{len(offline)}", flush=True)
                exec(compile(code, f"{path.name}[offline {i + 1}]", "exec"), namespace)
            out = Path(tmp) / "benchmark_results"
            missing = [
                n
                for n in ("results.md", "results.json", "embedding_benchmark.png")
                if not (out / n).exists()
            ]
            if missing:
                return [f"{path.name}: offline cells did not write {missing}"]
            rows = [
                line
                for line in (out / "results.md").read_text(encoding="utf-8").splitlines()
                if line.startswith("| `")
            ]
        finally:
            os.chdir(cwd)
    if len(rows) != 3:
        return [f"{path.name}: expected 3 offline result rows, got {len(rows)}"]
    lesson = LESSON05.read_text(encoding="utf-8")
    return [
        f"lessons/05-embeddings/README.fa.md: offline row not found verbatim (re-run the "
        f"notebook's offline cells and paste it):\n  {row}"
        for row in rows
        if row not in lesson
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="execute the offline cells too")
    args = parser.parse_args()
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        sys.exit("no notebooks found in notebooks/")
    errors = [e for p in paths for e in check_structure(p)]
    if not errors and args.run:
        errors += run_offline(NOTEBOOKS / "benchmark_real_embeddings.ipynb")
    for e in errors:
        print("ERROR:", e)
    if errors:
        sys.exit(1)
    print(f"OK: {len(paths)} notebook(s){' (offline cells executed)' if args.run else ''}")


if __name__ == "__main__":
    main()

"""Run the real-model benchmark notebook, then compare each model with the best offline baseline.

Downloads model weights and takes a long time on CPU. Run from the repository root:

    pip install -e ".[st]"
    python examples/benchmark_real_paired.py

Executes the cells of ``notebooks/benchmark_real_embeddings.ipynb``, saving after every model,
and adds ``benchmark_results/paired.json`` with the paired bootstrap of the MRR difference.
The output of the 2026-11-04 run is in ``docs/eval/real_embeddings/``.
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

nb = json.loads(Path("notebooks/benchmark_real_embeddings.ipynb").read_text(encoding="utf-8"))
code = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}

ns = {"__name__": "__main__"}
exec(code[4], ns)  # setup
exec(code[6], ns)  # run_benchmark

# Keep the per-question reciprocal ranks that the notebook computes, for paired comparisons.
PER_QUESTION = {}
_capture = []
_bootstrap_ci = ns["bootstrap_ci"]
_run_benchmark = ns["run_benchmark"]


def bootstrap_ci(values, **kwargs):
    _capture.append(list(values))
    return _bootstrap_ci(values, **kwargs)


def run_benchmark(name, embedding, note=""):
    _capture.clear()
    row = _run_benchmark(name, embedding, note)
    PER_QUESTION[name] = dict(zip(ns["QA_SETS"], _capture, strict=True))
    return row


ns["bootstrap_ci"] = bootstrap_ci
ns["run_benchmark"] = run_benchmark

from parsrag.eval.stats import paired_bootstrap  # noqa: E402

BASELINE = "tfidf-char 2-4 (offline)"


def save():
    exec(code[14], ns)  # table and results files
    print("\nPaired bootstrap, delta MRR against", BASELINE)
    paired = {}
    for name, sets in PER_QUESTION.items():
        if name == BASELINE:
            continue
        paired[name] = {}
        for set_name, rr in sets.items():
            ci = paired_bootstrap(PER_QUESTION[BASELINE][set_name], rr)
            paired[name][set_name] = {"delta": ci.mean, "low": ci.low, "high": ci.high}
            print(f"{name:<62} {set_name:<11} {ci.mean:+.3f} [{ci.low:+.3f}, {ci.high:+.3f}]")
    Path("benchmark_results/paired.json").write_text(
        json.dumps(
            {"baseline": BASELINE, "paired": paired, "per_question": PER_QUESTION}, indent=2
        ),
        encoding="utf-8",
    )


exec(code[8], ns)  # offline baselines
save()

# Same loop as the notebook's model cell, one model at a time; BGE-M3 is left out for download size.
for model in (
    '("intfloat/multilingual-e5-small", "query: ", "passage: ")',
    '("intfloat/multilingual-e5-base", "query: ", "passage: ")',
    '("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "")',
):
    cell, n = re.subn(r"MODELS = \[.*?\n\]\n", f"MODELS = [{model}]\n", code[10], flags=re.S)
    assert n == 1
    exec(cell, ns)
    save()
    print("SAVED AFTER", model, flush=True)

print("\nDONE")

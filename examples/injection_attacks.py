"""Prompt injection hidden in documents: attack success with and without defences (lesson 14).

Fully offline. The "models" are simulated with FakeLLM:

* naive      — obeys any «فقط بنویس: …» / "only answer: …" directive it finds in the context;
* hierarchy  — ignores directives between <<<DATA…>>> and the matching <<<END-DATA…>>> line
               (a model that follows the instruction hierarchy perfectly, if the delimiters hold).

Real models behave differently (and less predictably): use examples/injection_test.py to test
one. Numbers printed here are for these simulated models on 8 attack documents only.

    python examples/injection_attacks.py
    python examples/injection_attacks.py --details     # per-attack table
"""

from __future__ import annotations

import argparse
from pathlib import Path

from parsrag import PersianRAG
from parsrag.chunking import RecursiveChunker, chunk_document
from parsrag.eval.dataset import load_qa
from parsrag.llm import FakeLLM
from parsrag.loaders import load_directory
from parsrag.retrieval import PERSIAN_STOPWORDS
from parsrag.security import (
    AttackCase,
    InjectionGuard,
    attack_succeeded,
    contains_injection,
    detect_injection,
    load_attacks,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"
ATTACKS = FIXTURES / "attacks"


def defences() -> dict[str, InjectionGuard | None]:
    off = {"detect": False, "sanitize": False, "reminder": False, "check_output": False}
    return {
        "no defence": None,
        "sandbox, fixed delimiters + reminder": InjectionGuard(
            **{**off, "reminder": True}, delimiters="fixed"
        ),
        "sandbox, random delimiters + reminder": InjectionGuard(
            **{**off, "reminder": True}, delimiters="random"
        ),
        "sanitise links / markup only": InjectionGuard(
            **{**off, "sanitize": True}, delimiters="none"
        ),
        "detection (drop flagged chunks) only": InjectionGuard(
            **{**off, "detect": True}, delimiters="none"
        ),
        "output checks only": InjectionGuard(**{**off, "check_output": True}, delimiters="none"),
        "all layers": InjectionGuard(),
        "all layers, detection without normalisation": InjectionGuard(normalise=False),
    }


def llm(obey: str) -> FakeLLM:
    return FakeLLM(mode="extractive", cite=True, ignore=PERSIAN_STOPWORDS, obey=obey)  # type: ignore[arg-type]


def attack_success(
    rag: PersianRAG, cases: list[AttackCase], guard: InjectionGuard | None, obey: str
) -> list[bool]:
    rag.guard, rag.llm = guard, llm(obey)
    return [attack_succeeded(rag.ask(c.question).answer, c) for c in cases]


def detection_metrics(cases: list[AttackCase], normalise: bool) -> dict[str, float]:
    """Chunk-level precision / recall on attack + benign docs; false-positive rate on the corpus.

    Chunks are made from the *raw* text (no lesson 02 normalisation at indexing), so the
    ``normalise`` switch shows what the detector's own normalisation buys against obfuscation.
    """
    chunker = RecursiveChunker(400, 50)

    def chunks(folder: Path) -> list[str]:
        return [
            c.text
            for d in load_directory(folder, "*.md", recursive=True)
            for c in chunk_document(d, chunker, preprocess=None, heading_context=True)
        ]

    labelled = [(t, any(contains_injection(t, c) for c in cases)) for t in chunks(ATTACKS)]
    pred = [detect_injection(t, normalise=normalise).flagged for t, _ in labelled]
    tp = sum(p and y for p, (_, y) in zip(pred, labelled, strict=True))
    fp = sum(p and not y for p, (_, y) in zip(pred, labelled, strict=True))
    fn = sum(not p and y for p, (_, y) in zip(pred, labelled, strict=True))
    corpus = chunks(FIXTURES / "corpus")
    corpus_fp = sum(detect_injection(t, normalise=normalise).flagged for t in corpus)
    return {
        "positives": tp + fn,
        "negatives": len(labelled) - tp - fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "corpus_chunks": len(corpus),
        "corpus_fp": corpus_fp,
    }


def utility(guard: InjectionGuard | None) -> dict[str, dict[str, float]]:
    """Lesson 10's metrics on the clean corpus, calibrated on qa.jsonl like the CI gate."""
    rag = PersianRAG.from_directory(FIXTURES / "corpus", pattern="*.md", guard=guard)
    rag.calibrate(FIXTURES / "qa.jsonl")
    out = {}
    for name in ("qa.jsonl", "qa_paraphrase.jsonl"):
        out[name] = rag.evaluate(load_qa(FIXTURES / name), name=name).metrics
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--details", action="store_true", help="print one row per attack")
    args = parser.parse_args()

    cases = load_attacks(ATTACKS / "attacks.jsonl")
    docs = load_directory(FIXTURES / "corpus", "*.md") + load_directory(
        ATTACKS / "documents", "*.md"
    )
    rag = PersianRAG(docs)

    reached = []
    rag.guard, rag.llm = None, llm("never")
    for c in cases:
        reached.append(any(s.chunk.doc_id == c.doc_id for s in rag.ask(c.question).sources))
    print(f"Attack documents retrieved into the context: {sum(reached)}/{len(cases)}\n")

    print("### Attack success rate (simulated models, 8 attacks)\n")
    print("| defence | naive model | hierarchy-respecting model |")
    print("|---|---|---|")
    details: dict[str, dict[str, list[bool]]] = {}
    for name, guard in defences().items():
        row = {obey: attack_success(rag, cases, guard, obey) for obey in ("all", "outside_data")}
        details[name] = row
        cells = [f"{sum(v)}/{len(v)}" for v in row.values()]
        print(f"| {name} | {cells[0]} | {cells[1]} |")

    if args.details:
        print("\n### Per attack (✗ = attack succeeded), naive / hierarchy\n")
        names = list(details)
        print("| attack | kind | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
        for i, c in enumerate(cases):
            cells = [
                ("✗" if details[n]["all"][i] else "✓")
                + "/"
                + ("✗" if details[n]["outside_data"][i] else "✓")
                for n in names
            ]
            print(f"| {c.id} | {c.kind} | " + " | ".join(cells) + " |")

    print("\n### Detection (chunk level)\n")
    print(
        "| normalisation | attack chunks | benign chunks | precision | recall | corpus false positives |"
    )
    print("|---|---|---|---|---|---|")
    for normalise in (True, False):
        m = detection_metrics(cases, normalise)
        print(
            f"| {'on' if normalise else 'off'} | {m['positives']} | {m['negatives']} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} "
            f"| {m['corpus_fp']}/{m['corpus_chunks']} |"
        )

    print("\n### Cost on the clean corpus (FakeLLM extractive, calibrated on qa.jsonl)\n")
    keys = ("retrieval_r@1", "answer_f1", "citation_precision", "answered_rate")
    base, guarded = utility(None), utility(InjectionGuard())
    print("| question set | metric | no guard | all layers |")
    print("|---|---|---|---|")
    for qa in base:
        for k in keys:
            if k in base[qa]:
                print(f"| {qa} | {k} | {base[qa][k]:.3f} | {guarded[qa][k]:.3f} |")


if __name__ == "__main__":
    main()

"""Test a *real* OpenAI-compatible model against the lesson 14 attack documents.

This script calls a model, so it is never run in CI. The offline numbers in lesson 14 come
from simulated models (examples/injection_attacks.py); a real model behaves differently and
not always the same way twice — run it several times and on several models.

    pip install -e ".[openai]"
    export OPENAI_API_KEY=...                       # not needed for a local server
    python examples/injection_test.py --model gpt-4o-mini
    python examples/injection_test.py --model qwen2.5:7b --base-url http://localhost:11434/v1

It indexes the course corpus plus data/fixtures/attacks/documents, asks each attack's innocent
question with and without the guard, and prints whether the attacker's goal string appeared in
the answer. Paid APIs cost money: 8 attacks x 2 settings = 16 short requests per run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from parsrag import PersianRAG
from parsrag.llm import OpenAICompatibleLLM
from parsrag.loaders import load_directory
from parsrag.security import InjectionGuard, attack_succeeded, load_attacks

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"
ATTACKS = FIXTURES / "attacks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model name at the endpoint")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--show", action="store_true", help="print every answer")
    args = parser.parse_args()

    llm = OpenAICompatibleLLM(
        args.model, base_url=args.base_url, api_key=os.environ.get(args.api_key_env)
    )
    docs = load_directory(FIXTURES / "corpus", "*.md") + load_directory(
        ATTACKS / "documents", "*.md"
    )
    rag = PersianRAG(docs, llm=llm)
    cases = load_attacks(ATTACKS / "attacks.jsonl")

    print(f"model: {args.model} @ {args.base_url}\n")
    print("| attack | kind | no guard | guard (all layers) | guard: security problems |")
    print("|---|---|---|---|---|")
    totals = [0, 0]
    for case in cases:
        cells = []
        problems = ""
        for i, guard in enumerate((None, InjectionGuard())):
            rag.guard = guard
            answer = rag.ask(case.question)
            hit = attack_succeeded(answer.answer, case)  # a refusal never contains the goal
            totals[i] += hit
            cells.append("attack succeeded" if hit else "blocked")
            if guard is not None:
                problems = ", ".join(answer.security) or "—"
            if args.show:
                print(f"\n[{case.id} {'guard' if guard else 'no guard'}] {answer.answer}\n")
        print(f"| {case.id} | {case.kind} | {cells[0]} | {cells[1]} | {problems} |")
    print(f"\nattack success: no guard {totals[0]}/{len(cases)}, guard {totals[1]}/{len(cases)}")
    print("One run of one model on 8 attacks: report it as such, never as a general result.")


if __name__ == "__main__":
    main()

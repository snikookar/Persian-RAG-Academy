# 11 — Agentic RAG with LangGraph

**Persian lesson:** [RAG عامل‌محور](../lessons/11-agentic-rag.md) · **Module:** `parsrag.agent`

## What you build

An agent around the lesson 09 pipeline: analyse and rewrite the question → route (answer, ask
for clarification, out of scope) → retrieve → rerank → grade relevance → generate → self-check
faithfulness → bounded retry with a rewritten query. The same nodes run in a plain-Python runner
(core install) and as a LangGraph `StateGraph` with a checkpointer and `interrupt()` for human
review. `OfflineAgentLLM` answers every step with rules, so the agent is testable offline.

## Key ideas

- **Order by cost.** The cheap reranker narrows 20 candidates to 6 before the expensive LLM
  grading step, and all 6 are graded in one call.
- **Colloquial Persian → standard Persian.** Rules map «چقدره», «میشه», «تو تهران» to
  «چقدر است», «می‌شود», «در تهران» and split multi-part questions.
- **Agents cost calls.** Every node is another model call; the question is whether the gain is
  worth it, and that needs a measurement, not a diagram.
- **Leakage, admitted.** Some colloquial rules were written after looking at rewritten
  paraphrase questions. Future rules need a separate development set.

## Measured results

Same index, retriever, reranker and threshold for both systems
(`python examples/agentic_rag.py`):

| Metric | orig: lesson 09 | orig: agent | para: lesson 09 | para: agent | para: agent, no colloquial rules |
|---|---|---|---|---|---|
| retrieval R@1 | 0.881 | 0.881 | 0.370 | 0.478 | 0.370 |
| retrieval MRR@10 | 0.941 | 0.941 | 0.487 | 0.557 | 0.469 |
| answer F1 | 0.276 | 0.301 | 0.016 | 0.021 | 0.007 |
| citation precision | 0.842 | 0.877 | 0.800 | 0.562 | 0.700 |
| abstention F1 | 0.714 | 0.714 | 0.178 | 0.205 | 0.178 |
| answered rate | 0.966 | 0.966 | 0.217 | 0.348 | 0.217 |
| LLM calls / question (mean / max) | 0.89 / 1 | 3.89 / 4 | 0.22 / 1 | 3.49 / 6 | 3.27 / 6 |

Paired bootstrap, answer F1 agent − lesson 09: original +0.025 [0.000, +0.063], paraphrase
+0.005 [-0.010, +0.019] — neither shown.

- The agent costs about 4.4× the LLM calls on the original set for a gain whose interval
  touches zero.
- On paraphrases the whole retrieval gain (R@1 0.370 → 0.478) comes from the colloquial rules:
  without them the agent equals lesson 09 exactly. Of that gain, about 0.043 is optimistic
  because of the leakage above (one-off measurement: no rules 0.370, first-pass rules 0.435,
  all rules 0.478).
- The retry loop rescues little, and citation precision falls to 0.562 on the newly answered
  questions. The offline faithfulness self-check never fires, because extractive answers are
  always faithful; it only matters with a model that hallucinates.

## Takeaway

An agent is a cost you must justify with numbers. Rewriting colloquial questions helps; loops
and self-checks need a real model and a held-out set to prove their worth.

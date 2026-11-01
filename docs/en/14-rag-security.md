# 14 — RAG security

**Persian lesson:** [امنیت RAG](../lessons/14-rag-security.md) · **Module:** `parsrag.security`

## What you build

Defences against *indirect prompt injection* — instructions hidden inside documents that the
retriever hands to the model: `detect_injection` (Persian and English rules on normalised text),
`sanitize_text` (links and markup), a sandbox that wraps context in delimiters with a random
per-prompt nonce, an instruction hierarchy with a reminder, post-generation checks
(`check_answer`) and `GuardedRetriever`. All of it is opt-in: `PersianRAG(guard=True)`,
`[security]` in `parsrag.toml`, or `PARSRAG_SECURITY` for the API.

## Key ideas

- **The threat.** Eight original attack documents cover override (Persian and English), data
  exfiltration through a Markdown image, citation spoofing, role hijacking, obfuscation with
  Arabic letters and invisible characters, delimiter escape and a "stealth" request aimed at
  summarisers; three benign documents that talk about security serve as hard negatives.
- **Layers, because each one fails.** Sandboxing only works with a model that respects it;
  detection and output checks share rules and fail together; links can be removed but a
  supported, faithful answer can still be an attack, because the injected text *is* in the
  source.
- **Simulated models make it measurable offline.** `FakeLLM(obey="all")` obeys directives in its
  context; `FakeLLM(obey="outside_data")` ignores directives inside the data markers.

## Measured results

Successful attacks out of 8, each attack document retrieved for its innocent question
(`python examples/injection_attacks.py`):

| Defence | naive model | hierarchy-respecting model |
|---|---|---|
| no defence | 8/8 | 8/8 |
| sandbox, fixed delimiters + reminder | 8/8 | 1/8 |
| sandbox, random delimiters + reminder | 8/8 | 0/8 |
| sanitise links / markup only | 7/8 | 7/8 |
| detection (drop flagged chunks) only | 1/8 | 1/8 |
| output checks only | 1/8 | 1/8 |
| all layers | 1/8 | 0/8 |
| all layers, detection without normalisation | 1/8 | 0/8 |

Fixed delimiters can be closed by the document itself (the delimiter-escape attack); a random
nonce cannot. The stealth attack still fools the naive model with all layers on.

Detection on raw chunks (8 with an instruction, 8 harmless): precision 1.000 / recall 0.875
with normalisation, 1.000 / 0.750 without; 0 of 109 corpus chunks flagged. On the clean corpus
the guard costs nothing measurable: retrieval R@1 0.881, answer F1 0.276, citation precision
0.842 and answered rate 0.966 on the original set, identical with and without it.

**Optimistic by construction:** the rules and the attacks were written together. The only real
test is attacks written by someone else. Real-model numbers are left to
`examples/injection_test.py`.

## Takeaway

Treat retrieved text as untrusted input. Combine a random-nonce sandbox, detection and output
checks, and keep measuring with attacks you did not write.

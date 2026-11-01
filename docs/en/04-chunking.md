# 04 — Chunking Persian text

**Persian lesson:** [قطعه‌بندی متن فارسی](../lessons/04-chunking.md) · **Module:** `parsrag.chunking`

## What you build

A Persian sentence splitter, three chunking strategies (fixed-size snapped to words, recursive
heading → paragraph → line → sentence → word, and sentence windows), `chunk_document` that
attaches offsets, section, page and optional heading context to every chunk, and a byte-based
token estimate.

## Key ideas

- **Small vs large chunks.** Small chunks give focused vectors but may lose the answer or its
  context ("this fee" — which fee?); large chunks keep context but dilute the vector and send
  more text to the model.
- **Where does a Persian sentence end?** «؟» and «؛» are boundaries, «...» and «…» are one
  boundary, decimal points (2.5, ۴۲٫۵), URLs, Persian abbreviations (ق.ظ.), initials and list
  numbers are not, and a newline is a boundary too, because web and chat text is full of
  sentences without a full stop.
- **Heading context.** Prepending the document title and section path to the text that gets
  embedded gives small chunks the context they lack.
- **Persian costs more tokens.** Byte-level BPE tokenizers start from UTF-8 bytes: a Persian
  letter is two bytes (a ZWNJ three), and tokenizers learn fewer merges for Persian. Three
  equivalent sentences: English 154 characters, 154 bytes, an estimated 46 tokens; Persian 146
  characters, 269 bytes, 69 tokens. The estimate (`ceil(bytes / 4)` per word) is a deliberately
  simple assumption; count with your model's real tokenizer before budgeting.

## Measured results

Passage level means a retrieved chunk actually contains the evidence span, which is what the
model needs. `HashingEmbedding`, normalised, 59 answerable questions
(`python examples/chunking_benchmark.py`), selected rows:

| Strategy | size | passage R@1 | passage R@3 | passage MRR | doc R@3 |
|---|---|---|---|---|---|
| fixed | 200 | 0.525 | 0.763 | 0.660 | 0.983 |
| recursive | 200 | 0.576 | 0.864 | 0.731 | 0.966 |
| recursive + headings | 200 | 0.695 | 0.966 | 0.820 | 0.966 |
| recursive + headings (course default) | 400 | 0.729 | 0.983 | 0.841 | 0.983 |
| recursive + headings | 800 | 0.915 | 0.983 | 0.948 | 0.983 |
| sentence window w=1 | – | 0.610 | 0.881 | 0.763 | 0.966 |

- **Structure helps** at every size: recursive beats fixed (MRR at 200: 0.660 → 0.731).
- **Heading context is the cheapest win:** at 200 characters passage R@3 goes 0.864 → 0.966.
- **Document-level recall hides all of this** (doc R@3 stays around 0.97–0.98).
- Bigger chunks win here because most documents are short, but an 800-character chunk sends
  about three times more text per result (560 vs 187 characters). The best R@3 (0.983) is
  already reached at 400.

The course keeps `RecursiveChunker(400, 50)` with heading context from here on.

## Takeaway

Evaluate chunking at passage level, add heading context, and size chunks in tokens rather than
words for Persian. The numbers come from a small synthetic corpus: rerun the script on your own
documents.

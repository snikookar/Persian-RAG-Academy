# 02 — Persian normalization

**Persian lesson:** [نرمال‌سازی متن فارسی](../lessons/02-normalization.md) · **Module:** `parsrag.normalize`

## What you build

A configurable normalizer for Persian text: `normalize()` plus the individual steps
(`unify_characters`, `unify_digits`, `remove_diacritics`, `remove_tatweel`,
`fix_presentation_forms`, `normalize_whitespace`, `fix_zwnj`), and a measurement of how much
each step helps retrieval.

## Key ideas

- **Persian text has many spellings of the same string.** The common culprits:
  - Arabic letters instead of Persian ones (ي/ی, ك/ک), from Arabic keyboards and old texts;
  - three digit systems (Persian ۱۲, Arabic-Indic ١٢, Latin 12) that look alike but differ in
    code points;
  - the zero-width non-joiner (ZWNJ, U+200C) typed as a space or left out («می شود»,
    «میشود», «می‌شود»);
  - diacritics, tatweel (decorative stretching), presentation forms copied out of PDFs and
    stray whitespace.
- **Normalize both sides.** Documents and questions must go through the same function,
  otherwise a clean index still misses a messy query (lesson 05 measures this).
- **Measure each step.** An ablation switches one step off at a time; that is how you learn
  which steps matter on *your* data.

## Measured results

Document-level hit rate on the 59 answerable questions (`HashingEmbedding`,
`python examples/normalization_impact.py`):

| chunk size | raw @1 | normalized @1 | raw @3 | normalized @3 |
|---|---|---|---|---|
| 200 | 0.763 | 0.831 | 0.847 | 0.949 |
| 300 | 0.814 | 0.898 | 0.864 | 0.949 |
| 400 | 0.763 | 0.864 | 0.864 | 0.966 |

With 300-character chunks hit@1 rises from 0.814 to 0.898 (48 → 53 of 59 questions). All five
fixed questions belong to documents typed with Arabic letters, and no question got worse.

Ablation (chunk 300, hit@1 / hit@3): all steps 0.898 / 0.949; without character unification
0.814 / 0.831; without diacritic removal 0.915 / 0.949; every other step switched off leaves
0.898 / 0.949 unchanged.

- **Character unification is by far the most valuable step.**
- Switching off diacritic removal *raises* hit@1 by one question — exactly the size of noise on
  59 questions, so small differences must be read with care.
- The other steps show no measurable effect with a character-n-gram embedding on this small
  set. They stay because they are free and matter for display, for BM25 (lesson 07) and for
  neural models.

## Takeaway

For Persian, normalization is not optional: one cheap step (letter unification) moves hit@1 by
eight points. Always measure, because the step that "should" matter is not always the one that
does.

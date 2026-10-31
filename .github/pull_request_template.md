## What and why

<!-- What does this PR change, and why? Link the issue it closes, e.g. "Closes #12". -->

## Type of change

- [ ] Bug fix
- [ ] Library feature or improvement (`src/parsrag/`)
- [ ] Lesson text, starter or grader tests (`lessons/`)
- [ ] Fixtures (`data/fixtures/`)
- [ ] Docs site, README or community files
- [ ] CI, packaging or tooling

## Numbers

<!-- If retrieval, chunking, normalisation, generation or fixtures changed: paste the before/after
     numbers and the command that produced them (e.g. `python examples/hybrid_search.py`).
     Never type numbers by hand. Write "none" if nothing measurable changed. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `pytest`
- [ ] `python -m grader check-solutions` and `python -m grader check-starters`
- [ ] `python -m parsrag.eval gate --baseline docs/eval/baseline.json` (baseline updated only for an intended improvement, explained above)
- [ ] `python scripts/check_links.py` and `mkdocs build --strict` (if Markdown or docs changed)
- [ ] No paid API calls, network access or model downloads in tests or CI
- [ ] New Persian sample text is original; lesson terms follow `docs/glossary.fa.md`
- [ ] `CHANGELOG.md` updated under *Unreleased* (user-visible changes)

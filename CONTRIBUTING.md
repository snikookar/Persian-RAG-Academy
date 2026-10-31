# Contributing to Persian RAG Academy

> **فارسی:** از هر مشارکتی استقبال می‌کنیم: گزارش اشکال، نظر دربارهٔ یک درس، اصلاح متن یا
> کد، یا یک درس تازه. کد، توضیحات کد و پیام‌های commit به انگلیسی‌اند و متن درس‌ها به فارسی.
> اگر سؤالی دارید، یک [Issue](https://github.com/snikookar/Persian-RAG-Academy/issues/new/choose)
> باز کنید.

Thank you for helping. This repository is two things at once — a Persian course (`lessons/`)
and the `parsrag` library (`src/parsrag/`) — and every lesson builds one module of the library.
Please read this page before opening a pull request, and follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

1. **No paid API calls, no network, no downloads in tests or CI.** Tests, the grader, CI and the
   examples' default mode run with zero API keys. Use `FakeLLM` and `HashingEmbedding` (offline,
   deterministic). Real providers are only used behind explicit flags (`--model`, `--llm`, ...)
   on the contributor's own machine.
2. **Real providers are optional extras**, imported lazily inside functions — never at package
   top level, never required by a test (use `pytest.importorskip` for extras-only checks).
3. **The core install stays light:** `pydantic` and `numpy` only. Anything else goes in an extra
   in `pyproject.toml`.
4. **Every Persian sample text is original** (see [Writing fixtures](#writing-fixtures)).
5. **Never invent numbers.** Every number in a lesson, the README or the docs site must be
   produced by a script in the repository; say which one.

## Development setup

```bash
git clone https://github.com/snikookar/Persian-RAG-Academy.git
cd Persian-RAG-Academy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[loaders,faiss,agent,api,openai,dev,docs]" pytest ruff build twine
```

Python 3.10–3.12 are supported (CI runs all three plus a core-only install).

## Before you push

Run what CI runs:

```bash
ruff check . && ruff format --check .
pytest
python -m grader check-solutions        # the reference solution passes every lesson
python -m grader check-starters         # every blank starter loads and fails
python -m parsrag.eval gate --baseline docs/eval/baseline.json   # the eval gate
python scripts/check_links.py           # relative Markdown links (offline)
mkdocs build --strict                   # the documentation site
python -m build && python scripts/check_dist.py                  # what would ship to PyPI
```

If you change retrieval, chunking, normalisation or generation, also run
`python examples/make_readme_charts.py` and commit the updated charts and
`docs/eval/readme_numbers.json` (CI fails if the committed numbers no longer reproduce).

### The eval gate

`python -m parsrag.eval gate` re-runs the end-to-end course pipeline on both question sets and
fails if any metric drops more than the tolerance (0.02) below `docs/eval/baseline.json`. If a
change makes the pipeline *better* on purpose, update the baseline in the same PR and explain the
new numbers:

```bash
python -m parsrag.eval gate --baseline docs/eval/baseline.json --update
python -m parsrag.eval run --markdown docs/eval/report.md
```

Never update the baseline to hide a regression.

## Pull requests

- Small, focused PRs with [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `build:`, `chore:`).
- Code, identifiers, docstrings, comments, commit messages and PR descriptions in **English**;
  lesson text (`lessons/*/README.fa.md`) in **Persian**.
- Full type hints, `ruff` clean, tests for every behaviour change.
- Add an entry under *Unreleased* in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- Fill in the PR template; list the numbers that changed and the script that produced them.

## Lesson structure

Each lesson lives in `lessons/NN-slug/` and is discovered by the grader from its folder name:

```text
lessons/NN-slug/
├── README.fa.md        the lesson, in Persian
├── starter/            the learner's files, with TODOs (same names as the parsrag module)
│   └── module.py
└── tests/
    ├── conftest.py     fixture that loads the learner's module or the reference solution
    └── test_lessonNN_slug.py
```

The reference solution is the matching `parsrag` module — there is no separate solution folder.

### `README.fa.md`

Every lesson README follows the same skeleton, so learners always know where they are:

1. `# درس NN — عنوان (English term)` — the title.
2. A one-line info box right below it:
   `> **ماژول:** … · **کد شروع:** … · **تصحیح:** `python -m grader run NN` (N بررسی) · **پیش‌نیاز:** …`
   The number of checks must match `python -m grader check-solutions`.
3. Introduction and a Mermaid diagram of the idea.
4. Concept sections (free structure, with code from the real module).
5. A measurement section with a table from an example script (`python examples/…`), when the
   lesson makes a choice that can be measured. Numbers must match the ROADMAP baselines.
6. *With a real model, on your own machine* — optional, never run in CI.
7. `## تمرین` — the exercise: which TODOs to fill in, and `python -m grader run NN`.
8. `### برای کنجکاوها` — open questions and experiments.
9. `## مطالعهٔ بیشتر` — papers, docs and tools (real references only, with links when possible).
10. `## درس بعد` — one sentence and a link to the next lesson's README.

Writing style: clear, friendly Persian; technical terms in Persian with the English term in
parentheses on first use, e.g. «بازیابی (Retrieval)», using the terms of the
[glossary](docs/glossary.fa.md). If a term is missing there, add it in the same PR. Use relative
links between lessons (`../07-hybrid-search/README.fa.md`); they are rewritten for the docs site
automatically, and `scripts/check_links.py` checks them.

### Starters

- Same file name as the parsrag module (`retrieval.py`, `chunking.py`, ...), so
  `python -m grader run NN --path my/folder` works with the learner's own copy.
- Module docstring: what the lesson builds, `python -m grader run NN`, where the reference
  solution is, and a suggested order of the TODOs.
- Signatures, types, docstrings and boilerplate are given; bodies to write raise
  `NotImplementedError("TODO: ...")`. A blank starter must load and fail
  (`python -m grader check-starters`).

### Grader tests and hints

Tests load the code under test through `grader.load_impl("module", "parsrag.module")`, so the
same test grades the learner's file and the reference. Give every test a bilingual hint that
helps without giving the answer away:

```python
hint = pytest.mark.hint

@hint(fa="«ك» عربی را هم به «ک» فارسی تبدیل کنید.", en="Map Arabic KAF to Persian KEHEH too.")
def test_unify_kaf(nz: ModuleType) -> None:      # nz: fixture from conftest.py
    assert nz.normalize("كتاب") == "کتاب"
```

Checks that need an optional extra must skip (not fail) without it, with the install command in
the reason.

### Adding a new lesson

1. Open a [new lesson proposal](https://github.com/snikookar/Persian-RAG-Academy/issues/new/choose)
   first, to agree on scope and on where it fits in the course.
2. Build the module in `src/parsrag/` with unit tests in `tests/`.
3. Add `lessons/NN-slug/` (README, starter, tests) following the structure above.
4. Add an example script in `examples/` for every number the lesson shows; run it in CI if it is
   offline and fast.
5. Add the lesson to the lesson tables in `README.md`, `README.fa.md`, `docs/index.md`, the `nav`
   in `mkdocs.yml` and the checklist in `ROADMAP.md`.

## Writing fixtures

`data/fixtures/` holds a small Persian corpus (the fictional shop «کهربا» and library «مهرگان»)
and the QA sets. When you add or change data:

- **Write every sentence yourself.** Never copy or lightly edit text from news, books, websites,
  Wikipedia, or model output you did not review and rewrite. The data is MIT-licensed like the
  code, so it must be ours to license.
- Keep the fiction consistent: same prices, addresses and rules across documents.
- Messy on purpose: if you add a deliberate problem (Arabic letters, missing ZWNJ, tatweel,
  digits), add it to the table in `data/fixtures/README.md`.
- QA rows: `{"id", "question", "answer", "doc_id", "evidence"}`; `evidence` must be copied
  **verbatim** from the document (`parsrag.eval.validate_qa` checks it); unanswerable questions
  use `"doc_id": null`. Paraphrases go in `qa_paraphrase.jsonl` with `source_id`.
- Regenerate derived files with `python data/fixtures/build_fixtures.py` and check them with
  `--check`.
- Changing fixtures changes numbers: re-run the examples, update lesson tables and the baseline,
  and say so in the PR.

## Reporting bugs and giving lesson feedback

Use the [issue templates](https://github.com/snikookar/Persian-RAG-Academy/issues/new/choose):
*Bug report*, *Lesson feedback* or *New lesson proposal*. For security issues, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.

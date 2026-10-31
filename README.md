# Persian RAG Academy

[![CI](https://github.com/snikookar/Persian-RAG-Academy/actions/workflows/ci.yml/badge.svg)](https://github.com/snikookar/Persian-RAG-Academy/actions/workflows/ci.yml)

**Build, evaluate, and ship Retrieval-Augmented Generation for Persian text.**

An open-source, hands-on course (written in Persian) plus `parsrag`, a Python
library that you rebuild lesson by lesson. Every lesson has a Persian
explanation, starter code with TODOs, an automatic grader, and a reference
solution in `src/parsrag/`. Everything runs **offline with zero API keys**.

> 🚧 Work in progress — a fuller README with diagrams and benchmark charts is coming.
> See [ROADMAP.md](ROADMAP.md) for the plan.

## آکادمی RAG فارسی

یک دورهٔ رایگان و پروژه‌محور به زبان فارسی برای ساختن، ارزیابی و راه‌اندازی سیستم‌های **تولید تقویت‌شده با بازیابی (RAG)** روی متن فارسی. در هر درس یک بخش از کتابخانهٔ `parsrag` را خودتان می‌سازید و یک تصحیح‌کنندهٔ خودکار کارتان را بررسی می‌کند. همهٔ تمرین‌ها بدون اینترنت و بدون کلید API اجرا می‌شوند.

## Quick start

```bash
git clone https://github.com/snikookar/Persian-RAG-Academy.git
cd Persian-RAG-Academy
pip install -e . pytest ruff
python -m grader run 00
```

## Lessons / درس‌ها

| # | Lesson | درس |
|---|---|---|
| 00 | [Setup](lessons/00-setup/README.fa.md) | آماده‌سازی محیط |
| 01 | [RAG in 50 lines](lessons/01-rag-in-50-lines/README.fa.md) | RAG در ۵۰ خط |
| 02 | [Persian normalization](lessons/02-normalization/README.fa.md) | نرمال‌سازی متن فارسی |
| 03–13 | coming soon — see [ROADMAP.md](ROADMAP.md) | به‌زودی |

## License

MIT — see [LICENSE](LICENSE). All Persian sample data in `data/fixtures/` was written originally for this repository.

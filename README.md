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
| 03 | [Loading Persian documents](lessons/03-loading-documents/README.fa.md) | بارگذاری اسناد فارسی (PDF، DOCX، HTML) |
| 04 | [Chunking Persian text](lessons/04-chunking/README.fa.md) | قطعه‌بندی متن فارسی |
| 05 | [Embeddings & a retrieval benchmark](lessons/05-embeddings/README.fa.md) | برداری‌سازی و بنچمارک بازیابی |
| 06 | [Vector stores](lessons/06-vector-stores/README.fa.md) | انباره‌های برداری (FAISS، pgvector) |
| 07 | [Hybrid search](lessons/07-hybrid-search/README.fa.md) | جست‌وجوی ترکیبی (BM25 فارسی + برداری + RRF) |
| 08 | [Reranking](lessons/08-reranking/README.fa.md) | بازرتبه‌بندی |
| 09 | [Grounded generation](lessons/09-grounded-generation/README.fa.md) | تولید مستند با ارجاع و «نمی‌دانم» |
| 10 | [Evaluation](lessons/10-evaluation/README.fa.md) | ارزیابی و دروازهٔ CI |
| 11–13 | coming soon — see [ROADMAP.md](ROADMAP.md) | به‌زودی |

## License

MIT — see [LICENSE](LICENSE). All Persian sample data in `data/fixtures/` was written originally for this repository.

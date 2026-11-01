# کتابخانهٔ parsrag

`parsrag` همان کتابخانه‌ای است که در درس‌ها می‌سازید. نصب پایه فقط به `pydantic` و `numpy` وابسته است؛ هر چیز دیگری یک «افزونهٔ اختیاری» (Extra) است و فقط وقتی بارگذاری می‌شود که واقعاً به کارش ببرید.

```bash
pip install parsrag                      # هسته
pip install "parsrag[loaders,openai]"    # PDF/DOCX/HTML + مدل‌های سازگار با OpenAI
pip install "parsrag[all]"               # همه‌چیز
```

!!! note "نسخه‌ها"
    امکانات این صفحه که با «۰.۲.۰» علامت خورده‌اند هنوز منتشر نشده‌اند؛ تا انتشار آن نسخه، از داخل مخزن با `pip install -e ".[loaders]"` نصب کنید.

| افزونه | چه می‌آورد | درس |
|---|---|---|
| `loaders` | `pypdf`، `python-docx`، `beautifulsoup4` | ۰۳ |
| `openai` | `httpx` برای هر API سازگار با OpenAI (OpenAI، vLLM، OpenRouter، Ollama) | ۰۰، ۰۵، ۰۹ |
| `ollama` | کلاینت رسمی Ollama | ۰۰ |
| `st` | `sentence-transformers` (برداری‌ساز و بازرتبه‌ساز متقاطع محلی) | ۰۵، ۰۸ |
| `faiss` | `faiss-cpu` | ۰۶ |
| `pgvector` | `pgvector`، `psycopg` | ۰۶، ۱۲ |
| `agent` | `langgraph`، `langchain-core` | ۱۱ |
| `api` | `fastapi`، `uvicorn`، `python-multipart` | ۱۲ |
| `docs`، `dev` | ابزار ساخت همین سایت، نمودارها و فایل‌های نمونه | — |

## `PersianRAG`: کل دوره در یک شیء

```python
from parsrag import PersianRAG

rag = PersianRAG()                              # پیش‌فرض‌های دوره، آفلاین
rag.add_documents(["docs/"])                    # پوشه، فایل، متن یا Document
answer = rag.ask("شرایط مرجوعی کالا چیست؟")
answer.text, answer.citations, answer.abstained
rag.passage(answer.citations[0])                # متن دقیق منبع در سند اصلی
```

```python
PersianRAG(
    documents=(), *, embedding=None, llm=None, chunker=None, store=None,
    reranker=<lexical>, k=4, fetch_k=20, stem=True, heading_context=True,
    normalize_text=True, min_score=None, config=None, guard=None,
)
```

| متد | کار |
|---|---|
| `PersianRAG.from_directory(dir, pattern="*")` | ساختن و نمایه‌سازی یک پوشه |
| `PersianRAG.from_config("parsrag.toml")` | از فایل پیکربندی TOML یا JSON ([خط فرمان](cli.md)) |
| `PersianRAG.course_default()` | پیکرهٔ دوره، کالیبره‌شده روی `qa.jsonl` — همان عددهای دروازهٔ CI |
| `add_documents(items)`، `add_path(path)`، `add_texts(texts, ids)` | افزودن سند |
| `upsert_documents(items, prune=False)` | افزودن سند تازه، بازنمایه‌سازی سند تغییرکرده، ردکردن سند بی‌تغییر (۰.۲.۰) |
| `remove_documents(ids)`، `remove_document(id)` | حذف سند و قطعه‌ها و بردارهایش (۰.۲.۰) |
| `search(query, k=None, filters=None)` | فقط بازیابی (ترکیبی + بازرتبه‌بندی)؛ `filter=` هم پذیرفته می‌شود |
| `ask(question, timings=None, filters=None)` | پاسخ مستند با ارجاع یا «نمی‌دانم» |
| `stream(question, filters=None)` | همان، به‌صورت رویداد: منابع، توکن‌ها، پایان |
| `passage(citation)` | متن اصلیِ یک ارجاع |
| `calibrate(qa)` | تنظیم آستانهٔ «نمی‌دانم» روی یک مجموعهٔ توسعه |
| `evaluate(qa)` | گزارش کامل درس ۱۰ |
| `save(dir)`، `PersianRAG.load(dir)` | ذخیره و بارگذاری کامل نمایه (قالب نسخه‌دار) |
| `PersianRAG.index_info(dir)` | محتوای `rag.json` یک نمایهٔ ذخیره‌شده |

`guard=True` (یا یک `InjectionGuard`) محافظ تزریق پرامپت [درس ۱۴](../lessons/14-rag-security.md) را روشن می‌کند؛ پیش‌فرض خاموش است.

## کار با دادهٔ واقعی (۰.۲.۰)

### ذخیره و بارگذاری نمایه

برداری‌سازی با مدل واقعی کند و گاهی پولی است؛ پس یک‌بار نمایه بسازید و بعد فقط بارگذاری کنید:

```python
rag = PersianRAG.from_config("parsrag.toml")    # اسناد [documents] را هم می‌خواند
rag.save(".parsrag/index")

rag = PersianRAG.load(".parsrag/index")          # هیچ سندی دوباره قطعه‌بندی یا برداری نمی‌شود
rag = PersianRAG.load(".parsrag/index", llm=another_llm)   # هر جزء را می‌شود عوض کرد
```

پوشهٔ نمایه این فایل‌ها را دارد:

| فایل | محتوا |
|---|---|
| `rag.json` | `format_version` (اکنون ۲)، گزینه‌ها (`k`، `min_score`، …)، هش محتوای هر سند، کل پیکربندی |
| `documents.jsonl` | سندهای اصلی |
| `chunks.jsonl` | قطعه‌ها، دقیقاً همان‌طور که نمایه شده‌اند |
| `bm25.json` | نمایهٔ توکن‌شدهٔ BM25 (بدون توکن‌سازی دوباره هنگام بارگذاری) |
| `store/` | بردارهای انبارهٔ درون‌حافظه یا FAISS |

- `load()` بدون آرگومان، برداری‌ساز، بازرتبه‌ساز، مدل زبانی و محافظ را از پیکربندیِ ذخیره‌شده می‌سازد. آرگومان‌ها بر هر دو مقدم‌اند.
- اگر برداری‌سازِ پیکربندی با برداری‌سازی که نمایه با آن ساخته شده فرق کند، `ValueError` می‌گیرید (`embedding=...` را خودتان بدهید). اگر بُعد بردارها جور نباشد، بردارها از روی قطعه‌های ذخیره‌شده دوباره ساخته می‌شوند.
- دادن `chunker` (یا `heading_context`/`normalize_text`) یعنی «قطعه‌بندی را عوض کرده‌ام»: اگر قطعه‌ها فرق کنند، نمایه از نو ساخته می‌شود.
- نمایه‌های نسخهٔ ۰.۱.۰ (بدون شمارهٔ قالب) همچنان بارگذاری می‌شوند؛ با `save()` دوباره، به قالب ۲ ارتقا می‌یابند. قالبی جدیدتر از نسخهٔ نصب‌شده خطای روشن می‌دهد.
- ذخیره ابتدا در یک پوشهٔ موقت کنار مقصد انجام می‌شود و در پایان جایگزین می‌شود؛ ذخیرهٔ نیمه‌کاره نمایهٔ قبلی را خراب نمی‌کند.

### نمایه‌سازی افزایشی

```python
report = rag.upsert_documents(["docs/"])           # سند، فایل یا پوشه
print(report.summary())
# added 3, updated 1, unchanged 120, removed 0 documents (+9 / -2 chunks)

rag.upsert_documents(["docs/"], prune=True)        # سندهایی که دیگر در پوشه نیستند حذف می‌شوند
rag.remove_documents(["old-price-list"])           # حذف با شناسه؛ شناسهٔ ناشناخته KeyError می‌دهد
```

`upsert_documents` برای هر سند هش SHA-256 متن، عنوان و فرادادهٔ آن را می‌سازد (`document_hash`؛ مسیر فایل یعنی `source` در هش نیست). سند بی‌تغییر هیچ هزینه‌ای ندارد؛ سند تغییرکرده اول قطعه‌های قبلی‌اش حذف و بعد دوباره نمایه می‌شود. `add_documents` مثل قبل سختگیر است و شناسهٔ تکراری را رد می‌کند. انباره‌های درون‌حافظه، FAISS و pgvector حذف را پشتیبانی می‌کنند (`store.delete(ids)`)؛ برای برداری‌سازهایی مثل TF-IDF که روی کل مجموعه `fit` می‌شوند، هر تغییر یعنی برازش دوباره.

### فیلتر فراداده

```python
rag.ask("هزینهٔ ارسال چقدر است؟", filters={"doc_id": ["shipping", "faq"]})
rag.ask("…", filters={"format": "pdf"})                 # فرادادهٔ سند (بارگذارها format و source می‌گذارند)
rag.ask("…", filters={"lang": "fa", "tags": "billing"}) # فرادادهٔ دلخواه خودتان در Document.metadata
rag.search("…", filters={"page": [1, 2]})               # فرادادهٔ قطعه
```

- کلیدهای قطعه (`CHUNK_FILTER_KEYS`): `doc_id`، `id`، `section`، `section_path`، `page`، `index`، `start`، `end`، … . هر کلید دیگری (و `title`) روی سند بررسی می‌شود.
- مقدار ساده یعنی برابری؛ فهرست یعنی «هرکدام از این‌ها»؛ اگر فرادادهٔ سند خودش فهرست باشد (مثل `tags`)، داشتن یک مقدار مشترک کافی است.
- شرط‌های سند همین‌جا، روی `rag.documents`، به فهرستی از `doc_id` تبدیل می‌شوند (`rag.resolve_filter`)؛ برای همین با pgvector هم کار می‌کنند.
- اگر هیچ سندی جور نباشد، پاسخ «نمی‌دانم» است (`abstain_reason="no_results"`). تابع (`lambda chunk: ...`) هم مثل قبل پذیرفته می‌شود.

### پیش‌تنظیم‌های پیکربندی

```python
from parsrag.config import PRESETS, config_template, preset_config

PRESETS                              # ("offline", "ollama", "openai-compatible")
print(config_template("ollama"))     # همان متنی که parsrag init --preset ollama می‌نویسد
cfg = preset_config("ollama")        # همان، به‌صورت RAGConfig
```

نام مدل‌ها در پیش‌تنظیم‌های واقعی جای‌نگهدار است (`"<chat-model>"`)؛ تا ویرایششان نکنید، ساختن مدل خطای روشن می‌دهد. کلید API هیچ‌وقت در پیکربندی نیست: هر ارائه‌دهنده فقط *نام* متغیر محیطی (`api_key_env`) را دارد و کلیدی مثل `api_key` در فایل رد می‌شود. ارائه‌دهندهٔ `ollama` حالا برای برداری‌سازی هم هست (نقطهٔ پایانی سازگار با OpenAI در `http://localhost:11434/v1`). راهنمای قدم‌به‌قدم: [PDFهای خودتان با Ollama](../guides/own-pdfs-ollama.md).

## ماژول‌ها، به ترتیب درس‌ها

| ماژول | مهم‌ترین نام‌ها | درس |
|---|---|---|
| `parsrag.llm` | `LLM`، `FakeLLM` (حالت‌های echo و extractive؛ `obey` برای درس ۱۴)، `OpenAICompatibleLLM`، `CountingLLM` | ۰۰ |
| `parsrag.embeddings` | `Embedding`، `HashingEmbedding`، `TfidfEmbedding`، `SentenceTransformerEmbedding`، `OpenAICompatibleEmbedding` | ۰۰، ۰۵ |
| `parsrag.naive_rag` | `NaiveRAG`، `chunk_text`، `build_prompt` | ۰۱ |
| `parsrag.normalize` | `normalize`، `Normalizer`، `NormalizerConfig`، `fix_presentation_forms` | ۰۲ |
| `parsrag.loaders` | `load_document`، `load_directory`، `load_pdf`، `load_docx`، `load_html`، `repair_pdf_text`، `locate` | ۰۳ |
| `parsrag.chunking` | `FixedSizeChunker`، `RecursiveChunker`، `SentenceWindowChunker`، `chunk_document`، `split_sentences`، `estimate_tokens` | ۰۴ |
| `parsrag.eval` | `evaluate_retrieval`، `recall_at_k`، `reciprocal_rank`، `ndcg_at_k`، `token_f1`، `faithfulness`، `LLMJudge`، `bootstrap_ci`، `paired_bootstrap`، `validate_qa` | ۰۵، ۱۰ |
| `parsrag.stores` | `VectorStore`، `InMemoryVectorStore`، `FaissVectorStore`، `PgVectorStore` | ۰۶ |
| `parsrag.retrieval` | `tokenize`، `PersianTokenizer`، `BM25Retriever`، `DenseRetriever`، `HybridRetriever`، `reciprocal_rank_fusion` | ۰۷ |
| `parsrag.rerank` | `LexicalReranker`، `CrossEncoderReranker`، `LLMReranker`، `RerankingRetriever` | ۰۸ |
| `parsrag.generate` | `GroundedGenerator`، `Citation`، `validate_citations`، `is_refusal`، `calibrate_threshold` | ۰۹ |
| `parsrag.agent` | `AgenticRAG`، `OfflineAgentLLM`، `build_graph`، `draw_mermaid` | ۱۱ |
| `parsrag.api` | `create_app`، `Settings`، `AnswerCache`، `RateLimiter` | ۱۲ |
| `parsrag.rag`، `parsrag.config`، `parsrag.cli` | `PersianRAG`، `RAGConfig`، فرمان `parsrag` | ۱۳ |
| `parsrag.security` | `InjectionGuard`، `GuardedRetriever`، `detect_injection`، `sanitize_text`، `sandbox_context`، `load_attacks` | ۱۴ |

همهٔ نام‌ها docstring انگلیسی دارند؛ در پایتون با `help(parsrag.retrieval.BM25Retriever)` ببینید.

## نمونه: هر جزء جایگزین‌پذیر است

```python
from parsrag import PersianRAG
from parsrag.chunking import RecursiveChunker
from parsrag.embeddings import SentenceTransformerEmbedding   # pip install "parsrag[st]"
from parsrag.llm import OpenAICompatibleLLM                   # pip install "parsrag[openai]"

rag = PersianRAG(
    embedding=SentenceTransformerEmbedding("intfloat/multilingual-e5-base",
                                           query_prefix="query: ", document_prefix="passage: "),
    llm=OpenAICompatibleLLM("qwen2.5:7b", base_url="http://localhost:11434/v1"),
    chunker=RecursiveChunker(400, 50),
    k=4,
)
rag.add_documents(["docs/"])
rag.calibrate("dev.jsonl")          # آستانهٔ «نمی‌دانم» را روی پرسش‌های توسعه تنظیم کن
print(rag.evaluate("test.jsonl").metrics)
```

!!! warning "عدد واقعی را خودتان بسازید"
    عددهای این سایت با مدل‌های آفلاین به دست آمده‌اند. هر مدل واقعی را با همین ابزارها روی پرسش‌های خودتان بسنجید: [درس ۰۵](../lessons/05-embeddings.md) و [درس ۱۰](../lessons/10-evaluation.md).

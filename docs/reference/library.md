# کتابخانهٔ parsrag

`parsrag` همان کتابخانه‌ای است که در درس‌ها می‌سازید. نصب پایه فقط به `pydantic` و `numpy` وابسته است؛ هر چیز دیگری یک «افزونهٔ اختیاری» (Extra) است و فقط وقتی بارگذاری می‌شود که واقعاً به کارش ببرید.

```bash
pip install parsrag                      # هسته
pip install "parsrag[loaders,openai]"    # PDF/DOCX/HTML + مدل‌های سازگار با OpenAI
pip install "parsrag[all]"               # همه‌چیز
```

!!! note "هنوز روی PyPI نیست"
    تا انتشار نسخهٔ اول، از داخل مخزن با `pip install -e ".[loaders]"` نصب کنید.

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
| `search(query, k=None, filter=None)` | فقط بازیابی (ترکیبی + بازرتبه‌بندی) |
| `ask(question, timings=None)` | پاسخ مستند با ارجاع یا «نمی‌دانم» |
| `stream(question)` | همان، به‌صورت رویداد: منابع، توکن‌ها، پایان |
| `passage(citation)` | متن اصلیِ یک ارجاع |
| `calibrate(qa)` | تنظیم آستانهٔ «نمی‌دانم» روی یک مجموعهٔ توسعه |
| `evaluate(qa)` | گزارش کامل درس ۱۰ |
| `save(dir)`، `PersianRAG.load(dir)` | ذخیره و بارگذاری نمایه |

`guard=True` (یا یک `InjectionGuard`) محافظ تزریق پرامپت [درس ۱۴](../lessons/14-rag-security.md) را روشن می‌کند؛ پیش‌فرض خاموش است.

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

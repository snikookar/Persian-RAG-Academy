# API وب (HTTP)

سرویس FastAPI درس ۱۲ همان `PersianRAG` را پشت HTTP می‌گذارد، با استریم، کش، محدودیت نرخ، لاگ ساختاریافته و رابط وب درس ۱۳. جزئیات طراحی و اندازه‌گیری‌ها در [درس ۱۲](../lessons/12-production-api.md) است.

## اجرا

```bash
pip install "parsrag[api,loaders]"
parsrag serve --docs data/fixtures/corpus          # http://127.0.0.1:8000
```

یا با Docker و PostgreSQL + pgvector:

```bash
docker compose up --build                          # http://localhost:8000
PARSRAG_LLM_PROVIDER=ollama PARSRAG_LLM_MODEL=qwen2.5:7b docker compose up
```

مستندات تعاملی OpenAPI در `/docs` ساخته می‌شود.

## نقطه‌های پایانی (Endpoints)

| مسیر | کار | پاسخ |
|---|---|---|
| `GET /` | رابط وب راست‌به‌چپ (بارگذاری، پرسش، ارجاع قابل‌کلیک) | HTML |
| `GET /health` | سلامت سرویس | وضعیت، تعداد سند و قطعه، نسخهٔ نمایه، آمار کش |
| `POST /ingest` | افزودن سند به‌صورت JSON | `{"documents": [{"id", "text", "title"}]}` |
| `POST /ingest/files` | بارگذاری فایل (TXT، MD، HTML، DOCX، PDF) | مثل بالا |
| `GET /documents`، `GET /documents/{id}` | فهرست و متن کامل سندها | JSON |
| `POST /ask` | پرسش ← پاسخ با ارجاع | `AskResponse` |
| `POST /ask/stream` | همان، به‌صورت رویدادهای سرور (Server-Sent Events) | `sources` ← `token`… ← `done` |

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' \
     -d '{"question": "هزینهٔ پیک در تهران چقدر است؟"}'
```

```json
{"text": "هزینهٔ پیک در تهران 45 هزار تومان است.", "abstained": false, "cached": false,
 "citations": [{"number": 1, "doc_id": "kahroba-shipping", "section": "شیوه‌های ارسال سفارش",
                "start": 0, "end": 375, "passage": "# شیوه‌های ارسال سفارش\n\n..."}],
 "timings": {"retrieve_ms": 7.0, "generate_ms": 0.9, "total_ms": 7.9}}
```

`passage` متن **اصلی** سند است (با `start` و `end`)، نه متن نرمال‌شده.

### استریم

```text
event: sources
data: [{"number": 1, "doc_id": "kahroba-shipping", ...}]

event: token
data: {"text": "هزینهٔ "}

event: done
data: {"answer": "هزینهٔ پیک در تهران 45 هزار تومان است. [1]", "citations": [...], ...}
```

اگر مدل وسط کار خطا بدهد یا از مهلت بگذرد، به‌جای `done` یک `event: error` می‌آید. چون پرسش در بدنهٔ `POST` است، در مرورگر از `fetch` و `ReadableStream` استفاده کنید، نه `EventSource`.

## پیکربندی با متغیرهای محیطی

| متغیر | معنا | پیش‌فرض |
|---|---|---|
| `PARSRAG_CONFIG` | فایل `parsrag.toml` یا `.json` ([خط فرمان](cli.md)) | — |
| `PARSRAG_INDEX` | پوشه‌ای که `parsrag ingest` ساخته | — |
| `PARSRAG_DOCS` | پوشه‌ای که هنگام شروع نمایه می‌شود | — |
| `PARSRAG_DATABASE_URL` | PostgreSQL + pgvector به‌جای حافظه | — |
| `PARSRAG_LLM_PROVIDER` | `fake` / `openai` / `ollama` | از پیکربندی |
| `PARSRAG_LLM_MODEL` | نام مدل | از پیکربندی |
| `PARSRAG_LLM_BASE_URL` | نشانی API سازگار با OpenAI | از پیکربندی |
| `PARSRAG_LLM_TIMEOUT` | ثانیه تا خطای `504 llm_timeout` | 30 |
| `PARSRAG_CACHE_SIZE` | تعداد پاسخ‌های کش‌شده (۰ = بدون کش) | 256 |
| `PARSRAG_CACHE_TTL` | عمر هر پاسخ در کش (ثانیه) | 3600 |
| `PARSRAG_RATE_LIMIT` | درخواست در دقیقه برای هر کاربر (۰ = خاموش) | 60 |
| `PARSRAG_MAX_UPLOAD_MB` | بزرگ‌ترین فایل قابل‌بارگذاری | 20 |
| `PARSRAG_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` | INFO |

کلید کش، پرسشِ **نرمال‌شده** به‌اضافهٔ نسخهٔ نمایه است؛ برای همین همان پرسش با «ي» و «ك» عربی، بدون «؟» یا با فاصلهٔ اضافه هم به همان پاسخ کش‌شده می‌رسد (در درس ۱۲: ۱۱۷ از ۱۱۷ گونهٔ املایی، در برابر ۰ با کلید خام).

## خطاها

همهٔ خطاها یک بدنهٔ یکسان دارند: `{"error": "...", "message": "..."}`.

| وضعیت | کد | `error` |
|---|---|---|
| هنوز سندی نمایه نشده | 409 | `empty_index` |
| سند تکراری | 409 | `duplicate_document` |
| ورودی نامعتبر (پرسش خالی، بیش از ۲۰۰۰ نویسه) | 422 | اعتبارسنجی FastAPI |
| فایل بزرگ‌تر از حد / نوع پشتیبانی‌نشده | 413 / 415 | `file_too_large` / `unsupported_file` |
| درخواست زیاد | 429 | `rate_limited` و سرآیند `Retry-After` |
| مدل خطا داد | 502 | `llm_error` |
| مدل در مهلت جواب نداد | 504 | `llm_timeout` |

## از داخل پایتون

```python
from parsrag import PersianRAG
from parsrag.api import Settings, create_app

app = create_app(PersianRAG.from_directory("docs/"), Settings(rate_limit=0))
# uvicorn.run(app)  یا  fastapi.testclient.TestClient(app) در تست‌ها
```

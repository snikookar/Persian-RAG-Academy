# خط فرمان (CLI)

نصب `parsrag` فرمان `parsrag` را هم نصب می‌کند ([درس ۱۳](../lessons/13-capstone.md)). همهٔ فرمان‌ها بدون مدل واقعی و بدون اینترنت کار می‌کنند؛ مدل را در `parsrag.toml` انتخاب می‌کنید.

```text
parsrag init | ingest | remove | ask | eval | serve
```

## گردش کار پیشنهادی

```mermaid
flowchart LR
    I["parsrag init --preset ollama"] --> G["parsrag ingest docs/<br/>--calibrate dev.jsonl"]
    G -->|"فایل تازه یا تغییرکرده"| G
    G --> A["parsrag ask «…»"]
    G --> E["parsrag eval test.jsonl<br/>--baseline baseline.json"]
    E --> S["parsrag serve"]
```

## `parsrag init [path] [--force] [--preset NAME]`

یک `parsrag.toml` با توضیح همهٔ گزینه‌ها می‌نویسد. `--preset` تعیین می‌کند مدل‌ها از کجا بیایند:

| پیش‌تنظیم | برداری‌ساز و مدل زبانی | کلید |
|---|---|---|
| `offline` (پیش‌فرض) | `hashing` و `fake` — بدون دانلود و اینترنت | — |
| `ollama` | مدل‌های محلی از طریق Ollama (`http://localhost:11434/v1`) | لازم نیست |
| `openai-compatible` | هر سرور سازگار با OpenAI (`base_url` را ویرایش کنید) | فقط از متغیر محیطی `api_key_env` |

در دو پیش‌تنظیم واقعی، نام مدل‌ها جای‌نگهدار است (`"<embedding-model>"`، `"<chat-model>"`) و باید ویرایش شوند؛ اگر فراموش کنید، `parsrag ingest` خطای روشن می‌دهد. راهنمای کامل: [PDFهای خودتان با Ollama](../guides/own-pdfs-ollama.md).

پیش‌تنظیم `offline` این است:

```toml
[documents]
paths = ["docs"]               # پوشه‌ها یا فایل‌ها، نسبت به همین فایل
pattern = "*"
recursive = true

[index]
path = ".parsrag/index"        # خروجی parsrag ingest، ورودی ask و serve

[chunking]
size = 400                     # درس ۰۴
overlap = 50
heading_context = true

[retrieval]
k = 4                          # تعداد متن‌هایی که به مدل داده می‌شود (درس ۰۹)
fetch_k = 20                   # نامزدهای بازرتبه‌ساز (درس ۰۸)
rerank = "lexical"             # "lexical" | "cross-encoder" | "none"
stem = true                    # ریشه‌یابی سبک در BM25 (درس ۰۷)
# min_score = 4.0              # با parsrag eval --calibrate تنظیم می‌شود (درس ۰۹)

[embedding]
provider = "hashing"           # "hashing" (آفلاین) | "st" | "openai" | "ollama" | "tfidf"

[llm]
provider = "fake"              # "fake" (آفلاین، استخراجی) | "ollama" | "openai"
# model = "qwen2.5:7b"
# base_url = "http://localhost:11434/v1"
# api_key_env = "OPENAI_API_KEY"
timeout = 60

[security]
enabled = false                # محافظ تزریق پرامپت (درس ۱۴)
# action = "drop"              # "drop" | "flag" برای قطعه‌های مشکوک
```

کلید API هیچ‌وقت در فایل نوشته نمی‌شود؛ فقط *نام* متغیر محیطی‌اش (`api_key_env`). کلیدی مثل `api_key` در فایل با خطا رد می‌شود.

## `parsrag ingest [paths ...]`

نمایه را می‌سازد و **به‌طور پیش‌فرض ذخیره می‌کند**. اگر نمایه از قبل باشد، فقط سندهای تازه و تغییرکرده نمایه می‌شوند (هش محتوا) و سندهای بی‌تغییر دوباره برداری نمی‌شوند:

```text
$ parsrag ingest docs/
added 0, updated 1, unchanged 41, removed 0 documents (+3 / -2 chunks)
indexed 42 documents, 388 chunks → .parsrag/index
```

اگر از ساخت نمایه به بعد بخش `[chunking]` یا `[embedding]` یا `stem` عوض شده باشد، یا نمایه از نسخهٔ ۰.۱.۰ باشد، نمایه خودکار از نو ساخته می‌شود و دلیلش چاپ می‌شود.

| گزینه | کار |
|---|---|
| `paths` | پوشه‌ها یا فایل‌ها (پیش‌فرض: `[documents] paths` در پیکربندی) |
| `--config` | فایل TOML یا JSON (پیش‌فرض: `./parsrag.toml` اگر باشد) |
| `--index` | پوشهٔ نمایه (پیش‌فرض: `[index] path`) |
| `--calibrate QA.jsonl` | آستانهٔ «نمی‌دانم» را روی این مجموعه کالیبره و ذخیره کن |
| `--prune` | سندهای نمایه‌شده‌ای را که دیگر در این مسیرها نیستند حذف کن |
| `--rebuild` | نمایهٔ موجود را نادیده بگیر و همه‌چیز را از نو بساز |
| `--no-save` | نمایه بساز ولی ذخیره نکن (برای آزمایش) |

## `parsrag remove ID [ID ...]`

سندها را با شناسه از نمایهٔ ذخیره‌شده حذف می‌کند (قطعه‌ها، بردارها و BM25). شناسهٔ سند همان نام فایل بدون پسوند است، یا در خروجی `parsrag ask --json` و `GET /documents` دیده می‌شود.

## `parsrag ask "پرسش"`

| گزینه | کار |
|---|---|
| `--k` | تعداد متن‌هایی که به مدل داده می‌شود |
| `--json` | کل پاسخ (متن، ارجاع‌ها، بازه‌ها) به‌صورت JSON |
| `--filter KEY=VALUE` | فقط سندها یا قطعه‌های جور؛ تکرارپذیر، مقدار می‌تواند JSON باشد: `--filter doc_id=faq --filter 'page=[1,2]'` |
| `--config`، `--index` | مثل بالا |

معنای فیلترها در [کتابخانه](library.md) (بخش «فیلتر فراداده») آمده است.

## `parsrag eval QA.jsonl`

گزارش درس ۱۰ (بازیابی، F1 پاسخ، دقت ارجاع، وفاداری، «نمی‌دانم») با بازهٔ اطمینان بوت‌استرپ، و در صورت نیاز دروازهٔ افت کیفیت. پیش از ارزیابی، `validate_qa` پرسش‌هایی را که شاهدشان در سند پیدا نمی‌شود گزارش می‌کند.

| گزینه | کار |
|---|---|
| `--calibrate` | کالیبره روی همین فایل (فقط برای آزمایش؛ نتیجه خوش‌بینانه است) |
| `--calibrate-on DEV.jsonl` | کالیبره روی یک فایل دیگر (توصیه‌شده) |
| `--save` | آستانهٔ کالیبره‌شده را در نمایه ذخیره کن |
| `--json`، `--markdown` | نوشتن گزارش کامل یا خلاصه |
| `--baseline FILE` | مقایسه با خط پایه؛ اگر معیاری بیش از حد مجاز افت کرده باشد، با کد خطا خارج می‌شود |
| `--update-baseline` | به‌جای مقایسه، خط پایه را بنویس |
| `--tolerance` | افت مجاز (پیش‌فرض: مقدار ذخیره‌شده در خط پایه) |
| `--name` | نام گزارش یا خط پایه |

## `parsrag serve`

API و رابط وب ([API وب](http-api.md))؛ به افزونهٔ `api` نیاز دارد (`pip install "parsrag[api]"`).

| گزینه | کار |
|---|---|
| `--index` | نمایه‌ای که `parsrag ingest` ساخته |
| `--docs DIR` | به‌جای بارگذاری نمایه، این پوشه را هنگام شروع نمایه کن |
| `--host`، `--port` | پیش‌فرض `127.0.0.1:8000` |
| `--config` | مثل بالا |

## ابزارهای مخزن (فقط در کلون)

```bash
python -m grader list                    # فهرست درس‌ها
python -m grader run 07 --path my/dir    # تصحیح کد شما برای درس ۰۷
python -m grader progress                # جدول پیشرفت: قبول / ناتمام / شروع‌نشده (--root my/work)
python -m grader check-solutions         # پاسخ مرجع همهٔ درس‌ها را بسنج
python -m parsrag.eval run --markdown docs/eval/report.md
python -m parsrag.eval gate --baseline docs/eval/baseline.json
python -m parsrag.doctor                 # بررسی محیط (درس ۰۰)
```

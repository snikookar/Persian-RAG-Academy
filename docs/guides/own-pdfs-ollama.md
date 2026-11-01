# PDFهای خودتان با Ollama

این راهنما قدم‌به‌قدم نشان می‌دهد چطور `parsrag` را روی PDFهای خودتان و با مدل‌های محلی [Ollama](https://ollama.com) اجرا کنید: بی‌کلید، بی‌هزینه و بدون فرستادن سندها به جایی بیرون از رایانهٔ خودتان. پیش‌نیاز مفهومی‌اش [درس ۱۳](../lessons/13-capstone.md) است؛ جزئیات هر فرمان در [مرجع خط فرمان](../reference/cli.md) آمده است.

!!! note "نسخهٔ ۰.۲.۰"
    `--preset`، نمایه‌سازی افزایشی و `--filter` در نسخهٔ ۰.۲.۰ آمده‌اند. تا انتشار آن، از داخل مخزن نصب کنید: `pip install -e ".[loaders,openai]"`.

## ۱. نصب

```bash
pip install "parsrag[loaders,openai]"
```

`loaders` خواندن PDF، DOCX و HTML را اضافه می‌کند و `openai` کلاینت HTTP برای نقطهٔ پایانی سازگار با OpenAI را، که Ollama هم همان را ارائه می‌دهد. Ollama را از سایت خودش نصب کنید و مطمئن شوید در حال اجراست:

```bash
ollama list
```

## ۲. انتخاب و دریافت دو مدل

به دو مدل نیاز دارید: یک **برداری‌ساز** (Embedding) چندزبانه برای بازیابی و یک **مدل گفت‌وگو** برای نوشتن پاسخ. هر دو را با `ollama pull` دریافت کنید؛ برای نمونه:

```bash
ollama pull bge-m3          # نمونه‌ای از یک برداری‌ساز چندزبانه
ollama pull qwen2.5:7b      # نمونه‌ای از یک مدل گفت‌وگوی چندزبانه
```

این نام‌ها فقط نمونه‌اند، نه توصیهٔ سنجیده. کدام مدل برای متن فارسیِ *شما* بهتر است را باید خودتان اندازه بگیرید (گام ۷ و [درس ۰۵](../lessons/05-embeddings.md)). هر مدلی که فارسی را خوب می‌فهمد و روی سخت‌افزارتان جا می‌شود قابل استفاده است.

## ۳. ساختن پیکربندی

```bash
mkdir my-rag && cd my-rag
parsrag init --preset ollama
```

در `parsrag.toml` دو جای‌نگهدار را با نام مدل‌هایی که دریافت کرده‌اید عوض کنید:

```toml
[embedding]
provider = "ollama"
model = "bge-m3"               # به‌جای "<embedding-model>"
base_url = "http://localhost:11434/v1"

[llm]
provider = "ollama"
model = "qwen2.5:7b"           # به‌جای "<chat-model>"
base_url = "http://localhost:11434/v1"
timeout = 120
```

اگر جای‌نگهداری را جا بیندازید، `parsrag` پیش از هر کاری با پیام روشن متوقف می‌شود. بعضی برداری‌سازها (مثل خانوادهٔ E5) با پیشوند آموزش دیده‌اند؛ برای آن‌ها `query_prefix` و `document_prefix` را از حالت توضیح دربیاورید.

## ۴. نمایه‌سازی PDFها

PDFها (و هر فایل TXT، MD، HTML یا DOCX دیگری) را در پوشهٔ `docs/` کنار `parsrag.toml` بگذارید و نمایه بسازید:

```bash
parsrag ingest
```

```text
added 12, updated 0, unchanged 0, removed 0 documents (+418 / -0 chunks)
indexed 12 documents, 418 chunks → .parsrag/index
```

(عددهای بالا فقط شکل خروجی را نشان می‌دهند.) هر PDF صفحه‌به‌صفحه خوانده و تعمیر می‌شود: ترتیب دیداری، نیم‌فاصله‌های گم‌شده و «شکل‌های نمایشی» حروف ([درس ۰۳](../lessons/03-loading-documents.md)). شناسهٔ هر سند نام فایل بدون پسوند است. نمایه در `.parsrag/index` ذخیره می‌شود و بعد از این هیچ سندی بی‌دلیل دوباره برداری نمی‌شود.

!!! warning "PDF اسکن‌شده"
    PDFی که فقط تصویرِ صفحه است متنی برای استخراج ندارد و `parsrag` نویسه‌خوانی نوری (OCR) انجام نمی‌دهد. چنین فایل‌هایی را اول با یک ابزار OCR که فارسی را پشتیبانی می‌کند به متن تبدیل کنید.

## ۵. پرسیدن

```bash
parsrag ask "شرایط بازگرداندن کالا چیست؟"
parsrag ask "شرایط بازگرداندن کالا چیست؟" --json         # همراه با متن دقیق منبع هر ارجاع
parsrag ask "مبلغ قرارداد چقدر است؟" --filter doc_id=contract-1404
parsrag ask "…" --filter format=pdf                     # فقط PDFها
```

هر پاسخ با ارجاع‌های `[1]`، `[2]` به قطعه‌های منبع (سند، بخش و صفحه) برمی‌گردد. اگر پاسخ در سندها نباشد، پاسخ درست «نمی‌دانم» است؛ برای این‌که سیستم واقعاً بتواند بگوید «نمی‌دانم»، گام ۶ را انجام دهید.

## ۶. تنظیم آستانهٔ «نمی‌دانم»

بدون کالیبره‌کردن، سیستم هیچ‌وقت بر اساس امتیاز بازیابی از پاسخ‌دادن خودداری نمی‌کند ([درس ۰۹](../lessons/09-grounded-generation.md)). یک فایل کوچک از پرسش‌های واقعی خودتان بسازید، شامل چند پرسش که جوابشان در سندها **نیست** (`doc_id` خالی):

```json
{"id": "d1", "question": "مهلت بازگرداندن کالا چند روز است؟", "answer": "هفت روز", "doc_id": "returns-policy", "evidence": "تا هفت روز پس از تحویل"}
{"id": "d2", "question": "قیمت بلیت کنسرت هفتهٔ بعد چند است؟", "answer": "", "doc_id": null}
```

```bash
parsrag ingest --calibrate dev.jsonl
```

برای سنجیدن کیفیت، فایل دومی (`test.jsonl`) با پرسش‌های دیگر بسازید و آن را ارزیابی کنید؛ هرگز روی همان فایلی که کالیبره کرده‌اید نتیجه اعلام نکنید ([درس ۱۰](../lessons/10-evaluation.md)):

```bash
parsrag eval test.jsonl
```

## ۷. مقایسهٔ مدل‌ها

برای مقایسهٔ دو برداری‌ساز، `model` را در `[embedding]` عوض کنید و دوباره `parsrag ingest` و `parsrag eval test.jsonl` بزنید. `parsrag` تغییر برداری‌ساز را تشخیص می‌دهد و نمایه را خودکار از نو می‌سازد. به بازه‌های اطمینان گزارش نگاه کنید: با چند ده پرسش، اختلاف‌های کوچک اغلب معنادار نیستند.

## ۸. به‌روز نگه‌داشتن نمایه

```bash
cp new-report.pdf docs/ && parsrag ingest          # فقط فایل تازه نمایه می‌شود
parsrag ingest --prune                             # فایل‌های پاک‌شده از docs/ از نمایه هم حذف می‌شوند
parsrag remove old-price-list                      # حذف یک سند با شناسه
parsrag ingest --rebuild                           # ساختن دوباره از صفر
```

سندی که محتوایش عوض شده باشد (هش متن، عنوان و فراداده)، قطعه‌های قبلی‌اش را از دست می‌دهد و دوباره نمایه می‌شود؛ بقیه دست نمی‌خورند.

## ۹. رابط وب و API

```bash
pip install "parsrag[loaders,openai,api]"
parsrag serve                                      # http://127.0.0.1:8000
```

`serve` همان نمایهٔ ذخیره‌شده را بارگذاری می‌کند. رابط وب راست‌به‌چپ است و ارجاع‌ها قابل کلیک‌اند. با API هم می‌شود فیلتر داد یا سند حذف کرد ([API وب](../reference/http-api.md)):

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' \
     -d '{"question": "مبلغ قرارداد چقدر است؟", "filters": {"format": "pdf"}}'
```

## عیب‌یابی

| نشانه | علت محتمل و راه‌حل |
|---|---|
| `llm.model is still the placeholder` | نام مدل را در `parsrag.toml` ویرایش نکرده‌اید (گام ۳). |
| `Connection refused` یا `ConnectError` | Ollama اجرا نیست؛ `ollama serve` یا برنامهٔ Ollama را باز کنید. |
| خطای `model not found` از سرور | مدل را با `ollama pull` دریافت نکرده‌اید یا نامش در پیکربندی فرق دارد. |
| `504 llm_timeout` یا کندی شدید در اولین پرسش | بار اول مدل در حافظه بارگذاری می‌شود؛ `timeout` را در `[llm]` بیشتر کنید. |
| `needs pypdf` | افزونهٔ `loaders` نصب نیست. |
| متن PDF به‌هم‌ریخته یا خالی | PDF اسکن‌شده است (OCR لازم است) یا فونت‌هایش نگاشت یونیکد ندارند؛ با `parsrag.loaders.load_pdf(path, repair=False)` متن خام را ببینید. |
| پاسخ می‌دهد ولی باید «نمی‌دانم» بگوید | آستانه کالیبره نشده است (گام ۶). |

## از داخل پایتون

```python
from parsrag import PersianRAG

rag = PersianRAG.load(".parsrag/index")       # مدل‌ها از پیکربندی ذخیره‌شده ساخته می‌شوند
answer = rag.ask("شرایط بازگرداندن کالا چیست؟", filters={"format": "pdf"})
print(answer.text)
for c in answer.citations:
    print(c.number, c.doc_id, c.page, rag.passage(c)[:80])

rag.upsert_documents(["docs/"])               # سندهای تازه یا تغییرکرده
rag.save(".parsrag/index")
```

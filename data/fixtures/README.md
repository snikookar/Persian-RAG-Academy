# Fixtures

Small, **original** Persian test data written for this repository (MIT licensed,
like the rest of the repo). Nothing here is copied from news, books or websites.
«کهربا» (an online shop) and «سپیدار» (a public library) are fictional.

- `corpus/*.md` — 25 documents. The first line (`# ...`) is the title; the
  file name (without `.md`) is the document id. 21 are short (60–120 words);
  4 are long, multi-section documents (800–1100 words) so that chunking
  (lesson 04) actually matters: `kahroba-seller-guide`, `kahroba-privacy`,
  `sepidar-annual-report`, `sepidar-children`.
- `qa.jsonl` — 66 evaluation questions:
  `{"id", "question", "answer", "doc_id", "evidence"}`.
  `doc_id` is `null` for the 7 questions whose answer is **not** in the corpus
  (used to test honest "I don't know" answers in lesson 09). For every
  answerable question, `evidence` is a short span copied **verbatim** from the
  document that contains the answer — it enables passage-level evaluation
  ("did the retrieved chunk contain the answer?", lesson 04).
- `documents/` — files in other formats for the loaders (lesson 03), see below.

## The text is messy on purpose

The documents imitate real-world Persian text so that normalization
(lesson 02) has a measurable effect. Questions are written in clean Persian.

| Document | Deliberate problems |
|---|---|
| kahroba-returns | Arabic `ي`/`ك` everywhere, Arabic-Indic digits (`٧`, `٥`) |
| kahroba-shipping | ZWNJ replaced by spaces (`می شود`)، Latin digits, extra spaces before `.` |
| kahroba-warranty | tatweel (`گارانـــتی`), diacritics (`دَستگاه`, `فَنّی`), missing ZWNJ (`میشود`) |
| kahroba-faq | Arabic `ك`, `می` written as a separate word |
| kahroba-payment | Arabic `ي`/`ك`, Arabic-Indic digits |
| kahroba-club | double spaces, ZWNJ replaced by spaces, Latin digits |
| kahroba-seller-guide | ZWNJ replaced by spaces throughout, Latin digits |
| kahroba-privacy | `می` as a separate word, diacritics (`مُهمی`), Persian digits |
| kahroba-account, kahroba-discounts | Arabic `ك`; Arabic-Indic digits in discounts |
| kahroba-gift-cards, -installation, -stores, -trade-in | ZWNJ replaced by spaces / Arabic-Indic digits |
| sepidar-rules | Arabic `ي`/`ك`, diacritics (`كِتاب`, `مَمنوع`) |
| sepidar-membership | ZWNJ replaced by spaces, mixed Persian/Latin/Arabic digits |
| sepidar-hours | tatweel, Arabic-Indic digits |
| sepidar-digital | Arabic `ي`/`ك`, missing ZWNJ, Arabic presentation forms (`ﺭﺍﻳﮕﺎﻥ`, as copied from PDFs) |
| sepidar-annual-report | one section in Arabic `ي`/`ك`; Persian thousands/decimal separators (`۱۰٬۴۲۰`, `۹٫۶`) |
| sepidar-children | tatweel in headings, Arabic-Indic digits |
| sepidar-rooms | Arabic `ي`/`ك` |
| sepidar-interlibrary | ZWNJ replaced by spaces, Latin digits |

## `documents/` — other file formats (lesson 03)

Generated from the UTF-8 texts in `documents/sources/` by
`build_fixtures.py` (DOCX/TXT need only the standard library and are
byte-for-byte reproducible — a test checks this; the PDFs need
`pip install -e ".[dev]"`). The HTML page is written by hand.

| File | Problem it reproduces |
|---|---|
| `kahroba-notice.txt` | Windows-1256 encoding, Arabic `ي`/`ك`, CRLF line endings |
| `sepidar-events.html` | navigation, breadcrumb, cookie banner, sidebar and footer around the content; `&zwnj;` entities; an LRM mark |
| `kahroba-installation-guide.docx` | Title/Heading styles, a bullet list, a table, RLM marks around Latin model numbers |
| `kahroba-catalog.pdf` | 2 pages; glyphs stored as Arabic presentation forms (logical order); ZWNJs lost |
| `sepidar-newsletter.pdf` | presentation forms stored in **visual** (reversed) order; ZWNJs lost |

Both PDFs render correctly on screen; the problems only appear when text is
extracted. They follow parsrag's own shaping model, so real PDFs can be messier.

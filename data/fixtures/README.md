# Fixtures

Small, **original** Persian test data written for this repository (MIT licensed,
like the rest of the repo). Nothing here is copied from news, books or websites.
«کهربا» (an online shop) and «سپیدار» (a public library) are fictional.

- `corpus/*.md` — 10 short documents. The first line (`# ...`) is the title; the
  file name (without `.md`) is the document id.
- `qa.jsonl` — 28 evaluation questions: `{"id", "question", "answer", "doc_id"}`.
  `doc_id` is `null` for the 4 questions whose answer is **not** in the corpus
  (used to test honest "I don't know" answers in lesson 09).

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
| sepidar-rules | Arabic `ي`/`ك`, diacritics (`كِتاب`, `مَمنوع`) |
| sepidar-membership | ZWNJ replaced by spaces, mixed Persian/Latin/Arabic digits |
| sepidar-hours | tatweel, Arabic-Indic digits |
| sepidar-digital | Arabic `ي`/`ك`, missing ZWNJ, Arabic presentation forms (`ﺭﺍﻳﮕﺎﻥ`, as copied from PDFs) |

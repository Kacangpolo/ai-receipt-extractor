# AI Receipt Extractor

Turn photos of paper sales receipts (*nota*) into clean, queryable data — no manual typing.

**Live demo:** https://ai-receipt-extractor-2jmg52mqgfhowmoux8uugm.streamlit.app/ (read-only, 15 synthetic receipts)

Small businesses in Indonesia still run on paper receipts. Someone re-types them into Excel at the end of the day, which is slow and error-prone. This pipeline sends each photo to a vision LLM, gets back **validated structured data**, checks it, and loads it into PostgreSQL ready for dashboards.

## How it works

```
 receipt photos ──► LLM (vision + JSON schema output) ──► Pydantic validation ──► PostgreSQL ──► dashboard
   data/receipts        structured JSON output           + sum check             receipts,
                                                                                 receipt_items
```

- **Pluggable LLM backend:** Google Gemini (free tier) or Anthropic Claude, switched with one line in `.env` (`LLM_PROVIDER`).
- **Structured output, not free text:** both backends are constrained to a JSON schema generated from the Pydantic model (Gemini via response schema, Claude via a forced tool call), so every result has the same shape.
- **Rate-limit aware:** requests are spaced out and retried with backoff on HTTP 429, so the free tier doesn't break a batch.
- **Validation:** types are validated, and consistency checks use the receipt's own redundancy (`items = subtotal`, `subtotal + tax = total`) plus missing-field rules.
- **Second opinion:** receipts that fail the checks are re-read by a stronger model (`GEMINI_FALLBACK_MODEL`); the answer with fewer issues wins.
- **Idempotent:** images are hashed (SHA-256), so re-running the batch skips receipts already loaded.
- **Cost tracking:** input/output tokens are stored per receipt.
- **Measured accuracy:** `src/evaluate.py` compares output against hand-labelled ground truth.

## Results

15 synthetic receipts (`samples/`), primary model `gemini-3.5-flash-lite`, second opinion `gemini-3.8-flash`:

| Metric | v1 | v2 (date fix) | v3 (checks + second opinion) |
|---|---|---|---|
| Receipts evaluated | 3 | 14 of 15 | **15 of 15** |
| Merchant name accuracy | 100% | 100% | 100% |
| Date accuracy | 66.7% | 100% | 100% |
| Total amount accuracy | 100% | 100% | 100% |
| All fields correct | 66.7% | 100% | **100%** |
| Needed a second opinion | – | – | 2 of 15 |
| Left for human review | – | – | 0 |

The two receipts sent for a second opinion failed for different reasons: one came back with no merchant, no total and an unpriced item; the other listed its items twice (items 750,000 vs printed subtotal 375,000). Both were caught by the consistency checks and fixed by the stronger model.

_Next: evaluate on 30 real receipts with varied quality (crumpled, handwritten, low light), then compare models on accuracy, speed and cost._

## Lessons learned

**Don't let the LLM do deterministic conversions.** In v1 the model read `03/10/2026` (3 October, Indonesian day-first format) as `2026-03-03`. Prompting harder is fragile, so in v2 the model only *copies* the date exactly as printed (`date_printed`) and Python parses it with an explicit day-first rule (`parse_printed_date`). Date accuracy went from 66.7% to 100%, and a regression test locks the fix in.

**A wrong answer can be internally consistent.** On a re-run, the small model returned 1 of 5 items, no merchant, and copied the first item's price (12,000) into `total`, with `confidence: 1.0`. The v1 sum check (`items + tax = total`) could not catch it, because the only item had no line total. v3 checks the redundancy every receipt prints (`items = subtotal`, `subtotal + tax = total`), flags missing fields, and sends flagged receipts to a stronger model for a second opinion. The model's self-reported confidence is not used to decide anything.

## Tech stack

Python · Google Gemini API · Anthropic API (Claude) · Pydantic · pandas · Streamlit · PostgreSQL · Docker Compose

## Quick start

```bash
git clone https://github.com/Kacangpolo/ai-receipt-extractor.git
cd ai-receipt-extractor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY (free: aistudio.google.com/apikey)

# put receipt photos in data/receipts/, then:
python -m src.pipeline --no-db --limit 3   # try 3 receipts, JSON output only
```

With PostgreSQL:

```bash
docker compose up -d db            # Postgres with tables created from sql/schema.sql
python -m src.pipeline             # JSON + database
```

Fully containerised:

```bash
docker compose --profile run up --build
```

Open the dashboard (upload photos, charts, review queue, Excel export):

```bash
streamlit run dashboard.py
```

Export everything to Excel without the dashboard: `python -m src.report`

Measure accuracy (fill `data/labels.csv` with the true values first):

```bash
python -m src.evaluate
```

Run tests (offline, no API key needed): `python -m pytest -q`

## Deploy (Streamlit Community Cloud)

1. Run the pipeline on `samples/*.jpg` locally and copy the JSON results to `samples/output/` (the public demo reads these, so it costs no API calls).
2. Push to GitHub, then create an app at share.streamlit.io with main file `dashboard.py`.
3. In *Advanced settings → Secrets*, add:
   ```toml
   GEMINI_API_KEY = "..."
   APP_PASSWORD = "..."
   ```
   Root-level secrets are exposed as environment variables, so the code reads them like `.env`.

Visitors get a read-only demo; the password unlocks upload. Files written online are not guaranteed to persist across restarts, so a database is the next step for real use.

## Project structure

```
src/schema.py      Pydantic model + JSON schema for the LLM tool
src/extractor.py   Claude backend + provider factory (get_extractor)
src/gemini.py      Gemini backend with rate limiting and retries
src/pipeline.py    batch runner (hashing, skip, logging, DB load)
src/db.py          PostgreSQL writes
src/evaluate.py    accuracy against ground truth
src/report.py      summaries, review queue, Excel export
dashboard.py       Streamlit dashboard
sql/schema.sql     tables + daily spend view
tests/             offline tests with a fake LLM client
```

## Roadmap

- [x] Extraction pipeline with validation and PostgreSQL storage
- [x] Accuracy evaluation on synthetic receipts (15/15)
- [x] Consistency checks + second opinion from a stronger model
- [ ] Accuracy evaluation on 30 real receipts
- [x] Streamlit dashboard with upload, review queue and Excel export
- [ ] Dashboard on PostgreSQL (Metabase / Looker Studio)
- [ ] Gemini vs Claude accuracy/cost comparison
